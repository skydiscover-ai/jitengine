"""No landing slot is lost to a fetch that raises while its reads are in flight.

The landing region is fixed in size: every slot is free, parked until its write ends, or owned by a
fetch in progress. A raise after the reads were submitted (`during`, the overlap dispatch, the
victim choice) or after a split `ensure` returned and before `ensure_finish` (the caller's
correction or lookahead rebuild) must leave its slots in one of those, or every later demand miss
would read synchronously. On stand-in stores, no Metal kernel."""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import OffloadMoeCache  # noqa: E402

NAMES = [
    f"{p}.{t}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for t in ("weight", "scales", "biases")
]


class _Store:
    """Fills every byte of a record with a marker naming its id; demand reads land at once."""

    def read_many(self, jobs):
        for gid, bufs in jobs:
            for b in bufs:
                b.view(np.uint8).fill(gid % 251 + 1)
        return 0.0

    def read_into(self, gid, bufs):
        self.read_many([(gid, bufs)])

    def submit_demand(self, gid, bufs, read):
        fut = Future()
        read(gid, bufs)
        fut.set_result(None)
        return fut

    def stats(self):
        return {}


def _pool(slots=160, experts=100, layers=2, **options):
    """Two layers of 100 experts, so a 160-slot pool carries a landing region (`init_landing`)."""
    fields, off = [], 0
    for n in NAMES:
        w = n.endswith("weight")
        fields.append(
            {
                "name": n,
                "dtype": "U32",
                "shape": [2, 8] if w else [2, 1],
                "bytes": 64 if w else 8,
                "offset": off,
            }
        )
        off += fields[-1]["bytes"]
    return OffloadMoeCache(
        _Store(),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=64,
        **options,
    )


def _fill(pool) -> np.ndarray:
    """Make ids resident until no slot is free, so the next fetch must evict; returns 3 ids that
    are not resident."""
    g = 0
    while pool._free:
        pool.ensure(np.array([g]))
        g += 1
    return np.array([g, g + 1, g + 2])


def _region(pool):
    """Region slots free or parked: all of them unless some were lost."""
    return len(pool._landing_free) + len(pool._orphans)


def _boom():
    raise RuntimeError("[metal::malloc] simulated in the in-flight window")


@pytest.mark.parametrize("split", [False, True])
def test_a_raise_in_during_while_reads_are_in_flight_loses_no_slot(split):
    pool = _pool()
    ids = _fill(pool)  # the pool is full: the next misses take landing slots, then victims
    n, resident = pool.landing_n, int((pool.id_of_slot >= 0).sum())
    assert n >= 3 and _region(pool) == n and not pool._free
    with pytest.raises(RuntimeError, match="simulated"):
        pool.ensure(ids, during=_boom, split=split)
    assert _region(pool) == n and not pool._free and int((pool.id_of_slot >= 0).sum()) == resident
    assert (pool.slot_for_id[ids] < 0).all() and pool.open_fetches() == 0
    s = pool.stats()
    assert s["expert_landing_live"] == s["expert_landing_slots"] == n
    # and the region serves the next fetch
    before = pool.landing_fetches
    slots = pool.ensure(ids)
    assert pool.landing_fetches == before + 1 and (pool.slot_for_id[ids] == slots).all()


def test_a_split_fetch_its_caller_abandons_is_settled_and_cannot_be_finished():
    pool = _pool()
    ids = _fill(pool)
    n, free, resident = pool.landing_n, len(pool._free), int((pool.id_of_slot >= 0).sum())
    pend = pool.ensure(ids, split=True)
    assert pool.open_fetches() == 1 and _region(pool) == n - 3
    # the caller raised before `ensure_finish` (its correction, its lookahead rebuild)
    assert pool.abandon_open() == 3
    assert pool.open_fetches() == 0 and _region(pool) == n
    # the victims it chose went back empty: the pool holds what it held, minus them, and they are free
    assert len(pool._free) == free + 3 and int((pool.id_of_slot >= 0).sum()) == resident - 3
    with pytest.raises(InvariantError, match="abandon"):
        pool.ensure_finish(pend)
    assert pool.stats()["expert_landing_abandoned"] == 3


def test_a_finished_split_fetch_leaves_nothing_open():
    pool = _pool()
    ids = _fill(pool)[:2]
    pend = pool.ensure(ids, split=True)
    slots = pool.ensure_finish(pend)
    assert pool.open_fetches() == 0 and (pool.slot_for_id[ids] == slots).all()
    assert _region(pool) == pool.landing_n and pool.abandon_open() == 0


def test_a_victim_choice_that_raises_gives_back_the_free_slots_it_took():
    pool = _pool(slots=4, experts=6, layers=1, landing_slots=0)
    free = sorted(pool._free)
    with pytest.raises(RuntimeError, match="cannot hold this step"):
        pool.ensure(np.arange(6))  # 6 records into 4 slots: the free ones are taken, then no victim
    assert sorted(pool._free) == free and (pool.id_of_slot[free] == -1).all()
