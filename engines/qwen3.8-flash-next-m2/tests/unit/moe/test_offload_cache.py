"""A layer whose distinct experts exceed the pool is served in waves, fetched one at a time; a
fetch that fails gives back every slot it took and publishes nothing."""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m2.moe.offload_cache import OffloadMoeCache  # noqa: E402

NAMES = [
    f"{p}.{t}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for t in ("weight", "scales", "biases")
]


class _Store:
    """Fills every byte of a record with a marker naming its id."""

    def read_many(self, jobs):
        for gid, bufs in jobs:
            for b in bufs:
                b.view(np.uint8).fill(gid % 251 + 1)
        return 0.0

    def stats(self):
        return {}


class _Faulty(_Store):
    """`_Store`, failing as told: `read_many` raises `sync`; the demand read submitted after
    `submit_at` good ones raises `submit`; and every demand read's future ends in `land`."""

    def __init__(self, sync=None, submit=None, submit_at=0, land=None):
        self.sync, self.submit, self.submit_at, self.land, self.calls = (
            sync,
            submit,
            submit_at,
            land,
            0,
        )

    def read_many(self, jobs):
        if self.sync is not None:
            raise self.sync
        return super().read_many(jobs)

    def read_into(self, gid, bufs):
        super().read_many([(gid, bufs)])

    def submit_demand(self, gid, bufs, read):
        self.calls += 1
        if self.submit is not None and self.calls > self.submit_at:
            raise self.submit
        fut = Future()
        if self.land is None:
            read(gid, bufs)
            fut.set_result(None)
        else:
            fut.set_exception(self.land)
        return fut


def _pool(slots, experts, store=None, layers=1, **options):
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
        store or _Store(),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=64,
        **options,
    )


def _holds(pool, slot):
    return int(np.array(pool.gate_proj.weight[int(slot)]).view(np.uint8).reshape(-1)[0]) - 1


def test_each_wave_still_holds_its_experts_when_reached():
    pool = _pool(slots=2, experts=6)
    ids = np.array([0, 1, 2, 3, 4, 5, 0, 3])
    segs = pool.segments(0, ids, overlap=(mx.zeros(1),))
    assert len(segs) == 3
    seen = []
    for pos, slots in segs:
        assert [_holds(pool, s) for s in slots] == ids[pos].tolist()
        pool.reader_barrier()  # the caller's eval of this wave, then its barrier
        seen += pos.tolist()
    assert sorted(seen) == list(range(ids.size))
    assert pool.lookups == ids.size and pool.hits + pool.misses + pool.dedup_saved == ids.size


def test_a_layer_that_fits_is_one_wave():
    pool = _pool(slots=8, experts=6)
    ids = np.array([5, 1, 5, 2])
    ((pos, slots),) = pool.segments(0, ids)
    assert pos.tolist() == [0, 1, 2, 3]
    assert [_holds(pool, s) for s in slots] == ids.tolist()
    assert pool.wave_calls == 0


def test_a_failed_synchronous_fill_publishes_nothing_and_frees_its_slots():
    store = _Faulty(sync=RuntimeError("bad read"))
    pool = _pool(slots=4, experts=6, store=store, landing_slots=0)
    free = sorted(pool._free)
    with pytest.raises(RuntimeError, match="bad read"):
        pool.ensure(np.array([0, 1]))
    assert (pool.slot_for_id[[0, 1]] < 0).all() and sorted(pool._free) == free
    assert (pool.id_of_slot[free] == -1).all()


def test_a_failed_submit_parks_the_submitted_read_and_returns_the_other_slots():
    """Two layers of 100 experts, so a 160-slot pool carries a landing region (`init_landing`)."""
    pool = _pool(
        slots=160, experts=100, layers=2, store=_Faulty(submit=OSError("queue full"), submit_at=1)
    )
    region = len(pool._landing_free)
    with pytest.raises(OSError, match="queue full"):
        pool.ensure(np.array([0, 1, 2]))
    assert len(pool._orphans) == 1 and len(pool._landing_free) == region - 1
    assert (pool.slot_for_id[[0, 1, 2]] < 0).all()


def test_a_bad_landed_read_returns_its_slots_to_the_region():
    pool = _pool(slots=160, experts=100, layers=2, store=_Faulty(land=RuntimeError("bad bytes")))
    region = len(pool._landing_free)
    with pytest.raises(RuntimeError, match="bad bytes"):
        pool.ensure(np.array([0, 1, 2]))
    assert len(pool._landing_free) == region and not pool._orphans
    assert (pool.slot_for_id[[0, 1, 2]] < 0).all()


def test_a_transient_landing_fault_is_read_again_and_served():
    pool = _pool(slots=160, experts=100, layers=2, store=_Faulty(land=OSError("short read")))
    region = len(pool._landing_free)
    slots = pool.ensure(np.array([0, 1, 2]))
    assert [_holds(pool, s) for s in slots] == [0, 1, 2] and pool.read_retries == 1
    assert len(pool._landing_free) == region
