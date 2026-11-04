"""The cycle-aware victim rule beside the annex: releasing the annex drops its records, so every layer
head the rule cached may name a slot that no longer holds that layer's record. The release stales the
heads, as the extension's does, and the next pick after a regrow takes a resident record, never an
empty annex slot (an empty slot keys +inf, so a stale head naming one would be picked first)."""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import OffloadMoeCache  # noqa: E402

NAMES = [
    f"{p}.{t}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for t in ("weight", "scales", "biases")
]


class _Store:
    """Fills every byte of a record with a marker naming its id; reads land at once."""

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

    def submit(self, gid, bufs):
        return self.submit_demand(gid, bufs, self.read_into)

    def stats(self):
        return {}


def _pool(**options):
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
        _Store(), num_layers=4, num_experts=60, fields=fields, group_size=64, **options
    )


def _bad_heads(pool) -> list:
    """Layers whose cached head is live (not stale) but names a slot that holds no record of that layer."""
    E = pool.num_experts
    bad = []
    for lay in range(pool.num_layers):
        h = int(pool._cyc_head[lay])
        if pool._cyc_stale[lay] or h < 0:
            continue
        g = int(pool.id_of_slot[h])
        if g < 0 or g // E != lay:
            bad.append(lay)
    return bad


@pytest.mark.parametrize("planted", [False, True], ids=["released", "planted-no-stale"])
def test_releasing_the_annex_stales_the_cycle_aware_heads(monkeypatch, planted):
    """Single-row decode under the cycle-aware rule fills main, extension and annex; then the annex is
    released and regrown. Released: no live head names a dropped record's slot, and the next pick is a
    resident. Planted (the release without the stale, i.e. the annex's own code before composition):
    both checks fail, so the test sees the hazard."""
    pool = _pool(slots=40, ext_slots=10, ann_slots=100, decay_tokens=32, landing_slots=0)
    assert pool.regrow_ann() == 100
    E, L = pool.num_experts, pool.num_layers
    rng = np.random.default_rng(3)
    w = 1.0 / np.arange(1, E + 1) ** 0.8
    w /= w.sum()
    pool.cyc_tokens = 64
    for t in range(400):  # 240 ids over 150 slots: the rule picks victims all along
        lay = t % L
        ids = np.sort(rng.choice(E, size=6, replace=False, p=w)).astype(np.int64)
        ((_pos, slots),) = pool.segments_finish(pool.segments_begin(lay, ids))
        assert all(int(pool.id_of_slot[s]) == lay * E + int(i) for s, i in zip(slots, ids))
        pool.reader_barrier()
    assert pool.evictions_cyc > 0
    pool._cyc_pick(np.zeros(pool.slots, dtype=bool))  # every head current, none stale
    assert not pool._cyc_stale.any() and not _bad_heads(pool)
    in_ann = [lay for lay in range(L) if pool._cyc_head[lay] >= pool.ann0]
    assert in_ann, "no head in the annex: the scenario did not arise"
    pool.cyc_tokens = 0  # the annex is released between requests, outside the armed decode
    if planted:
        monkeypatch.setattr(type(pool), "cyc_invalidate", lambda self, gids=None: None)
    assert pool.release_ann() > 0
    monkeypatch.undo()
    bad = _bad_heads(pool)
    assert pool.regrow_ann() == 100 and pool._free  # empty, unprotected annex slots
    pool.cyc_tokens = 64
    prot = pool._protect_unusable_slots(np.zeros(pool.slots, dtype=bool))
    v = pool._cyc_pick(prot)
    if planted:
        assert set(bad) == set(in_ann)
        assert v >= pool.ann0 and pool.id_of_slot[v] < 0  # an empty annex slot as the victim
    else:
        assert not bad
        assert v >= 0 and pool.id_of_slot[v] >= 0 and not prot[v]
