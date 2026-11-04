"""A layer whose distinct experts exceed the pool is served in waves, fetched one at a time."""

import numpy as np
import pytest

pytest.importorskip("mlx.core")

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


def _pool(slots, experts):
    from jitengine_qwen3_6_35b_a3b_m4pro.moe.offload_cache import OffloadMoeCache

    fields = [
        {
            "name": n,
            "dtype": "U32",
            "shape": [2, 8] if n.endswith("weight") else [2, 1],
            "bytes": 64 if n.endswith("weight") else 8,
            "offset": 0,
        }
        for n in NAMES
    ]
    off = 0
    for f in fields:
        f["offset"], off = off, off + f["bytes"]
    return OffloadMoeCache(
        _Store(), num_layers=1, num_experts=experts, slots=slots, fields=fields, group_size=64
    )


def _holds(pool, slot):
    return int(np.array(pool.gate_proj.weight[int(slot)]).view(np.uint8).reshape(-1)[0]) - 1


def test_each_wave_still_holds_its_experts_when_reached():
    pool = _pool(slots=2, experts=6)
    ids = np.array([0, 1, 2, 3, 4, 5, 0, 3])
    segs = pool.segments(0, ids)
    assert len(segs) == 3
    seen = []
    for pos, slots in segs:
        assert [_holds(pool, s) for s in slots] == ids[pos].tolist()
        seen += pos.tolist()
    assert sorted(seen) == list(range(ids.size))
    assert pool.wave_calls == 1


def test_a_layer_that_fits_is_one_wave():
    pool = _pool(slots=8, experts=6)
    ids = np.array([5, 1, 5, 2])
    ((pos, slots),) = pool.segments(0, ids)
    assert pos.tolist() == [0, 1, 2, 3]
    assert [_holds(pool, s) for s in slots] == ids.tolist()
    assert pool.wave_calls == 0
