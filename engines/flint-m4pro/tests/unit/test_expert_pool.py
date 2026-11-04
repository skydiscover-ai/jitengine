"""The expert pool (offload mode) against the resident banks, on a small synthetic checkpoint: the MoE layer's output
from pooled experts is bit-identical to the one from stacked expert tensors, through both gathers (decode's unsorted
one and prefill's sorted one), both landings, and evictions. Runs on whatever device MLX defaults to (the CPU on the
Linux CI, the GPU on a Mac)."""

from types import SimpleNamespace

import numpy as np
import pytest

from jitengine_flint_m4pro import expert_pool
from jitengine_flint_m4pro.expert_pool import ExpertPool
from jitengine_flint_m4pro.loader import QLinear
from jitengine_flint_m4pro.model import model as graph

H, INTER, E, LAYERS = 256, 128, 16, (2, 3)
PROJ = (("gate_proj", INTER, H), ("up_proj", INTER, H), ("down_proj", H, INTER))


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def make_checkpoint(tmp_path, seed=0):
    """A raw file holding each routed layer's nine [E, ...] tensors, its tensor index (config.tensor_index's form) and
    the same tensors as numpy arrays."""
    rng = np.random.default_rng(seed)
    path = tmp_path / "experts.bin"
    index, arrays, blobs, off = {}, {}, [], 0
    for layer in LAYERS:
        for proj, rows, cols in PROJ:
            parts = {
                "weight": ("U8", rng.integers(0, 256, size=(E, rows, cols // 2), dtype=np.uint8)),
                "scales": ("BF16", _bf16_bits(rng.normal(0, 0.02, (E, rows, cols // 64)))),
                "biases": ("BF16", _bf16_bits(rng.normal(0, 0.01, (E, rows, cols // 64)))),
            }
            for suffix, (dt, arr) in parts.items():
                name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                raw = arr.tobytes()
                index[name] = (str(path), dt, list(arr.shape), off, len(raw))
                arrays[name] = arr
                blobs.append(raw)
                off += len(raw)
    path.write_bytes(b"".join(blobs))
    cfg = SimpleNamespace(
        moe_layers=LAYERS, n_experts=E, n_layers=4, bits=4, group_size=64, top_k=6, rms_eps=1e-6
    )
    return cfg, index, arrays


def stacked(arrays, layer):
    """The resident form: one [E, ...] QLinear per projection."""
    import mlx.core as mx

    out = []
    for proj, _, _ in PROJ:
        p = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        w = mx.array(arrays[p + ".weight"].view(np.uint32))
        s, b = (mx.array(arrays[f"{p}.{k}"]).view(mx.bfloat16) for k in ("scales", "biases"))
        out.append(QLinear(w, s, b, 4, 64))
    return tuple(out)


def layer_weights(arrays, layer, seed=1):
    import mlx.core as mx

    rng = np.random.default_rng(seed + layer)
    router = mx.array(_bf16_bits(rng.normal(0, 0.05, (E, H)))).view(mx.bfloat16)
    shared = tuple(QLinear(q.w[0], q.s[0], q.b[0], 4, 64) for q in stacked(arrays, layer))
    return SimpleNamespace(
        index=layer, dense=None, router=router, experts=stacked(arrays, layer), shared=shared
    )


def bits(a) -> np.ndarray:
    import mlx.core as mx

    return np.array(a.view(mx.uint16))


@pytest.mark.parametrize("slots", [E, 3 * E])
@pytest.mark.parametrize("landing", ["auto", "scatter"])
def test_pooled_moe_is_bit_identical(tmp_path, monkeypatch, landing, slots):
    import mlx.core as mx

    if landing == "scatter":
        monkeypatch.setattr(expert_pool, "_writable_host_view", lambda a: None)
    cfg, index, arrays = make_checkpoint(tmp_path)
    # slots = E: every switch of layer evicts; 3 E: more slots than experts, so a gather over the pool's arrays
    # would see other rows per expert than the banks' (MLX picks the sorted gather's kernel from that, on Metal)
    pool = ExpertPool(cfg, index, slots=slots, log=lambda m: None)
    assert landing == "auto" or pool.landing == "scatter"
    pool.selfcheck()
    resident = SimpleNamespace(cfg=cfg, pool=None)
    pooled = SimpleNamespace(cfg=cfg, pool=pool)
    rng = np.random.default_rng(7)
    try:
        for step in range(6):
            layer = LAYERS[step % 2]
            lw = layer_weights(arrays, layer)
            for n in (1, 16, 64):  # 6 rows: the unsorted gather; 96 and 384 rows: the sorted one
                x = mx.array(_bf16_bits(rng.normal(0, 1, (1, n, H)))).view(mx.bfloat16)
                want = graph._mlp(resident, lw, x)
                got = graph._mlp(pooled, SimpleNamespace(**{**vars(lw), "experts": None}), x)
                mx.eval(want, got)
                assert np.array_equal(bits(want), bits(got)), (step, layer, n)
        st = pool.stats()
        assert st["misses"] > 0 and st["hits"] > 0 and (st["evictions"] > 0) == (slots == E), st
        assert st["read_bytes"] == st["misses"] * pool.record_bytes
        assert st["landing"] == pool.landing and pool.resident() <= min(slots, len(LAYERS) * E)
    finally:
        pool.close()


def test_pool_bookkeeping(tmp_path):
    """Slots map both ways, protected slots are never victims, a failed read publishes nothing."""
    cfg, index, _ = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="at least"):
        ExpertPool(cfg, index, slots=E - 1, log=lambda m: None)
    pool = ExpertPool(cfg, index, slots=2 * E, log=lambda m: None)
    try:
        a = np.arange(E, dtype=np.int64) + LAYERS[0] * E
        b = np.arange(E, dtype=np.int64) + LAYERS[1] * E
        sa, sb = pool.ensure(a), pool.ensure(b)
        assert len(set(sa) | set(sb)) == 2 * E and pool.resident() == 2 * E
        assert np.array_equal(pool.ensure(a), sa) and pool.stats()["hits"] == E
        assert all(pool.id_of_slot[s] == g for s, g in zip(sa, a))
        broken = SimpleNamespace(**{**vars(cfg)})
        pool2 = ExpertPool(broken, index, slots=E, log=lambda m: None)
        pool2.ensure(a[:4])

        def fail(gid, dsts):
            raise OSError("injected")

        pool2._read_record = fail
        with pytest.raises(OSError, match="injected"):
            pool2.ensure(b[:4])
        assert (pool2.slot_for_id[b[:4]] < 0).all() and pool2.resident() == 4
        pool2.close()
    finally:
        pool.close()
