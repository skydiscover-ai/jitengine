"""The subset readout on lm_head's own kernel (model.draft_vocab): real-shape slices of a 4-bit g32
lm_head (2,560 inputs, bf16 metadata) on the row-blocked kernel with lm_head's config at float32 and
bfloat16 activations. The copies' logits are lm_head's rows by bytes, and the draft is the lowest id
among the full logits' maxima over S, which is the full argmax whenever that lies in S."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as dk  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.model import draft_vocab as dv  # noqa: E402

N, K_IN, VOCAB = 8192, 2560, 248320


def _lm(dt, seed=11):
    """An N-row slice at lm_head's shape, served the way lm_head is: its (KL, NSG, RB) at `dt`."""
    w = mx.random.normal((N, K_IN), key=mx.random.key(seed)) * K_IN**-0.5
    q, s, b = mx.quantize(w, group_size=32, bits=4)
    lin = QLinear(q, s.astype(mx.bfloat16), b.astype(mx.bfloat16), 32, 4)
    cfg = dk.config_for(VOCAB, K_IN, dt)
    assert cfg is not None and dk.supports(4, 32, K_IN, True, mx.bfloat16, dt)
    lin.kernel_cfg, lin.kernel_dtype = cfg, dt
    mx.eval(lin.weight, lin.scales, lin.biases)
    return lin


def _bits(a):
    a = np.asarray(a.astype(mx.float32)).reshape(-1)
    return a.view(np.uint32)


@pytest.mark.parametrize("dt", [mx.float32, mx.bfloat16], ids=["float32", "bfloat16"])
def test_subset_rows_and_draft_on_the_row_blocked_kernel(requires_mlx, dt):
    lm = _lm(dt)
    rng = np.random.default_rng(3)
    base = np.sort(rng.choice(N, N // 8, replace=False)).astype(np.int32)
    ro = dv.SubsetReadout(lm, base, p_max=256)
    assert ro.base.kernel_cfg == lm.kernel_cfg and ro.tail.kernel_cfg == lm.kernel_cfg
    inside = outside = 0
    for i in range(24):
        n, _ = ro.prepare(rng.choice(N, 400).tolist())
        x = mx.random.normal((1, 1, K_IN), key=mx.random.key(500 + i)).astype(dt)
        full = lm(x)
        fb = _bits(full)
        assert np.array_equal(_bits(ro.base(x.reshape(1, K_IN))), fb[base])
        assert np.array_equal(_bits(ro.tail(x.reshape(1, K_IN)))[:n], fb[ro.tail_ids[:n]])
        f = np.asarray(full.astype(mx.float32)).reshape(-1)
        s_ids = np.concatenate([base, ro.tail_ids[:n]])
        v = f[s_ids]
        want = int(s_ids[v == v.max()].min())
        got = int(np.asarray(ro(x))[0])
        assert got == want, (i, got, want)
        top = int(np.asarray(dv.draft_id(lm, None, x))[0])
        if top in set(s_ids.tolist()):
            inside += 1
            assert got == top
        else:
            outside += 1
    assert inside > 0 and outside > 0, (inside, outside)


def test_a_negative_control_swapped_rows_are_caught(requires_mlx):
    """Two base rows swapped in the copy change exactly those two logits (the byte compare above can
    fail), and only there."""
    lm = _lm(mx.float32)
    base = np.arange(0, N, 4, dtype=np.int32)
    ro = dv.SubsetReadout(lm, base, p_max=16)
    x = mx.random.normal((1, K_IN), key=mx.random.key(9))
    good = _bits(ro.base(x))
    i, j = 3, 40
    for name in ("weight", "scales", "biases"):
        a = getattr(ro.base, name)
        t = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)
        t[[i, j]] = t[[j, i]]
        b = mx.array(t)
        setattr(ro.base, name, b.view(mx.bfloat16) if a.dtype == mx.bfloat16 else b)
    bad = _bits(ro.base(x))
    assert sorted(np.flatnonzero(bad != good).tolist()) == [i, j]
