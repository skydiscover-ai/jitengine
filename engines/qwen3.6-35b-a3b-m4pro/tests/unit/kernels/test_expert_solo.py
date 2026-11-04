"""The exact expert kernels' coverage test (no Metal needed)."""

from __future__ import annotations


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_expert_exact_unsupported_shapes_fall_back(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_solo as ee
    from jitengine_qwen3_6_35b_a3b_m4pro.linear import QLinear

    gu = QLinear(*_qz(mx, (4, 128, 128), 4, 3), 64)  # K = 128: not a qmv_fast shape
    dn = QLinear(*_qz(mx, (4, 128, 64), 4, 4), 64)
    assert not ee.supported(gu, dn, mx.bfloat16)
