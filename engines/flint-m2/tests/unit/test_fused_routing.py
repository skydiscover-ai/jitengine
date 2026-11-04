"""The decode MoE's compiled combine (routing.combine_fused) and fused route are bit-identical to the eager ops: over every bf16
value as an expert output or logit, and over draws that span bf16 magnitudes. The route tail's float ops stay eager (a compiled fp32
sigmoid is 1 ulp off on some bf16 logits)."""

import numpy as np
import pytest

from jitengine_flint_m2.model import routing

H, K, E = 2048, 6, 256


def _bits(a):
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _same(a, b):
    import mlx.core as mx

    nan = np.isnan(np.array(a.astype(mx.float32))) & np.isnan(np.array(b.astype(mx.float32)))
    return np.array_equal(np.where(nan, 0, _bits(a)), np.where(nan, 0, _bits(b)))


def test_combine_fused_is_bit_identical_on_every_bf16_value():
    import mlx.core as mx

    rng = np.random.default_rng(0)
    allv = np.arange(65536, dtype=np.uint16)
    for r in range(
        4
    ):  # every value in every rank position, against weights drawn over all bf16 patterns too
        y = np.stack(
            [np.roll(allv, r * 4099 + j * 7919).reshape(16, 1, 4096) for j in range(K)], axis=2
        )
        y = mx.array(y).view(
            mx.bfloat16
        )  # [16, 1, K, 4096]: each rank position holds all 65,536 values once
        w = mx.array(rng.integers(0, 65536, (16, 1, K), dtype=np.uint16)).view(mx.bfloat16)
        c_e, c_f = routing.combine(y, w), routing.combine_fused(y, w)
        mx.eval(c_e, c_f)
        assert _same(c_e, c_f)


@pytest.mark.parametrize("seed", range(8))
def test_combine_fused_is_bit_identical_on_routed_draws(seed):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    for _ in range(25):
        scale = rng.choice([1e-3, 0.1, 1.0, 8.0, 60.0])
        x = mx.array((rng.normal(0, 1, (1, 1, H)) * scale).astype(np.float32)).astype(mx.bfloat16)
        w = mx.array(rng.normal(0, 0.05, (E, H)).astype(np.float32)).astype(mx.bfloat16)
        if rng.random() < 0.3:  # exact score ties between experts
            w = mx.concatenate([w[:8], w[:8], w[16:]], axis=0)
        _, w_e = routing.route(x, w, K)
        y = mx.array((rng.normal(0, 1, (1, 1, K, H)) * scale).astype(np.float32)).astype(
            mx.bfloat16
        )
        c_e, c_f = routing.combine(y, w_e), routing.combine_fused(y, w_e)
        mx.eval(c_e, c_f)
        assert _same(c_e, c_f)


def test_fused_route_is_bit_identical_on_every_bf16_logit():
    """The fused route compiles only the integer select; one-row calls whose logits cover every bf16 value, in order
    and shuffled, give the eager ids and weights bit for bit."""
    import mlx.core as mx

    rng = np.random.default_rng(0)
    allv = np.arange(65536, dtype=np.uint16)
    for v in (allv, rng.permutation(allv)):
        lg = mx.array(v.reshape(256, 1, 256)).view(mx.bfloat16)
        for i in range(256):
            a, b = (
                routing._route_tail(lg[i : i + 1], K),
                routing._route_tail(lg[i : i + 1], K, fused=True),
            )
            mx.eval(*a, *b)
            assert np.array_equal(_bits(a[0]), _bits(b[0])) and _same(a[1], b[1])
