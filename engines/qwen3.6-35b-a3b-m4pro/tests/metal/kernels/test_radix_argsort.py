"""The radix argsort equals mx.argsort."""

from __future__ import annotations


def test_radix_argsort_is_mx_argsort(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import radix_argsort

    V = 248320
    for t in range(10):
        key = mx.random.key(300 + t)
        R = 1 + t % 4
        if t % 3 == 0:
            lg = mx.random.randint(-3, 3, (R, V), key=key).astype(mx.bfloat16)  # massive ties
        elif t % 3 == 1:
            lg = (mx.random.normal((R, V), key=key) * 1e-3).astype(mx.bfloat16)  # signed zeros
        else:
            lg = (mx.random.normal((R, V), key=key) * 8).astype(mx.bfloat16)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        for a in (lg, lp):
            assert mx.array_equal(
                radix_argsort(a), mx.argsort(a, axis=-1).astype(mx.uint32)
            ).item(), t
