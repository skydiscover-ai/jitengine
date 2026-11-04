"""Decode attention against MLX's scaled_dot_product_attention."""

from __future__ import annotations


def test_sdpa_decode_bitwise_vs_stock(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import sdpa_decode

    for n in (7, 300, 1023, 1024, 1500, 4095, 4096, 6000):
        q = mx.random.normal((1, 16, 1, 256), key=mx.random.key(n)).astype(mx.bfloat16)
        kb = mx.random.normal((1, 2, n + 64, 256), key=mx.random.key(n + 1)).astype(mx.bfloat16)
        vb = mx.random.normal((1, 2, n + 64, 256), key=mx.random.key(n + 2)).astype(mx.bfloat16)
        want = mx.fast.scaled_dot_product_attention(
            q, kb[:, :, :n], vb[:, :, :n], scale=0.0625, mask=None
        )
        assert mx.array_equal(sdpa_decode(q, kb, vb, n, 0.0625), want).item(), n


def test_two_pass_rows_bitwise_vs_stock(requires_mlx):
    """The two-pass rows kernel at grouped-query lengths, where stock runs its own two passes on any
    GPU, at stock's block count for each length: every row is stock's bits."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import (
        mlx_2pass_blocks,
        sdpa_decode_rows_2pass,
    )

    for n in (4096, 6000, 9000, 20000):
        qs = mx.random.normal((3, 16, 1, 256), key=mx.random.key(n)).astype(mx.bfloat16)
        ks = [
            mx.random.normal((1, 2, n + 32 * r, 256), key=mx.random.key(n + 10 + r)).astype(
                mx.bfloat16
            )
            for r in range(3)
        ]
        vs = [
            mx.random.normal((1, 2, n + 32 * r, 256), key=mx.random.key(n + 20 + r)).astype(
                mx.bfloat16
            )
            for r in range(3)
        ]
        got = sdpa_decode_rows_2pass(qs, ks, vs, [n] * 3, 0.0625, mlx_2pass_blocks(n, 8))
        for r in range(3):
            want = mx.fast.scaled_dot_product_attention(
                qs[r : r + 1], ks[r][:, :, :n], vs[r][:, :, :n], scale=0.0625, mask=None
            )
            assert mx.array_equal(got[r : r + 1], want).item(), (n, r)
