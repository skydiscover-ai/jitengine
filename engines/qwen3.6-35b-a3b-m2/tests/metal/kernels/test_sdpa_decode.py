"""Decode SDPA: the same bits as mx.fast.scaled_dot_product_attention at every length."""


def test_sdpa_decode_bitwise_vs_stock(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels.sdpa_decode import decode

    for n in (7, 300, 1023, 1024, 1500, 4095, 4096, 6000):
        q = mx.random.normal((1, 16, 1, 256), key=mx.random.key(n)).astype(mx.bfloat16)
        kb = mx.random.normal((1, 2, n + 64, 256), key=mx.random.key(n + 1)).astype(mx.bfloat16)
        vb = mx.random.normal((1, 2, n + 64, 256), key=mx.random.key(n + 2)).astype(mx.bfloat16)
        want = mx.fast.scaled_dot_product_attention(
            q, kb[:, :, :n], vb[:, :, :n], scale=0.0625, mask=None
        )
        assert mx.array_equal(decode(q, kb, vb, n, 0.0625), want).item(), n
