"""The large-tile prefill qmm and the QLinear route to it: the same bits as stock MLX."""


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_dense_prefill_qmm_bitwise_vs_stock(requires_mlx):
    """Large-tile prefill qmm: the same bits as mx.quantized_matmul wherever stock is not split-K."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import qmm_prefill as dp

    for N, K in ((4096, 2048), (2048, 4096), (2048, 512)):
        w, s, b = _qz(mx, (N, K), 3, 7 * N + K)
        for M in (192, 200, 231, 256, 300):
            if not dp.eligible(M, N, K, 3):
                continue
            x = mx.random.normal((M, K), key=mx.random.key(M + N)).astype(mx.bfloat16)
            want = mx.quantized_matmul(
                x, w, scales=s, biases=b, transpose=True, group_size=64, bits=3
            )
            assert mx.array_equal(dp.qmm_t(x, w, s, b, 3, 64), want).item(), (N, K, M)


def test_qlinear_routes_prefill_rows_and_keeps_stock_bits(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2 import linear as Q

    lin = Q.QLinear(*_qz(mx, (4096, 2048), 3, 11), 64)
    x = mx.random.normal((1, 256, 2048), key=mx.random.key(12)).astype(mx.bfloat16)
    assert lin.calibrate_pf()
    assert mx.array_equal(lin(x), lin._stock(x)).item()
