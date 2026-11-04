"""The prescaled multi-row GEMVs (rows in registers, rows through L1): each row stock MLX's bits."""


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_dense_v4_rows_bitwise_vs_stock_solo(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_prescaled import qmv_rows_in_registers

    for N, K in ((1024, 2048), (512, 4096), (2048, 512)):
        w, s, b = _qz(mx, (N, K), 3, N + K)
        x = mx.random.normal((5, K), key=mx.random.key(K)).astype(mx.bfloat16)
        solo = mx.concatenate(
            [
                mx.quantized_matmul(
                    x[i : i + 1], w, scales=s, biases=b, transpose=True, group_size=64, bits=3
                )
                for i in range(5)
            ]
        )
        for M in range(1, 6):
            for mb in (1, 2, 3):
                got = qmv_rows_in_registers(x[:M], w, s, b, mb=mb)
                assert mx.array_equal(got, solo[:M]).item(), (N, K, M, mb)


def test_v10_rows_bitwise_vs_stock_solo(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_prescaled import qmv_rows_in_l1

    for N, K in ((1024, 2048), (512, 4096), (2048, 512)):
        w, s, b = _qz(mx, (N, K), 3, 3 * N + K)
        x = mx.random.normal((6, K), key=mx.random.key(K + 1)).astype(mx.bfloat16)
        solo = mx.concatenate(
            [
                mx.quantized_matmul(
                    x[i : i + 1], w, scales=s, biases=b, transpose=True, group_size=64, bits=3
                )
                for i in range(6)
            ]
        )
        for M in (2, 4, 5, 6):
            assert mx.array_equal(qmv_rows_in_l1(x[:M], w, s, b), solo[:M]).item(), (N, K, M)
