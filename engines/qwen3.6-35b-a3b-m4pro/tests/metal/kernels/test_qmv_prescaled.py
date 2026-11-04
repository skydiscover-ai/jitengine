"""The prescaled GEMVs (rows in registers, rows through L1, one row with the prologue inline): the stock bits."""

from __future__ import annotations


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_dense_v4_rows_bitwise_vs_stock_solo(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import qmv_rows_in_registers

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

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import qmv_rows_in_l1

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


def test_v4_inline_prologue_bitwise(requires_mlx):
    """Single-row v4 with the prologue inline == prescale + v4, for short and long rows."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import (
        qmv_inline_prescale,
        qmv_rows_in_registers,
    )

    for N, K in ((2048, 512), (512, 2048), (1024, 4096)):
        w, s, b = _qz(mx, (N, K), 3, 7 * N + K)
        for t in range(6):
            x = (mx.random.normal((1, K), key=mx.random.key(t + K)) * (0.5 + t)).astype(mx.bfloat16)
            ref = qmv_rows_in_registers(x, w, s, b, mb=1, rps=8, nsg=2, pf=1)
            assert mx.array_equal(qmv_inline_prescale(x, w, s, b, rps=8, nsg=2), ref).item(), (
                N,
                K,
                t,
            )
