"""qmv_row_blocked, Flash-Next's dense decode GEMV: a row in a batch is the row alone, and stock's accuracy.

qmv_row_blocked.py claims row exactness: a row's k-order is set by KL alone (not VECS, NSG, RB or M), so a
row computed with up to `MAX_M - 1` others is bit-identical to the row computed alone, and
`QLinear.rows` keeps that past MAX_M by splitting. Against stock `mx.quantized_matmul` (what
`dense_kernel="stock"` runs) only the summation order differs, and no tolerance is stated, so the
kernel must be as accurate as stock against an exact float64 product, in both dtypes.
"""

LM_HEAD_ROWS = 16384 + 37  # lm_head's 248,320 rows, cut: a row's arithmetic does not depend on N


def _shape(n, k):
    return (min(n, LM_HEAD_ROWS), k)


def test_rows_in_a_batch_equal_the_row_alone(requires_mlx, q4):
    """Every (shape, dtype) the config table serves, at every row count up to MAX_M."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as dk

    for (n, k, dt_name), (kl, nsg, rb) in sorted(dk.CONFIGS.items()):
        dt = getattr(mx, dt_name)
        w, s, b = q4(_shape(n, k), seed=n + k)
        x = mx.random.normal((dk.MAX_M, k), key=mx.random.key(k)).astype(dt)

        def run(xs, w=w, s=s, b=b, dt=dt, kl=kl, nsg=nsg, rb=rb):
            vecs = dk.vecs_for(xs.shape[0], dt)
            return dk.qmv_wide_rb(xs, w, s, b, vecs, kl, nsg, rb, group_size=32)

        solo = mx.concatenate([run(x[i : i + 1]) for i in range(dk.MAX_M)])
        for m in range(2, dk.MAX_M + 1):
            batch = run(x[:m])
            assert mx.array_equal(batch, solo[:m]).item(), (n, k, dt_name, m)


def test_qlinear_rows_past_max_m_equal_each_row_alone(requires_mlx, q4):
    """The decode path: `QLinear.rows` over 12 rows of [B, 1, K] splits at MAX_M, rows unchanged."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear

    for dt in (mx.float32, mx.bfloat16):
        lin = QLinear(*q4((6144, 2560), seed=7), 32, 4)
        served = lin.enable_dense_kernel("rb", dt)
        assert served, dt
        x = mx.random.normal((12, 1, 2560), key=mx.random.key(8)).astype(dt)
        solo = mx.concatenate([lin(x[i : i + 1]) for i in range(12)])
        rows = lin.rows(x)
        assert mx.array_equal(rows, solo).item(), dt


def test_as_accurate_as_stock_quantized_matmul(requires_mlx, q4, accuracy):
    """Every (shape, dtype) the table serves, at 1 and MAX_M rows. Stock is what the model runs
    with the kernel off: float32-stored metadata at float32, bfloat16 metadata at bfloat16."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as dk

    for (n, k, dt_name), (kl, nsg, rb) in sorted(dk.CONFIGS.items()):
        dt = getattr(mx, dt_name)
        w, s, b = q4(_shape(n, k), seed=3 * n + k)
        exact_w = accuracy.weights(w, s, b)
        for m in (1, dk.MAX_M):
            x = mx.random.normal((m, k), key=mx.random.key(n + m)).astype(dt)
            got = dk.qmv_wide_rb(x, w, s, b, dk.vecs_for(m, dt), kl, nsg, rb, group_size=32)
            stock = mx.quantized_matmul(
                x,
                w,
                scales=s.astype(dt),
                biases=b.astype(dt),
                transpose=True,
                group_size=32,
                bits=4,
            )
            exact = accuracy.np64(x) @ exact_w.T
            accuracy.no_worse_than_stock(got, stock, exact, what=f"{n}x{k} {dt_name} M={m}")
