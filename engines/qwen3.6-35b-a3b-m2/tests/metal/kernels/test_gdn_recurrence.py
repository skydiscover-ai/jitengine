"""The batched GDN recurrence's entry point gives every row the single-row kernel's bits."""


def test_the_rows_launch_equals_row_by_row(requires_mlx):
    """`recurrence_rows` with the rows kernel (one launch) and without it (`gdn_recurrence` per
    row), on Qwen3.6's GDN heads: the same outputs and new states, bit for bit."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels.gdn_recurrence import recurrence_rows

    B, Hk, Hv, D = 3, 16, 32, 128
    r = lambda s, k: mx.random.normal(s, key=mx.random.key(k))  # noqa: E731
    q, k = r((B, 1, Hk, D), 1).astype(mx.bfloat16), r((B, 1, Hk, D), 2).astype(mx.bfloat16)
    v = r((B, 1, Hv, D), 3).astype(mx.bfloat16)
    g, beta = mx.sigmoid(r((B, 1, Hv), 4)), mx.sigmoid(r((B, 1, Hv), 5)).astype(mx.bfloat16)
    states = [r((1, Hv, D, D), 6 + b) for b in range(B)]
    y1, s1 = recurrence_rows(q, k, v, g, beta, states, kernel=True)
    y2, s2 = recurrence_rows(q, k, v, g, beta, states, kernel=False)
    assert mx.array_equal(y1, y2).item()
    assert all(mx.array_equal(a, b).item() for a, b in zip(s1, s2))
