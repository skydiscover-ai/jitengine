"""The GDN recurrence kernels give the single-row kernel's bits: the one that stores every position's
state, and the batched rows' entry point."""

from __future__ import annotations


def test_gdn_recurrence_steps_equals_chained_single_steps(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import (
        gdn_recurrence,
        gdn_recurrence_steps,
    )

    B, T, Hk, Dk, Hv, Dv = 1, 4, 4, 128, 8, 128
    r = lambda *sh: mx.random.normal(sh).astype(mx.bfloat16)  # noqa: E731
    q, k, v = r(B, T, Hk, Dk), r(B, T, Hk, Dk), r(B, T, Hv, Dv)
    g = mx.random.uniform(shape=(B, T, Hv))
    beta = mx.random.uniform(shape=(B, T, Hv)).astype(mx.bfloat16)
    S0 = mx.random.normal((B, Hv, Dv, Dk))
    y, states = gdn_recurrence_steps(q, k, v, g, beta, S0)
    s = S0
    for t in range(T):
        yt, s = gdn_recurrence(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            g[:, t : t + 1],
            beta[:, t : t + 1],
            s,
        )
        assert mx.array_equal(y[:, t : t + 1], yt).item(), t
        assert mx.array_equal(states[t], s).item(), t


def test_the_rows_launch_equals_row_by_row(requires_mlx):
    """`recurrence_rows` with the rows kernel (one launch) and without it (`gdn_recurrence` per
    row), on Qwen3.6's GDN heads: the same outputs and new states, bit for bit."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import recurrence_rows

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
