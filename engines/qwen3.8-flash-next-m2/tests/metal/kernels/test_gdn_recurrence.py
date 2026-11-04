"""The GatedDeltaNet recurrence as Flash-Next runs it: the staged prefill variant and the decode one.

gdn_recurrence.py claims `gdn_recurrence_staged` (Flash-Next's prefill, `tb=8`) is bit-identical to
`gdn_recurrence`: the same operations, operands and order with the inputs staged in threadgroup
memory. `gdn_recurrence_ops` is the plain-ops definition (`GDN_IMPL = "ops"`); against it the
kernel only differs in summation order (simd_sum trees), so it must be as accurate as the ops
against an exact float64 run of the same recurrence. Geometry is Flash-Next's: 16 key heads and 48
value heads of 128, float32 activations and state.
"""

import numpy as np

HK, HV, D = 16, 48, 128


def _inputs(mx, T, seed):
    """q and k l2-normalised and scaled as the block does, g in (0, 1), beta in (0, 1)."""
    key = lambda i: mx.random.key(seed + i)  # noqa: E731
    q = mx.fast.rms_norm(mx.random.normal((1, T, HK, D), key=key(0)), None, 1e-6 / D) / D
    k = mx.fast.rms_norm(mx.random.normal((1, T, HK, D), key=key(1)), None, 1e-6 / D) * D**-0.5
    v = mx.random.normal((1, T, HV, D), key=key(2))
    g = mx.exp(-mx.exp(0.3 * mx.random.normal((1, T, HV), key=key(3))) * 0.5)
    beta = mx.sigmoid(mx.random.normal((1, T, HV), key=key(4)))
    state = 0.1 * mx.random.normal((1, HV, D, D), key=key(5))
    return q, k, v, g, beta, state


def test_staged_equals_the_single_launch(requires_mlx):
    """Prefill lengths below, at and across the staging tile: Flash-Next's tile (8) and the default."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.gdn_recurrence import (
        gdn_recurrence,
        gdn_recurrence_staged,
    )

    for T in (1, 7, 8, 9, 33, 100):
        for tb in (8, 16):
            q, k, v, g, beta, state = _inputs(mx, T, seed=T)
            y, s = gdn_recurrence(q, k, v, g, beta, state)
            ys, ss = gdn_recurrence_staged(q, k, v, g, beta, state, tb=tb)
            assert mx.array_equal(ys, y).item() and mx.array_equal(ss, s).item(), (T, tb)


def _exact(accuracy, q, k, v, g, beta, state):
    q, k, v, g, beta, S = (accuracy.np64(a) for a in (q, k, v, g, beta, state))
    rep = HV // HK
    q, k = np.repeat(q, rep, axis=2), np.repeat(k, rep, axis=2)
    ys = []
    for t in range(q.shape[1]):
        S = S * g[:, t, :, None, None]
        kv = (S * k[:, t, :, None, :]).sum(-1)
        d = (v[:, t] - kv) * beta[:, t, :, None]
        S = S + k[:, t, :, None, :] * d[..., None]
        ys.append((S * q[:, t, :, None, :]).sum(-1))
    return np.stack(ys, axis=1), S


def test_kernel_as_accurate_as_the_plain_ops(requires_mlx, accuracy):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.gdn_recurrence import (
        gdn_recurrence,
        gdn_recurrence_ops,
    )

    for T in (1, 16):
        q, k, v, g, beta, state = _inputs(mx, T, seed=50 + T)
        y, s = gdn_recurrence(q, k, v, g, beta, state)
        yo, so = gdn_recurrence_ops(q, k, v, g, beta, state)
        ye, se = _exact(accuracy, q, k, v, g, beta, state)
        accuracy.no_worse_than_stock(y, yo, ye, what=f"y T={T}")
        accuracy.no_worse_than_stock(s, so, se, what=f"state T={T}")
