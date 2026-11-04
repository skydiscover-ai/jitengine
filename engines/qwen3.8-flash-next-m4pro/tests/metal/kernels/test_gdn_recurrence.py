"""The GatedDeltaNet recurrence as Flash-Next runs it: the staged prefill variant and the decode one.

gdn_recurrence.py claims `gdn_recurrence_staged` (Flash-Next's prefill, `tb=8`) is bit-identical to
`gdn_recurrence`: the same operations, operands and order with the inputs staged in threadgroup
memory. `gdn_recurrence_ops` is the plain-ops definition (`GDN_IMPL = "ops"`); against it the
kernel only differs in summation order (simd_sum trees), so it must be as accurate as the ops
against an exact float64 run of the same recurrence. Geometry is Flash-Next's: 16 key heads and 48
value heads of 128, float32 activations and state. `gdn_recurrence_pair` (the MTP verify pass's
two rows in one launch, `GDN_LAZY`) claims two `gdn_recurrence_step` launches' bits.
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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import (
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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import (
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


def _rows(mx, a, r):
    """Row r of [1, T, ...] inputs, unexpanded as `gdn_recurrence_step` takes them."""
    return tuple(x[0, r] for x in a)


def _bits(mx, a):
    return np.asarray(a.view(mx.uint32 if a.dtype.size == 4 else mx.uint16)).tobytes()


def test_pair_is_two_single_launches_bit_for_bit(requires_mlx):
    """On 36 distinct states (one per GDN layer of the model, as one verify pass sees them), each
    advanced by real steps first, and at both compute dtypes: y0, y1 and the final state are the
    two single launches' bits. A one-ulp change to row 1's decay is caught (the comparison sees
    a single rounding), and a non-float32 state is refused."""
    import mlx.core as mx
    import pytest

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import (
        gdn_recurrence,
        gdn_recurrence_pair,
        gdn_recurrence_step,
    )

    dims = (HK, D, HV, D)
    for dt in (mx.float32, mx.bfloat16):
        for layer in range(36):
            q, k, v, g, beta, state = _inputs(mx, 6, seed=1000 + 10 * layer)
            q, k, v, beta = (x.astype(dt) for x in (q, k, v, beta))
            _, state = gdn_recurrence(q[:, :4], k[:, :4], v[:, :4], g[:, :4], beta[:, :4], state)
            a = (q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:])
            r0, r1 = _rows(mx, a, 0), _rows(mx, a, 1)
            y0, s0 = gdn_recurrence_step(*r0, state, *dims)
            y1, s1 = gdn_recurrence_step(*r1, s0, *dims)
            got = gdn_recurrence_pair(r0, r1, state, *dims)
            mx.eval(y0, y1, s1, *got)
            for name, x, ref in zip(("y0", "y1", "state"), got, (y0, y1, s1), strict=True):
                assert x.dtype == ref.dtype and _bits(mx, x) == _bits(mx, ref), (dt, layer, name)
            if layer == 0 and dt == mx.float32:
                g1 = mx.array(np.nextafter(np.asarray(r1[3]), np.float32(2)).astype(np.float32))
                bad = gdn_recurrence_pair(r0, (*r1[:3], g1, r1[4]), state, *dims)
                assert _bits(mx, bad[0]) == _bits(mx, y0), "row 0 does not read row 1's decay"
                assert _bits(mx, bad[1]) != _bits(mx, y1) and _bits(mx, bad[2]) != _bits(mx, s1)
    with pytest.raises(ValueError, match="float32 state"):
        gdn_recurrence_pair(r0, r1, state.astype(mx.bfloat16), *dims)
