"""hc_decode.py, Flash-Next's decode hyper-connection in three launches (pre, down, up).

Qwen4ExpTextGatedResidual runs batched decode rows through these kernels because "each row is its
own grid.z
with the solo path's per-row arithmetic, so batched rows match solo rows bit for bit": a row in a
batch must equal the row alone, with and without the folded-in residual inject. The module says
the arithmetic is stock's "in a different, fixed reduction order" and states no tolerance, so
against the stock ops of `Qwen4ExpTextGatedResidual._decode_body`, compiled as the model compiles
them (the rms
norm, mix_down, silu, mix_up, sigmoid and stream mean, and the inject gate), the kernels must be
as accurate as stock against an exact float64 result. Geometry and launch configs are Flash-Next's: 4 streams of 2560, rank 320.
"""

import numpy as np

HC, HID, RANK, EPS = 4, 2560, 320, 1e-6


def _block(mx, q4, seed):
    from jitengine_qwen3_8_flash_next_m2.linear import QLinear

    mix_down = QLinear(*q4((RANK, HC * HID), seed=seed), 32, 4)
    mix_up = QLinear(*q4((HC * HID, RANK), seed=seed + 1), 32, 4)
    inject = QLinear(*q4((HC, HC * HID), seed=seed + 2), 32, 4)
    w1 = 1.0 + 0.1 * mx.random.normal((HC * HID,), key=mx.random.key(seed + 3))
    mx.eval(w1)  # the compiled stock path captures it (tests/metal/kernels/conftest.py)
    return mix_down, mix_up, inject, w1


def _fused(mx, h, w1, lins, block_out=None, inject_in=None):
    """`Qwen4ExpTextGatedResidual.fused` without the reshapes: the engine's launch configs."""
    from jitengine_qwen3_8_flash_next_m2.kernels.hc_decode import hc_fused
    from jitengine_qwen3_8_flash_next_m2.model import hyper_connections as hcm

    return hc_fused(
        h,
        w1,
        *lins,
        HC,
        HID,
        EPS,
        block_out=block_out,
        inject_in=inject_in,
        down_cfg=hcm.HC_FUSED_DOWN,
        up_cfg=hcm.HC_FUSED_UP,
        pre=True,
        pre_nt=hcm.HC_FUSED_PRE_NT,
    )


def _inputs(mx, rows, seed):
    h = mx.random.normal((rows, HC * HID), key=mx.random.key(seed))
    block_out = mx.random.normal((rows, HID), key=mx.random.key(seed + 1))
    inject_in = 2.0 * mx.sigmoid(mx.random.normal((rows, HC), key=mx.random.key(seed + 2)))
    mx.eval(h, block_out, inject_in)
    return h, block_out, inject_in


def test_rows_in_a_batch_equal_the_row_alone(requires_mlx, q4):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.hc_decode import supports

    md, mu, mi, w1 = _block(mx, q4, seed=0)
    assert supports(HC, HID, RANK, 32, 4, md, mu, mi)
    h, bo, ij = _inputs(mx, 8, seed=10)
    for folded in (False, True):
        extra = dict(block_out=bo, inject_in=ij) if folded else {}
        batch = _fused(mx, h, w1, (md, mu, mi), **extra)
        for i in range(8):
            one = {k: v[i : i + 1] for k, v in extra.items()}
            alone = _fused(mx, h[i : i + 1], w1, (md, mu, mi), **one)
            for name, got, want in zip(("mixed", "inject", "h2"), batch, alone):
                assert mx.array_equal(got[i : i + 1], want).item(), (folded, i, name)


def _stock(mx, h, w1, lins, block_out=None, inject_in=None):
    """`Qwen4ExpTextGatedResidual._decode_body` on each row, compiled as Qwen4ExpTextGatedResidual
    compiles it, after
    `hyper_inject` when an inject is folded in; the linears are stock `quantized_matmul` over
    float32 metadata (dense_kernel="stock")."""
    import mlx.nn as nn

    from jitengine_qwen3_8_flash_next_m2.model.hyper_connections import hyper_inject

    def lin(x, q):
        return mx.quantized_matmul(
            x,
            q.weight,
            scales=q.scales.astype(mx.float32),
            biases=q.biases.astype(mx.float32),
            transpose=True,
            group_size=32,
            bits=4,
        )

    md, mu, mi = lins

    def body(h, *folded):
        if folded:
            bo, ij = folded
            h = hyper_inject(bo[:, None], h[:, None], ij[:, None])[:, 0]
        rows = h.shape[0]
        xg = h.reshape(rows, HC, HID)
        normed = (mx.fast.rms_norm(xg, None, EPS) * w1.reshape(HC, HID)).astype(xg.dtype)
        normed = normed.reshape(rows, HC * HID)
        w = mx.sigmoid(lin(nn.silu(lin(normed, md) / HC), mu))
        mixed = mx.mean(w.reshape(rows, HC, HID) * normed.reshape(rows, HC, HID), axis=-2)
        inject = 2.0 * mx.sigmoid(lin(normed, mi) / HC)
        return mixed, inject, h

    args = (h,) if block_out is None else (h, block_out, inject_in)
    return mx.compile(body)(*args)


def _exact(accuracy, h, w1, lins, block_out=None, inject_in=None):
    """The same formulas in float64 with numpy, on the exact quantized weights."""
    md, mu, mi = (accuracy.weights(q.weight, q.scales, q.biases) for q in lins)
    h = accuracy.np64(h)
    if block_out is not None:
        bo, ij = accuracy.np64(block_out), accuracy.np64(inject_in)
        h = h + (ij[:, :, None] * bo[:, None, :]).reshape(h.shape)
    rows = h.shape[0]
    xg = h.reshape(rows, HC, HID)
    rstd = 1.0 / np.sqrt((xg * xg).mean(axis=-1, keepdims=True) + EPS)
    normed = (xg * rstd * accuracy.np64(w1).reshape(HC, HID)).reshape(rows, HC * HID)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    y = normed @ md.T / HC
    w = sig((y * sig(y)) @ mu.T)
    mixed = (w.reshape(rows, HC, HID) * normed.reshape(rows, HC, HID)).mean(axis=-2)
    inject = 2.0 * sig(normed @ mi.T / HC)
    return mixed, inject, h


def test_as_accurate_as_the_stock_ops(requires_mlx, q4, accuracy):
    import mlx.core as mx

    md, mu, mi, w1 = _block(mx, q4, seed=20)
    lins = (md, mu, mi)
    for rows in (1, 8):
        h, bo, ij = _inputs(mx, rows, seed=30 + rows)
        for folded in (False, True):
            extra = dict(block_out=bo, inject_in=ij) if folded else {}
            got = _fused(mx, h, w1, lins, **extra)
            stock = _stock(mx, h, w1, lins, **extra)
            exact = _exact(accuracy, h, w1, lins, **extra)
            for name, g, s, e in zip(("mixed", "inject", "h2"), got, stock, exact):
                accuracy.no_worse_than_stock(g, s, e, what=f"{name} rows={rows} folded={folded}")
