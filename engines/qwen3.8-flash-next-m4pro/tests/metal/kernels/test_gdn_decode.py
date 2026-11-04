"""gdn_decode.py, Flash-Next's decode glue: qmv_multi, gdn_pre and gdn_post.

qmv_multi claims a row's value does not depend on which weight sets share the launch; that is
checked bit for bit for its three callers (the GDN projections, the attention projections with the
indexer's, and the shared expert's gate/up and SwiGLU down). The module puts the difference from
stock at "about 1e-7 relative"; the check here is the property behind that number, each launch
being as accurate as stock against an exact float64 result, since a fixed relative threshold
depends on the data (on random activations float32 stock is itself about 1e-6 of the largest
output from the exact result at K = 2560, on MLX's CPU build). Stock is what the model runs
without the glue: `QLinear` over float32 metadata for the projections, and the ops of
`Qwen4ExpTextGatedDeltaNet._decode_body` for gdn_pre and gdn_post. Geometry is Flash-Next's
(`Qwen4ExpTextConfig()`), in float32, the served dtype.
"""

import numpy as np


def _lins(q4, shapes, seed):
    """The weight sets as the model holds them at float32: the dense policy keeps bfloat16
    metadata on the shapes the row-blocked kernel serves and stores the rest as float32, so one
    launch can read both."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear, apply_dense_policy

    lins = [QLinear(*q4(shape, seed=seed + i), 32, 4) for i, shape in enumerate(shapes)]
    apply_dense_policy(lins, mx.float32, "rb")
    return lins


def _stock_rows(mx, x, lin):
    return mx.quantized_matmul(
        x,
        lin.weight,
        scales=lin.scales.astype(mx.float32),
        biases=lin.biases.astype(mx.float32),
        transpose=True,
        group_size=32,
        bits=4,
    )


def _callers():
    """(name, weight shapes, launch config, swiglu) of each qmv_multi call site in the model."""
    from jitengine_qwen3_8_flash_next_m4pro.model.attention import INPUT_PROJS_CFG
    from jitengine_qwen3_8_flash_next_m4pro.model.common import GDN_FUSED_MULTI
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    c = Qwen4ExpTextConfig()
    h = c.hidden_size
    idx = (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim
    attn = c.num_attention_heads * c.head_dim * 2, c.num_key_value_heads * c.head_dim
    gdn = c.conv_dim, c.value_dim, c.linear_num_value_heads, c.linear_num_value_heads
    shared = c.shared_expert_intermediate_size
    return (
        ("gdn in_proj qkv/z/b/a", [(n, h) for n in gdn], GDN_FUSED_MULTI, False),
        (
            "attention q/k/v/indexer",
            [(attn[0], h), (attn[1], h), (attn[1], h), (idx, h)],
            INPUT_PROJS_CFG,
            False,
        ),
        ("shared expert gate/up", [(shared, h), (shared, h)], GDN_FUSED_MULTI, False),
        ("shared expert down", [(h, shared)], GDN_FUSED_MULTI, True),
    )


def test_qmv_multi_rows_do_not_depend_on_the_other_weight_sets(requires_mlx, q4):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_decode import qmv_multi, supports_multi

    for t, (name, shapes, cfg, swiglu) in enumerate(_callers()):
        if len(shapes) == 1:  # a second set, so the launch has one to share with
            shapes = shapes * 2
        lins = _lins(q4, shapes, seed=10 * t)
        assert supports_multi(lins), name
        k = shapes[0][1] * (2 if swiglu else 1)
        x = mx.random.normal((1, k), key=mx.random.key(t))
        together = qmv_multi(x, lins, cfg, swiglu=swiglu)
        at = 0
        for i, lin in enumerate(lins):
            alone = qmv_multi(x, [lin], cfg, swiglu=swiglu)
            assert mx.array_equal(together[:, at : at + lin.out_dims], alone).item(), (name, i)
            at += lin.out_dims
        assert at == together.shape[1]


def test_qmv_multi_as_accurate_as_the_stock_projections(requires_mlx, q4, accuracy):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_decode import qmv_multi
    from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu as stock_swiglu

    for t, (name, shapes, cfg, swiglu) in enumerate(_callers()):
        lins = _lins(q4, shapes, seed=100 + 10 * t)
        k = shapes[0][1]
        x = mx.random.normal((1, 2 * k if swiglu else k), key=mx.random.key(50 + t))
        got = qmv_multi(x, lins, cfg, swiglu=swiglu)
        xs = accuracy.np64(x)
        if swiglu:  # x = [gate | up]: the input is silu(gate) * up, the shared expert's swiglu
            x = stock_swiglu(x[:, :k], x[:, k:])
            xs = xs[:, :k] / (1.0 + np.exp(-xs[:, :k])) * xs[:, k:]
        stock = mx.concatenate([_stock_rows(mx, x, lin) for lin in lins], axis=-1)
        exact = np.concatenate(
            [xs @ accuracy.weights(q.weight, q.scales, q.biases).T for q in lins], axis=-1
        )
        accuracy.no_worse_than_stock(got, stock, exact, what=name)


def _gdn_inputs(mx, seed):
    from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

    c = Qwen4ExpTextConfig()
    C, KC, HV = c.conv_dim, c.linear_conv_kernel_dim, c.linear_num_value_heads
    offs = (0, C, C + c.value_dim, C + c.value_dim + HV)  # qkv | z | b | a, as the block lays out
    key = lambda i: mx.random.key(seed + i)  # noqa: E731
    ycat = mx.random.normal((offs[3] + HV,), key=key(0))
    conv = mx.random.normal((KC - 1, C), key=key(1))
    conv_w = 0.5 * mx.random.normal((C, KC, 1), key=key(2))  # the depthwise conv1d weight
    a_log = 0.5 + 0.3 * mx.random.normal((HV,), key=key(3))
    dt_bias = 0.5 * mx.random.normal((HV,), key=key(4))
    # the compiled stock path captures the weights (tests/metal/kernels/conftest.py)
    mx.eval(ycat, conv, conv_w, a_log, dt_bias)
    return c, offs, ycat, conv, conv_w, a_log, dt_bias


def test_gdn_pre_as_accurate_as_the_stock_ops(requires_mlx, accuracy):
    """Conv state update (an exact copy), depthwise conv and silu, the scaled l2 norms of q and k,
    beta and g, against the same steps of `Qwen4ExpTextGatedDeltaNet._decode_body`, compiled as the
    block compiles it. g's allowance adds the Metal fast log's absolute error bound on [0.5, 2],
    2^-21 (MSL specification, table 8.2): the kernel's softplus takes log(1 + e^-|x|) there, so its
    error in g is at most 2^-21 g e^A_log."""
    import mlx.core as mx
    import mlx.nn as nn

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_decode import gdn_pre, supports_gdn
    from jitengine_qwen3_8_flash_next_m4pro.model.gdn import GDN_FUSED_NSG

    c, offs, ycat, conv, conv_w, a_log, dt_bias = _gdn_inputs(mx, seed=0)
    assert supports_gdn(c)
    C, KC = c.conv_dim, c.linear_conv_kernel_dim
    HK, DK, HV, DV = (
        c.linear_num_key_heads,
        c.linear_key_head_dim,
        c.linear_num_value_heads,
        c.linear_value_head_dim,
    )
    l2_eps, q_scale, k_scale = 1e-6 / DK, 1.0 / DK, DK**-0.5  # the block's constants
    got = gdn_pre(
        conv,
        ycat,
        conv_w.reshape(C, KC),
        a_log,
        dt_bias,
        c,
        l2_eps,
        q_scale,
        k_scale,
        offs,
        GDN_FUSED_NSG,
    )

    @mx.compile
    def stock_pre(conv, ycat):  # _decode_body from the projections on
        qkv = ycat[:C].reshape(1, 1, C)
        b, a = ycat[offs[2] : offs[2] + HV], ycat[offs[3] : offs[3] + HV]
        conv_input = mx.concatenate([conv[None], qkv], axis=1)
        conv_out = nn.silu(mx.conv1d(conv_input, conv_w, groups=C))
        q, k, v = mx.split(conv_out, [HK * DK, 2 * HK * DK], axis=-1)
        q = q_scale * mx.fast.rms_norm(q.reshape(HK, DK), None, l2_eps)
        k = k_scale * mx.fast.rms_norm(k.reshape(HK, DK), None, l2_eps)
        g = mx.exp(-mx.exp(a_log) * nn.softplus(a + dt_bias))
        return q, k, v.reshape(HV, DV), g, mx.sigmoid(b), conv_input[0, 1:]

    *stock, new_conv = stock_pre(conv, ycat)
    assert mx.array_equal(got[5], new_conv).item()  # the new conv state: a copy
    # exact, float64
    y, cv, w = accuracy.np64(ycat), accuracy.np64(conv), accuracy.np64(conv_w)[:, :, 0]
    taps = np.concatenate([cv, y[None, :C]], axis=0)  # [KC, C]
    co = (taps.T * w).sum(axis=1)
    co = co / (1.0 + np.exp(-co))
    qe, ke = co[: HK * DK].reshape(HK, DK), co[HK * DK : 2 * HK * DK].reshape(HK, DK)
    l2 = lambda z, s: s * z / np.sqrt((z * z).mean(axis=-1, keepdims=True) + l2_eps)  # noqa: E731
    sp = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)  # noqa: E731
    bx, ax = y[offs[2] : offs[2] + HV], y[offs[3] : offs[3] + HV]
    decay = np.exp(accuracy.np64(a_log))
    g = np.exp(-decay * sp(ax + accuracy.np64(dt_bias)))
    exact = (
        l2(qe, q_scale),
        l2(ke, k_scale),
        co[2 * HK * DK :].reshape(HV, DV),
        g,
        1 / (1 + np.exp(-bx)),
    )
    slack = {"g": 2.0**-21 * float((g * decay).max())}
    for name, gt, st, ex in zip(("q", "k", "v", "g", "beta"), got[:5], stock, exact):
        accuracy.no_worse_than_stock(gt, st, ex, what=name, slack=slack.get(name, 0.0))


def test_gdn_post_as_accurate_as_the_stock_ops(requires_mlx, accuracy):
    """sigmoid(z) * rms_norm(y, w) against `_decode_body`'s gated norm, compiled."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_decode import gdn_post
    from jitengine_qwen3_8_flash_next_m4pro.model.gdn import GDN_FUSED_NSG

    c, offs, ycat, *_ = _gdn_inputs(mx, seed=10)
    HV, DV, eps = c.linear_num_value_heads, c.linear_value_head_dim, c.rms_norm_eps
    y = mx.random.normal((HV, DV), key=mx.random.key(20))
    norm_w = 1.0 + 0.1 * mx.random.normal((DV,), key=mx.random.key(21))
    mx.eval(y, norm_w)  # the compiled stock path captures norm_w (tests/metal/kernels/conftest.py)
    got = gdn_post(y, ycat, norm_w, c, eps, offs[1], GDN_FUSED_NSG)
    z = ycat[offs[1] : offs[2]].reshape(HV, DV)
    stock = mx.compile(lambda y, z: (mx.sigmoid(z) * mx.fast.rms_norm(y, norm_w, eps)).reshape(-1))
    stock = stock(y, z)
    ye, ze = accuracy.np64(y), accuracy.np64(z)
    normed = ye / np.sqrt((ye * ye).mean(axis=-1, keepdims=True) + eps) * accuracy.np64(norm_w)
    exact = (normed / (1.0 + np.exp(-ze))).reshape(-1)
    accuracy.no_worse_than_stock(got, stock, exact, what="gdn_post")
