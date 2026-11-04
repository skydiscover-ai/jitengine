"""Speculative verification: M consecutive positions of one stream in one pass, bit-exact per row.

Every piece reuses a path that is bit-exact against solo decode, so greedy speculation is
token-identical to plain decoding; sampling is too if each position uses plain decoding's key.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from jitengine_qwen3_6_35b_a3b_m4pro.kernels.elementwise import gated_silu_f32
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import gdn_recurrence_steps
from jitengine_qwen3_6_35b_a3b_m4pro.model.attention import Qwen3_5MoeAttention
from jitengine_qwen3_6_35b_a3b_m4pro.model.gdn import Qwen3_5MoeGatedDeltaNet, compute_g


def _gdn_verify_glue(blk: Qwen3_5MoeGatedDeltaNet, x: mx.array, slot):
    """_gdn_verify with the conv/silu/split/norm/scale chain and the gated norm as one launch each."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import gdn_exact as _gx

    cfg = blk.cfg
    M = x.shape[0]
    w2, offs = blk._glue_consts()
    ycat = blk.in_proj.lin.rows(x).reshape(M, -1)  # one exact launch over the fused projection rows
    NK = blk.n_keep
    q, k, v, newconv = _gx.gdn_pre_rows(slot.conv.reshape(NK, -1), ycat, w2, cfg, offs[0])
    convs = [newconv[j : j + 1] for j in range(M)]
    b = ycat[:, offs[2] : offs[3]].reshape(M, 1, -1)
    a = ycat[:, offs[3] :].reshape(M, 1, -1)
    beta = mx.sigmoid(b)
    g = compute_g(blk.A_log, a, blk.dt_bias)
    y, states = gdn_recurrence_steps(
        q.reshape(1, M, *q.shape[1:]),
        k.reshape(1, M, *k.shape[1:]),
        v.reshape(1, M, *v.shape[1:]),
        g.reshape(1, M, -1),
        beta.reshape(1, M, -1),
        slot.ssm,
    )
    out = _gx.gdn_post_rows(y.reshape(M, *y.shape[2:]), ycat, blk.norm_w, cfg, blk.eps, offs[1])
    ssms = [states[j] for j in range(M)]
    return blk.out_proj.rows(out.reshape(M, 1, -1)), (convs, ssms)


def _gdn_verify(blk: Qwen3_5MoeGatedDeltaNet, x: mx.array, slot):
    """x [M, 1, D] (M consecutive positions as rows) -> (out [M, 1, D], (convs, ssms)) where
    convs[j] / ssms[j] are the slot state after position j."""
    if blk.paths.gdn_exact and blk.in_proj is not None:
        return _gdn_verify_glue(blk, x, slot)
    cfg = blk.cfg
    M = x.shape[0]
    if blk.in_proj is not None:
        qkv, z, b, a = blk.in_proj.rows(x)
    else:
        qkv, z = blk.in_proj_qkv.rows(x), blk.in_proj_z.rows(x)
        b, a = blk.in_proj_b.rows(x), blk.in_proj_a.rows(x)
    z = z.reshape(M, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

    nk = blk.n_keep
    seq = mx.concatenate([slot.conv, qkv.reshape(1, M, -1)], axis=1)  # [1, nk + M, C]
    windows = mx.concatenate([seq[:, j : j + nk + 1, :] for j in range(M)], axis=0)  # [M, nk+1, C]
    convs = [mx.contiguous(seq[:, j + 1 : j + 1 + nk, :]) for j in range(M)]
    conv_out = nn.silu(mx.conv1d(windows, blk.conv_w, groups=cfg.conv_dim))[:, -1:, :]  # [M, 1, C]

    kd = cfg.key_dim
    q, k, v = mx.split(conv_out, [kd, 2 * kd], axis=-1)
    q = q.reshape(M, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
    k = k.reshape(M, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
    v = v.reshape(M, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
    inv_scale = cfg.linear_key_head_dim**-0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(blk.A_log, a, blk.dt_bias)

    def seq1(t):  # [M, 1, ...] rows -> [1, M, ...] time
        return t.reshape(1, M, *t.shape[2:])

    y, states = gdn_recurrence_steps(seq1(q), seq1(k), seq1(v), seq1(g), seq1(beta), slot.ssm)
    y = y.reshape(M, 1, *y.shape[2:])
    out = gated_silu_f32(y, z, mx.fast.rms_norm(y, blk.norm_w, blk.eps))
    ssms = [states[j] for j in range(M)]
    return blk.out_proj.rows(out.reshape(M, 1, -1)), (convs, ssms)


def _attn_verify(blk: Qwen3_5MoeAttention, x: mx.array, slot, t: int):
    """x [M, 1, D] -> out [M, 1, D]; row j is the solo decode attention over keys[:t+j+1].

    One rope call over the M positions is bitwise M single-position calls.
    """
    cfg = blk.cfg
    M = x.shape[0]
    H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    if blk.qkv_proj is not None:
        qg, k, v = blk.qkv_proj.rows(x)
    else:
        qg, k, v = blk.q_proj.rows(x), blk.k_proj.rows(x), blk.v_proj.rows(x)
    qg = qg.reshape(M, 1, H, 2 * D)
    q, gate = mx.split(qg, 2, axis=-1)
    gate = gate.reshape(M, 1, -1)
    k = k.reshape(M, 1, Hkv, D)
    v = v.reshape(M, 1, Hkv, D)
    q = mx.fast.rms_norm(q, blk.q_norm_w, blk.eps)  # [M, 1, H, D]
    k = mx.fast.rms_norm(k, blk.k_norm_w, blk.eps)
    # one sequence of M positions: [1, H, M, D]
    q = q.reshape(1, M, H, D).transpose(0, 2, 1, 3)
    k = k.reshape(1, M, Hkv, D).transpose(0, 2, 1, 3)
    v = v.reshape(1, M, Hkv, D).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, blk.rope_dims, traditional=False, base=blk.rope_base, scale=1.0, offset=t)
    k = mx.fast.rope(k, blk.rope_dims, traditional=False, base=blk.rope_base, scale=1.0, offset=t)
    start = mx.array([0, 0, t, 0], dtype=mx.int32)
    slot.keys = mx.slice_update(slot.keys, k.astype(slot.keys.dtype), start, axes=(0, 1, 2, 3))
    slot.values = mx.slice_update(
        slot.values, v.astype(slot.values.dtype), start, axes=(0, 1, 2, 3)
    )
    slot.offset = t + M
    outs = []
    exact = blk.paths.exact_sdpa
    if exact:
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import (
            sdpa_decode,
            sdpa_decode_rows,
            supported,
        )

        exact = supported(q[:, :, :1], slot.keys)
        # the M positions as M rows of one launch over the same KV buffer with lengths t+1 .. t+M
        # (each row bitwise its own sdpa_decode call, per-row MLX routing included)
        if M > 1 and M <= 16:
            qr = q.transpose(2, 1, 0, 3)  # [M, H, 1, D]
            lens = [t + j + 1 for j in range(M)]
            outs = [
                sdpa_decode_rows(
                    qr[c : c + 8],
                    [slot.keys] * len(lens[c : c + 8]),
                    [slot.values] * len(lens[c : c + 8]),
                    lens[c : c + 8],
                    blk.scale,
                )
                for c in range(0, M, 8)
            ]  # the rows kernel takes <= 8
            out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
            out = out.reshape(M, 1, -1)
            return blk.o_proj.rows(out * mx.sigmoid(gate))
    for j in range(M):
        L = t + j + 1
        if exact:
            outs.append(sdpa_decode(q[:, :, j : j + 1], slot.keys, slot.values, L, blk.scale))
            continue
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, j : j + 1],
                slot.keys[:, :, :L, :],
                slot.values[:, :, :L, :],
                scale=blk.scale,
                mask=None,
            )
        )
    out = mx.concatenate(outs, axis=2)  # [1, H, M, D]
    out = out.transpose(0, 2, 1, 3).reshape(M, 1, -1)
    return blk.o_proj.rows(out * mx.sigmoid(gate))


ASYNC_EVERY = 2  # submit the graph to the GPU every N layers while the host builds the rest


def verify_step(model, tokens: mx.array, state, telem=None):
    """tokens [M] int32 (may be lazy: drafted on the GPU). Advances `state` by M positions and
    returns (logits [M, V], final hidden after the norm [M, D], undo). `commit(state, undo, keep)`
    keeps the first `keep` verified positions."""
    if model.strategies.verify_relaxed:  # dense rows through MLX's small-M kernel (not bit-exact)
        from jitengine_qwen3_6_35b_a3b_m4pro.linear import relaxed_rows

        with relaxed_rows():
            return _verify_step(model, tokens, state, telem)
    return _verify_step(model, tokens, state, telem)


def _verify_step(model, tokens: mx.array, state, telem=None):
    M = tokens.shape[0]
    t = state.tokens
    h = model.embed.lookup(tokens.reshape(M, 1))
    undo = []
    for layer, slot in zip(model.layers, state.layers):
        x = mx.fast.rms_norm(h, layer.ln1, layer.eps)
        if isinstance(layer.block, Qwen3_5MoeAttention):
            slot.ensure_room(M)
            # the previous KV arrays are not kept: holding them would stop slice_update from
            # donating the buffer and copy the whole KV every layer, every round
            a = _attn_verify(layer.block, x, slot, t)
            undo.append(("kv", slot, None))
        else:
            a, rec = _gdn_verify(layer.block, x, slot)
            undo.append(("delta", slot, rec))
        h = h + a
        h = layer.moe.with_residual(mx.fast.rms_norm(h, layer.ln2, layer.eps), h, telem, rows=True)
        li = len(undo)
        if ASYNC_EVERY and li % ASYNC_EVERY == 0 and li < len(model.layers):
            mx.async_eval(h)
    state.tokens = t + M
    hn = mx.fast.rms_norm(h[:, -1, :], model.norm_w, model.eps)
    return model.lm_head.rows(hn), hn, (t, undo)


def commit(state, undo, keep: int) -> None:
    """Keep the first `keep` verified positions (1 <= keep <= M) and drop the rest."""
    t, recs = undo
    for kind, slot, rec in recs:
        if kind == "kv":
            slot.offset = t + keep
        else:
            convs, ssms = rec
            slot.conv, slot.ssm = convs[keep - 1], ssms[keep - 1]
    state.tokens = t + keep
