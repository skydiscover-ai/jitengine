"""Flint forward pass as an MLX graph over the weight banks (the correctness anchor of this engine).

Semantics (config.json + the vendor export): 49 layers; 14 full-attention layers without RoPE over the
whole history, 35 sliding layers with RoPE (theta 1e4, non-traditional) over a 513-token window
(current + 512); per-head q/k RMSNorm; GQA 32:4, head_dim 128, scale 1/sqrt(128), a learned sink logit
per query head (an extra softmax term with no value); layers 0-1 dense SwiGLU, the rest 6-of-256
routed SwiGLU experts plus an always-on shared expert (routing.py). Hidden state bf16 throughout.

Two entry points: `prefill_chunk` (n positions at once, writing the history decode will read) and
`decode_step` (one position, writing its K/V first, then attending). In offload mode (W.pool set) the routed experts
are gathered from the expert pool by slot instead of from the banks by expert id: the same bytes and the same kernels.
"""

from __future__ import annotations

import math

from jitengine_flint_m4pro.kv_cache import History
from jitengine_flint_m4pro.loader import LayerWeights, QLinear, Weights
from jitengine_flint_m4pro.model.routing import combine, route

# Below this many (token, expert) rows the MoE uses the unsorted gather (decode, tiny chunks); above it,
# rows are sorted by expert for the rhs-gather kernel (mlx-lm uses the same 64-row threshold).
SORTED_GATHER_MIN = 64


def qmm(x, q: QLinear):
    import mlx.core as mx

    return mx.quantized_matmul(x, q.w, q.s, q.b, transpose=True, group_size=q.group, bits=q.bits)


def gather_qmm(x, q: QLinear, idx, sorted_indices=False):
    import mlx.core as mx

    return mx.gather_qmm(
        x,
        q.w,
        q.s,
        q.b,
        rhs_indices=idx,
        transpose=True,
        group_size=q.group,
        bits=q.bits,
        sorted_indices=sorted_indices,
    )


def swiglu(g, u):
    import mlx.core as mx

    return (g * mx.sigmoid(g)) * u


def embed(W: Weights, tokens):
    """tokens int32 [..] -> bf16 [.., hidden] (8-bit affine embedding, dequantized rows)."""
    import mlx.core as mx

    e = W.embed
    flat = tokens.reshape(-1)
    rows = mx.dequantize(
        mx.take(e.w, flat, axis=0),
        mx.take(e.s, flat, axis=0),
        mx.take(e.b, flat, axis=0),
        group_size=e.group,
        bits=e.bits,
    )
    return rows.reshape(*tokens.shape, -1)


def head(W: Weights, h):
    import mlx.core as mx

    return qmm(mx.fast.rms_norm(h, W.norm, W.cfg.rms_eps), W.lm_head)


def _qkv(W: Weights, lw: LayerWeights, x, pos0: int):
    import mlx.core as mx

    cfg = W.cfg
    B, L, _ = x.shape
    D = cfg.head_dim
    q = mx.fast.rms_norm(qmm(x, lw.q).reshape(B, L, cfg.n_heads, D), lw.q_norm, cfg.rms_eps)
    k = mx.fast.rms_norm(qmm(x, lw.k).reshape(B, L, cfg.n_kv_heads, D), lw.k_norm, cfg.rms_eps)
    v = qmm(x, lw.v).reshape(B, L, cfg.n_kv_heads, D)
    q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    if lw.sliding:
        q = mx.fast.rope(q, D, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos0)
        k = mx.fast.rope(k, D, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos0)
    return q, k, v


def _sdpa(W: Weights, lw: LayerWeights, q, k, v, mask):
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(W.cfg.head_dim), mask=mask, sinks=lw.sinks
    )


def _mlp(W: Weights, lw: LayerWeights, x):
    """x [B, L, hidden] bf16 -> [B, L, hidden] bf16."""
    import mlx.core as mx

    cfg = W.cfg
    if lw.dense is not None:
        g, u, d = lw.dense
        return qmm(swiglu(qmm(x, g), qmm(x, u)), d)
    B, L, H = x.shape
    ids, w = route(x, lw.router, cfg.top_k)  # [B, L, k]
    n = B * L
    if W.pool is None:
        eg, eu, ed = lw.experts
    elif n * cfg.top_k < SORTED_GATHER_MIN:
        # offload, unsorted gather: the same experts, read from their pool slots (expert_pool.py; evaluates ids)
        ids = W.pool.slots_for(lw.index, ids)
        eg, eu, ed = W.pool.experts
    else:
        # offload, sorted gather: the layer's [256, ...] expert table taken from the pool, indexed by expert id as the
        # banks are, since MLX picks the sorted gather's kernel from the rows per expert of the weight array
        eg, eu, ed = W.pool.layer_table(lw.index, ids)
    if n * cfg.top_k < SORTED_GATHER_MIN:
        # unsorted gather: x [n, 1, 1, H] against indices [n, k] -> [n, k, 1, H] (no weight copies)
        xe = x.reshape(n, 1, 1, H)
        flat = ids.reshape(n, cfg.top_k)
        a = swiglu(gather_qmm(xe, eg, flat), gather_qmm(xe, eu, flat))
        y = gather_qmm(a, ed, flat)
        y = y.reshape(B, L, cfg.top_k, H)
    else:
        flat = ids.reshape(-1)
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        xs = mx.take(x.reshape(n, 1, H), order // cfg.top_k, axis=0)
        si = mx.take(flat, order)
        a = swiglu(gather_qmm(xs, eg, si, True), gather_qmm(xs, eu, si, True))
        y = mx.take(gather_qmm(a, ed, si, True), inv, axis=0).reshape(B, L, cfg.top_k, H)
    routed = combine(y, w)
    sg, su, sd = lw.shared
    return routed + qmm(swiglu(qmm(x, sg), qmm(x, su)), sd)


def _layer_tail(W, lw, h, attn):
    import mlx.core as mx

    h = h + qmm(attn, lw.o)
    return h + _mlp(W, lw, mx.fast.rms_norm(h, lw.ln2, W.cfg.rms_eps))


def decode_step(W: Weights, hist: History, token, p: int, ring, trace=None):
    """One decode pass for the token at position p (int32 [1], may be lazy). Writes its K/V, returns
    the bf16 logits [vocab] of position p + 1's distribution. `ring` = hist.ring_mask(p).
    trace (a list) receives h_0 .. h_49 (the residual entering each layer, then the last layer's output)."""
    import mlx.core as mx

    cfg = W.cfg
    h = embed(W, token.reshape(1, 1))
    n_slots, rmask = ring
    for lw in W.layers:
        if trace is not None:
            trace.append(h.reshape(-1))
        x = mx.fast.rms_norm(h, lw.ln1, cfg.rms_eps)
        q, k, v = _qkv(W, lw, x, p)
        if lw.sliding:
            hist.write_ring(lw.index, p, k, v)
            kk = hist.rk[lw.index][:, :, :n_slots, :]
            vv = hist.rv[lw.index][:, :, :n_slots, :]
            o = _sdpa(W, lw, q, kk, vv, rmask)
        else:
            hist.write_full(lw.index, p, k, v)
            kk, vv = hist.full_kv(lw.index, p + 1)
            o = _sdpa(W, lw, q, kk, vv, None)
        h = _layer_tail(W, lw, h, o.transpose(0, 2, 1, 3).reshape(1, 1, -1))
    if trace is not None:
        trace.append(h.reshape(-1))
    return head(W, h).reshape(-1)


def _prefill_layer(W: Weights, hist: History, lw, h, start: int, swa_mask):
    """One layer of a prefill chunk at positions start .. start+n-1: writes the layer's K/V, returns (h, swa_mask) (the
    sliding layers' mask, built once per chunk)."""
    import mlx.core as mx

    cfg = W.cfg
    n = h.shape[1]
    x = mx.fast.rms_norm(h, lw.ln1, cfg.rms_eps)
    q, k, v = _qkv(W, lw, x, start)
    if lw.sliding:
        kw, vw, w = hist.ring_window_ordered(lw.index, start)
        kk = mx.concatenate([kw, k], axis=2) if w else k
        vv = mx.concatenate([vw, v], axis=2) if w else v
        if swa_mask is None:
            qi = mx.arange(n)[:, None] + w
            kj = mx.arange(w + n)[None, :]
            swa_mask = (kj <= qi) & ((qi - kj) < cfg.sliding_window)
        o = _sdpa(W, lw, q, kk, vv, swa_mask)
        hist.write_ring(lw.index, start, k, v)
    else:
        hist.write_full(lw.index, start, k, v)
        kk, vv = hist.full_kv(lw.index, start + n)
        o = _sdpa(W, lw, q, kk, vv, "causal" if n > 1 else None)
    return _layer_tail(W, lw, h, o.transpose(0, 2, 1, 3).reshape(1, n, -1)), swa_mask


def prefill_chunk(W: Weights, hist: History, tokens, start: int, rows=None):
    """Positions start .. start+n-1 (tokens int32 [n]). Writes their K/V into the history and returns
    the bf16 logits of the requested chunk rows (int list, default: the last row) as [len(rows), vocab]."""
    import mlx.core as mx

    cfg = W.cfg
    n = tokens.shape[0]
    h = embed(W, tokens.reshape(1, n))
    swa_mask = None
    for lw in W.layers:
        h, swa_mask = _prefill_layer(W, hist, lw, h, start, swa_mask)
        if n * cfg.top_k >= SORTED_GATHER_MIN:
            # The sorted expert gather materialises a contiguous copy of each strided expert view
            # (~450 MB per layer); evaluating layer by layer bounds those transients to one layer
            # instead of letting all 47 layers' copies coexist in one lazy chunk graph.
            mx.eval(h)
    if rows is None:
        rows = [n - 1]
    sel = mx.take(h.reshape(n, -1), mx.array(rows, dtype=mx.int32), axis=0)
    return head(W, sel)


def prefill_layer_major(W: Weights, hist: History, tokens, chunks: list, check=None) -> list:
    """Offload mode's prefill: the chunks [(c0, c1), ...] of `tokens` (int32 [n]) layer by layer, every chunk through
    layer i before any reaches layer i + 1, so a layer's experts are read from the checkpoint once per prompt instead
    of once per chunk. Each chunk at layer i reads only what earlier chunks wrote at layer i, as prefill_chunk's order
    has it, so the history and every hidden state are the same. check() runs between layers (cancellation). Returns
    the last layer's hidden states, one [1, m, hidden] per chunk (rows -> logits is the caller's, with head)."""
    import mlx.core as mx

    hs = [embed(W, tokens[c0:c1].reshape(1, c1 - c0)) for c0, c1 in chunks]
    masks = [None] * len(chunks)
    for lw in W.layers:
        if check is not None:
            check()
        for i, (c0, _) in enumerate(chunks):
            hs[i], masks[i] = _prefill_layer(W, hist, lw, hs[i], c0, masks[i])
            mx.eval(hs[i], hist.state_arrays())
    return hs
