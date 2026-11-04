"""Probabilistic (Leviathan / Chen) acceptance for sampled speculative rounds, the relaxed variant.

A draft d drawn from q is accepted with probability min(1, p(d) / q(d)); the first rejection draws
from the residual max(p - q, 0), and after k accepted drafts the bonus token is drawn from p of the
last position. The emitted sequence has exactly the target's sampling distribution but not the
plain path's draws for the same keys. Greedy requests never come here (argmax matching stays exact).

p and q are the request's filtered distributions in mlx_lm's order (top_p, top_k, then softmax over
the kept tokens), computed on the top `kk` candidates (kk = top_k, or 64 without top_k). Randomness
comes from keys split off the request's Sampler.key(step), so a seeded request is reproducible.
"""

from __future__ import annotations

import mlx.core as mx


def _kk(top_k: int) -> int:
    return top_k if 0 < top_k <= 256 else 64


def filtered_sparse(
    lp: mx.array, temperature: float, top_p: float, top_k: int, logits: bool = False
):
    """The request's sampling distribution on its top-kk candidates: (ids [R, kk] int32, probs [R, kk] float32).

    lp is [R, V] log-probabilities (raw logits with logits=True); probs are zero outside the kept
    set. The candidates come from topk_fast instead of a full-vocabulary sort.
    """
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.topk_fast import topk

    kk = _kk(top_k)
    v, ids = topk(lp, kk)  # descending
    if logits:
        v = v - mx.logsumexp(lp.astype(mx.float32), axis=-1, keepdims=True)
    keep = mx.ones(v.shape, dtype=mx.bool_)
    if 0.0 < top_p < 1.0:
        # mass ranked strictly above (full-vocab probs)
        above = mx.cumsum(mx.exp(v), axis=-1, inclusive=False)
        keep = above < top_p
    if 0 < top_k < kk:
        keep = keep & (mx.arange(kk)[None, :] < top_k)
    x = mx.where(keep, v / temperature, -float("inf"))
    w = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
    return ids, w / mx.sum(w, axis=-1, keepdims=True)


def draw_sparse(ids: mx.array, probs: mx.array, key) -> mx.array:
    """One draw per row from sparse distributions: ids/probs [R, kk] -> [R] int32."""
    j = mx.random.categorical(mx.log(mx.maximum(probs, 1e-38)), axis=-1, key=key)
    return mx.take_along_axis(ids, j[:, None].astype(mx.int32), axis=-1).reshape(-1)


def accept(
    lp: mx.array,
    drafts: mx.array,
    qids: mx.array,
    qprob: mx.array,
    temperature: float,
    top_p: float,
    top_k: int,
    keys: list,
):
    """Accept or resample each draft: lazy (acc [k] bool, resample [k] int32, bonus [1] int32).

    lp is [k+1, V] target logits of the verify rows, drafts [k], q = (qids, qprob) [k, kq]; the
    first of `keys` is used, split three ways. The emitted tokens are
    drafts[:a] + [resample[a] if a < k else bonus], a = the number of leading accepted drafts.
    """
    k = drafts.shape[0]
    pids, pprob = filtered_sparse(lp, temperature, top_p, top_k, logits=True)  # [k+1, kk]
    ku, kr, kb = mx.random.split(keys[0], 3)
    d = drafts.reshape(k, 1).astype(mx.int32)
    p_d = mx.sum(mx.where(pids[:k] == d, pprob[:k], 0.0), axis=-1)  # p(d_i)
    q_d = mx.sum(mx.where(qids == d, qprob, 0.0), axis=-1)  # q(d_i)
    u = mx.random.uniform(shape=(k,), key=ku)
    acc = (q_d > 0) & (u * q_d < p_d)
    # residual max(p - q, 0) on p's support (tokens outside it have p = 0)
    q_at = mx.sum(
        mx.where(pids[:k, :, None] == qids[:, None, :], qprob[:, None, :], 0.0), axis=-1
    )  # [k, kk]
    res = mx.maximum(pprob[:k] - q_at, 0.0)
    tot = mx.sum(res, axis=-1, keepdims=True)
    # p == q exactly: any draw from p (never reached on accept-all)
    res = mx.where(tot > 0, res, pprob[:k])
    resample = draw_sparse(pids[:k], res, kr)
    bonus = draw_sparse(pids[k : k + 1], pprob[k : k + 1], kb)
    return acc, resample, bonus


def one_hot_q(drafts: mx.array):
    """The draft distribution of copied tokens (all mass on the copied token)."""
    k = drafts.shape[0]
    return drafts.reshape(k, 1).astype(mx.int32), mx.ones((k, 1), dtype=mx.float32)
