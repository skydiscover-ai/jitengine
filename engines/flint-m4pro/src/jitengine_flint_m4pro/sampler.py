"""Token selection and log-probabilities, inside the MLX graph (no host round trip per step).

Greedy: argmax of the logits (the lowest id wins an exact tie).
Sampled: temperature T, then top-k (0 = off), then the smallest top-p nucleus of the renormalised
top-k distribution, renormalised; the draw is an inverse-CDF lookup with a uniform that is a pure
function of (seed, completion index) (`uniform`), so the same seed reproduces the same tokens whatever
the release grouping, streaming, restarts or interleaved traffic.
Log-probabilities: fp32 log_softmax of the logits; top-N ids sorted best first.
"""

from __future__ import annotations

import os

MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def uniform(seed: int, index: int) -> float:
    """U[0, 1) for completion index `index` of a request seeded with `seed` (53-bit resolution)."""
    h = _splitmix64(_splitmix64(seed & MASK64) ^ (index & MASK64))
    return (h >> 11) / float(1 << 53)


def fresh_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") >> 1


def greedy(logits):
    import mlx.core as mx

    return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)  # [1]


def sample(logits, temperature: float, top_k: int, top_p: float, u: float):
    """One draw from the filtered distribution; returns int32 [1]."""
    import mlx.core as mx

    lf = logits.astype(mx.float32) * (1.0 / temperature)
    V = lf.shape[-1]
    if 0 < top_k < V:
        cand = mx.argpartition(-lf, kth=top_k - 1)[:top_k]
        cv = mx.take(lf, cand)
        order = mx.argsort(-cv)
        cand, cv = mx.take(cand, order), mx.take(cv, order)
    else:
        cand = mx.argsort(-lf)
        cv = mx.take(lf, cand)
    p = mx.softmax(cv)
    if top_p < 1.0:
        before = mx.cumsum(p) - p
        p = mx.where(before < top_p, p, mx.zeros_like(p))
    cdf = mx.cumsum(p)
    total = cdf[-1]
    idx = mx.sum((cdf <= u * total).astype(mx.int32))
    idx = mx.minimum(idx, cand.shape[0] - 1)
    return mx.take(cand, idx.reshape(1)).astype(mx.int32)


def choose(logits, temperature: float, top_k: int, top_p: float, seed: int, index: int):
    if temperature <= 0.0 or top_k == 1:
        return greedy(logits)
    return sample(logits, temperature, top_k, top_p, uniform(seed, index))


def logprob_outputs(logits, token, n_top: int):
    """(logprob of `token` [1], top ids [n_top], top logprobs [n_top]) as lazy arrays."""
    import mlx.core as mx

    lf = logits.astype(mx.float32)
    lp = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    chosen = mx.take(lp, token.reshape(1))
    if n_top <= 0:
        return chosen, None, None
    ids = mx.argpartition(-lf, kth=n_top - 1)[:n_top]
    vals = mx.take(lp, ids)
    order = mx.argsort(-vals)
    return chosen, mx.take(ids, order), mx.take(vals, order)
