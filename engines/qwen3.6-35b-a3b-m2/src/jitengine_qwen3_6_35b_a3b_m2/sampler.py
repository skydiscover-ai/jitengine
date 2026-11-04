"""Token sampling in mlx_lm's order (top_p, then top_k, then categorical), or argmax when greedy.

Each request draws under its own key, never the global RNG, so seeded requests are reproducible and
can share a batch.
"""

from __future__ import annotations

import os

import mlx.core as mx

KEY_BLOCK = 256
# Sampler._sample_exact draws what mlx_lm's filters + categorical draw for the same key, from one
# argsort. Relies on mlx 0.32.2 (see tests/metal/test_sampler.py): argpartition is the stable merge sort
# argsort uses, so apply_top_k keeps the K largest by (value desc, index asc), and one-sample
# categorical is an fp32 exclusive cumsum of exp(x - max) in vocab order, then
# searchsorted(cdf, u * sum, "right") - 1.
# minimum width of _sample_exact's top window: a performance knob, not a correctness one
WINDOW_MIN = 256


def _argsort(lp: mx.array, radix: bool) -> mx.array:
    """mx.argsort over the last axis; `radix` runs radix_argsort, which equals it (a stable sort's
    permutation is unique) and which the engine enables once its check passes on this device."""
    if radix and lp.dtype == mx.bfloat16 and lp.ndim == 2:
        from jitengine_qwen3_6_35b_a3b_m2.kernels.radix_argsort import radix_argsort

        return radix_argsort(lp)
    return mx.argsort(lp, axis=-1)


def logprobs_from_logits(logits: mx.array) -> mx.array:
    """mlx_lm.generate: logprobs = logits - logsumexp(logits) (same dtype as the logits)."""
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative = mx.take_along_axis(cumulative, inverse, axis=-1)
    return mx.where(cumulative > 1 - top_p, logprobs, -float("inf"))


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1)


class Sampler:
    """One per request. `__call__(logits [1, V], step)` -> token [1] (lazy)."""

    def __init__(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        vocab: int,
        radix: bool = False,
    ):
        self.temperature = float(temperature or 0.0)
        self.radix = radix
        self.top_p = float(top_p if top_p is not None else 1.0)
        self.top_k = int(top_k or 0)
        self.greedy = self.temperature <= 0.0
        self._root = None
        self._chain = None  # chain key of the next block to derive
        self._block = -1  # index of the block held in self._keys
        self._keys = None
        if not self.greedy:
            if self.top_k >= vocab:
                self.top_k = 0
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "little")
            self._root = self._chain = mx.random.key(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def key(self, step: int) -> mx.array:
        """Return the key for `step`, deriving keys lazily one KEY_BLOCK block at a time.

        Block b has (block key, chain_{b+1}) = split(chain_b, 2) with chain_0 = root, and its step
        keys are split(block key, KEY_BLOCK); an earlier block is re-derived from the root.
        """
        b = step // KEY_BLOCK
        if b != self._block:
            if b < self._block:
                self._chain, self._block = self._root, -1
            while self._block < b:
                blk, self._chain = mx.random.split(self._chain, 2)
                self._block += 1
            self._keys = mx.random.split(blk, KEY_BLOCK)
        return self._keys[step % KEY_BLOCK]

    def __call__(self, logits: mx.array, step: int) -> mx.array:
        lp = logprobs_from_logits(logits)
        if self.greedy:
            return mx.argmax(lp, axis=-1)
        if lp.ndim == 2 and lp.shape[0] == 1:
            return self._sample_exact(lp, step)
        if 0.0 < self.top_p < 1.0:
            lp = apply_top_p(lp, self.top_p)
        if self.top_k > 0:
            lp = apply_top_k(lp, self.top_k)
        return mx.random.categorical(lp * (1.0 / self.temperature), axis=-1, key=self.key(step))

    def _sample_exact(self, lp: mx.array, step: int) -> mx.array:
        """`_sample_exact_full` computed on two short slices of the sorted order: the same draw.

        After the top_p cut the sorted values are non-decreasing, so with v = sv[n-k] the kept set
        is [first, first + (k - above)) union [n - above, n): at most k ties at v starting at
        `first`, and a strictly-larger suffix that fits the top window because above < k. Every
        other position has weight exactly +0.0 in the full path too, so the weight vector, cumsum
        and draw are identical.
        """
        n = lp.shape[-1]
        k = self.top_k
        if k <= 0 or 4 * k > n:
            return self._sample_exact_full(lp, step)
        w_top = self._window(n)
        idx = _argsort(lp, self.radix)
        sv = mx.take_along_axis(lp, idx, axis=-1)
        if 0.0 < self.top_p < 1.0:
            cum = mx.cumsum(mx.take_along_axis(mx.exp(lp), idx, axis=-1), axis=-1)
            sv = mx.where(cum > 1 - self.top_p, sv, -float("inf"))  # stays non-decreasing
        v = sv[..., n - k : n - k + 1]
        above = mx.sum(sv > v, axis=-1, keepdims=True)
        first = mx.sum(sv < v, axis=-1, keepdims=True)
        cut = first + (k - above)
        inv_t = 1.0 / self.temperature

        # window A: the top w_top sorted positions, which always hold the strictly-larger survivors
        pos_a = mx.arange(n - w_top, n, dtype=mx.int32)
        sv_a = sv[..., n - w_top :]
        idx_a = idx[..., n - w_top :]
        # window B: k positions anchored at `first`, which always hold the kept ties at v
        start = mx.minimum(first.reshape(-1).astype(mx.int32), mx.array(n - k, dtype=mx.int32))
        pos_b = start + mx.arange(k, dtype=mx.int32)
        sv_b = mx.slice(sv, start, (1,), (1, k))
        idx_b = mx.slice(idx, start, (1,), (1, k))

        # where the windows overlap, both copies carry the same value and the scatter writes it
        # twice; the max must span both windows (with all-equal logits the top one can be all -inf)
        svw = mx.concatenate([sv_a.reshape(-1), sv_b.reshape(-1)])
        posw = mx.concatenate([pos_a, pos_b])
        keep = (svw > v.reshape(-1)) | ((svw == v.reshape(-1)) & (posw < cut.reshape(-1)))
        xw = (mx.where(keep, svw, -float("inf")) * inv_t).astype(mx.float32)
        m = mx.max(xw)
        ww = mx.where(mx.isinf(m), (xw == m).astype(mx.float32), mx.exp(xw - m))
        w = mx.put_along_axis(
            mx.zeros((n,), dtype=mx.float32),
            mx.concatenate([idx_a.reshape(-1), idx_b.reshape(-1)]),
            ww,
            axis=-1,
        )
        cdf = mx.cumsum(w, axis=0, inclusive=False)
        u = mx.random.uniform(shape=(1,), dtype=mx.float32, key=self.key(step)) * mx.sum(w)
        return (mx.sum(cdf[None, :] <= u[:, None], axis=-1) - 1).astype(mx.uint32)

    def _window(self, n: int) -> int:
        """Width of the static top window; it must be at least top_k to hold the kept suffix."""
        return min(n, max(WINDOW_MIN, self.top_k))

    def _sample_exact_full(self, lp: mx.array, step: int) -> mx.array:
        """Bit-for-bit the stock draw (apply_top_p, apply_top_k, categorical) for one row [1, V].

        Uses one stable ascending argsort: top_k keeps values above the K-th largest v plus the
        first (lowest-index) positions holding v, and the draw counts cdf <= u instead of a search.
        """
        n = lp.shape[-1]
        idx = mx.argsort(lp, axis=-1)
        sv = mx.take_along_axis(lp, idx, axis=-1)
        if 0.0 < self.top_p < 1.0:
            cum = mx.cumsum(mx.take_along_axis(mx.exp(lp), idx, axis=-1), axis=-1)
            sv = mx.where(cum > 1 - self.top_p, sv, -float("inf"))  # stays non-decreasing
        if self.top_k > 0:
            v = sv[..., n - self.top_k : n - self.top_k + 1]
            above = mx.sum(sv > v, axis=-1, keepdims=True)
            first = mx.sum(sv < v, axis=-1, keepdims=True)
            pos = mx.arange(n, dtype=mx.int32)
            keep = (sv > v) | ((sv == v) & (pos < first + (self.top_k - above)))
            sv = mx.where(keep, sv, -float("inf"))
        x = mx.put_along_axis(mx.zeros_like(lp), idx, sv, axis=-1) * (1.0 / self.temperature)
        x = x.reshape(-1).astype(mx.float32)
        m = mx.max(x)
        w = mx.where(mx.isinf(m), (x == m).astype(mx.float32), mx.exp(x - m))
        cdf = mx.cumsum(w, axis=0, inclusive=False)
        u = mx.random.uniform(shape=(1,), dtype=mx.float32, key=self.key(step)) * mx.sum(w)
        return (mx.sum(cdf[None, :] <= u[:, None], axis=-1) - 1).astype(mx.uint32)


def rows_key(s: Sampler, n: int):
    """Samplers with equal keys can draw as one batch (sample_rows); None = draw alone."""
    if s.greedy or s.top_k <= 0 or 4 * s.top_k > n:
        return None
    return (s.temperature, s.top_p, s.top_k)


def sample_rows(lp: mx.array, samplers: list, steps: list) -> mx.array:
    """`Sampler._sample_exact` for B rows sharing one rows_key (lp [B, V]); the same draw per row.

    sum(w) is the one float reduction whose order could depend on the shape, so it stays a 1-D sum
    per row, as in the single-row path.
    """
    B, n = lp.shape
    s0 = samplers[0]
    k, top_p, inv_t = s0.top_k, s0.top_p, 1.0 / s0.temperature
    w_top = s0._window(n)
    idx = _argsort(lp, s0.radix)
    sv = mx.take_along_axis(lp, idx, axis=-1)
    if 0.0 < top_p < 1.0:
        cum = mx.cumsum(mx.take_along_axis(mx.exp(lp), idx, axis=-1), axis=-1)
        sv = mx.where(cum > 1 - top_p, sv, -float("inf"))
    v = sv[:, n - k : n - k + 1]
    above = mx.sum(sv > v, axis=-1, keepdims=True)
    first = mx.sum(sv < v, axis=-1, keepdims=True)
    cut = first + (k - above)
    pos_a = mx.arange(n - w_top, n, dtype=mx.int32)
    sv_a = sv[:, n - w_top :]
    idx_a = idx[:, n - w_top :]
    start = mx.minimum(first.astype(mx.int32), mx.array(n - k, dtype=mx.int32))
    pos_b = start + mx.arange(k, dtype=mx.int32)[None, :]
    sv_b = mx.take_along_axis(sv, pos_b, axis=-1)
    idx_b = mx.take_along_axis(idx, pos_b, axis=-1)
    svw = mx.concatenate([sv_a, sv_b], axis=-1)
    posw = mx.concatenate([mx.broadcast_to(pos_a[None, :], (B, w_top)), pos_b], axis=-1)
    keep = (svw > v) | ((svw == v) & (posw < cut))
    xw = (mx.where(keep, svw, -float("inf")) * inv_t).astype(mx.float32)
    m = mx.max(xw, axis=-1, keepdims=True)
    ww = mx.where(mx.isinf(m), (xw == m).astype(mx.float32), mx.exp(xw - m))
    w = mx.put_along_axis(
        mx.zeros((B, n), dtype=mx.float32), mx.concatenate([idx_a, idx_b], axis=-1), ww, axis=-1
    )
    cdf = mx.cumsum(w, axis=-1, inclusive=False)
    tot = mx.stack([mx.sum(w[b]) for b in range(B)]).reshape(B, 1)
    u = (
        mx.concatenate(
            [
                mx.random.uniform(shape=(1,), dtype=mx.float32, key=s.key(st))
                for s, st in zip(samplers, steps)
            ]
        ).reshape(B, 1)
        * tot
    )
    return (mx.sum(cdf <= u, axis=-1) - 1).astype(mx.uint32)
