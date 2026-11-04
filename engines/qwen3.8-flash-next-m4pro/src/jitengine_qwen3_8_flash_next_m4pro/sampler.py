"""Token sampling in mlx_lm.sample_utils order (top_p, top_k, then temperature), or greedy argmax.

Draws use a per-request key chain, never the global RNG, so seeded requests are reproducible and
stay batchable.
"""

from __future__ import annotations

import os

import mlx.core as mx

KEY_BLOCK = 256


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
    ):
        self.temperature = float(temperature or 0.0)
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
        """RNG key for `step`, derived lazily so max_tokens never sizes an allocation.

        Block b = step // KEY_BLOCK has (block key, chain_{b+1}) = split(chain_b, 2) with
        chain_0 = root, and its step keys are split(block key, KEY_BLOCK).
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
        if self.greedy:
            # argmax of the raw logits: log-softmax only shifts a row by a constant, and skipping
            # it avoids that shift's rounding and a vocab-sized temporary
            return mx.argmax(logits, axis=-1)
        lp = logprobs_from_logits(logits)
        if 0.0 < self.top_p < 1.0:
            lp = apply_top_p(lp, self.top_p)
        if self.top_k > 0:
            lp = apply_top_k(lp, self.top_k)
        return mx.random.categorical(lp * (1.0 / self.temperature), axis=-1, key=self.key(step))
