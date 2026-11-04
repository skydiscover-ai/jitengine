"""The KV cache of a full-attention layer: a growable slot per request, and the copy a snapshot
takes of its filled prefix."""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.engine.errors import InvariantError
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import Qwen3_5MoeConfig

KV_STEP = 256


def _copy_prefix(x: mx.array, n: int) -> mx.array:
    """Copy x[:, :, :n, :] into a fresh buffer ([1, n_kv, capacity, D] -> [1, n_kv, n, D]).

    Uses an elementwise `+ 0`: slicing, mx.contiguous, mx.slice and mx.array may share the buffer.
    """
    return x[:, :, :n, :] + mx.array(0, dtype=x.dtype)


def _copy_range(x: mx.array, a: int, b: int) -> mx.array:
    """A materialized copy of x[:, :, a:b, :] in a fresh buffer (as `_copy_prefix`)."""
    return x[:, :, a:b, :] + mx.array(0, dtype=x.dtype)


class KVSlot:
    """keys/values [1, n_kv, capacity, head_dim]; `offset` tokens are valid."""

    __slots__ = ("keys", "values", "offset", "capacity", "dtype")

    def __init__(self, cfg: Qwen3_5MoeConfig, capacity: int, dtype=mx.bfloat16):
        capacity = max(KV_STEP, -(-capacity // KV_STEP) * KV_STEP)
        shape = (1, cfg.num_key_value_heads, capacity, cfg.head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)
        self.offset = 0
        self.capacity = capacity
        self.dtype = dtype

    def load(
        self,
        keys: mx.array,
        values: mx.array,
        more_keys: mx.array | None = None,
        more_values: mx.array | None = None,
    ) -> None:
        """Copy [1, n_kv, n, D] prefix tensors, optionally followed by [1, n_kv, m, D] more, into this (fresh, empty) slot's own buffers."""
        p = keys.shape[2]
        n = p + (more_keys.shape[2] if more_keys is not None else 0)
        if n > self.capacity:
            self._grow(n + 1)
        if self.offset != 0 or n >= self.capacity:
            raise InvariantError("restore into a fresh slot with room to append")
        self.keys[:, :, :p, :] = keys
        self.values[:, :, :p, :] = values
        if more_keys is not None:
            self.keys[:, :, p:n, :] = more_keys
            self.values[:, :, p:n, :] = more_values
        self.offset = n

    def _grow(self, needed: int) -> None:
        extra = -(-(needed - self.capacity) // KV_STEP) * KV_STEP
        pad = mx.zeros((1, self.keys.shape[1], extra, self.keys.shape[3]), dtype=self.dtype)
        self.keys = mx.concatenate([self.keys, pad], axis=2)
        self.values = mx.concatenate([self.values, pad], axis=2)
        self.capacity += extra

    def ensure_room(self, n: int) -> bool:
        """Make room for `n` more positions (reallocates, so only between steps); True if grown."""
        if self.offset + n <= self.capacity:
            return False
        self._grow(self.offset + n)
        return True

    def append(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """k, v: [1, n_kv, L, D]. Writes them in place and returns the valid [.., :offset+L, ..] views."""
        L = k.shape[2]
        end = self.offset + L
        if end > self.capacity:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]
