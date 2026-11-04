"""The KV cache of a full-attention layer: a growable slot per request (keys, values and the QSA
indexer's raw keys), and the copy a snapshot takes of its filled prefix."""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig

KV_STEP = 256


def _copy_prefix(x: mx.array, n: int) -> mx.array:
    """Copy of x[:, :, :n, :] in a fresh buffer.

    Uses an elementwise op because slicing, mx.slice, mx.contiguous and mx.array can share the
    source buffer when the window is contiguous.
    """
    return x[:, :, :n, :] + mx.array(0, dtype=x.dtype)


class KVSlot:
    """One layer's attention KV plus the QSA indexer's raw key cache.

    keys/values are [1, n_kv, capacity, head_dim]; `idx` is [1, 1, capacity, indexer_head_dim], the
    un-normed, un-roped indexer key of each position. The first `offset` positions are valid.
    """

    __slots__ = ("keys", "values", "idx", "offset", "capacity", "dtype", "idx_bk")

    def __init__(self, cfg: Qwen4ExpTextConfig, capacity: int, dtype=mx.bfloat16):
        capacity = max(KV_STEP, -(-capacity // KV_STEP) * KV_STEP)
        shape = (1, cfg.num_key_value_heads, capacity, cfg.head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)
        self.idx = mx.zeros((1, 1, capacity, cfg.indexer_head_dim), dtype=dtype)
        self.offset = 0
        self.capacity = capacity
        self.dtype = dtype
        # QSA block keys keyed by complete-block count; valid because idx[:offset] is never
        # rewritten (load fills a fresh slot, append writes past offset, _grow pads after it)
        self.idx_bk: dict = {}

    def load(self, keys: mx.array, values: mx.array, idx: mx.array) -> None:
        """Copy [1, n_kv, n, D] / [1, 1, n, Di] prefixes into this fresh slot's own buffers."""
        n = keys.shape[2]
        if idx.shape[2] != n:
            raise ValueError(f"idx prefix {idx.shape[2]} != kv prefix {n}")
        if n > self.capacity:
            self._grow(n + 1)
        if self.offset != 0 or n >= self.capacity:
            raise InvariantError("restore into a fresh slot with room to append")
        self.keys[:, :, :n, :] = keys
        self.values[:, :, :n, :] = values
        self.idx[:, :, :n, :] = idx
        self.offset = n
        self.idx_bk = {}  # a restored prefix is a different key cache

    def _grow(self, needed: int) -> None:
        extra = -(-(needed - self.capacity) // KV_STEP) * KV_STEP
        pad = mx.zeros((1, self.keys.shape[1], extra, self.keys.shape[3]), dtype=self.dtype)
        self.keys = mx.concatenate([self.keys, pad], axis=2)
        self.values = mx.concatenate([self.values, pad], axis=2)
        self.idx = mx.concatenate(
            [self.idx, mx.zeros((1, 1, extra, self.idx.shape[3]), dtype=self.dtype)], axis=2
        )
        self.capacity += extra

    def ensure_room(self, n: int) -> bool:
        """Make room for `n` more positions ahead of a step. Returns True if the buffers grew."""
        if self.offset + n <= self.capacity:
            return False
        self._grow(self.offset + n)
        return True

    def append(self, k: mx.array, v: mx.array, raw_idx_k: mx.array) -> tuple[mx.array, mx.array]:
        """Write k, v [1, n_kv, L, D] and raw_idx_k [1, 1, L, Di] in place; return K/V views."""
        L = k.shape[2]
        if raw_idx_k.shape[2] != L:
            raise ValueError(f"idx append {raw_idx_k.shape[2]} positions != kv append {L}")
        end = self.offset + L
        if end > self.capacity:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.idx[:, :, self.offset : end, :] = raw_idx_k
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def idx_keys(self, n: int | None = None) -> mx.array:
        """The raw indexer keys of the first `n` (default: all valid) positions: [1, 1, n, Di]."""
        n = self.offset if n is None else int(n)
        return self.idx[:, :, :n, :]
