"""Per-request state: KV per full-attention layer, DeltaNet conv + SSM state per linear layer.

KV buffers are sized for the prompt plus a bounded reserve and grow in `KV_STEP` increments.
Snapshots and restores always copy into fresh buffers, so a live request never aliases prefix-store
memory or another request's state, and in-place KV appends keep their buffer donation.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import KVSlot, _copy_prefix
from jitengine_qwen3_6_35b_a3b_m2.kvcache.linear_state import LinearStateSlot, _materialize
from jitengine_qwen3_6_35b_a3b_m2.kvcache.prefix_cache import Entry
from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig


class RequestState:
    """All layer states of one sequence plus its position counter."""

    def __init__(self, cfg: Qwen3_5MoeConfig, prompt_len: int, max_new: int, reserve: int = 512):
        """KV is sized for the prompt plus min(max_new, reserve) tokens; longer generations grow it."""
        self.cfg = cfg
        self.tokens = 0  # positions already processed
        capacity = prompt_len + min(max(0, max_new), reserve)
        self.layers: list = []
        for i in range(cfg.num_hidden_layers):
            self.layers.append(
                KVSlot(cfg, capacity) if cfg.is_attention(i) else LinearStateSlot(cfg)
            )

    def ensure_room(self, n: int = 1) -> bool:
        """Pre-size every KV slot for `n` more positions, so no step reallocates midway."""
        grew = False
        for s in self.layers:
            if isinstance(s, KVSlot):
                grew |= s.ensure_room(n)
        return grew

    @property
    def kv_allocated_bytes(self) -> int:
        """Attention KV bytes allocated for this row so far (`Req.kv_bytes` is the reservation)."""
        return sum(s.keys.nbytes + s.values.nbytes for s in self.layers if isinstance(s, KVSlot))

    def arrays(self) -> list[mx.array]:
        out = []
        for s in self.layers:
            if isinstance(s, KVSlot):
                out.extend((s.keys, s.values))
            else:
                out.extend((s.conv, s.ssm))
        return out

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays())

    def snapshot(self, key) -> Entry:
        """Immutable, evaluated copy of the state after exactly `key` (len(key) == self.tokens)."""
        n = len(key)
        if n != self.tokens:
            raise ValueError(f"snapshot key length {n} != processed positions {self.tokens}")
        layers = []
        for s in self.layers:
            if isinstance(s, KVSlot):
                if s.offset != n:
                    raise ValueError(f"KV offset {s.offset} != {n}")
                layers.append(("kv", _copy_prefix(s.keys, n), _copy_prefix(s.values, n)))
            else:
                layers.append(("delta", _materialize(s.conv), _materialize(s.ssm)))
        entry = Entry(key, layers)
        mx.eval(*entry.arrays())
        return entry

    def restore(self, entry: Entry) -> None:
        """Load an entry into this fresh state; afterwards `tokens == entry.tokens`.

        KV and conv/ssm are copied, never referenced: one entry is often restored into several
        concurrent requests, which must not share mutable buffers with each other or the entry.
        """
        if self.tokens != 0:
            raise ValueError("restore requires a fresh state")
        layers = entry.layers
        if len(layers) != len(self.layers):
            raise ValueError("entry/layer count mismatch")
        for s, (kind, a, b) in zip(self.layers, layers):
            if isinstance(s, KVSlot):
                if kind != "kv":
                    raise ValueError("entry layer kind mismatch")
                s.load(a, b)
            else:
                if kind != "delta":
                    raise ValueError("entry layer kind mismatch")
                s.conv = _materialize(a)
                s.ssm = _materialize(b)
        self.tokens = entry.tokens
        mx.eval(*self.arrays())
