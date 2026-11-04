"""Per-request state: KV per full-attention layer, DeltaNet conv + SSM state per linear layer.

KV buffers are sized for the prompt plus a bounded reserve and grow in `KV_STEP` increments.
Snapshots and restores always copy into fresh buffers, so a live request never aliases prefix-store
memory or another request's state, and in-place KV appends keep their buffer donation.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.kv_cache import KVSlot, _copy_prefix, _copy_range
from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.linear_state import LinearStateSlot, _materialize
from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import Entry
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import Qwen3_5MoeConfig


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

    def snapshot(self, key, parent: Entry | None = None) -> Entry:
        """Immutable, evaluated copy of the state after exactly `key` (len(key) == self.tokens).

        With `parent` (a full entry whose key is a proper prefix of `key`) the KV layers store only the
        positions past it."""
        n = len(key)
        if n != self.tokens:
            raise ValueError(f"snapshot key length {n} != processed positions {self.tokens}")
        # the shared prefix is stored once, in the parent: the KV layers hold positions [parent.tokens, n)
        if parent is not None and (
            parent.parent is not None
            or not 0 < parent.tokens < n
            or tuple(key[: parent.tokens]) != parent.key
        ):
            parent = None
        p = parent.tokens if parent is not None else 0
        layers = []
        for s in self.layers:
            if isinstance(s, KVSlot):
                if s.offset != n:
                    raise ValueError(f"KV offset {s.offset} != {n}")
                if parent is not None:
                    layers.append(("kvs", _copy_range(s.keys, p, n), _copy_range(s.values, p, n)))
                else:
                    layers.append(("kv", _copy_prefix(s.keys, n), _copy_prefix(s.values, n)))
            else:
                layers.append(("delta", _materialize(s.conv), _materialize(s.ssm)))
        # the speculative drafter's own KV (one attention layer), when it covers exactly these
        # positions: stored as a trailing pseudo-layer so a prefix hit restores the drafter too
        mtp = getattr(self, "mtp", None)
        if mtp is not None and mtp.offset == n and n > 0:
            pm = [rec for rec in parent.layers if rec[0] == "mtp"] if parent is not None else []
            if pm and pm[0][1].shape[2] == p:
                layers.append(("mtp", _copy_range(mtp.keys, p, n), _copy_range(mtp.values, p, n)))
            else:
                layers.append(("mtp", _copy_prefix(mtp.keys, n), _copy_prefix(mtp.values, n)))
        entry = Entry(key, layers, parent)
        mx.eval(*entry.arrays())
        return entry

    def restore(self, entry: Entry) -> None:
        """Load an entry into this fresh state; afterwards `tokens == entry.tokens`.

        KV and conv/ssm are copied, never referenced: one entry is often restored into several
        concurrent requests, which must not share mutable buffers with each other or the entry.
        """
        if self.tokens != 0:
            raise ValueError("restore requires a fresh state")
        layers = [rec for rec in entry.layers if rec[0] != "mtp"]
        extra = [rec for rec in entry.layers if rec[0] == "mtp"]
        if len(layers) != len(self.layers):
            raise ValueError("entry/layer count mismatch")
        mtp = getattr(self, "mtp", None)
        if mtp is not None:
            mtp.keys = mtp.values = None
            mtp.offset = 0
            if extra:
                k, v = extra[0][1], extra[0][2]
                pm = (
                    [rec for rec in entry.parent.layers if rec[0] == "mtp"]
                    if entry.parent is not None
                    else []
                )
                if pm and k.shape[2] < entry.tokens:  # own positions only: the rest is the parent's
                    k = mx.concatenate([pm[0][1], k], axis=2)
                    v = mx.concatenate([pm[0][2], v], axis=2)
                    mtp.update_and_fetch(k, v)
                else:
                    mtp.update_and_fetch(_materialize(k), _materialize(v))
        base = (
            [rec for rec in entry.parent.layers if rec[0] != "mtp"]
            if entry.parent is not None
            else None
        )
        for i, (s, (kind, a, b)) in enumerate(zip(self.layers, layers)):
            if isinstance(s, KVSlot):
                if kind == "kvs" and base is not None and base[i][0] == "kv":
                    s.load(base[i][1], base[i][2], a, b)
                elif kind == "kv":
                    s.load(a, b)
                else:
                    raise ValueError("entry layer kind mismatch")
            else:
                if kind != "delta":
                    raise ValueError("entry layer kind mismatch")
                s.conv = _materialize(a)
                s.ssm = _materialize(b)
        self.tokens = entry.tokens
        mx.eval(
            *self.arrays(),
            *([mtp.keys, mtp.values] if mtp is not None and mtp.keys is not None else []),
        )
