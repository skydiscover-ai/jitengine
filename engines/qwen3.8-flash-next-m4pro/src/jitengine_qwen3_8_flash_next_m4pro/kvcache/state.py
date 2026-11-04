"""Per-request model state: attention KV, DeltaNet conv/SSM state, and PLE context/conv state.

KV buffers start at prompt plus a bounded reserve and grow in `KV_STEP` increments. Prefix-store
snapshots (`Entry`) hold materialized copies, never views: `KVSlot.append` writes in place and MLX
donates only uniquely referenced buffers, so a lingering view would force a full copy every step.
`restore` copies too, since one entry is often restored into several live requests. PLE state
lives in `RequestState.ple`, outside `layers`, so `layers` stays one slot per model layer.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.kvcache.kv_cache import KVSlot, _copy_prefix
from jitengine_qwen3_8_flash_next_m4pro.kvcache.linear_state import LinearStateSlot, _materialize
from jitengine_qwen3_8_flash_next_m4pro.kvcache.ple_state import PLESlot
from jitengine_qwen3_8_flash_next_m4pro.kvcache.prefix_cache import Entry
from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig


class RequestState:
    """All layer states of one sequence plus its position counter."""

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        prompt_len: int,
        max_new: int,
        reserve: int = 512,
        dtype=mx.bfloat16,
    ):
        """KV is sized for the prompt plus min(max_new, reserve) tokens; longer generations grow it."""
        self.cfg = cfg
        self.dtype = dtype
        self.tokens = 0  # positions already processed
        capacity = prompt_len + min(max(0, max_new), reserve)
        self.layers: list = []
        self.ple: dict[int, PLESlot] = {}
        for i in range(cfg.num_hidden_layers):
            self.layers.append(
                KVSlot(cfg, capacity, dtype=dtype)
                if cfg.is_attention(i)
                else LinearStateSlot(cfg, dtype=dtype)
            )
            if cfg.has_ple(i):
                self.ple[i] = PLESlot(cfg, dtype=dtype)

    def ensure_room(self, n: int = 1) -> bool:
        """Pre-size every KV slot for `n` more decode positions."""
        grew = False
        for s in self.layers:
            if isinstance(s, KVSlot):
                grew |= s.ensure_room(n)
        return grew

    @property
    def kv_allocated_bytes(self) -> int:
        """Bytes of attention KV currently allocated for this row."""
        return sum(
            s.keys.nbytes + s.values.nbytes + s.idx.nbytes
            for s in self.layers
            if isinstance(s, KVSlot)
        )

    def arrays(self) -> list[mx.array]:
        out = []
        for s in self.layers:
            if isinstance(s, KVSlot):
                out.extend((s.keys, s.values, s.idx))
            else:
                out.extend((s.conv, s.ssm))
        for li in sorted(self.ple):
            p = self.ple[li]
            out.extend((p.context, p.conv))
        return out

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays())

    def snapshot(self, key) -> Entry:
        """Evaluated, immutable entry of the first len(key) == `self.tokens` positions."""
        n = len(key)
        if n != self.tokens:
            raise ValueError(f"snapshot key length {n} != processed positions {self.tokens}")
        layers = []
        for s in self.layers:
            if isinstance(s, KVSlot):
                if s.offset != n:
                    raise ValueError(f"KV offset {s.offset} != {n}")
                layers.append(
                    (
                        "kv",
                        _copy_prefix(s.keys, n),
                        _copy_prefix(s.values, n),
                        _copy_prefix(s.idx, n),
                    )
                )
            else:
                layers.append(("delta", _materialize(s.conv), _materialize(s.ssm)))
        ple = [
            ("ple", li, _materialize(self.ple[li].context), _materialize(self.ple[li].conv))
            for li in sorted(self.ple)
        ]
        entry = Entry(key, layers, ple)
        mx.eval(*entry.arrays())
        return entry

    def restore(self, entry: Entry) -> None:
        """Load `entry` into this fresh state as private copies; then `tokens == entry.tokens`."""
        self._check_entry_fits(entry)
        self._load_entry_layers(entry)
        self.tokens = entry.tokens
        mx.eval(*self.arrays())

    def _check_entry_fits(self, entry: Entry) -> None:
        """Raise ValueError unless this state is fresh and entry has the same layers and PLE layers."""
        if self.tokens != 0:
            raise ValueError("restore requires a fresh state")
        if len(entry.layers) != len(self.layers):
            raise ValueError("entry/layer count mismatch")
        if [li for _, li, _, _ in entry.ple] != sorted(self.ple):
            raise ValueError("entry/PLE layer mismatch")

    def _load_entry_layers(self, entry: Entry) -> None:
        """Load each layer's entry into its slot or state, then each PLE layer's, as private copies."""
        for s, e in zip(self.layers, entry.layers):
            kind = e[0]
            if isinstance(s, KVSlot):
                if kind != "kv" or len(e) != 4:
                    raise ValueError(
                        "entry layer kind mismatch (an attention entry carries keys, values, idx)"
                    )
                s.load(e[1], e[2], e[3])
            else:
                if kind != "delta":
                    raise ValueError("entry layer kind mismatch")
                s.conv = _materialize(e[1])
                s.ssm = _materialize(e[2])
        for _, li, ctx, conv in entry.ple:
            self.ple[li].context = _materialize(ctx)
            self.ple[li].conv = _materialize(conv)
