"""The per-layer hashed n-gram embedding (PLE) of Qwen3.8-Flash-Next: the index and the layer.

Transcribed from transformers' `qwen4_exp` (`Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer`).
The index is int64 throughout and wraps like torch's int64, and it never looks back more than
`ngram_size - 1` positions, so a chunked forward with the carried context gives exactly the ids of a
full-sequence pass.
"""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.linear import QLinear, qlinear_from

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def _to_i64(values) -> list[int]:
    """Python ints of an I64 buffer / list, two's-complement folded into int64 range."""
    out = []
    for v in values.tolist() if hasattr(values, "tolist") else list(values):
        v = int(v)
        if v >= 1 << 63:
            v -= 1 << 64
        out.append(v)
    return out


class NGramIndex:
    """[B, S] token ids (+ the carried context) -> [B, S, ngram_heads] global row ids, int64."""

    def __init__(
        self,
        *,
        vocab_size: int,
        ngram_size: int,
        heads_per_ngram: int,
        ngram_vocab_size_base: int,
        make_divisible_by: int,
        seed: int,
        eos_token_id: int,
        ple_layer_index: int = 0,
        layer_multipliers=None,
        head_offsets=None,
        head_vocab_sizes=None,
    ):
        self.ngram_size = int(ngram_size)
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = int(heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.ple_layer_index = int(ple_layer_index)
        self.seed = int(seed)
        self.make_divisible_by = int(make_divisible_by)
        self.ngram_vocab_size_base = int(ngram_vocab_size_base)

        # the recomputation (HF's constructor), always available for `check_buffers`
        sizes, offsets, total = [], [], 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.computed_sizes = sizes
        self.computed_offsets = offsets
        self.computed_multipliers = build_layer_multipliers(
            self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed
        )
        self.total_vocab_size = total
        self.padded_vocab_size = math.ceil(total / self.make_divisible_by) * self.make_divisible_by

        # what the index uses: the checkpoint's buffers when given, else the recomputation
        self.layer_multipliers = (
            _to_i64(layer_multipliers)
            if layer_multipliers is not None
            else list(self.computed_multipliers)
        )
        self.head_offsets_l = _to_i64(head_offsets) if head_offsets is not None else list(offsets)
        self.head_vocab_sizes_l = (
            _to_i64(head_vocab_sizes) if head_vocab_sizes is not None else list(sizes)
        )
        if (
            len(self.layer_multipliers) != self.ngram_size
            or len(self.head_offsets_l) != self.ngram_heads
            or len(self.head_vocab_sizes_l) != self.ngram_heads
        ):
            raise ValueError("n-gram buffers have the wrong length for this config")
        self.head_vocab_sizes = mx.array(self.head_vocab_sizes_l, dtype=mx.int64)
        self.head_offsets = mx.array(self.head_offsets_l, dtype=mx.int64)

    @classmethod
    def from_config(
        cls,
        cfg,
        w: dict | None = None,
        layer: int | None = None,
        ple_layer_index: int = 0,
        check: bool = True,
    ) -> NGramIndex:
        """Build from `Qwen4ExpTextConfig`; with `w` and `layer`, use the checkpoint's I64
        buffers."""
        bufs = {}
        if w is not None and layer is not None:
            p = f"model.layers.{layer}.ple.ple_embedding."
            for k in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
                if p + k in w:
                    bufs[k] = np.asarray(w[p + k].astype(mx.int64))
        idx = cls(
            vocab_size=cfg.vocab_size,
            ngram_size=cfg.ngram_size,
            heads_per_ngram=cfg.heads_per_ngram,
            ngram_vocab_size_base=cfg.ngram_vocab_size_base,
            make_divisible_by=cfg.make_ngram_vocab_size_divisible_by,
            seed=cfg.ngram_seed,
            eos_token_id=cfg.ngram_eos_id,
            ple_layer_index=ple_layer_index,
            layer_multipliers=bufs.get("layer_multipliers"),
            head_offsets=bufs.get("ngram_heads_offsets"),
            head_vocab_sizes=bufs.get("ngram_heads_vocab_sizes"),
        )
        if check and bufs:
            idx.check_buffers()
        return idx

    def check_buffers(self) -> None:
        """The checkpoint's stored constants must equal the recomputation (seed and primes)."""
        if self.layer_multipliers != list(self.computed_multipliers):
            raise ValueError(
                f"layer_multipliers {self.layer_multipliers} != recomputed {self.computed_multipliers}"
            )
        if self.head_offsets_l != list(self.computed_offsets):
            raise ValueError("ngram_heads_offsets differ from the recomputation")
        if self.head_vocab_sizes_l != list(self.computed_sizes):
            raise ValueError("ngram_heads_vocab_sizes differ from the recomputation")

    def initial_context(self, batch: int = 1) -> mx.array:
        return mx.full((batch, self.context_len), self.eos_token_id, mx.int64)

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int) -> mx.array:
        """Shift right by `shift`, filling with EOS any position that would reach past an EOS."""
        if shift == 0:
            return token_ids
        batch, seq = token_ids.shape
        positions = mx.arange(seq, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, mx.array(-1, mx.int64))
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), previous_eos_inclusive[:, :-1]], axis=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None, :] - segment_start
        source_positions = positions - shift
        gather_positions = mx.maximum(source_positions, 0)
        shifted = mx.take_along_axis(
            token_ids, mx.broadcast_to(gather_positions[None, :], (batch, seq)), axis=1
        )
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return mx.where(valid, shifted, mx.full(token_ids.shape, self.eos_token_id, mx.int64))

    def __call__(self, input_ids: mx.array, previous_context: mx.array | None = None) -> mx.array:
        """input_ids [B, S], previous_context [B, ngram_size - 1] (EOS when None) -> row ids."""
        input_ids = input_ids.astype(mx.int64)
        batch, seq = input_ids.shape
        if previous_context is None:
            previous_context = self.initial_context(batch)
        token_history = mx.concatenate([previous_context.astype(mx.int64), input_ids], axis=-1)
        shifted = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            sizes = self.head_vocab_sizes[start:end]
            offs = self.head_offsets[start:end]
            ngram_ids = mx.remainder(mixed[..., None], sizes[None, None, :])
            blocks.append(ngram_ids + offs[None, None, :])
        return mx.concatenate(blocks, axis=-1)[:, -seq:]

    def rows_np(self, history: np.ndarray) -> np.ndarray:
        """Host (numpy) replica of `__call__` for one row: history [context_len + S] -> [S, heads].

        Same int64 wraparound multiply, xor and divisor-sign remainder as the mx path.
        """
        hist = np.asarray(history, dtype=np.int64).reshape(-1)
        seq = hist.size
        eos = self.eos_token_id
        positions = np.arange(seq, dtype=np.int64)
        eos_pos = np.where(hist == eos, positions, -1)
        prev_incl = np.maximum.accumulate(eos_pos)
        prev = np.concatenate([np.array([-1], dtype=np.int64), prev_incl[:-1]])
        seg_start = prev + 1
        pos_in_seg = positions - seg_start
        shifted = []
        for sh in range(self.ngram_size):
            if sh == 0:
                shifted.append(hist.copy())
                continue
            src = positions - sh
            gat = hist[np.maximum(src, 0)]
            valid = (pos_in_seg >= sh) & (src >= 0)
            shifted.append(np.where(valid, gat, eos))
        mults = [np.int64(m) for m in self.layer_multipliers]
        sizes = np.asarray(self.head_vocab_sizes_l, dtype=np.int64)
        offs = np.asarray(self.head_offsets_l, dtype=np.int64)
        blocks = []
        with np.errstate(over="ignore"):
            for ngram in range(2, self.ngram_size + 1):
                start = (ngram - 2) * self.heads_per_ngram
                end = start + self.heads_per_ngram
                mixed = shifted[0] * mults[0]
                for position in range(1, ngram):
                    mixed = np.bitwise_xor(mixed, shifted[position] * mults[position])
                ids = np.remainder(mixed[:, None], sizes[None, start:end])
                blocks.append(ids + offs[None, start:end])
        out = np.concatenate(blocks, axis=-1)
        return out[-(seq - self.context_len) :] if seq > self.context_len else out[-1:]

    @staticmethod
    def next_context(token_history: mx.array, context_len: int) -> mx.array:
        """The context to carry after `token_history` [B, L]: its last `context_len` ids."""
        return token_history[:, -context_len:]


# ----------------------------------------------------------------------------- the layer
# Grouped `(1 + w)` RMSNorm in one compiled launch, as hyper_connections.py's `_hc_norm` (no
# `.shape` read inside the shapeless region); this module keeps its own copy.


@partial(mx.compile, shapeless=True)
def _ple_norm(xg: mx.array, w1g: mx.array, eps: float) -> mx.array:
    return (mx.fast.rms_norm(xg, None, eps) * w1g).astype(xg.dtype)


class PLEState:
    """Per-request PLE state: the 2-token n-gram context and the conv's last `state_len` normed rows."""

    __slots__ = ("context", "conv", "cached")

    def __init__(self, context: mx.array, conv: mx.array):
        self.context = context  # [1, ngram_size - 1] int64
        self.conv = conv  # [1, state_len, hc x hidden] compute dtype
        # (history bytes, rows) of the last single-token call. Rows depend only on the token
        # history, so a rebuilt speculative layer reuses them: each position's rows are read once.
        self.cached = None


class Qwen4ExpTextPLELayer:
    """`Qwen4ExpTextPLELayer` in the engine's dtype over the FNXPLE01 stream."""

    def __init__(
        self,
        cfg,
        w: dict,
        p: str,
        store,
        ngram: NGramIndex,
        compute_dtype=mx.float32,
        hc_path: str = "compiled",
    ):
        self.cfg = cfg
        self.store = store
        self.ngram = ngram
        self.dtype = compute_dtype
        self.hid = int(cfg.hidden_size)
        self.hc = int(cfg.hc_count)
        self.eps = float(cfg.rms_norm_eps)
        self.key_proj: QLinear = qlinear_from(w, p + "key_proj", cfg)  # [hc*hid, ple_embed]
        self.value_proj: QLinear = qlinear_from(w, p + "value_proj", cfg)  # [hid, ple_embed]
        self.norm_key_w1 = 1.0 + w[p + "norm_key.weight"].astype(mx.float32)
        self.norm_query_w1 = 1.0 + w[p + "norm_query.weight"].astype(mx.float32)
        self.norm_conv_w1 = 1.0 + w[p + "norm_conv.weight"].astype(mx.float32)
        # the same three constants, grouped, for the compiled norm region (a reshape: same values)
        self.hc_path = str(hc_path)
        self.norm_key_w1g = self.norm_key_w1.reshape(self.hc, self.hid)
        self.norm_query_w1g = self.norm_query_w1.reshape(self.hc, self.hid)
        self.norm_conv_w1g = self.norm_conv_w1.reshape(self.hc, self.hid)
        conv_w = w[p + "conv1d.weight"]  # [C, K, 1]
        self.conv_k = int(conv_w.shape[1])
        self.conv_w32 = conv_w.astype(mx.float32).reshape(conv_w.shape[0], self.conv_k)  # [C, K]
        self.dilation = int(cfg.ngram_size)
        self.state_len = (self.conv_k - 1) * self.dilation
        self.ple_embed = int(cfg.ple_embed_dim)
        self.values_per_row = int(store.values_per_row)
        # materialized here: a lazy weight-derived node would be bound to the constructing thread's
        # stream and fail to evaluate on the model worker thread (see model.py)
        mx.eval(
            self.norm_key_w1,
            self.norm_query_w1,
            self.norm_conv_w1,
            self.norm_key_w1g,
            self.norm_query_w1g,
            self.norm_conv_w1g,
            self.conv_w32,
            self.ngram.head_vocab_sizes,
            self.ngram.head_offsets,
        )
        if self.values_per_row * self.ngram.ngram_heads != self.ple_embed:
            raise ValueError(
                f"PLE rows of {self.values_per_row} values x {self.ngram.ngram_heads} heads "
                f"!= ple_embed_dim {self.ple_embed}"
            )

    # ------------------------------------------------------------------ pieces
    def fresh_state(self, batch: int = 1) -> PLEState:
        return PLEState(
            self.ngram.initial_context(batch),
            mx.zeros((batch, self.state_len, self.hc * self.hid), dtype=self.dtype),
        )

    def _gnorm(self, x: mx.array, w1: mx.array, w1g: mx.array | None = None) -> mx.array:
        """Grouped `(1 + w)` RMSNorm: normalise each `hid`-wide stream, scale by the fp32 `1 + w`."""
        shp = x.shape
        xg = x.reshape(*shp[:-1], self.hc, self.hid)
        if self.hc_path == "compiled" and w1g is not None:
            return _ple_norm(xg, w1g, self.eps).reshape(shp)
        g = mx.fast.rms_norm(xg, None, self.eps).reshape(shp)
        return (g * w1).astype(x.dtype)

    def embed(
        self, ids: mx.array, context: mx.array, row: int | None = None
    ) -> tuple[mx.array, mx.array]:
        """[B, S] ids + [B, 2] context -> ([B, S, ple_embed] rows in compute dtype, row ids)."""
        row_ids = self.ngram(ids, context)  # [B, S, heads] int64
        b, s, h = row_ids.shape
        row_of = (
            np.full(b * s * h, int(row), dtype=np.int64)
            if row is not None
            else np.repeat(np.arange(b, dtype=np.int64), s * h)
        )
        rows = self.store.fetch(np.asarray(row_ids).reshape(-1), row_of)  # [B*S*heads, values]
        return rows.reshape(b, s, h * rows.shape[-1]).astype(self.dtype), row_ids

    def _conv(self, normed: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        """Dilated causal depthwise conv + silu; returns (out, last `state_len` pre-conv rows)."""
        _b, s, _c = normed.shape
        xt = mx.concatenate([state, normed], axis=1)  # [B, state_len + S, C]
        x32 = xt.astype(mx.float32)
        acc = None
        for j in range(self.conv_k):
            off = j * self.dilation
            term = x32[:, off : off + s, :] * self.conv_w32[:, j][None, None, :]
            acc = term if acc is None else acc + term
        out = (acc * mx.sigmoid(acc)).astype(normed.dtype)
        return out, xt[:, -self.state_len :, :].astype(
            self.dtype
        )  # the state stays in the compute dtype

    def apply(
        self, h: mx.array, emb: mx.array, state_conv: mx.array, rows: bool = False
    ) -> tuple[mx.array, mx.array]:
        """The layer body given the gathered rows: h [B, S, hc*hid], emb [B, S, ple_embed].

        `rows=True` treats B as independent single positions and routes the dense projections
        through `QLinear.rows`, so each row equals its solo result bit for bit.
        """
        hc, hid = self.hc, self.hid
        dense = (lambda lin: lin.rows) if rows else (lambda lin: lin)
        key_normed = self._gnorm(dense(self.key_proj)(emb), self.norm_key_w1, self.norm_key_w1g)
        key_normed = key_normed.reshape(*key_normed.shape[:-1], hc, hid)
        value = dense(self.value_proj)(emb)  # [B, S, hid]
        query_normed = self._gnorm(h, self.norm_query_w1, self.norm_query_w1g).reshape(
            *h.shape[:-1], hc, hid
        )
        gate = mx.sum(key_normed * query_normed, axis=-1, keepdims=True) / math.sqrt(hid)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated_value = mx.sigmoid(gate) * mx.expand_dims(value, -2)  # [B, S, hc, hid]
        flat = gated_value.reshape(*gated_value.shape[:-2], hc * hid)
        normed = self._gnorm(flat, self.norm_conv_w1, self.norm_conv_w1g)
        conv, new_state = self._conv(normed, state_conv)
        return flat + conv, new_state

    # ------------------------------------------------------------------ entry points
    def __call__(
        self, h: mx.array, ids: mx.array, state: PLEState, trace: dict | None = None, row: int = 0
    ) -> mx.array:
        """One row: h [1, S, hc*hid], ids [1, S]; advances `state` (context and conv) in place."""
        if trace is None and ids.shape[-1] == 1 and h.shape[1] == 1:
            tok = np.asarray(ids).astype(np.int64).reshape(-1)  # materialized by layer 0's sync
            ctx = np.asarray(state.context).astype(np.int64).reshape(-1)
            hist = np.concatenate([ctx, tok])
            key = hist.tobytes()
            cached = getattr(state, "cached", None)
            if cached is not None and cached[0] == key:
                emb = cached[1]  # the rebuild of an undone build
            else:
                rows_np_ = self.ngram.rows_np(hist)  # [1, heads], no GPU work
                rows = self.store.fetch(
                    rows_np_.reshape(-1), np.zeros(rows_np_.size, dtype=np.int64)
                )
                emb = rows.reshape(1, 1, -1).astype(self.dtype)
                state.cached = (key, emb)
            out, new_conv = self.apply(h, emb, state.conv)
            state.context = mx.array(hist[-self.ngram.context_len :])[None]
            state.conv = new_conv * mx.array(1, dtype=new_conv.dtype)
            return out
        emb, row_ids = self.embed(ids, state.context, row=row)
        out, new_conv = self.apply(h, emb, state.conv)
        history = mx.concatenate([state.context, ids.astype(mx.int64)], axis=-1)
        state.context = NGramIndex.next_context(history, self.ngram.context_len)
        state.conv = new_conv * mx.array(1, dtype=new_conv.dtype)  # a fresh buffer, never a view
        if trace is not None:
            trace["ple_row_ids"] = row_ids
            trace["ple_emb"] = emb
        return out

    def step(self, h: mx.array, ids: mx.array, states: list) -> mx.array:
        """B independent decode rows: h [B, 1, hc*hid], ids [B, 1], one PLEState per row."""
        b = h.shape[0]
        context = (
            mx.concatenate([s.context for s in states], axis=0) if b > 1 else states[0].context
        )
        conv_state = mx.concatenate([s.conv for s in states], axis=0) if b > 1 else states[0].conv
        emb, _ = self.embed(ids, context)
        out, new_conv = self.apply(h, emb, conv_state, rows=True)
        history = mx.concatenate([context, ids.astype(mx.int64)], axis=-1)
        nxt = NGramIndex.next_context(history, self.ngram.context_len)
        for i, s in enumerate(states):
            s.context = nxt[i : i + 1]
            s.conv = new_conv[i : i + 1] * mx.array(1, dtype=new_conv.dtype)
        return out
