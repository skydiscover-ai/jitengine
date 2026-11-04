"""The state of a PLE layer: per request, the n-gram context and the conv's last normed rows, and
the bytes one row's PLE state takes over all PLE layers."""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig


class PLESlot:
    """context [1, ngram_size-1] int64 (EOS-filled at start); conv [1, state_len, hc*hidden] (zeros)."""

    __slots__ = ("context", "conv", "cached")

    def __init__(self, cfg: Qwen4ExpTextConfig, dtype=mx.bfloat16):
        self.context = mx.full((1, cfg.ngram_size - 1), cfg.ngram_eos_id, mx.int64)
        self.conv = mx.zeros((1, cfg.ple_conv_state_len, cfg.stream_dim), dtype=dtype)
        self.cached = None  # (history bytes, gathered rows) of the last single-token call (ple.py)
