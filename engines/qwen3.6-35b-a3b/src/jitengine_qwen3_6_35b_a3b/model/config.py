"""Model configuration parsed from the checkpoint's config.json (text_config of the VL conversion)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Qwen3_5MoeConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=list)
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    # attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # moe
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    # quantization defaults
    q_group_size: int = 64
    q_bits: int = 4
    eos_token_ids: list[int] = field(default_factory=lambda: [248046, 248044])

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    def is_attention(self, i: int) -> bool:
        if self.layer_types:
            return self.layer_types[i] == "full_attention"
        return (i + 1) % self.full_attention_interval == 0


def load_config(model_path: str) -> Qwen3_5MoeConfig:
    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)
    t = raw.get("text_config", raw)
    cfg = Qwen3_5MoeConfig()
    for k in (
        "hidden_size",
        "num_hidden_layers",
        "full_attention_interval",
        "layer_types",
        "rms_norm_eps",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "norm_topk_prob",
    ):
        if k in t:
            setattr(cfg, k, t[k])
    rp = t.get("rope_parameters") or {}
    cfg.rope_theta = float(rp.get("rope_theta", t.get("rope_theta", cfg.rope_theta)))
    cfg.partial_rotary_factor = float(
        rp.get("partial_rotary_factor", t.get("partial_rotary_factor", cfg.partial_rotary_factor))
    )
    q = raw.get("quantization") or {}
    cfg.q_group_size = int(q.get("group_size", cfg.q_group_size))
    cfg.q_bits = int(q.get("bits", cfg.q_bits))
    eos = raw.get("eos_token_id", cfg.eos_token_ids)
    gen_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            g = json.load(f)
        eos = g.get("eos_token_id", eos)
    cfg.eos_token_ids = [int(e) for e in (eos if isinstance(eos, list) else [eos])]
    return cfg
