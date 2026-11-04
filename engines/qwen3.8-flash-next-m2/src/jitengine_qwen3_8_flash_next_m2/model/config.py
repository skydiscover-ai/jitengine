"""Model configuration parsed from the checkpoint's config.json.

Defaults are this checkpoint's (Qwen3.8-Flash-Next) values, so a missing key cannot silently yield
another model's geometry. `eos_token_ids` (stops generation) and `ngram_eos_id` (the PLE n-gram
boundary) are different ids. `is_attention` tests `!= "linear_attention"` because the HF config
class renames `full_attention` to `qwen_sparse_attention`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# HF's default `Qwen4ExpTextConfig.seed`; the text_config has no `seed` key (see `check_buffers`).
NGRAM_SEED_DEFAULT = 1234


@dataclass
class Qwen4ExpTextConfig:
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=list)
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    hidden_act: str = "silu"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    # QSA indexer
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # moe
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # PLE (per-layer hashed n-gram embedding)
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    ple_conv_kernel_size: int = 4
    ple_embed_dim: int = 2560
    ngram_seed: int = NGRAM_SEED_DEFAULT
    ngram_eos_id: int = 248044
    # quantization: default plus per-module overrides by prefix-stripped path; see `quant_of`
    q_group_size: int = 32
    q_bits: int = 4
    q_overrides: dict = field(default_factory=dict)
    tensor_prefix: str = "language_model."
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

    @property
    def stream_dim(self) -> int:
        """Width of the carried hyper-connection residual: hc_count copies of the hidden size."""
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ple_conv_state_len(self) -> int:
        """Rows of the pre-conv tensor the dilated PLE conv needs from the past: (K - 1) x dilation."""
        return (self.ple_conv_kernel_size - 1) * self.ngram_size

    def is_attention(self, i: int) -> bool:
        if self.layer_types:
            return self.layer_types[i] != "linear_attention"
        return (i + 1) % self.full_attention_interval == 0

    def has_ple(self, i: int) -> bool:
        """Layer index `i` (0-based) carries the PLE injection iff `i + 1` is in `ple_layer_ids`."""
        return (i + 1) in self.ple_layer_ids

    def quant_of(self, module: str) -> tuple[int, int]:
        """(bits, group_size) of a module path; never inferred from shapes, which are ambiguous."""
        ov = self.q_overrides.get(module)
        if ov is not None:
            return ov
        return self.q_bits, self.q_group_size


def _derive_prefix(model_path: str) -> str | None:
    """The tensor prefix in front of `model.`, read from the safetensors index."""
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import PackError, detect_prefix

    # an unreadable index gives no prefix here; the pack module refuses it loudly where it matters
    try:
        with open(idx) as f:
            return detect_prefix(json.load(f)["weight_map"])
    except OSError, ValueError, KeyError, PackError:
        return None


def load_config(model_path: str) -> Qwen4ExpTextConfig:
    model_path = os.path.expanduser(model_path)
    raw = _read_config_json(model_path)
    t = raw.get("text_config", raw)
    cfg = Qwen4ExpTextConfig()
    _apply_text_config(cfg, t)

    _apply_quantization(cfg, raw, model_path)

    _apply_eos_token_ids(cfg, raw, model_path)
    return cfg


def _read_config_json(model_path: str) -> dict:
    """The parsed config.json of the checkpoint."""
    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)
    return raw


def _apply_text_config(cfg: Qwen4ExpTextConfig, t: dict) -> None:
    """Copy the text config's fields, rope parameters, n-gram seed and boundary id onto cfg."""
    for k in (
        "hidden_size",
        "num_hidden_layers",
        "full_attention_interval",
        "layer_types",
        "rms_norm_eps",
        "vocab_size",
        "hidden_act",
        "hc_count",
        "hc_lowrank",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "output_gate_type",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "norm_topk_prob",
        "ple_layer_ids",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "ple_conv_kernel_size",
        "ple_embed_dim",
    ):
        if k in t and t[k] is not None:
            setattr(cfg, k, t[k])
    cfg.layer_types = list(cfg.layer_types or [])
    cfg.ple_layer_ids = [int(x) for x in (cfg.ple_layer_ids or [])]
    # the gated norm's activation is `output_gate_type or hidden_act`
    cfg.output_gate_type = str(
        t.get("output_gate_type") or t.get("hidden_act") or cfg.output_gate_type
    )
    rp = t.get("rope_parameters") or {}
    cfg.rope_theta = float(rp.get("rope_theta", t.get("rope_theta", cfg.rope_theta)))
    cfg.partial_rotary_factor = float(
        rp.get("partial_rotary_factor", t.get("partial_rotary_factor", cfg.partial_rotary_factor))
    )
    if rp.get("mrope_section"):
        cfg.mrope_section = [int(x) for x in rp["mrope_section"]]
    cfg.ngram_seed = int(t.get("seed") or NGRAM_SEED_DEFAULT)

    # the n-gram boundary id: the text config's single eos (a list's first element if it is one)
    te = t.get("eos_token_id", cfg.ngram_eos_id)
    cfg.ngram_eos_id = int(te[0] if isinstance(te, list) else te)


def _apply_quantization(cfg: Qwen4ExpTextConfig, raw: dict, model_path: str) -> None:
    """Set cfg's quantization group size, bits, tensor prefix and per-tensor overrides."""
    q = raw.get("quantization") or {}
    cfg.q_group_size = int(q.get("group_size", cfg.q_group_size))
    cfg.q_bits = int(q.get("bits", cfg.q_bits))
    prefix = _derive_prefix(model_path)
    if prefix is not None:
        cfg.tensor_prefix = prefix
    overrides: dict[str, tuple[int, int]] = {}
    for k, v in q.items():
        if not isinstance(v, dict):
            continue
        name = k[len(cfg.tensor_prefix) :] if k.startswith(cfg.tensor_prefix) else k
        overrides[name] = (int(v["bits"]), int(v["group_size"]))
    cfg.q_overrides = overrides


def _apply_eos_token_ids(cfg: Qwen4ExpTextConfig, raw: dict, model_path: str) -> None:
    """Set cfg's stop ids from generation_config.json when present, else from config.json."""
    # what stops a generation: generation_config.json's list, else config.json's
    eos = raw.get("eos_token_id", cfg.eos_token_ids)
    gen_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            g = json.load(f)
        eos = g.get("eos_token_id", eos)
    cfg.eos_token_ids = [int(e) for e in (eos if isinstance(eos, list) else [eos])]
