"""Model configuration and startup validation (everything here runs BEFORE any weight is allocated).

Refusals are loud and early: unknown or unbuilt profile, context limit outside 1..262,144, a context
limit the GPU wired limit cannot hold (resolve_context_limit: an explicit limit is refused, never
truncated; `auto` sizes it to the wired limit), a checkpoint whose small files differ from the digests
pinned in its own manifest (flint-int4.json), or safetensors headers that do not match the geometry
config.json declares. Which GPUs are admitted is device.py's.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field

CONTEXT_LIMIT_MAX = 262144
PROFILES = ("main", "kv8", "relaxed")
BUILT_PROFILES = ("main", "kv8")
# attention-history bits per profile: main keeps every layer's K / V in bf16 (human ruling); kv8 stores the 14 full
# layers' K and V at 8 bits in groups of 64 (config.json kv_key_bits / kv_value_bits / kv_group_size), rings bf16
KV_BITS = {"main": 16, "kv8": 8}
CONTEXT_AUTO = "auto"
CONTEXT_AUTO_MIN = 4096  # `auto` refuses a wired limit that holds fewer tokens than this
CONTEXT_AUTO_QUANTUM = 1024  # `auto` rounds the tokens that fit down to a multiple of this
HEADROOM_BYTES = 3 * 2**30  # prefill transients + allocator slack, on top of weights + history
# a suggested wired limit leaves this much physical memory to the system
SYSTEM_RESERVE_BYTES = 6 * 2**30
# Files hashed at every start (small); the five safetensors are checked by header geometry instead,
# and by full sha256 only when verify_weights=True (22.6 GB of reads).
SMALL_FILES = (
    "config.json",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


class StartupError(ValueError):
    """A configuration the engine refuses to start with (raised before weights are allocated)."""


@dataclass(frozen=True)
class FlintConfig:
    model_path: str
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    dense_inter: int
    dense_layers: tuple
    full_layers: tuple
    sliding_layers: tuple
    sliding_window: int
    rope_theta: float
    rms_eps: float
    eos_ids: tuple
    group_size: int
    bits: int
    embed_bits: int
    gen_defaults: dict = field(default_factory=dict)
    kv_variant: tuple = (
        8,
        8,
        64,
    )  # config.json (kv_key_bits, kv_value_bits, kv_group_size): the kv8 profile

    @property
    def moe_layers(self) -> tuple:
        return tuple(i for i in range(self.n_layers) if i not in self.dense_layers)

    @property
    def history_bytes_per_token(self) -> int:
        return len(self.full_layers) * 2 * self.n_kv_heads * self.head_dim * 2

    def history_bytes_per_token_bits(self, kv_bits: int) -> int:
        """Full-layer history bytes per position: 16-bit rows, or 8-bit codes + a bf16 scale and bias per 64."""
        if kv_bits == 16:
            return self.history_bytes_per_token
        return (
            len(self.full_layers) * 2 * self.n_kv_heads * (self.head_dim + self.head_dim // 64 * 4)
        )


def load_config(model_path: str) -> FlintConfig:
    with open(os.path.join(model_path, "config.json")) as f:
        c = json.load(f)
    gen = {}
    gp = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gp):
        with open(gp) as f:
            gen = json.load(f)
    types = c["layer_types"]
    full = tuple(i for i, t in enumerate(types) if t == "full_attention")
    sliding = tuple(i for i, t in enumerate(types) if t == "sliding_attention")
    if tuple(sorted(c.get("no_rope_layer_indices", []))) != full:
        raise StartupError("config: no_rope_layer_indices must equal the full-attention layers")
    q = c.get("quantization", {})
    eos = c.get("eos_token_id")
    eos = tuple(eos) if isinstance(eos, list) else (eos,)
    rope = (c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", 10000.0))
    return FlintConfig(
        model_path=model_path,
        hidden=c["hidden_size"],
        n_layers=c["num_hidden_layers"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        vocab=c["vocab_size"],
        n_experts=c["num_experts"],
        top_k=c["num_experts_per_tok"],
        moe_inter=c["moe_intermediate_size"],
        shared_inter=c["shared_expert_intermediate_size"],
        dense_inter=c["intermediate_size"],
        dense_layers=tuple(c.get("mlp_only_layers", [])),
        full_layers=full,
        sliding_layers=sliding,
        sliding_window=c["sliding_window"],
        rope_theta=float(rope),
        rms_eps=float(c["rms_norm_eps"]),
        eos_ids=eos,
        group_size=int(q.get("group_size", 64)),
        bits=int(q.get("bits", 4)),
        embed_bits=int((c.get("mlx_export") or {}).get("embedding_bits", 8)),
        gen_defaults=gen,
        kv_variant=(
            int(c.get("kv_key_bits", 8)),
            int(c.get("kv_value_bits", 8)),
            int(c.get("kv_group_size", 64)),
        ),
    )


def check_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise StartupError(f"unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
    if profile not in BUILT_PROFILES:
        raise StartupError(
            f"profile {profile!r} is declared but not built yet in this engine; "
            f"only {', '.join(BUILT_PROFILES)} can run (refusing rather than "
            f"silently serving the main profile)"
        )


def check_kv8_config(cfg: FlintConfig) -> None:
    """The kv8 kernels are built for 8-bit K and V in groups of 64 over 128-wide heads, as config.json declares."""
    if cfg.kv_variant != (8, 8, 64) or cfg.head_dim != 128:
        raise StartupError(
            f"profile 'kv8' needs config.json kv_key_bits = kv_value_bits = 8, kv_group_size = 64 and "
            f"head_dim 128; the checkpoint declares {cfg.kv_variant}, head_dim {cfg.head_dim}"
        )


def check_context_limit(limit) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= CONTEXT_LIMIT_MAX:
        raise StartupError(
            f"context limit must be an integer in 1..{CONTEXT_LIMIT_MAX}, got {limit!r}"
        )
    return limit


def _mib(nbytes: int) -> int:
    return -(-int(nbytes) // 2**20)


def tokens_that_fit(wired_bytes: int, weights_bytes: int, bytes_per_token: int) -> int:
    """The history tokens the wired limit holds beside the weights and the headroom, rounded down to a multiple of
    CONTEXT_AUTO_QUANTUM and capped at CONTEXT_LIMIT_MAX."""
    fit = max(0, (int(wired_bytes) - weights_bytes - HEADROOM_BYTES) // bytes_per_token)
    return min(CONTEXT_LIMIT_MAX, fit // CONTEXT_AUTO_QUANTUM * CONTEXT_AUTO_QUANTUM)


def _raise_hint(
    need: int, wired_bytes: int, memory_size: int, weights_bytes: int, bytes_per_token: int
) -> str:
    """How to hold `need` bytes: the sysctl that does, capped at physical memory less SYSTEM_RESERVE_BYTES."""
    m = _mib(need)
    cap = (int(memory_size) - SYSTEM_RESERVE_BYTES) // 2**20 if memory_size else None
    if cap is None or m <= cap:
        return f"raise the limit: sudo sysctl iogpu.wired_limit_mb={m}"
    fit = tokens_that_fit(cap * 2**20, weights_bytes, bytes_per_token)
    if fit < CONTEXT_AUTO_MIN:
        return (
            f"this Mac's {memory_size / 1e9:.1f} GB cannot hold it: a wired limit of {max(cap, 0):,} MiB "
            f"(physical memory less 6 GiB for the system) is short"
        )
    if cap * 2**20 <= wired_bytes:
        return "the wired limit is already at physical memory less 6 GiB for the system; this Mac holds no more"
    return (
        f"raise the limit as far as sudo sysctl iogpu.wired_limit_mb={cap} (physical memory less 6 GiB for "
        f"the system), which holds {fit:,} tokens"
    )


def resolve_context_limit(
    limit,
    *,
    weights_bytes: int,
    bytes_per_token: int,
    wired_bytes: int,
    memory_size: int,
    profile: str = "main",
    budget: bool = False,
) -> tuple:
    """(context limit, "auto" | "explicit"). `auto` (or None): the tokens the GPU wired limit holds beside the weights
    and HEADROOM_BYTES, at most 262,144, rounded down to a multiple of 1,024; fewer than 4,096 is refused. An explicit
    limit is range-checked and refused when it does not fit; it is never truncated. Every refusal names what fits and
    the sysctl that would hold it, or, with budget=True (wired_bytes is a memory budget below the wired limit), the
    budget that would."""
    W = int(wired_bytes)
    fit = tokens_that_fit(W, weights_bytes, bytes_per_token)
    if budget:
        wtxt = f"the memory budget is {W / 2**30:.2f} GiB"

        def hint(need: int) -> str:
            return f"raise --memory-budget-gib to at least {need / 2**30:.2f}"
    else:
        wtxt = f"the GPU wired limit is {W / 1e9:.1f} GB ({W // 2**20:,} MiB)"

        def hint(need: int) -> str:
            return _raise_hint(need, W, memory_size, weights_bytes, bytes_per_token)

    if limit is None or limit == CONTEXT_AUTO:
        if fit < CONTEXT_AUTO_MIN:
            hist = CONTEXT_AUTO_MIN * bytes_per_token
            need = weights_bytes + hist + HEADROOM_BYTES
            raise StartupError(
                f"context limit auto: Flint needs {_mib(need):,} MiB for {CONTEXT_AUTO_MIN:,} tokens "
                f"(weights {weights_bytes / 1e9:.1f} + history {hist / 1e9:.1f} + headroom "
                f"{HEADROOM_BYTES / 1e9:.1f} GB, profile {profile}) but {wtxt}; {hint(need)}"
            )
        return fit, "auto"
    if isinstance(limit, str):
        raise StartupError(
            f"context limit must be {CONTEXT_AUTO!r} or an integer in 1..{CONTEXT_LIMIT_MAX}, got {limit!r}"
        )
    n = check_context_limit(limit)
    hist = n * bytes_per_token
    need = weights_bytes + hist + HEADROOM_BYTES
    if need > W:
        if fit:
            most = f"the most that fits is {fit:,} tokens" + (
                " (--context-limit auto)" if fit >= CONTEXT_AUTO_MIN else ""
            )
        else:
            most = "no history fits beside the weights and the headroom"
        raise StartupError(
            f"context limit {n:,} needs {need / 1e9:.1f} GB (weights {weights_bytes / 1e9:.1f} + "
            f"history {hist / 1e9:.1f} + headroom {HEADROOM_BYTES / 1e9:.1f}, profile {profile}) but "
            f"{wtxt}; {most}, or {hint(need)}"
        )
    return n, "explicit"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 64 * 2**20:
            raise StartupError(
                f"{os.path.basename(path)}: implausible safetensors header length {n}"
            )
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def expected_tensors(cfg: FlintConfig) -> dict:
    """name -> (dtype, shape) for every tensor this engine reads, from config.json geometry."""
    g, H, D = cfg.group_size, cfg.hidden, cfg.head_dim
    out: dict = {}

    def qlin(name, n_out, n_in, bits=4, lead=()):
        out[name + ".weight"] = ("U8", [*lead, n_out, n_in * bits // 8])
        out[name + ".scales"] = ("BF16", [*lead, n_out, n_in // g])
        out[name + ".biases"] = ("BF16", [*lead, n_out, n_in // g])

    qlin("model.embed_tokens", cfg.vocab, H, bits=cfg.embed_bits)
    qlin("lm_head", cfg.vocab, H)
    out["model.norm.weight"] = ("BF16", [H])
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}"
        out[p + ".input_layernorm.weight"] = ("BF16", [H])
        out[p + ".post_attention_layernorm.weight"] = ("BF16", [H])
        a = p + ".self_attn"
        qlin(a + ".q_proj", cfg.n_heads * D, H)
        qlin(a + ".k_proj", cfg.n_kv_heads * D, H)
        qlin(a + ".v_proj", cfg.n_kv_heads * D, H)
        qlin(a + ".o_proj", H, cfg.n_heads * D)
        out[a + ".q_norm.weight"] = ("BF16", [D])
        out[a + ".k_norm.weight"] = ("BF16", [D])
        out[a + ".softmax_offset"] = ("BF16", [cfg.n_heads])
        m = p + ".mlp"
        if i in cfg.dense_layers:
            qlin(m + ".gate_proj", cfg.dense_inter, H)
            qlin(m + ".up_proj", cfg.dense_inter, H)
            qlin(m + ".down_proj", H, cfg.dense_inter)
        else:
            out[m + ".gate.weight"] = ("BF16", [cfg.n_experts, H])
            qlin(m + ".shared_expert.gate_proj", cfg.shared_inter, H)
            qlin(m + ".shared_expert.up_proj", cfg.shared_inter, H)
            qlin(m + ".shared_expert.down_proj", H, cfg.shared_inter)
            E = (cfg.n_experts,)
            qlin(m + ".switch_mlp.gate_proj", cfg.moe_inter, H, lead=E)
            qlin(m + ".switch_mlp.up_proj", cfg.moe_inter, H, lead=E)
            qlin(m + ".switch_mlp.down_proj", H, cfg.moe_inter, lead=E)
    return out


def tensor_index(model_path: str, cfg: FlintConfig) -> dict:
    """name -> (file, dtype, shape, byte offset in file, nbytes), validated against expected_tensors."""
    want = expected_tensors(cfg)
    found: dict = {}
    files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    if not files:
        raise StartupError(f"no safetensors files in {model_path}")
    for fn in files:
        path = os.path.join(model_path, fn)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        for name, meta in safetensors_header(path).items():
            a, b = meta["data_offsets"]
            found[name] = (path, meta["dtype"], list(meta["shape"]), 8 + n + a, b - a)
    missing = sorted(set(want) - set(found))
    if missing:
        raise StartupError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
    for name, (dt, shape) in want.items():
        if found[name][1] != dt or found[name][2] != shape:
            raise StartupError(
                f"{name}: header says {found[name][1]} {found[name][2]}, config implies {dt} {shape}"
            )
    return {k: found[k] for k in want}


def check_checkpoint(model_path: str, verify_weights: bool = False) -> dict:
    """Digests of the small files against the checkpoint's own manifest (flint-int4.json)."""
    man_path = os.path.join(model_path, "flint-int4.json")
    if not os.path.exists(man_path):
        raise StartupError(
            f"{model_path}: no flint-int4.json manifest; not a Flint INT4 checkpoint"
        )
    with open(man_path) as f:
        man = json.load(f)
    if man.get("kv_format") != "bf16":
        raise StartupError(
            f"manifest kv_format {man.get('kv_format')!r}; the main profile needs bf16"
        )
    digests = man.get("checkpoint_files") or {}
    names = list(SMALL_FILES) + (
        [n for n in digests if n.endswith(".safetensors")] if verify_weights else []
    )
    for name in names:
        want = digests.get(name)
        path = os.path.join(model_path, name)
        if want is None:
            continue
        if not os.path.exists(path):
            raise StartupError(f"checkpoint file missing: {name}")
        got = _sha256(path)
        if got != want:
            raise StartupError(
                f"checkpoint file {name} sha256 {got[:16]}... differs from the manifest {want[:16]}..."
            )
    return man
