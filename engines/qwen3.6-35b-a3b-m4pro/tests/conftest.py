"""Fixtures for this engine's tests: the device checks, and tiny Qwen3.6 (qwen3_5_moe) checkpoints
and engines.

The tiny checkpoints have the real tensor names, layouts and quantization over random weights. Their
tokenizer files are copied from a real Qwen3.6 checkpoint when one is found
($JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_TOKENIZER, $JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_CHECKPOINT or ~/models/Qwen3.6-35B-A3B-4bit).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import weakref

import pytest

MIB = 1 << 20


# ---------------------------------------------------------------- device checks
@pytest.fixture
def requires_mlx():
    """Skips the test unless MLX runs on Metal."""
    try:
        import mlx.core as mx

        mx.eval(mx.array([1]) + 1)
        metal = mx.metal.is_available()
    except Exception:  # noqa: BLE001
        metal = False
    if not metal:
        pytest.skip("no MLX/Metal")


@pytest.fixture
def requires_mlx_any():
    """MLX on any device, Metal or MLX's CPU build: for tests that run no custom kernel."""
    pytest.importorskip("mlx.core")


# ---------------------------------------------------------------- tiny qwen3_5_moe checkpoints
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

TINY = dict(
    hidden_size=128,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    rms_norm_eps=1e-6,
    norm_topk_prob=True,
    partial_rotary_factor=0.25,
)


def _real_tokenizer_dir() -> str | None:
    cand = [
        os.environ.get("JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_TOKENIZER"),
        os.environ.get("JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_CHECKPOINT"),
        os.path.expanduser("~/models/Qwen3.6-35B-A3B-4bit"),
    ]
    for c in cand:
        if c and os.path.isfile(os.path.join(c, "tokenizer.json")):
            return c
    return None


def _tiny_config(
    bits: int = 4,
    group_size: int = 64,
    vocab: int = 248320,
    experts: int | None = None,
    hidden: int | None = None,
) -> dict:
    t = dict(TINY)
    if experts is not None:
        t["num_experts"] = experts
    if hidden is not None:  # the hidden size and both expert widths
        t["hidden_size"] = t["moe_intermediate_size"] = hidden
        t["shared_expert_intermediate_size"] = hidden
    n = t["num_hidden_layers"]
    t["layer_types"] = [
        "full_attention" if (i + 1) % t["full_attention_interval"] == 0 else "linear_attention"
        for i in range(n)
    ]
    t["vocab_size"] = vocab
    t["model_type"] = "qwen3_5_moe_text"
    t["rope_parameters"] = {"rope_theta": 1e7, "partial_rotary_factor": 0.25}
    q = {"group_size": group_size, "bits": bits, "mode": "affine"}
    for i in range(n):
        q[f"language_model.model.layers.{i}.mlp.gate"] = {"group_size": group_size, "bits": 8}
        q[f"language_model.model.layers.{i}.mlp.shared_expert_gate"] = {
            "group_size": group_size,
            "bits": 8,
        }
    return {
        "model_type": "qwen3_5_moe",
        "text_config": t,
        "quantization": q,
        "quantization_config": q,
        "tie_word_embeddings": False,
        "eos_token_id": [248046, 248044],
        "vision_config": {"depth": 1},
    }


def _shapes(cfg: dict) -> dict:
    """{name: (kind, shape)}: 'q' quantized ([out, in] or [E, out, in]), 'f' bf16 or 'f32'."""
    t = cfg["text_config"]
    D = t["hidden_size"]
    out = {
        "model.embed_tokens": ("q", (t["vocab_size"], D)),
        "lm_head": ("q", (t["vocab_size"], D)),
        "model.norm.weight": ("f", (D,)),
    }
    kd = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vd = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    conv = 2 * kd + vd
    H, Hkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    E, F, S = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    for i in range(t["num_hidden_layers"]):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = ("f", (D,))
        out[p + "post_attention_layernorm.weight"] = ("f", (D,))
        if t["layer_types"][i] == "full_attention":
            a = p + "self_attn."
            out[a + "q_proj"] = ("q", (H * hd * 2, D))
            out[a + "k_proj"] = ("q", (Hkv * hd, D))
            out[a + "v_proj"] = ("q", (Hkv * hd, D))
            out[a + "o_proj"] = ("q", (D, H * hd))
            out[a + "q_norm.weight"] = ("f", (hd,))
            out[a + "k_norm.weight"] = ("f", (hd,))
        else:
            a = p + "linear_attn."
            Hv = t["linear_num_value_heads"]
            out[a + "in_proj_qkv"] = ("q", (conv, D))
            out[a + "in_proj_z"] = ("q", (vd, D))
            out[a + "in_proj_b"] = ("q", (Hv, D))
            out[a + "in_proj_a"] = ("q", (Hv, D))
            out[a + "out_proj"] = ("q", (D, vd))
            out[a + "conv1d.weight"] = ("f", (conv, t["linear_conv_kernel_dim"], 1))
            out[a + "A_log"] = ("f32", (Hv,))
            out[a + "dt_bias"] = ("f", (Hv,))
            out[a + "norm.weight"] = ("f", (t["linear_value_head_dim"],))
        m = p + "mlp."
        out[m + "gate"] = ("q", (E, D))
        out[m + "shared_expert_gate"] = ("q", (1, D))
        out[m + "shared_expert.gate_proj"] = ("q", (S, D))
        out[m + "shared_expert.up_proj"] = ("q", (S, D))
        out[m + "shared_expert.down_proj"] = ("q", (D, S))
        out[m + "switch_mlp.gate_proj"] = ("q", (E, F, D))
        out[m + "switch_mlp.up_proj"] = ("q", (E, F, D))
        out[m + "switch_mlp.down_proj"] = ("q", (E, D, F))
    return out


def _bits_for(cfg: dict, module: str) -> int:
    q = cfg["quantization"]
    o = q.get("language_model." + module)
    return int(o["bits"]) if isinstance(o, dict) else int(q["bits"])


def _write_common(path: str, cfg: dict, tokenizer_from) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if tokenizer_from is False:
        return
    src = tokenizer_from or _real_tokenizer_dir()
    if src:
        for fn in TOKENIZER_FILES:
            p = os.path.join(src, fn)
            if os.path.isfile(p):
                shutil.copy(p, os.path.join(path, fn))


def _make_tiny_checkpoint(
    path: str,
    bits: int = 4,
    group_size: int = 64,
    seed: int = 0,
    shards: int = 2,
    experts: int | None = None,
    tokenizer_from: str | None = None,
    hidden: int | None = None,
) -> str:
    """Write a loadable tiny checkpoint (random weights, real layout). Returns `path`."""
    import mlx.core as mx

    cfg = _tiny_config(bits, group_size, experts=experts, hidden=hidden)
    _write_common(path, cfg, tokenizer_from)
    key = mx.random.key(seed)
    tensors: dict = {}
    for i, (name, (kind, shape)) in enumerate(sorted(_shapes(cfg).items())):
        k = (
            mx.random.fold_in(key, i)
            if hasattr(mx.random, "fold_in")
            else mx.random.key(seed * 100003 + i)
        )
        full = "language_model." + name
        if kind == "q":
            w = (mx.random.normal(shape, key=k) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
            b = _bits_for(cfg, name)
            qw, sc, bi = mx.quantize(w, group_size=group_size, bits=b)
            tensors[full + ".weight"] = qw
            tensors[full + ".scales"] = sc.astype(mx.bfloat16)
            tensors[full + ".biases"] = bi.astype(mx.bfloat16)
        elif kind == "f32":
            tensors[full] = (mx.random.uniform(shape=shape, key=k) - 1.0).astype(mx.float32)
        else:
            base = 1.0 if name.endswith("norm.weight") or "layernorm" in name else 0.0
            tensors[full] = (base + 0.1 * mx.random.normal(shape, key=k)).astype(mx.bfloat16)
    tensors["vision_tower.patch_embed.proj.weight"] = mx.zeros((16, 3, 2, 2), dtype=mx.bfloat16)
    mx.eval(*tensors.values())
    names = sorted(tensors)
    per = -(-len(names) // shards)
    weight_map = {}
    for s in range(shards):
        chunk = names[s * per : (s + 1) * per]
        if not chunk:
            continue
        fn = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        mx.save_safetensors(os.path.join(path, fn), {n: tensors[n] for n in chunk})
        weight_map.update({n: fn for n in chunk})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return path


def _packed_shape(shape: tuple, bits: int, gs: int) -> tuple[tuple, tuple]:
    *lead, n_in = shape
    return (*lead, n_in * bits // 32), (*lead, n_in // gs)


def _write_fake_checkpoint(
    path: str,
    bits: int = 4,
    group_size: int = 64,
    mutate=None,
    cfg: dict | None = None,
    vocab: int = 512,
) -> str:
    """Valid safetensors headers over zero bytes; `mutate(entries, cfg)` builds malformed cases."""
    cfg = cfg or _tiny_config(bits, group_size, vocab=vocab)
    entries: dict = {}
    for name, (kind, shape) in _shapes(cfg).items():
        full = "language_model." + name
        if kind == "q":
            b = _bits_for(cfg, name)
            w, s = _packed_shape(shape, b, group_size)
            entries[full + ".weight"] = ("U32", w)
            entries[full + ".scales"] = ("BF16", s)
            entries[full + ".biases"] = ("BF16", s)
        else:
            entries[full] = ("F32" if kind == "f32" else "BF16", shape)
    entries["vision_tower.patch_embed.proj.weight"] = ("BF16", (16, 3, 2, 2))
    if mutate is not None:
        mutate(entries, cfg)
    _write_common(path, cfg, tokenizer_from=False)
    sizes = {"U32": 4, "BF16": 2, "F32": 4}
    hdr, off = {}, 0
    for n in sorted(entries):
        dt, shp = entries[n]
        nb = sizes[dt]
        for d in shp:
            nb *= d
        hdr[n] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(hdr).encode()
    raw += b" " * (-len(raw) % 8)
    fn = "model-00001-of-00001.safetensors"
    with open(os.path.join(path, fn), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + off)
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {n: fn for n in hdr}}, f)
    return path


@pytest.fixture(scope="session")
def tiny_config():
    """`tiny_config(bits=4, group_size=64, vocab=248320, experts=None, hidden=None)`: a tiny
    config.json dict."""
    return _tiny_config


@pytest.fixture(scope="session")
def fake_checkpoint():
    """`fake_checkpoint(path, bits=4, group_size=64, mutate=None, cfg=None, vocab=512)`: a
    checkpoint of valid headers over zero bytes, written without MLX; `mutate(entries, cfg)` edits
    the tensor entries and config first, to build malformed cases."""
    return _write_fake_checkpoint


@pytest.fixture(scope="session")
def tiny_checkpoint():
    """`tiny_checkpoint(bits=4, experts=8, hidden=None)`: the path of a loadable tiny checkpoint
    (needs MLX), built once per machine in the temp directory and reused across tests and sessions.
    `hidden` (default 128) sets the hidden size and both expert widths; at 512 the custom
    kernels' shape conditions hold (K a multiple of 512), as on the measured model."""

    def get(bits: int = 4, experts: int = 8, hidden: int | None = None) -> str:
        name = f"jitengine-qwen3.6-35b-a3b-m4pro-tiny-v1-b{bits}-e{experts}" + (
            f"-h{hidden}" if hidden else ""
        )
        p = os.path.join(tempfile.gettempdir(), name)
        if not os.path.isfile(os.path.join(p, "model.safetensors.index.json")):
            tmp = p + f".tmp{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            _make_tiny_checkpoint(tmp, bits=bits, experts=experts, hidden=hidden)
            try:
                os.rename(tmp, p)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
        return p

    return get


# Small memory buckets, so a tiny engine plans in megabytes
SMALL = dict(
    scratch_bytes=64 * MIB,
    host_bytes=64 * MIB,
    safety_bytes=32 * MIB,
    prefix_cache_bytes=16 * MIB,
    allocator_cache_bytes=16 * MIB,
    system_reserve_bytes=0,
    log_requests=False,
)


@pytest.fixture
def tiny_engine(tmp_path, tiny_checkpoint):
    """`tiny_engine(bits=4, experts=8, hidden=None, **options)`: an engine on a tiny checkpoint
    (needs Metal).

    Offload engines of one test build and share one expert pack in its `tmp_path`. Every engine the
    test made and did not close is closed at teardown.
    """
    made = []

    def make(bits: int = 4, experts: int = 8, hidden: int | None = None, **kw):
        from jitengine_qwen3_6_35b_a3b_m4pro import create_engine

        opts = dict(SMALL)
        opts.update(kw)
        if opts.get("mode") == "offload" or "expert_pack_path" in kw:
            opts.setdefault("build_expert_pack", True)
            opts.setdefault("expert_pack_path", os.path.join(str(tmp_path), "tiny.experts"))
        opts.setdefault("process_budget_bytes", 2048 * MIB)
        engine = create_engine(tiny_checkpoint(bits, experts, hidden), **opts)
        made.append(weakref.ref(engine))
        return engine

    yield make
    for ref in made:
        engine = ref()
        if engine is not None:
            engine.close()
