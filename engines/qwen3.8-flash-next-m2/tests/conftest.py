"""Fixtures for this engine's tests: the device check, and a tiny Qwen3.8-Flash-Next (`qwen4_exp`)
checkpoint, its expert pack, and engines on them.

The checkpoint has the real tensor names, quantization (4-bit affine, group 32; routers 8-bit,
group 64) and every block type at toy size: four layers (three Gated DeltaNet, one sparse
attention), four hyper-connection streams of 64, eight experts with top-2, and a QSA indexer whose
budget of 16 positions prunes every prompt past 20 tokens. It has no PLE layer, so no PLE pack
(`tiny_flashnext_ple` is the same checkpoint with one PLE layer and its pack); widths that are
multiples of 32 let the fused decode kernels run (the dense kernel's table serves only the real
shapes). Weights are random and fixed, the tokenizer is a five-word stand-in (the tests pass
token ids), and the expert pack is written by the package's own pack writer.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest


# ---------------------------------------------------------------- device check
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


# ---------------------------------------------------------------- tiny Flash-Next checkpoint
VERSION = 1  # bump when the geometry or the weights change: the cache key
PLE_VERSION = 1  # the same, for the checkpoint with a PLE layer
EOS = [248046, 248044]
TEXT = dict(
    model_type="qwen4_exp_text",
    hidden_size=64,
    num_hidden_layers=4,
    full_attention_interval=4,
    hc_count=4,
    hc_lowrank=32,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    linear_num_value_heads=6,
    linear_num_key_heads=2,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    partial_rotary_factor=0.25,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    indexer_compress_ratio=4,
    indexer_budget=16,
    ple_layer_ids=[],
    vocab_size=248320,
    eos_token_id=EOS[1],
    rms_norm_eps=1e-6,
    norm_topk_prob=True,
    tie_word_embeddings=False,
    rope_parameters=dict(rope_theta=10000000, partial_rotary_factor=0.25),
)
# The variant with one PLE layer (layer index 1: `ple_layer_ids` are 1-based): two n-gram heads of
# 32 values each fill `ple_embed_dim` = 64; the table is 2 shards of 64 rows, the 112 ids of the two
# heads' prime sizes (53 and 59) padded to a multiple of 128.
PLE_TEXT = dict(
    ple_layer_ids=[2],
    ngram_size=3,
    heads_per_ngram=1,
    ngram_vocab_size_base=50,
    make_ngram_vocab_size_divisible_by=128,
    ple_conv_kernel_size=4,
    ple_embed_dim=64,
)
PLE_SHARDS, PLE_ROWS_PER_SHARD = 2, 64
Q = {"group_size": 32, "bits": 4, "mode": "affine"}
Q_ROUTER = {"group_size": 64, "bits": 8, "mode": "affine"}
LM, MDL = "language_model.", "language_model.model."


class _Weights:
    """The checkpoint's tensors as they are made: random bf16 weights, one key per draw in a fixed
    order, quantized as the real checkpoint is."""

    def __init__(self, mx):
        self.mx, self.out, self.overrides, self.draws = mx, {}, {}, 0

    def rnd(self, shape, std=1.0, mean=0.0):
        mx = self.mx
        self.draws += 1
        return (mean + std * mx.random.normal(shape, key=mx.random.key(self.draws))).astype(
            mx.bfloat16
        )

    def linear(self, name, n, k, q=Q, std=None):
        w = self.rnd((n, k), k**-0.5 if std is None else std)
        self.out[name + ".weight"], self.out[name + ".scales"], self.out[name + ".biases"] = (
            self.mx.quantize(w, group_size=q["group_size"], bits=q["bits"])
        )
        if q is not Q:
            self.overrides[name] = dict(q)

    def experts(self, name, n, k):
        w = self.rnd((TEXT["num_experts"], n, k), k**-0.5)
        self.out[name + ".weight"], self.out[name + ".scales"], self.out[name + ".biases"] = (
            self.mx.quantize(w, group_size=32, bits=4)
        )

    def hyper(self, p, inject):
        t = TEXT
        hcH = t["hc_count"] * t["hidden_size"]
        self.out[p + "hc_norm.weight"] = self.rnd((hcH,), 0.1)
        self.linear(p + "input_mix_weight_down", t["hc_lowrank"], hcH)
        self.linear(p + "input_mix_weight_up", hcH, t["hc_lowrank"])
        if inject:
            self.linear(p + "block_inject_weight", t["hc_count"], hcH)


def _linear_attention(w, a):
    t = TEXT
    H, Hv = t["hidden_size"], t["linear_num_value_heads"]
    kd = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    vd = Hv * t["linear_value_head_dim"]
    w.out[a + "A_log"] = w.rnd((Hv,), 0.3, 0.5)
    w.out[a + "conv1d.weight"] = w.rnd((2 * kd + vd, t["linear_conv_kernel_dim"], 1), 0.5)
    w.out[a + "dt_bias"] = w.rnd((Hv,), 0.5)
    w.linear(a + "in_proj_a", Hv, H)
    w.linear(a + "in_proj_b", Hv, H)
    w.linear(a + "in_proj_qkv", 2 * kd + vd, H)
    w.linear(a + "in_proj_z", vd, H)
    w.out[a + "norm.weight"] = w.rnd((t["linear_value_head_dim"],), 0.1, 1.0)
    w.linear(a + "out_proj", H, vd)


def _sparse_attention(w, a):
    t = TEXT
    H, hd = t["hidden_size"], t["head_dim"]
    nh, nkv = t["num_attention_heads"], t["num_key_value_heads"]
    idx_out = (t["indexer_n_heads"] + t["indexer_kv_heads"]) * t["indexer_head_dim"]
    w.linear(a + "indexer.index_qk_proj", idx_out, H)
    w.out[a + "indexer.k_layernorm.weight"] = w.rnd((t["indexer_head_dim"],), 0.1)
    w.out[a + "indexer.q_layernorm.weight"] = w.rnd((t["indexer_head_dim"],), 0.1)
    # q/k norm deltas near +1 and a large o_proj make attention pick tokens, so the QSA mask
    # changes the output
    w.out[a + "k_norm.weight"] = w.rnd((hd,), 0.1, 1.0)
    w.out[a + "q_norm.weight"] = w.rnd((hd,), 0.1, 1.0)
    w.linear(a + "q_proj", nh * hd * 2, H)
    w.linear(a + "k_proj", nkv * hd, H)
    w.linear(a + "v_proj", nkv * hd, H)
    w.linear(a + "o_proj", H, nh * hd, std=6.0 * (nh * hd) ** -0.5)


def _moe(w, m):
    t = TEXT
    H, shared, inter = (
        t["hidden_size"],
        t["shared_expert_intermediate_size"],
        t["moe_intermediate_size"],
    )
    w.linear(m + "gate", t["num_experts"], H, Q_ROUTER)
    w.linear(m + "shared_expert.gate_proj", shared, H)
    w.linear(m + "shared_expert.up_proj", shared, H)
    w.linear(m + "shared_expert.down_proj", H, shared)
    w.linear(m + "shared_expert_gate", 1, H, Q_ROUTER)
    w.experts(m + "switch_mlp.gate_proj", inter, H)
    w.experts(m + "switch_mlp.up_proj", inter, H)
    w.experts(m + "switch_mlp.down_proj", H, inter)


def _ple_tensors(w, layer):
    """The PLE layer's tensors, drawn after every other one so that a checkpoint with a PLE layer
    has the default checkpoint's weights everywhere else."""
    from jitengine_qwen3_8_flash_next_m2.model.ple import NGramIndex

    t, mx = TEXT, w.mx
    hcH, p = t["hc_count"] * t["hidden_size"], f"{MDL}layers.{layer}.ple."
    w.linear(p + "key_proj", hcH, PLE_TEXT["ple_embed_dim"])
    w.linear(p + "value_proj", t["hidden_size"], PLE_TEXT["ple_embed_dim"])
    for name in ("norm_key", "norm_query", "norm_conv"):
        w.out[p + name + ".weight"] = w.rnd((hcH,), 0.1)
    w.out[p + "conv1d.weight"] = w.rnd((hcH, PLE_TEXT["ple_conv_kernel_size"], 1), 0.5)
    index = NGramIndex(
        vocab_size=t["vocab_size"],
        ngram_size=PLE_TEXT["ngram_size"],
        heads_per_ngram=PLE_TEXT["heads_per_ngram"],
        ngram_vocab_size_base=PLE_TEXT["ngram_vocab_size_base"],
        make_divisible_by=PLE_TEXT["make_ngram_vocab_size_divisible_by"],
        seed=1234,
        eos_token_id=EOS[1],
    )
    for name, vals in (
        ("layer_multipliers", index.computed_multipliers),
        ("ngram_heads_offsets", index.computed_offsets),
        ("ngram_heads_vocab_sizes", index.computed_sizes),
    ):
        w.out[p + "ple_embedding." + name] = mx.array(vals, dtype=mx.int64)
    for shard in range(PLE_SHARDS):
        table = w.rnd((PLE_ROWS_PER_SHARD, 32), 1.0)
        base = f"{p}ple_embedding.ngram_embedding.shards.{shard}."
        w.out[base + "weight"], w.out[base + "scales"], w.out[base + "biases"] = mx.quantize(
            table, group_size=32, bits=4
        )


def _tensors(mx, ple=False):
    """{name: array} of the whole checkpoint, and the per-tensor quantization overrides."""
    t, w = TEXT, _Weights(mx)
    w.linear(MDL + "embed_tokens", t["vocab_size"], t["hidden_size"], std=1.0)
    for i in range(t["num_hidden_layers"]):
        p = f"{MDL}layers.{i}."
        w.hyper(p + "attn_hyper_connection.", True)
        w.hyper(p + "mlp_hyper_connection.", True)
        if (i + 1) % t["full_attention_interval"]:
            _linear_attention(w, p + "linear_attn.")
        else:
            _sparse_attention(w, p + "self_attn.")
        _moe(w, p + "mlp.")
    w.hyper(MDL + "hyper_connection_mixer.", False)
    w.linear(LM + "lm_head", t["vocab_size"], t["hidden_size"], std=3.0 * t["hidden_size"] ** -0.5)
    if ple:
        _ple_tensors(w, layer=PLE_TEXT["ple_layer_ids"][0] - 1)
    return w.out, w.overrides


def _tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "user", "assistant", "hello", "."]
    core = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.save_pretrained(path)


def _expert_pack(model_path, pack_path):
    """The package's expert pack writer over the checkpoint's own geometry, which `ExpertScan`
    takes as it is (`expert_writer.scan_checkpoint`, which pins the real checkpoint's, is skipped)."""
    from jitengine_qwen3_8_flash_next_m2.checkpoint import expert_writer
    from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import ExpertScan

    expert_writer.build_pack(ExpertScan(model_path), pack_path, log=lambda *_: None)


def _ple_pack(model_path, pack_path):
    """The package's PLE pack writer over the checkpoint's own table, which `PLEScan` takes as it
    is (`ple_writer.scan_checkpoint`, which pins the real table's geometry, is skipped)."""
    from jitengine_qwen3_8_flash_next_m2.checkpoint import ple_writer
    from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import PLEScan, segments_for

    scan = PLEScan(model_path)
    shards = list(range(scan.num_shards))
    rows = {s: (0, scan.rows_per_shard) for s in shards}
    ple_writer.build_pack(scan, pack_path, segments_for(scan, shards, rows), log=lambda *_: None)


def _build(path, ple=False):
    import mlx.core as mx

    tensors, overrides = _tensors(mx, ple=ple)
    mx.eval(*tensors.values())
    os.makedirs(path)
    names = sorted(tensors)
    half = names[: len(names) // 2], names[len(names) // 2 :]
    weight_map = {}
    for i, part in enumerate(half):
        fn = f"model-0000{i + 1}-of-00002.safetensors"
        mx.save_safetensors(os.path.join(path, fn), {n: tensors[n] for n in part})
        weight_map.update(dict.fromkeys(part, fn))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    text = dict(TEXT, **PLE_TEXT) if ple else dict(TEXT)
    n = text["num_hidden_layers"]
    text["layer_types"] = [
        "full_attention" if (i + 1) % text["full_attention_interval"] == 0 else "linear_attention"
        for i in range(n)
    ]
    q = dict(Q, **overrides)
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "quantization": q,
        "quantization_config": q,
        "eos_token_id": EOS,
        "tie_word_embeddings": False,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump({"eos_token_id": EOS}, f)
    _tokenizer(path)


def _complete(root: str, files) -> bool:
    """Whether a cached build holds every file it is used for."""
    return all(os.path.isfile(os.path.join(root, f)) for f in files)


def cached_build(root: str, files, build) -> str:
    """`root`, holding `files`, built by `build(dir)` into a private directory renamed onto `root`.

    A root that exists but lacks a file (a build that was interrupted, or a husk another process
    left) is never returned: it is moved aside and replaced, and if a complete root still cannot be
    put in place the build raises rather than handing its tests an empty checkpoint.
    """
    if _complete(root, files):
        return root
    # the pack's header names the checkpoint by its shards' size and mtime and its index hash,
    # which renaming the directory keeps
    tmp = root + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    build(tmp)
    for _ in range(3):
        try:
            os.rename(tmp, root)
            return root
        except OSError:
            if _complete(root, files):  # another process built it first
                shutil.rmtree(tmp, ignore_errors=True)
                return root
            stale = root + f".stale{os.getpid()}"
            shutil.rmtree(stale, ignore_errors=True)
            try:
                os.rename(root, stale)  # the incomplete root, out of the way
            except OSError:
                pass  # gone already, or it cannot move: retry, then give up below
            shutil.rmtree(stale, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError(f"cannot put a complete test checkpoint at {root}")


@pytest.fixture(scope="session")
def tiny_flashnext():
    """`tiny_flashnext()`: (checkpoint path, expert pack path), built once per machine in the
    temp directory (needs MLX)."""

    def build(tmp):
        _build(os.path.join(tmp, "model"))
        _expert_pack(os.path.join(tmp, "model"), os.path.join(tmp, "tiny.experts"))

    def get():
        name = f"jitengine-qwen3.8-flash-next-m2-tiny-v{VERSION}"
        root = cached_build(
            os.path.join(tempfile.gettempdir(), name),
            ("model/config.json", "tiny.experts"),
            build,
        )
        return os.path.join(root, "model"), os.path.join(root, "tiny.experts")

    return get


@pytest.fixture(scope="session")
def tiny_flashnext_ple():
    """`tiny_flashnext_ple()`: (checkpoint path, expert pack path, PLE pack path) of the tiny
    checkpoint with one PLE layer (see `PLE_TEXT`), built once per machine in the temp directory
    (needs MLX). The default tiny checkpoint, which every other test uses, has none."""

    def build(tmp):
        model = os.path.join(tmp, "model")
        _build(model, ple=True)
        _expert_pack(model, os.path.join(tmp, "tiny.experts"))
        _ple_pack(model, os.path.join(tmp, "tiny.ple"))

    def get():
        root = cached_build(
            os.path.join(
                tempfile.gettempdir(),
                f"jitengine-qwen3.8-flash-next-m2-tiny-ple-v{VERSION}.{PLE_VERSION}",
            ),
            ("model/config.json", "tiny.experts", "tiny.ple"),
            build,
        )
        return (
            os.path.join(root, "model"),
            os.path.join(root, "tiny.experts"),
            os.path.join(root, "tiny.ple"),
        )

    return get


@pytest.fixture
def flashnext_engine(tiny_flashnext):
    """`flashnext_engine(**options)`: a Flash-Next engine on the tiny checkpoint (needs Metal).
    The engine has no close(); its worker thread is a daemon that waits on an empty queue."""

    def make(**options):
        from jitengine_qwen3_8_flash_next_m2.engine.engine import create_engine

        model, pack = tiny_flashnext()
        options.setdefault("log_requests", False)
        return create_engine(model, expert_pack_path=pack, **options)

    return make


@pytest.fixture(scope="session")
def cpu_flashnext_engine(tiny_flashnext, tiny_flashnext_ple):
    """`cpu_flashnext_engine(ple=False, **options)`: a Flash-Next engine on the tiny checkpoint
    (with `ple=True`, the one with a PLE layer) that runs on MLX's CPU build (or Metal). The
    load-time kernel checks, which launch Metal kernels, are replaced by ones that fail every
    kernel, so the engine serves on the stock ops (as in tests/unit/engine/test_engine.py);
    `F_NOCACHE`, a macOS read flag, is off. The engine has no close(); its worker thread is a
    daemon that waits on an empty queue."""

    def make(ple=False, **options):
        from jitengine_qwen3_8_flash_next_m2.engine.engine import create_engine
        from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck
        from jitengine_qwen3_8_flash_next_m2.model import kernel_checks

        def checks(cfg, act_dtype, dense, moe, variant, mtp=False):
            out = selfcheck.Checks()
            for name in kernel_checks.NAMES:
                out.run(name, lambda: False)
            return out

        if ple:
            model, pack, ple_pack = tiny_flashnext_ple()
            options.setdefault("ple_pack_path", ple_pack)
            options.setdefault("ple_io_nocache", False)
        else:
            model, pack = tiny_flashnext()
        options.setdefault("log_requests", False)
        options.setdefault("expert_io_nocache", False)
        options.setdefault("embed_io_nocache", False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(kernel_checks, "run_kernel_checks", checks)
            return create_engine(model, expert_pack_path=pack, **options)

    return make
