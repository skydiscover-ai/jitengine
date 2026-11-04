"""Header-only checkpoint validation on fake checkpoints (valid headers over zero bytes); no MLX."""

from __future__ import annotations

import json
import os
import sys

from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.inventory import (
    CheckpointError,
    QuantSpec,
    family,
    inventory,
    parse_quantization,
)
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import load_config


def _raises(fn, needle: str):
    try:
        fn()
    except CheckpointError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc!r}"
        return
    raise AssertionError(f"no CheckpointError (wanted {needle!r})")


def _scan(path):
    return inventory(str(path), load_config(str(path)))


def test_scan_does_not_import_mlx(tmp_path, fake_checkpoint):
    before = "mlx.core" in sys.modules
    fake_checkpoint(str(tmp_path / "m"))
    _scan(tmp_path / "m")
    if not before:
        assert "mlx.core" not in sys.modules


def test_four_bit_fake_is_valid_and_families_add_up(tmp_path, fake_checkpoint):
    p = fake_checkpoint(str(tmp_path / "m"), bits=4)
    inv = _scan(p)
    assert inv.default_quant == QuantSpec(4, 64)
    assert {s.bits for s in inv.overrides.values()} == {8}
    assert inv.quant_of("model.layers.0.mlp.gate").bits == 8
    assert inv.quant_of("model.layers.0.mlp.switch_mlp.up_proj").bits == 4
    assert inv.family_bytes["vision"] > 0 and inv.dropped_bytes == inv.family_bytes["vision"]
    total = sum(t.nbytes for t in inv.tensors.values())
    assert inv.text_bytes + inv.dropped_bytes == total


def test_three_bit_fake_is_valid(tmp_path, fake_checkpoint):
    inv = _scan(fake_checkpoint(str(tmp_path / "m"), bits=3))
    assert inv.default_quant.bits == 3
    assert inv.quant_of("model.layers.1.mlp.switch_mlp.down_proj").bits == 3
    assert set(inv.bits_histogram) == {3, 8}


def test_three_bit_is_smaller_than_four_bit(tmp_path, fake_checkpoint):
    a = _scan(fake_checkpoint(str(tmp_path / "a"), bits=3))
    b = _scan(fake_checkpoint(str(tmp_path / "b"), bits=4))
    assert a.expert_bytes < b.expert_bytes


def test_declared_bits_must_match_packed_shape(tmp_path, fake_checkpoint):
    # packed as 4-bit, config says 3-bit
    def mutate(entries, cfg):
        cfg["quantization"]["bits"] = 3
        cfg["quantization_config"]["bits"] = 3

    p = fake_checkpoint(str(tmp_path / "m"), bits=4, mutate=mutate)
    _raises(lambda: _scan(p), "implies 4 bits")


def test_override_width_must_match(tmp_path, fake_checkpoint):
    def mutate(entries, cfg):
        cfg["quantization"]["language_model.model.layers.0.mlp.gate"] = {
            "group_size": 64,
            "bits": 4,
        }

    p = fake_checkpoint(str(tmp_path / "m"), bits=4, mutate=mutate)
    _raises(lambda: _scan(p), "model.layers.0.mlp.gate")


def test_mixed_group_size_rejected(tiny_config):
    cfg = tiny_config(4)
    cfg["quantization"]["language_model.model.layers.0.mlp.gate"] = {"group_size": 32, "bits": 8}
    _raises(lambda: parse_quantization(cfg), "mixed group sizes")


def test_unknown_mode_rejected(tiny_config):
    cfg = tiny_config(4)
    cfg["quantization"]["mode"] = "mxfp4"
    _raises(lambda: parse_quantization(cfg), "not supported")


def test_missing_expert_tensor_rejected(tmp_path, fake_checkpoint):
    def mutate(entries, cfg):
        for suffix in (".weight", ".scales", ".biases"):
            entries.pop(f"language_model.model.layers.2.mlp.switch_mlp.down_proj{suffix}")

    p = fake_checkpoint(str(tmp_path / "m"), mutate=mutate)
    _raises(lambda: _scan(p), "required text tensors are missing")


def test_expert_count_must_match_config(tmp_path, fake_checkpoint):
    def mutate(entries, cfg):
        cfg["text_config"]["num_experts"] = 16

    p = fake_checkpoint(str(tmp_path / "m"), mutate=mutate)
    # mutate runs after the shapes are built, so the stacks keep 8 experts; the config says 12
    raw = json.load(open(os.path.join(p, "config.json")))
    raw["text_config"]["num_experts"] = 12
    json.dump(raw, open(os.path.join(p, "config.json"), "w"))
    _raises(lambda: _scan(p), "experts, config says 12")


def test_missing_biases_rejected(tmp_path, fake_checkpoint):
    def mutate(entries, cfg):
        entries.pop("language_model.model.layers.0.linear_attn.out_proj.biases")

    p = fake_checkpoint(str(tmp_path / "m"), mutate=mutate)
    _raises(lambda: _scan(p), "needs .biases")


def test_extra_decoder_layer_rejected(tmp_path, fake_checkpoint):
    def mutate(entries, cfg):
        entries["language_model.model.layers.4.input_layernorm.weight"] = ("BF16", (128,))

    p = fake_checkpoint(str(tmp_path / "m"), mutate=mutate)
    _raises(lambda: _scan(p), "more decoder layers")


def test_truncated_shard_rejected(tmp_path, fake_checkpoint):
    p = fake_checkpoint(str(tmp_path / "m"))
    shard = os.path.join(p, "model-00001-of-00001.safetensors")
    size = os.path.getsize(shard)
    with open(shard, "r+b") as f:
        f.truncate(size - 1024)
    _raises(lambda: _scan(p), "truncated shard")


def test_index_naming_absent_tensor_rejected(tmp_path, fake_checkpoint):
    p = fake_checkpoint(str(tmp_path / "m"))
    ip = os.path.join(p, "model.safetensors.index.json")
    idx = json.load(open(ip))
    idx["weight_map"]["language_model.model.layers.0.ghost.weight"] = (
        "model-00001-of-00001.safetensors"
    )
    json.dump(idx, open(ip, "w"))
    _raises(lambda: _scan(p), "absent from the shards")


def test_family_classification():
    assert family("vision_tower.blocks.0.attn.qkv.weight") == "vision"
    assert family("language_model.model.layers.3.mlp.switch_mlp.up_proj.weight") == "expert"
    assert family("language_model.model.layers.3.mlp.shared_expert.up_proj.weight") == "dense"
    assert family("mtp.layers.0.mlp.gate.weight") == "mtp"
    assert family("lm_head.weight") == "dense"


def test_real_checkpoint_if_present():
    """The locally available 4-bit checkpoint scans clean and matches its known totals."""
    p = os.environ.get("JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_CHECKPOINT") or os.path.expanduser(
        "~/models/Qwen3.6-35B-A3B-4bit"
    )
    if not os.path.isfile(os.path.join(p, "model.safetensors.index.json")):
        return
    inv = inventory(p, load_config(p))
    assert inv.model_type == "qwen3_5_moe"
    assert inv.default_quant.group_size == 64
    assert {s.bits for s in inv.overrides.values()} == {8}
    assert inv.dense_bytes > 0 and inv.expert_bytes > 10 * inv.dense_bytes
