"""Checks of a model directory against what this engine was built and measured for."""

from __future__ import annotations

import json
import os
import re

from jitengine_qwen3_1_7b_m4pro.provenance import MODEL_REPO, MODEL_REVISION

EXPECTED = {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}
EXPECTED_QUANT = {"bits": 4, "group_size": 64}


def revision_from_path(path: str) -> str | None:
    """The commit hash when the directory is a Hugging Face cache snapshot (`snapshots/<hash>`)."""
    name = os.path.basename(os.path.realpath(path.rstrip("/")))
    return name if re.fullmatch(r"[0-9a-f]{40}", name) else None


def validate(path: str) -> dict:
    """{'ok', 'problems', 'warnings', 'revision'} for the model directory at `path`."""
    problems, warnings = [], []
    cfg_file = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_file):
        return {
            "ok": False,
            "problems": [f"no config.json in {path}"],
            "warnings": [],
            "revision": None,
        }
    with open(cfg_file) as f:
        cfg = json.load(f)
    for key, want in EXPECTED.items():
        if cfg.get(key) != want:
            problems.append(f"config.json {key} is {cfg.get(key)!r}, this engine serves {want!r}")
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    for key, want in EXPECTED_QUANT.items():
        if quant.get(key) != want:
            problems.append(
                f"quantization {key} is {quant.get(key)!r}, this engine serves {want!r}"
            )
    if quant.get("mode", "affine") != "affine":
        problems.append(f"quantization mode is {quant.get('mode')!r}, this engine serves 'affine'")
    if not any(n.endswith(".safetensors") for n in os.listdir(path)):
        problems.append("no .safetensors weights in the model directory")
    revision = revision_from_path(path)
    if revision is None:
        warnings.append(
            f"cannot tell which revision of {MODEL_REPO} this is; the engine was measured on {MODEL_REVISION}"
        )
    elif revision != MODEL_REVISION:
        warnings.append(f"revision {revision} differs from the measured {MODEL_REVISION}")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "revision": revision}
