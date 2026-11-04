"""Checkpoint loading: only the language model, lazily via mx.load, evaluated once.

Expects the sanitized mlx_lm layout (`switch_mlp.*`, conv1d `[C, K, 1]`); the only renaming is
stripping the `language_model.` prefix.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

PREFIX = "language_model."


def shard_files(model_path: str) -> list[str]:
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        files = sorted({os.path.join(model_path, v) for v in wm.values()})
        if files:
            return files
    return sorted(glob.glob(os.path.join(model_path, "*.safetensors")))


def load_language_weights(
    model_path: str, skip_experts: bool = False, materialize: bool = True
) -> dict[str, mx.array]:
    """Return {name_without_prefix: array} for every language-model tensor.

    `skip_experts` drops the routed experts, which then come from the expert pack; mx.load is lazy,
    so a dropped tensor costs no I/O. `materialize=False` returns the tensors unevaluated.
    """
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.inventory import canonical, family

    t0 = time.monotonic()
    weights: dict[str, mx.array] = {}
    dropped: dict[str, int] = {}
    for path in shard_files(model_path):
        shard = mx.load(path)
        for k, v in shard.items():
            fam = family(k)  # the same classification the header scan and the plan used
            if fam in ("vision", "mtp") or (skip_experts and fam == "expert"):
                dropped[fam] = dropped.get(fam, 0) + 1
                continue
            weights[canonical(k)] = v
    nbytes = sum(v.nbytes for v in weights.values())
    logger.info(
        f"[loader] {len(weights)} language tensors, {nbytes / 2**30:.2f} GiB; not loaded: "
        + (", ".join(f"{n} {f}" for f, n in sorted(dropped.items())) or "nothing")
        + "; reading..."
    )
    if not materialize:
        # the caller builds fused copies layer by layer, so no layer is ever resident twice in full
        return weights
    mx.eval(*weights.values())
    logger.info(f"[loader] resident in {time.monotonic() - t0:.1f}s")
    return weights
