"""Load the checkpoint's dense language-model weights lazily via mx.load, then evaluate them once.

Names come back with the checkpoint's prefix (`expert_pack.prefix_of`) stripped. The PLE n-gram
shards are never referenced (they stream from the PLE pack and must not be resident), nor are the
routed experts under `skip_experts`; `mx.load` is lazy, so a skipped name costs no I/O.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import prefix_of

logger = logging.getLogger(__name__)


def shard_files(model_path: str) -> list[str]:
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        files = sorted({os.path.join(model_path, v) for v in wm.values()})
        if files:
            return files
    return sorted(glob.glob(os.path.join(model_path, "*.safetensors")))


EXPERTS = ".mlp.switch_mlp."
PLE_SHARDS = ".ple.ple_embedding.ngram_embedding.shards."


def load_language_weights(
    model_path: str, skip_experts: bool = False, keep=None, skip_embed: bool = False
) -> dict[str, mx.array]:
    """Return {name_without_prefix: array} for every language-model tensor, materialized.

    `keep(stripped_name) -> bool`, if given, further filters the names; the expert and PLE-shard
    exclusions apply regardless. With `skip_embed`, `model.embed_tokens.*` is left out too, since
    `embed_store.StreamedEmbedding` streams it.
    """
    t0 = time.monotonic()
    model_path = os.path.expanduser(model_path)
    prefix = prefix_of(model_path)
    weights: dict[str, mx.array] = {}
    skipped = 0
    skipped_ple = 0
    skipped_embed = 0
    for path in shard_files(model_path):
        shard = mx.load(path)
        skipped, skipped_ple, skipped_embed = _take_language_tensors(
            shard,
            prefix,
            skip_experts,
            keep,
            weights,
            skipped,
            skipped_ple,
            skip_embed,
            skipped_embed,
        )
    _log_loaded(weights, prefix, skipped, skipped_ple, skipped_embed)
    mx.eval(*weights.values())
    logger.debug(f"[loader] resident in {time.monotonic() - t0:.1f}s")
    return weights


def _take_language_tensors(
    shard: dict[str, mx.array],
    prefix: str,
    skip_experts: bool,
    keep,
    weights: dict[str, mx.array],
    skipped: int,
    skipped_ple: int,
    skip_embed: bool,
    skipped_embed: int,
) -> tuple[int, int, int]:
    """Add a shard's language-model tensors to weights; returns the expert, PLE, embed skips."""
    for k, v in shard.items():
        if PLE_SHARDS in k:
            skipped_ple += 1
            continue
        if skip_experts and EXPERTS in k:
            skipped += 1
            continue
        if prefix and not k.startswith(prefix):
            continue  # not part of the language tower (the vision tower)
        name = k[len(prefix) :]
        if skip_embed and name.startswith("model.embed_tokens."):
            skipped_embed += 1
            continue
        if keep is not None and not keep(name):
            continue
        weights[name] = v
    return skipped, skipped_ple, skipped_embed


def _log_loaded(
    weights: dict[str, mx.array], prefix: str, skipped: int, skipped_ple: int, skipped_embed: int
) -> None:
    """Log how many dense tensors and bytes were collected and what was never referenced."""
    nbytes = sum(v.nbytes for v in weights.values())
    logger.info(
        f"[loader] prefix={prefix!r}; {len(weights)} dense tensors, {nbytes / 2**30:.3f} GiB "
        f"({skipped} expert tensors, {skipped_ple} PLE shard tensors, {skipped_embed} streamed embedding tensors never "
        f"referenced); reading..."
    )
