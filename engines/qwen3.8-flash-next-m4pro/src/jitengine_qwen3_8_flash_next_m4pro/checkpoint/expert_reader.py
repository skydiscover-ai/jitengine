"""The engine's side of the FNXEPK01 expert pack: open and validate it, and scan the checkpoint.

The format, its validation and its default path are pack_format.py's. `CheckpointScan` is that
module's `ExpertScan` (the checkpoint's own geometry, the prefix derived from the index) plus what
the budget planner reads; the pack header's checkpoint identity must match the scan's exactly, so
a changed checkpoint is rejected. `prefix_of` gives the loader the tensor prefix to strip.
"""

from __future__ import annotations

import json
import os

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    MAGIC,
    PLE_SUBMODULE,
    SUBMODULE,
    ExpertScan,
    PackError,
    default_pack_path,
    detect_prefix,
    read_header_block,
    validate,
)


def prefix_of(model_path: str) -> str:
    """`detect_prefix` of the checkpoint's `model.safetensors.index.json`."""
    with open(os.path.join(os.path.expanduser(model_path), "model.safetensors.index.json")) as f:
        return detect_prefix(json.load(f)["weight_map"])


class CheckpointScan(ExpertScan):
    """`ExpertScan` plus what the budget planner reads: every shard header, the dense byte total.

    Reads only the safetensors headers, so the budget planner can run it before any allocation.
    """

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.weight_map = self.ckpt.weight_map
        self.headers = {name: self.ckpt.header(name) for name in self.ckpt.shard_names}

        dense = 0
        ple = 0
        for name, shard in self.weight_map.items():
            if not name.startswith(self.prefix):
                continue
            hdr, _ = self.headers[shard]
            lo, hi = hdr[name]["data_offsets"]
            if f".{SUBMODULE}." in name:
                continue
            if f".{PLE_SUBMODULE}.shards." in name:
                ple += hi - lo
                continue
            dense += hi - lo
        self.dense_bytes = dense
        self.ple_table_bytes = ple
        self.expert_bytes = self.record_bytes * self.num_layers * self.num_experts


# ------------------------------------------------------------------ pack file
def open_pack(
    model_path: str, build: bool = False, path: str | None = None, log=print
) -> tuple[str, dict, CheckpointScan]:
    """Validate the pack for `model_path` and return (path, header, scan); never builds one.

    `build` exists for call-site compatibility; True raises `PackError`.
    """
    if build:
        raise PackError(
            "the engine never builds an expert pack; build it with `jitengine-qwen3.8-flash-next-m4pro pack`"
        )
    scan = CheckpointScan(model_path)
    path = os.path.expanduser(path) if path else default_pack_path(scan)
    if not os.path.exists(path):
        raise PackError(f"expert pack {path} does not exist")
    ok, why = validate(path, scan)
    if not ok:
        log(f"[pack] rejecting {path}: {why}")
        raise PackError(f"expert pack {path} unusable: {why}")
    return path, read_header_block(path, MAGIC), scan
