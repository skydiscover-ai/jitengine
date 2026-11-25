"""Where the checkpoint's MTP head is.

mlx-community ships the head as a repository of its own beside the checkpoint's (`...-4bit` and `...-MTP-4bit`), so in a
directory layout the engine finds it by name next to the checkpoint. A checkpoint that came from the Hugging Face cache
(`--model-path mlx-community/Qwen3.6-35B-A3B-4bit`) has no neighbour: its head is the pinned MTP repository
(pinned.py), fetched into the cache the same way. MLX-free, so the lookup works anywhere."""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from jitengine_qwen3_6_35b_a3b_m4pro import pinned
from jitengine_runtime import hub

HEAD_FILE = "model.safetensors"


def local_head(model_path: str) -> str | None:
    """The head beside a checkpoint directory (`X-4bit` -> `X-MTP-4bit`) or inside it (`mtp.safetensors`), if any."""
    base = model_path.rstrip("/")
    for candidate in (
        re.sub(r"(-\d+bit)$", r"-MTP\1", base) + "/model.safetensors",  # X-4bit -> X-MTP-4bit
        base + "-MTP-4bit/model.safetensors",
        os.path.join(base, "mtp.safetensors"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def hub_head(
    model_path: str, log: Callable[[str], None], warn: Callable[[str], None]
) -> str | None:
    """The pinned head, fetched, when `model_path` is a folder of the Hugging Face cache holding the pinned checkpoint.

    Any other checkpoint has no head to fetch. A head that cannot be fetched is a warning, not a failure: the engine
    then decodes one token per pass, as it does without a head."""
    repo = hub.snapshot_repo(model_path)
    if repo is None or repo.lower() != pinned.CHECKPOINT.lower():
        return None
    try:
        folder = hub.fetch_snapshot(
            pinned.MTP_CHECKPOINT, pinned.MTP_REVISION, allow_patterns=[HEAD_FILE], log=log
        )
    except hub.ModelPathError as exc:
        warn(f"[engine] no MTP head, so one token is decoded per pass: {exc}")
        return None
    head = os.path.join(folder, HEAD_FILE)
    return head if os.path.isfile(head) else None
