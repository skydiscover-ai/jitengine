"""Where this build's packs are: the file a variable names, else the checkpoint's pack in the pack directory.

The variables carry this build's name (envs.py), so this stays in each build while the formats themselves
(pack_format.py) are the same for both.
"""

from __future__ import annotations

import os

from jitengine_qwen3_8_flash_next.checkpoint.pack_format import ExpertScan, PackError, PLEScan
from jitengine_qwen3_8_flash_next_m4pro import envs
from jitengine_runtime import hub


def _pack_dir(file_var: envs.EnvPath, what: str) -> str:
    """The pack directory, or a PackError naming the two variables that say where a pack is."""
    root = envs.PACK_DIR.get()
    if not root:
        raise PackError(
            f"set {envs.PACK_DIR.name} to the directory holding the packs (built by "
            f"`jitengine-qwen3.8-flash-next-m4pro pack`), or {file_var.name} to the {what}"
        )
    return root


def default_pack_path(scan: ExpertScan, layers: list[int] | None = None) -> str:
    """`JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_EXPERT_PACK`, else `<checkpoint>-<index sha12>.experts` in the pack dir.

    A builder's layer subset gets `-L<layers>` before the extension, so it never takes the whole
    pack's name; the engine always looks up the whole pack.
    """
    path = envs.EXPERT_PACK.get()
    if path:
        return path
    root = _pack_dir(envs.EXPERT_PACK, "expert pack")
    tag = f"{hub.checkpoint_name(scan.model_path)}-{scan.ckpt.index_sha256[:12]}"
    if layers is not None and len(layers) != scan.num_layers:
        tag += f"-L{'_'.join(str(layer) for layer in layers)}"
    return os.path.join(root, f"{tag}.experts")


def default_ple_pack_path(scan: PLEScan, segments: list[dict] | None = None) -> str:
    """`JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PACK`, else `<checkpoint>-<index sha12>.ple` in the pack dir.

    A builder's row selection gets `-R<rows>` before the extension; the engine always looks up the
    whole table.
    """
    path = envs.PLE_PACK.get()
    if path:
        return path
    root = _pack_dir(envs.PLE_PACK, "PLE pack")
    tag = f"{hub.checkpoint_name(scan.model_path)}-{scan.ckpt.index_sha256[:12]}"
    if segments is not None and sum(s["rows"] for s in segments) != scan.total_rows:
        tag += f"-R{sum(s['rows'] for s in segments)}"
    return os.path.join(root, f"{tag}.ple")
