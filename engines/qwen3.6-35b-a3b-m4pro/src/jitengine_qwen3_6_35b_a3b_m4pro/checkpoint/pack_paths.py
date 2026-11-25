"""Where this build keeps its expert pack: the file JITENGINE_..._EXPERT_PACK names, else one named after the checkpoint
in the pack directory. The variables carry this build's name (envs.py), so this stays in the build while the pack
format (checkpoint/pack.py) is shared with the other Qwen3.6 build."""

from __future__ import annotations

import os

from jitengine_qwen3_6_35b_a3b.checkpoint.pack import VERSION, CheckpointScan
from jitengine_qwen3_6_35b_a3b_m4pro import envs
from jitengine_runtime import hub


def default_pack_path(scan: CheckpointScan) -> str:
    env = envs.EXPERT_PACK.get()
    if env:
        return env
    base = envs.PACK_DIR.get() or os.path.expanduser("~/.cache/jitengine")
    tag = f"{hub.checkpoint_name(scan.model_path)}-{scan.index_sha256[:12]}"
    return os.path.join(os.path.expanduser(base), f"{tag}.v{VERSION}.experts")
