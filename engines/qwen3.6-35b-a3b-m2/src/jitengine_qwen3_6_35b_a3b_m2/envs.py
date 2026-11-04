"""The environment variables this engine reads, each defined once.

Settings are CLI flags or configuration; these variables only say where files are.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvPath:
    """A path from the environment, with `~` expanded; None when unset or empty."""

    name: str

    def get(self) -> str | None:
        value = os.environ.get(self.name)
        return os.path.expanduser(value) if value else None


# the directory holding expert packs, each named after its checkpoint (default ~/.cache/jitengine)
PACK_DIR = EnvPath("JITENGINE_QWEN3_6_35B_A3B_M2_PACK_DIR")
# one expert pack file, in place of the one PACK_DIR would name
EXPERT_PACK = EnvPath("JITENGINE_QWEN3_6_35B_A3B_M2_EXPERT_PACK")
