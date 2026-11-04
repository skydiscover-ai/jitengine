"""MLX inference engine for Qwen3.8-Flash-Next (4-bit, `qwen4_exp`) on an Apple silicon Mac with 24 GB or more.

The checkpoint does not fit in memory: dense weights stay resident, routed experts live in a slot
pool filled on demand from the FNXEPK01 pack, and PLE rows stream per call from the FNXPLE01 pack.
"""

import os as _os

# Commit a Metal command buffer every 10 encoded ops so the GPU starts a layer's first kernels while
# the host encodes the rest. MLX reads this at its first command buffer: set it before any GPU work.
_os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "10")

__version__ = "0.1.0a1"


def __getattr__(name):
    # lazily, so the planner and tokenizer modules import without MLX
    if name in ("Engine", "create_engine"):
        from jitengine_qwen3_8_flash_next_m4pro.engine import engine as _engine

        return getattr(_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
