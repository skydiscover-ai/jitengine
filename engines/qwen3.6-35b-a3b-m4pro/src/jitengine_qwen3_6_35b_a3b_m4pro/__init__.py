"""jitengine_qwen3_6_35b_a3b_m4pro: budgeted local serving of Qwen3.6-35B-A3B (qwen3_5_moe) on Apple Silicon.

Importing this package does not import MLX: `Engine` and `create_engine` resolve lazily, so the
CLI's help/version/config paths and the pure planning modules work on any platform.
"""

import os as _os

# MLX splits Metal work into command buffers capped by these variables and the GPU idles at each
# split, so the op cap is raised for decode's many small kernels and the MB cap is lifted. MLX
# reads both when the Metal device is created: set them before the first GPU operation.
_os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100")
_os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "100000")

__version__ = "0.1.0a1"

__all__ = ["Engine", "__version__", "create_engine"]


def __getattr__(name):
    if name in ("Engine", "create_engine"):
        from jitengine_qwen3_6_35b_a3b_m4pro.engine import engine as _engine

        return getattr(_engine, name)
    raise AttributeError(f"module 'jitengine_qwen3_6_35b_a3b_m4pro' has no attribute {name!r}")
