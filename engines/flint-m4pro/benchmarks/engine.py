"""Entry point the benchmark driver loads the engine through.

`$FLINT_IMPL_DIR` names the directory to import the engine package from; otherwise this file's own
directory is used.

    python engine.py serve --model DIR [--port 8080] [--context-limit N] [--profile main] ...
"""

from __future__ import annotations

import os
import sys

_TREE = os.path.abspath(
    os.environ.get("FLINT_IMPL_DIR") or os.path.dirname(os.path.abspath(__file__))
)
if _TREE not in sys.path:
    sys.path.insert(0, _TREE)

from jitengine_flint_m4pro.cli import main as _main  # noqa: E402
from jitengine_flint_m4pro.engine.engine import Engine  # noqa: E402
from jitengine_flint_m4pro.engine.io_struct import (  # noqa: E402,F401
    GenRequest,
    GenResult,
    ScoredPosition,
    StepOutput,
)

def create_engine(model_path: str, **options) -> Engine:
    """Default options: profile main, context limit auto."""
    return Engine(model_path, **options)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
