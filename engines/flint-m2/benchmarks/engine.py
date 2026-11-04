"""Entry point the benchmark driver loads the engine through.

`$FLINT_IMPL_DIR` names the directory to import the engine package from; otherwise this file's own
directory is used. `$FLINT_MEMORY_BUDGET_GIB` sets the memory budget (the published numbers ran at 18
and 12); unset, the engine's default applies.

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

from jitengine_flint_m2.cli import main as _main  # noqa: E402
from jitengine_flint_m2.engine.engine import Engine  # noqa: E402
from jitengine_flint_m2.engine.io_struct import (  # noqa: E402,F401
    GenRequest,
    GenResult,
    ScoredPosition,
    StepOutput,
)

def create_engine(model_path: str, **options) -> Engine:
    """Default options: profile main, context limit auto, the budget from $FLINT_MEMORY_BUDGET_GIB if set."""
    gib = os.environ.get("FLINT_MEMORY_BUDGET_GIB")
    if gib and "memory_budget_bytes" not in options:
        options["memory_budget_bytes"] = int(float(gib) * 2**30)
    return Engine(model_path, **options)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
