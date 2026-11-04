"""Imported at startup by the end-to-end test's server process, whose PYTHONPATH leads with this directory.

When the process first imports MLX, after the engine package's own import, this writes the MLX
command-buffer variables the process then has to $JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PROBE_OUT (MLX reads them
when it makes its first command buffer). Then it runs the sitecustomize it shadows, if any.
"""

import json
import os
import sys

_OUT = os.environ.get("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PROBE_OUT")


class _RecordAtMlxImport:
    def find_spec(self, name, path=None, target=None):
        if name == "mlx.core":
            sys.meta_path.remove(self)
            with open(_OUT, "w") as f:
                json.dump(
                    {
                        k: os.environ.get(k)
                        for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")
                    },
                    f,
                )
        return None


if _OUT:
    sys.meta_path.insert(0, _RecordAtMlxImport())

_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    _path = os.path.join(_entry or os.getcwd(), "sitecustomize.py")
    if os.path.dirname(os.path.abspath(_path)) != _here and os.path.isfile(_path):
        with open(_path) as _f:
            exec(
                compile(_f.read(), _path, "exec"), {"__name__": "sitecustomize", "__file__": _path}
            )
        break
