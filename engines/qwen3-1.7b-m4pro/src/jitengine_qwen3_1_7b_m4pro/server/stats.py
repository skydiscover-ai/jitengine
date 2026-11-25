"""`/stats`: what this server applied and checked. Added to mlx-lm's handler without editing it."""

from __future__ import annotations

import json
import os
import time

from jitengine_qwen3_1_7b_m4pro import __version__
from jitengine_qwen3_1_7b_m4pro.engine.state import STATE


def _prompt_cache(handler) -> dict | None:
    try:
        cache = handler.response_generator.prompt_cache
        return {"sequences": len(cache), "bytes": cache.nbytes}
    except AttributeError:
        return None


def _memory() -> dict | None:
    """MLX's own counters, in bytes: live arrays, freed buffers it keeps for reuse, and the peak."""
    try:
        import mlx.core as mx

        return {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        }
    except ImportError, AttributeError:
        return None


def payload(handler=None) -> dict:
    from jitengine_qwen3_1_7b_m4pro import hardware

    return {
        "engine": "jitengine-qwen3-1.7b-m4pro",
        "version": __version__,
        "model": os.path.basename(str(STATE["model_path"]).rstrip("/"))
        if STATE.get("model_path")
        else None,
        "patches": STATE.get("patches"),
        "selfcheck": STATE.get("selfcheck"),
        "admission": STATE["budget"].snapshot() if STATE.get("budget") else None,
        "prompt_cache": _prompt_cache(handler) if handler is not None else None,
        "memory": _memory(),
        "power": hardware.power(),
    }


def _entry(name: str) -> dict:
    entry = {"id": name, "object": "model", "created": _CREATED}
    if STATE.get("budget"):
        entry["max_model_len"] = STATE["budget"].max_context  # the launcher reads this
    return entry


def models() -> dict:
    """/v1/models lists the one served model; stock lists every model in the Hugging Face cache."""
    name = STATE.get("served_primary")
    return {"object": "list", "data": [_entry(name)] if name else []}


_CREATED = int(time.time())


def install(handler_class) -> None:
    """Answer GET /stats and /v1/models; every other path goes to the original handler."""
    original = handler_class.do_GET

    def do_GET(self):
        if self.path == "/stats":
            body = json.dumps(payload(self), default=str).encode()
        elif self.path.split("?")[0].rstrip("/") == "/v1/models" and STATE.get("served_names"):
            body = json.dumps(models()).encode()
        else:
            original(self)
            return
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    handler_class.do_GET = do_GET
