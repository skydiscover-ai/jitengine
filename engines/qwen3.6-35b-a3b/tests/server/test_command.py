"""The server names the build that runs it: `/v1/models` reports the command `serve` was given as the owner (each
build's Engine.serve passes its own, tests/unit/test_build_names.py in each build). The real handler over a socket,
with a stand-in engine; no MLX."""

import json
import socket
import threading
import time
import urllib.request

from jitengine_qwen3_6_35b_a3b.server import api_server


class Engine:
    model_name = "m"
    max_kv_positions = 4096


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _models(port: int) -> dict:
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                return json.loads(r.read())
        except OSError:
            time.sleep(0.05)
    raise AssertionError("the server did not answer")


def test_v1_models_names_the_command_serve_was_given():
    port = _free_port()
    threading.Thread(
        target=api_server.serve,
        args=(Engine(), "127.0.0.1", port),
        kwargs={"command": "jitengine-qwen3.6-35b-a3b-test"},
        daemon=True,
    ).start()
    (model,) = _models(port)["data"]
    assert model["id"] == "m" and model["owned_by"] == "jitengine-qwen3.6-35b-a3b-test"


def test_a_server_built_directly_names_the_model():
    assert api_server.Server.command == "jitengine-qwen3.6-35b-a3b"
