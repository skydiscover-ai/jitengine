"""The end-to-end test's plumbing: this engine's server as a subprocess on a real checkpoint, and
this engine's benchmark client against it. The test runs only on a Mac with Metal and only when its
checkpoint is named in the environment; otherwise it skips and says what it needs."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ENGINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK = os.path.join(ENGINE, "benchmarks", "benchmark.py")
PACKAGE = "jitengine_qwen3_6_35b_a3b_m4pro"
# MLX's command-buffer limits, which an engine package sets at import in the process importing it
MLX_BUFFER_VARS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")


def gate_model(env_var: str, what: str) -> str:
    """The checkpoint directory in `env_var`, or a skip naming what the test needs."""
    try:
        import mlx.core as mx

        metal = mx.metal.is_available()
    except Exception:  # noqa: BLE001
        metal = False
    if not metal:
        pytest.skip("the end-to-end test needs a Mac with Metal")
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(f"set {env_var} to {what} to run this test")
    path = os.path.expanduser(path)
    assert os.path.isfile(os.path.join(path, "config.json")), f"{env_var}={path}: no config.json"
    return path


def env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def server_env(extra: dict | None = None) -> dict:
    """The server's environment: this process's without MLX's command-buffer variables, so the
    server's own package sets them as it does for a user (this process may have imported the other
    engine's package, which sets its values), and with PYTHONPATH leading with the source root of
    the package these tests import (src/ under pytest's settings), so the server runs that code."""
    env = {k: v for k, v in os.environ.items() if k not in MLX_BUFFER_VARS}
    src = os.path.dirname(os.path.dirname(importlib.util.find_spec(PACKAGE).origin))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    env.update(extra or {})
    return env


class Server:
    """A server process with its output in a log file; `url` once `/health` answers ok."""

    def __init__(self, cmd: list[str], env: dict, log_path: str, port: int):
        self.log_path = log_path
        self.url = f"http://127.0.0.1:{port}"
        self._log = open(log_path, "w")  # closed in stop()
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env, cwd=ENGINE
        )

    def log(self) -> str:
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            return json.loads(r.read())

    def wait_ready(self, timeout_s: float) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"the server exited ({self.proc.returncode}):\n{self.log()[-4000:]}"
                )
            try:
                if self.get("/health").get("status") == "ok":
                    return
            except OSError:
                pass
            time.sleep(2)
        raise AssertionError(f"the server was not ready in {timeout_s:.0f}s:\n{self.log()[-4000:]}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()


@pytest.fixture(scope="module")
def gate_server(tmp_path_factory):
    """`gate_server(args, env=None, ready_s=1800)`: start `python <args...> --port P` with
    `server_env(env)` and wait for `/health`; every server a module started is stopped (SIGTERM)
    after its last test."""
    started = []

    def start(args: list[str], env: dict | None = None, ready_s: float = 1800) -> Server:
        port = free_port()
        log = tmp_path_factory.mktemp("server") / "server.log"
        server = Server(
            [sys.executable, *args, "--port", str(port)], server_env(env), str(log), port
        )
        started.append(server)
        server.wait_ready(ready_s)
        return server

    yield start
    for server in started:
        server.stop()


def run_gate(server: Server, model_path: str, out_json: str) -> dict:
    """The published greedy-512 check against `server`, with this engine's client:

        python benchmarks/benchmark.py --model-path M --mode interactive --temperature 0
            --max-tokens 512 --repeats 3 --output-json OUT

    plus `--url` for the server's port and `--server-pid` for its memory samples. Returns the JSON.
    """
    cmd = [
        sys.executable,
        BENCHMARK,
        "--url",
        server.url,
        "--model-path",
        model_path,
        "--mode",
        "interactive",
        "--temperature",
        "0",
        "--max-tokens",
        "512",
        "--repeats",
        "3",
        "--server-pid",
        str(server.proc.pid),
        "--output-json",
        out_json,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE)
    assert os.path.isfile(out_json), (
        f"benchmark.py wrote no result (exit {r.returncode}; it needs pip install '.[benchmark]'):"
        f"\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    )
    with open(out_json) as f:
        return json.load(f)


def text_shas(result: dict) -> list[str]:
    """Every stream's `text_sha`: one per repeat."""
    return [s["text_sha"] for rep in result["repeats_detail"] for s in rep["streams"]]


class Gate:
    """The helpers above, for the end-to-end test modules (test modules cannot import conftest)."""

    model = staticmethod(gate_model)
    run = staticmethod(run_gate)
    shas = staticmethod(text_shas)
    env_float = staticmethod(env_float)


@pytest.fixture(scope="session")
def gate() -> Gate:
    return Gate()
