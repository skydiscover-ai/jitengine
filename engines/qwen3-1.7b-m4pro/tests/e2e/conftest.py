"""Plumbing for tests that start this engine's server on the real checkpoint.

The tests run only on a Mac with Metal and only when the checkpoint is named in the environment.
Metal and live-server tests share one GPU: a session lock serializes them, and each server gets
its own free port."""

from __future__ import annotations

import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ENV = "JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT"
LOCK = os.path.join(os.environ.get("TMPDIR", "/tmp"), "jitengine-qwen3-1.7b-m4pro-gpu.lock")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def checkpoint():
    path = os.environ.get(ENV)
    if not path:
        pytest.skip(f"set {ENV} to the model directory to run the end-to-end tests")
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            pytest.skip("the end-to-end tests need a Mac with Metal")
    except ImportError:
        pytest.skip("the end-to-end tests need MLX")
    return os.path.expanduser(path)


@pytest.fixture(scope="session", autouse=True)
def _gpu_lock(request):
    """One test session at a time uses the GPU."""
    if not os.environ.get(ENV):
        yield
        return
    with open(LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


class Server:
    def __init__(self, checkpoint: str, *extra: str):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jitengine_qwen3_1_7b_m4pro",
                "serve",
                "--model-path",
                checkpoint,
                "--port",
                str(self.port),
                *extra,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("the server exited before it was ready")
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=2)
                return
            except OSError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError("the server was not ready in 180 s")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError, PermissionError:
            pass
        self.proc.wait(timeout=30)


@pytest.fixture
def server(checkpoint):
    srv = Server(checkpoint)
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(scope="module")
def shared(checkpoint):
    """One server for the tests of a module that only read from it."""
    srv = Server(checkpoint)
    try:
        yield srv
    finally:
        srv.stop()
