"""Start and stop a server for a measurement, and describe the machine it ran on."""

from __future__ import annotations

import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

# A named tuple: the repository formatter would drop the parentheses of an inline `except (A, B)`,
# which Python 3.13 cannot parse.
_RUN_ERRORS = (OSError, subprocess.SubprocessError)
_GONE = (ProcessLookupError, PermissionError)
CHAT_TEMPLATE_ARGS = '{"enable_thinking": false}'
CONTENDED_CPU_PERCENT = 15.0  # a foreign process above this is recorded as contention


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def command(arm: str, model_path: str, port: int) -> list[str]:
    """The server command of an arm. `stock` is the installed mlx-lm, unpatched."""
    if arm == "engine":
        return [
            sys.executable,
            "-m",
            "jitengine_qwen3_1_7b_m4pro",
            "serve",
            "--model-path",
            model_path,
            "--port",
            str(port),
        ]
    if arm == "stock":
        return [
            sys.executable, "-m", "mlx_lm", "server", "--model", model_path, "--port", str(port),
            "--chat-template-args", CHAT_TEMPLATE_ARGS,
        ]  # fmt: skip
    raise ValueError(f"unknown arm {arm!r}")


class Server:
    """A server started cold (no warmup) in its own process group; only that group is stopped."""

    def __init__(self, arm: str, model_path: str, log_path: str, ready_timeout: float = 240):
        self.port = free_port()
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            command(arm, model_path, self.port), stdout=self.log, stderr=subprocess.STDOUT,
            cwd="/tmp", start_new_session=True,
        )  # fmt: skip
        start = time.time()
        while time.time() - start < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the {arm} server exited before it was ready; see {log_path}")
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=2)
                self.ready_seconds = round(time.time() - start, 1)
                return
            except OSError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"the {arm} server was not ready in {ready_timeout} s")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.url}/v1/chat/completions"

    def stop(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except _GONE:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except _RUN_ERRORS:
        return ""


def machine() -> dict:
    """What the numbers depend on: chip, memory, OS, power source and mode, package versions."""
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    batt = _run("pmset", "-g", "batt")
    mode = re.search(r"^\s*powermode\s+(\d+)", _run("pmset", "-g"), re.M)
    return {
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string").strip() or platform.processor(),
        "memory_gib": round(int(_run("sysctl", "-n", "hw.memsize").strip() or 0) / 2**30, 1),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "power_source": (re.search(r"Now drawing from '([^']+)'", batt) or [None, None])[1],
        "power_mode": int(mode.group(1)) if mode else None,
        "uptime": _run("uptime").strip(),
        "mlx": version("mlx"),
        "mlx-lm": version("mlx-lm"),
        "transformers": version("transformers"),
        "engine": version("jitengine-qwen3-1.7b-m4pro"),
    }


def busy_processes() -> list[dict]:
    """Processes using more than CONTENDED_CPU_PERCENT of one core right now (a snapshot, not a kill list)."""
    out = _run("top", "-l", "2", "-n", "8", "-o", "cpu", "-s", "2", "-stats", "pid,command,cpu")
    last = out.split("PID")[-1].splitlines()[1:]
    found = []
    for line in last:
        parts = line.split()
        if len(parts) >= 3:
            try:
                cpu = float(parts[-1])
            except ValueError:
                continue
            if cpu >= CONTENDED_CPU_PERCENT:
                found.append({"pid": int(parts[0]), "command": " ".join(parts[1:-1]), "cpu": cpu})
    return found


def dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
