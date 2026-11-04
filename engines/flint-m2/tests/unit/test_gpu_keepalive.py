"""The GPU keep-alive child: runs only when resumed, and never outlives its parent (closed, or killed while paused)."""

import os
import subprocess
import sys
import time

from jitengine_flint_m2 import gpu_keepalive
from jitengine_flint_m2.gpu_keepalive import KeepAlive


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(st) and not st.startswith("Z")


def _cpu_s(pid: int) -> float:
    t = subprocess.run(
        ["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    m, s = t.split(":")[-2:]
    return int(m) * 60 + float(s)


def _wait_gone(pid: int, secs: float = 10.0) -> bool:
    end = time.time() + secs
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _goes_idle(pid: int, secs: float) -> bool:
    """Whether, within `secs`, the child uses under 0.2 s of CPU in one second. Waits poll with long
    deadlines (a slow MLX import or a loaded host only makes the test slower), where fixed sleeps
    guessed how long the child takes."""
    end = time.time() + secs
    while time.time() < end:
        t0 = _cpu_s(pid)
        time.sleep(1.0)
        if _cpu_s(pid) - t0 < 0.2:
            return True
    return False


def _works(pid: int, secs: float = 60.0) -> bool:
    """Whether the child uses more than 0.3 s of CPU within `secs`."""
    t0, end = _cpu_s(pid), time.time() + secs
    while time.time() < end:
        time.sleep(0.25)
        if _cpu_s(pid) - t0 > 0.3:
            return True
    return False


def test_enabled_defaults_to_offload_and_env_overrides(monkeypatch):
    monkeypatch.delenv("FLINT_GPU_KEEPALIVE", raising=False)
    assert gpu_keepalive.enabled(True) and not gpu_keepalive.enabled(False)
    monkeypatch.setenv("FLINT_GPU_KEEPALIVE", "0")
    assert not gpu_keepalive.enabled(True)
    monkeypatch.setenv("FLINT_GPU_KEEPALIVE", "1")
    assert gpu_keepalive.enabled(False)


def test_runs_only_when_resumed_and_exits_on_close():
    ka = KeepAlive(log=lambda *_: None)
    pid = ka.proc.pid
    assert _goes_idle(pid, 60.0), "paused child must not submit work"  # it imports mlx, then waits
    ka.resume()
    assert _works(pid), "resumed child must be streaming kernels"
    ka.pause()
    assert _goes_idle(pid, 10.0), "paused again, the child must stop submitting work"
    ka.close()
    assert not _alive(pid)
    ka.close()  # idempotent


def test_child_dies_with_a_killed_parent_even_when_paused():
    src = (
        f"import sys, time; sys.path[:0] = {sys.path!r}\n"
        "from jitengine_flint_m2.gpu_keepalive import KeepAlive\n"
        "ka = KeepAlive(log=lambda *_: None); print(ka.proc.pid, flush=True); time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", src], stdout=subprocess.PIPE, text=True)
    pid = int(parent.stdout.readline())
    time.sleep(1.0)
    parent.kill()
    parent.wait()
    assert _wait_gone(pid), "keep-alive child outlived a killed parent"
