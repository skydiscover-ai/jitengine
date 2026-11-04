"""Darwin telemetry bindings; the footprint is validated against footprint(1)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time


def _until(done, secs: float = 30.0) -> None:
    """Wait until `done()` holds, at most `secs`: a fixed sleep sees fewer samples on a loaded host."""
    end = time.monotonic() + secs
    while not done() and time.monotonic() < end:
        time.sleep(0.02)


def test_capability_report_has_errors_not_guesses():
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.monitor import capability_report

    r = capability_report()
    for k in ("process_footprint", "system_memory"):
        assert "ok" in r[k]
        assert ("value" in r[k]) == r[k]["ok"] and (("error" in r[k]) != r[k]["ok"])


def test_footprint_matches_footprint_tool():
    if sys.platform != "darwin" or not shutil.which("footprint"):
        return
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.monitor import process_rusage

    code = (
        "import time; b = bytearray(300 << 20); b[::4096] = b'x' * len(b[::4096]); time.sleep(20)"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        time.sleep(3)
        ours = process_rusage(child.pid)["footprint_bytes"]
        out = subprocess.run(["footprint", str(child.pid)], capture_output=True, text=True).stdout
        m = re.search(r"Footprint:\s+([\d.]+)\s*(KB|MB|GB)", out)
        assert m, out[:400]
        tool = float(m.group(1)) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[m.group(2)]
        assert ours > 300 << 20
        assert abs(ours - tool) / tool < 0.05, (ours, tool)
    finally:
        child.kill()
        child.wait()


def test_monitor_thread_samples_and_stops():
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.monitor import MemoryMonitor

    seen = []
    m = MemoryMonitor(0.05, callback=seen.append).start()
    _until(lambda: len(seen) >= 3)
    m.stop()
    assert not m.running and len(seen) >= 3
    snap = m.snapshot()
    assert snap["samples"] >= 3 and snap["available_is_estimate"]
    if sys.platform == "darwin":
        assert seen[-1].footprint_bytes and seen[-1].available_bytes and seen[-1].error is None


def test_monitor_survives_a_failing_callback_and_sampler():
    from jitengine_qwen3_6_35b_a3b_m4pro.memory.monitor import MemoryMonitor, MemorySample

    def bad_cb(s):
        raise RuntimeError("controller bug")

    def bad_sampler(pid):
        return MemorySample(time.monotonic(), None, None, None, "denied"), {
            "error": "denied",
            "timestamp_monotonic": time.monotonic(),
        }

    m = MemoryMonitor(0.05, callback=bad_cb, sampler=bad_sampler).start()
    _until(lambda: m.snapshot()["callback_errors"] >= 2 and m.snapshot()["errors"] >= 2)
    m.stop()
    s = m.snapshot()
    assert s["callback_errors"] >= 2 and s["errors"] >= 2 and s["samples"] >= 2
