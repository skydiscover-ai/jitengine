"""External measurement helpers for the benchmark harness (standard library only; never trusts the engine).

  * process tree footprint: proc_pid_rusage(RUSAGE_INFO_V4) -> ri_phys_footprint (what footprint(1)
    reports; unified-memory Metal buffers count here and not in RSS) and the kernel's own
    ri_lifetime_max_phys_footprint, summed over the engine process and every descendant;
  * swap: vm.swapusage (bytes used) and the host's cumulative swap-out page counter, before/after;
  * box exclusivity: flock on <bench dir>/box.lock, plus a memory predicate (available bytes, wired bytes)
    waited on before a model is loaded, never a fixed sleep.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import json
import os
import subprocess
import threading
import time

RUSAGE_INFO_V4 = 4
HOST_VM_INFO64 = 4

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


class _RusageV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
        "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
        "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time")]


class _VmStats64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64), ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32), ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32), ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]


class _XswUsage(ctypes.Structure):
    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64),
                ("xsu_used", ctypes.c_uint64), ("xsu_pagesize", ctypes.c_uint32),
                ("xsu_encrypted", ctypes.c_int32)]


assert ctypes.sizeof(_RusageV4) == 296 and ctypes.sizeof(_VmStats64) == 152 and ctypes.sizeof(_XswUsage) == 32


def _sysctl(name: str, ctype):
    buf = ctype()
    size = ctypes.c_size_t(ctypes.sizeof(buf))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(buf), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name}) failed")
    return buf


def rusage(pid: int) -> dict | None:
    info = _RusageV4()
    if _libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), ctypes.byref(info)) != 0:
        return None
    return {"footprint": int(info.ri_phys_footprint), "lifetime_max": int(info.ri_lifetime_max_phys_footprint)}


def system() -> dict:
    stats = _VmStats64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    if _libc.host_statistics64(ctypes.c_uint32(_libc.mach_host_self()), ctypes.c_int(HOST_VM_INFO64),
                               ctypes.byref(stats), ctypes.byref(count)) != 0:
        raise OSError("host_statistics64 failed")
    page = int(_sysctl("hw.pagesize", ctypes.c_int64).value)
    sw = _sysctl("vm.swapusage", _XswUsage)
    avail = (stats.free_count + stats.inactive_count + stats.speculative_count + stats.purgeable_count) * page
    return {"available_bytes": avail, "wired_bytes": stats.wire_count * page, "swap_used_bytes": int(sw.xsu_used),
            "swapouts_pages": int(stats.swapouts), "page_bytes": page}


def tree(root: int) -> set:
    out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout
    kids: dict = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2:
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(kids.get(p, []))
    return seen


class BoxLock:
    """Exclusive flock on <bench dir>/box.lock for the lifetime of the measurement."""

    def __init__(self, path: str):
        self.path, self.fd = path, None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def wait_memory(need_bytes: int, wired_max_bytes: int = 4 * 2**30, max_wait_s: float = 1800.0, log=print) -> dict:
    """Block until `need_bytes` are available and wired memory has unwound (a predicate, not a clock)."""
    t0 = time.monotonic()
    k = 0
    while True:
        s = system()
        if s["available_bytes"] >= need_bytes and s["wired_bytes"] < wired_max_bytes:
            return {"waited_s": round(time.monotonic() - t0, 1), "available_gb": round(s["available_bytes"] / 1e9, 2),
                    "wired_gb": round(s["wired_bytes"] / 1e9, 2), "swap_used_gb": round(s["swap_used_bytes"] / 1e9, 3)}
        if time.monotonic() - t0 > max_wait_s:
            raise SystemExit(f"memory never settled: available {s['available_bytes'] / 1e9:.1f} GB, "
                             f"wired {s['wired_bytes'] / 1e9:.2f} GB")
        if k % 15 == 0:
            log(f"waiting for memory: available {s['available_bytes'] / 1e9:.1f} GB, wired {s['wired_bytes'] / 1e9:.2f} GB")
        k += 1
        time.sleep(1.0)


def other_big_processes(exclude: set, min_bytes: int = 2 * 2**30) -> list:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    big = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        pid = int(parts[0])
        if pid in exclude or pid == os.getpid():
            continue
        r = rusage(pid)
        if r and r["footprint"] >= min_bytes:
            big.append({"pid": pid, "footprint_gb": round(r["footprint"] / 1e9, 2),
                        "command": (parts[1] if len(parts) > 1 else "")[:160]})
    return big


class TreeSampler:
    """Samples the engine process tree from outside; tags each sample with the driver's current phase."""

    def __init__(self, root: int, phase_file: str | None = None, interval_s: float = 0.25):
        self.root, self.phase_file, self.interval = root, phase_file, interval_s
        self.peak = 0
        self.lifetime_max = 0
        self.per_phase: dict = {}
        self.samples = 0
        s = system()
        self.swap0, self.swapouts0 = s["swap_used_bytes"], s["swapouts_pages"]
        self.swap_max = self.swap0
        self.cotenants: dict = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def _phase(self) -> str:
        if not self.phase_file:
            return "-"
        try:
            with open(self.phase_file) as f:
                return f.read().strip() or "-"
        except OSError:
            return "-"

    def _run(self):
        k = 0
        while not self._stop.is_set():
            pids = tree(self.root)
            tot = 0
            for pid in pids:
                r = rusage(pid)
                if r:
                    tot += r["footprint"]
                    if pid == self.root:
                        self.lifetime_max = max(self.lifetime_max, r["lifetime_max"])
            ph = self._phase()
            self.per_phase[ph] = max(self.per_phase.get(ph, 0), tot)
            self.peak = max(self.peak, tot)
            self.swap_max = max(self.swap_max, system()["swap_used_bytes"])
            if k % 40 == 0:
                for b in other_big_processes(pids):
                    self.cotenants.setdefault(b["pid"], b)
            self.samples += 1
            k += 1
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=10)
        s = system()
        return {"peak_tree_footprint_gb": round(self.peak / 1e9, 3),
                "kernel_lifetime_max_footprint_gb": round(self.lifetime_max / 1e9, 3),
                "per_phase_peak_gb": {k: round(v / 1e9, 3) for k, v in self.per_phase.items()},
                "samples": self.samples, "interval_s": self.interval,
                "swap_used_start_gb": round(self.swap0 / 1e9, 3), "swap_used_max_gb": round(self.swap_max / 1e9, 3),
                "swap_used_growth_gb": round((self.swap_max - self.swap0) / 1e9, 3),
                "swapouts_delta_pages": int(s["swapouts_pages"] - self.swapouts0),
                "cotenants_ge_2gb": list(self.cotenants.values())}


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
