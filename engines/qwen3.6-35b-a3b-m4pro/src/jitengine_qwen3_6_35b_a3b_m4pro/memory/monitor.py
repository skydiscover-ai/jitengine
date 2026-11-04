"""Process and system memory telemetry on Darwin through native interfaces (ctypes, no MLX).

The process figure is the physical footprint (`proc_pid_rusage`), not RSS, which misses most
unified-memory Metal buffers. A probe that fails yields `None` plus the error text, never a guess.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass

RUSAGE_INFO_V2 = 2
HOST_VM_INFO64 = 4
KERN_SUCCESS = 0


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


class _VmStatistics64(ctypes.Structure):
    # <mach/vm_statistics.h> struct vm_statistics64 (natural_t = uint32, 64-bit counters)
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]


# SDK struct sizes; a mismatch means the binding is wrong for this OS and the probes report errors
_LAYOUT_OK = (
    ctypes.sizeof(_RusageInfoV2) == 160
    and ctypes.sizeof(_VmStatistics64) == 152
    and ctypes.sizeof(_XswUsage) == 32
)

_libc = None
_libproc = None
if sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
        _libproc = ctypes.CDLL(
            ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True
        )
    except OSError:
        _libc = _libproc = None


def _sysctl_bytes(name: str, ctype):
    buf = ctype()
    size = ctypes.c_size_t(ctypes.sizeof(buf))
    rc = _libc.sysctlbyname(
        name.encode(), ctypes.byref(buf), ctypes.byref(size), None, ctypes.c_size_t(0)
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name}) failed")
    return buf


def process_rusage(pid: int | None = None) -> dict:
    """{'footprint_bytes', 'resident_bytes', 'wired_bytes', 'diskio_read_bytes', ...} or raises."""
    if not _LAYOUT_OK or _libproc is None:
        raise OSError("proc_pid_rusage binding unavailable on this platform")
    info = _RusageInfoV2()
    rc = _libproc.proc_pid_rusage(
        ctypes.c_int(pid or os.getpid()), ctypes.c_int(RUSAGE_INFO_V2), ctypes.byref(info)
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    if info.ri_resident_size == 0:
        raise OSError("proc_pid_rusage returned an implausible record (resident size 0)")
    return {
        "footprint_bytes": int(info.ri_phys_footprint),
        "resident_bytes": int(info.ri_resident_size),
        "wired_bytes": int(info.ri_wired_size),
        "diskio_read_bytes": int(info.ri_diskio_bytesread),
        "diskio_write_bytes": int(info.ri_diskio_byteswritten),
        "pageins": int(info.ri_pageins),
    }


def system_memory() -> dict:
    """Page counts -> bytes, the swap figures and the kernel's memorystatus level. Raises on error."""
    if not _LAYOUT_OK or _libc is None:
        raise OSError("host_statistics64 binding unavailable on this platform")
    host = _libc.mach_host_self()
    stats = _VmStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    rc = _libc.host_statistics64(
        ctypes.c_uint32(host),
        ctypes.c_int(HOST_VM_INFO64),
        ctypes.byref(stats),
        ctypes.byref(count),
    )
    if rc != KERN_SUCCESS:
        raise OSError(f"host_statistics64 failed ({rc})")
    page = int(_sysctl_bytes("hw.pagesize", ctypes.c_int64).value)
    total = int(_sysctl_bytes("hw.memsize", ctypes.c_uint64).value)
    sw = _sysctl_bytes("vm.swapusage", _XswUsage)
    try:
        level = int(_sysctl_bytes("kern.memorystatus_level", ctypes.c_int32).value)
    except OSError:
        level = None
    return {
        "page_bytes": page,
        "free_bytes": stats.free_count * page,
        "inactive_bytes": stats.inactive_count * page,
        "speculative_bytes": stats.speculative_count * page,
        "purgeable_bytes": stats.purgeable_count * page,
        "wired_bytes": stats.wire_count * page,
        "active_bytes": stats.active_count * page,
        "compressed_bytes": stats.compressor_page_count * page,
        # an estimate of what could be handed out without paging: free + reclaimable file cache
        "available_bytes_estimate": (
            stats.free_count
            + stats.inactive_count
            + stats.speculative_count
            + stats.purgeable_count
        )
        * page,
        # what macOS can hand out without paging out app memory: all of it but wired memory, the compressor
        # and app (anonymous) pages, which purgeable pages are part of. Cached files count: macOS drops them
        # on demand, and after reading a large file they can fill most of memory while the estimate above
        # counts only the ones in the inactive queue.
        "available_bytes": max(
            0,
            total
            - (
                stats.wire_count
                + stats.compressor_page_count
                + stats.internal_page_count
                - stats.purgeable_count
            )
            * page,
        ),
        "swap_total_bytes": int(sw.xsu_total),
        "swap_used_bytes": int(sw.xsu_used),
        "memorystatus_level": level,
    }


@dataclass(frozen=True)
class MemorySample:
    timestamp: float
    footprint_bytes: int | None
    available_bytes: int | None
    swap_used_bytes: int | None
    error: str | None = None


def take_sample(pid: int | None = None) -> tuple[MemorySample, dict]:
    """One sample (for the policy) plus the full raw record (for /stats). Never raises."""
    now = time.monotonic()
    raw: dict = {"timestamp_monotonic": now, "time": time.time()}
    errors = []
    fp = avail = swap = None
    try:
        r = process_rusage(pid)
        raw["process"] = r
        fp = r["footprint_bytes"]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"process: {exc}")
    try:
        s = system_memory()
        raw["system"] = s
        avail = s["available_bytes_estimate"]
        swap = s["swap_used_bytes"]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"system: {exc}")
    err = "; ".join(errors) or None
    raw["error"] = err
    return MemorySample(now, fp, avail, swap, err), raw


class MemoryMonitor:
    """A sampler thread that calls `callback(sample)` after each sample; it never touches MLX."""

    def __init__(
        self, interval_s: float = 1.0, callback=None, pid: int | None = None, sampler=take_sample
    ):
        self.interval_s = float(interval_s)
        self.callback = callback
        self.pid = pid
        self._sampler = sampler
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last: dict | None = None
        self._last_sample: MemorySample | None = None
        self.samples = 0
        self.errors = 0
        self.callback_errors = 0
        self.peak_footprint_bytes = 0
        self._thread = threading.Thread(
            target=self._loop, name="jitengine-qwen3.6-35b-a3b-m4pro-memory-monitor", daemon=True
        )

    def start(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def sample_now(self) -> dict:
        sample, raw = self._sampler(self.pid)
        self._record(sample, raw)
        return self._public(raw)

    def _record(self, sample: MemorySample, raw: dict) -> None:
        with self._lock:
            self._last, self._last_sample = raw, sample
            self.samples += 1
            if sample.error:
                self.errors += 1
            if sample.footprint_bytes is not None:
                self.peak_footprint_bytes = max(self.peak_footprint_bytes, sample.footprint_bytes)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                sample, raw = self._sampler(self.pid)
            except Exception as exc:  # noqa: BLE001  the sampler never raises; belt and braces
                sample, raw = MemorySample(t0, None, None, None, repr(exc)), {"error": repr(exc)}
            self._record(sample, raw)
            if self.callback is not None:
                try:
                    self.callback(sample)
                except Exception:  # noqa: BLE001  a controller fault must not kill telemetry
                    with self._lock:
                        self.callback_errors += 1
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - t0)))

    def _public(self, raw: dict | None) -> dict:
        if raw is None:
            return {"available": False}
        out = dict(raw)
        out["age_s"] = round(time.monotonic() - raw.get("timestamp_monotonic", time.monotonic()), 3)
        out["available_is_estimate"] = True
        return out

    def last(self) -> MemorySample | None:
        with self._lock:
            return self._last_sample

    def snapshot(self) -> dict:
        with self._lock:
            out = self._public(self._last)
            out.update(
                samples=self.samples,
                errors=self.errors,
                callback_errors=self.callback_errors,
                peak_footprint_bytes=self.peak_footprint_bytes,
                interval_s=self.interval_s,
                running=self._thread.is_alive(),
            )
            return out


def capability_report() -> dict:
    """What `jitengine-qwen3.6-35b-a3b-m4pro doctor` shows: which probes work here, with the exact error when one does not."""
    out = {"layout_ok": _LAYOUT_OK, "platform": sys.platform}
    for name, fn in (("process_footprint", process_rusage), ("system_memory", system_memory)):
        try:
            out[name] = {"ok": True, "value": fn()}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": str(exc)}
    return out


__all__ = [
    "MemoryMonitor",
    "MemorySample",
    "asdict",
    "capability_report",
    "process_rusage",
    "system_memory",
    "take_sample",
]
