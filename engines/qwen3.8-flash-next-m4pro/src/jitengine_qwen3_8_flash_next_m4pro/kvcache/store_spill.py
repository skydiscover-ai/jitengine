"""Prefix-store spill: the store's entries written to a scratch file so their memory can be lent.

One-row decode never reads the prefix store (it is read at admission and written at the end of a
prefill), so while one row decodes its 0.5 GiB can hold expert records instead (the pool's annex).
Entries are immutable, so each is written once, then only read.

Writes are behind the decode: `write_behind` hands the entries not yet on file to one writer thread
and returns at once; their arrays stay held (by the entry and by the write) until the write is done,
so the memory is lent only once every byte is on file (`PrefixCache.spill` drops the arrays then).
A write the decode would otherwise wait for costs the decode nothing; a decode that ends first only
lends nothing that turn.

Where an entry's bytes are is recorded on the entry itself (`Entry.spill_at`), so the file can never
answer for an entry it did not write. Space is reused: a new entry goes in the first gap between
live entries' regions that fits, else at the end, so evicted entries' space is reclaimed without
moving any byte; the file never holds a byte of a live entry's that another write could overwrite.
`read_back` reads an entry's arrays byte for byte; one that fails costs only that entry (a store
miss, the next request recomputes the prefill).

The file is opened with `F_NOCACHE` and unlinked at once: its bytes never sit in the page cache
(which would hold the store outside the process's budget), and it disappears with the process. The
engine puts it on the expert pack's volume but outside the pack directory, which it only reads
(`<pack dir>/../jitengine-spill`, option `store_spill_dir`; `open_store_spill`): the spill's traffic
shares the device the expert reads use, and is measured that way. A place it cannot write turns
the spill off, never the server. Every transfer is whole 16 KiB pages (an array's tail is
padded), so `page_bytes_written` / `page_bytes_read` are what the device moves, counted with the
engine's other reads in the device cross-check. A directory the spill created is removed at `close`.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import queue
import tempfile
import threading
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import F_NOCACHE

logger = logging.getLogger(__name__)

_NP = {
    mx.float32: np.float32,
    mx.int64: np.int64,
    mx.int32: np.int32,
    mx.uint32: np.uint32,
    mx.bfloat16: np.uint16,  # numpy has no bfloat16: the bits travel as u16
    mx.float16: np.uint16,
}
ALIGN = 16 << 10


def _host(a: mx.array) -> np.ndarray:
    """The array's bytes as a numpy view (no copy: MLX's view and buffer share the storage)."""
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return np.asarray(a)


def _device(raw: np.ndarray, dtype) -> mx.array:
    a = mx.array(raw)
    return a.view(dtype) if dtype in (mx.bfloat16, mx.float16) else a


def _aligned(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


class _Write:
    """One entry's write: its region, the host views it writes from, and its outcome."""

    __slots__ = ("entry", "start", "length", "regs", "views", "done", "error")

    def __init__(self, entry, start: int, length: int, regs: list, views: list):
        self.entry, self.start, self.length = entry, int(start), int(length)
        self.regs, self.views = regs, views  # regs: [(offset, nbytes, shape, dtype)] per array
        self.done = threading.Event()
        self.error: BaseException | None = None


class StoreSpill:
    """The scratch file behind `PrefixCache.spill` / `read_back`. Its methods are called from the
    model thread only; the writer thread touches nothing but the file and its own `_Write`s."""

    def __init__(self, directory: str | None = None, nocache: bool = True):
        directory = os.path.abspath(directory or tempfile.gettempdir())
        self.created_dir = not os.path.isdir(directory)
        os.makedirs(directory, exist_ok=True)
        try:
            fd, path = tempfile.mkstemp(prefix="jitengine-store-spill-", dir=directory)
        except BaseException:
            if self.created_dir:  # leave nothing behind for a spill that never opened
                with contextlib.suppress(OSError):
                    os.rmdir(directory)
            raise
        os.unlink(path)  # gone with the process; nothing to clean up
        if nocache:
            try:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            except BaseException:
                os.close(fd)
                raise
        self.fd, self.dir, self.nocache = fd, directory, bool(nocache)
        self._pending: list[_Write] = []  # writes handed to the writer, not yet collected
        self._jobs: queue.Queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._write_loop, name="store-spill-writer", daemon=True
        )
        self._writer.start()
        self.file_bytes = 0  # end of the furthest live region
        self.spills = self.reads = 0
        self.entries_written = self.write_failures = 0
        self.bytes_written = self.bytes_read = 0
        self.page_bytes_written = self.page_bytes_read = 0  # whole pages: what the device moves
        self.write_seconds = self.read_seconds = 0.0
        self.restore_failures = 0

    # ------------------------------------------------------------------ writing
    def _write_loop(self) -> None:
        while True:
            w = self._jobs.get()
            if w is None:
                return
            t0 = time.monotonic()
            try:
                for (off, nbytes, _s, _d), raw in zip(w.regs, w.views):
                    mv = memoryview(raw).cast("B")
                    whole = nbytes - nbytes % ALIGN
                    self._pwrite_all(mv[:whole], off)
                    if whole < nbytes:  # the tail as one whole page, zero padded
                        page = bytearray(ALIGN)
                        page[: nbytes - whole] = mv[whole:]
                        self._pwrite_all(memoryview(page), off + whole)
                    self.page_bytes_written += _aligned(nbytes)
            except BaseException as exc:  # noqa: BLE001  reported by `_collect`; the entry stays resident
                w.error = exc
            w.views = None  # the arrays' storage is released with the entry's arrays, not before
            self.write_seconds += time.monotonic() - t0
            w.done.set()

    def _pwrite_all(self, mv: memoryview, off: int) -> None:
        done = 0
        while done < len(mv):
            k = os.pwrite(self.fd, mv[done:], off + done)
            if k <= 0:
                raise OSError(f"short write of the store spill at {off + done}")
            done += k

    def _collect(self) -> None:
        """Book the finished writes: a written entry gets its regions, a failed one stays resident."""
        keep = []
        for w in self._pending:
            if not w.done.is_set():
                keep.append(w)
                continue
            if w.error is None:
                w.entry.spill_at = w.regs
                self.entries_written += 1
                self.bytes_written += sum(n for _o, n, _s, _d in w.regs)
            else:
                w.entry.spill_failed = True
                self.write_failures += 1
                logger.warning(
                    f"[store] spill write of a {w.entry.tokens}-token entry failed: {w.error!r}"
                )
        self._pending = keep

    def _place(self, length: int, live: list) -> int:
        """First gap of at least `length` bytes between the live regions [(start, end)], else the end."""
        cur = 0
        for s, e in sorted(live):
            if s - cur >= length:
                return cur
            cur = max(cur, e)
        return cur

    def write_behind(self, entries: list) -> bool:
        """Start the writes of the resident entries not yet on file; True once every resident entry
        is on file (or its write failed and it stays resident). Never blocks on the device."""
        self._collect()
        if self._pending:
            return False
        todo = [e for e in entries if e.resident and e.spill_at is None and not e.spill_failed]
        if not todo:
            return True
        live = [(r[0][0], r[-1][0] + _aligned(r[-1][1])) for e in entries if (r := e.spill_at)]
        for e in todo:
            arrs = e.arrays()
            sizes = [int(a.nbytes) for a in arrs]
            length = sum(_aligned(n) for n in sizes)
            start = self._place(length, live)
            regs, views, off = [], [], start
            for a, n in zip(arrs, sizes):
                regs.append((off, n, tuple(a.shape), a.dtype))
                views.append(_host(a))
                off += _aligned(n)
            live.append((start, start + length))
            w = _Write(e, start, length, regs, views)
            self._pending.append(w)
            self._jobs.put(w)
        self.file_bytes = max(e for _s, e in live)
        self.spills += 1
        return False

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every handed-out write has finished (each at most `timeout` s), then book
        them; False if one is still in flight."""
        for w in list(self._pending):
            w.done.wait(timeout)
        self._collect()
        if self._pending:
            logger.warning(
                f"[store] {len(self._pending)} spill write(s) still in flight after {timeout} s"
            )
        return not self._pending

    def trim(self, entries: list) -> None:
        """Give the file's tail past the last live entry back to the volume."""
        if self._pending:
            return
        ends = [r[-1][0] + _aligned(r[-1][1]) for e in entries if (r := e.spill_at)]
        end = max(ends, default=0)
        if end < self.file_bytes:
            os.ftruncate(self.fd, end)
        self.file_bytes = end

    # ------------------------------------------------------------------ reading
    def read_back(self, entry) -> bool:
        """Read a spilled entry's arrays back into it; False (and a warning) when that fails."""
        t0 = time.monotonic()
        try:
            if entry.spill_at is None:
                raise OSError("entry not on file")
            arrs = [self._read(o, n, s, d) for o, n, s, d in entry.spill_at]
            mx.eval(*arrs)
            entry.install_arrays(arrs)
            nread = sum(n for _o, n, _s, _d in entry.spill_at)
            self.bytes_read += nread
            logger.info(
                f"[store] read back a {entry.tokens}-token entry: {nread / 1e6:.1f} MB in "
                f"{time.monotonic() - t0:.3f} s"
            )
            ok = True
        except Exception as exc:  # noqa: BLE001  a lost entry is a store miss, nothing worse
            logger.warning(f"[store] read-back of a {entry.tokens}-token entry failed: {exc!r}")
            self.restore_failures += 1
            ok = False
        self.reads += 1
        self.read_seconds += time.monotonic() - t0
        return ok

    def _read(self, off: int, nbytes: int, shape, dtype) -> mx.array:
        """One array's region as whole pages (the padding is dropped)."""
        span = _aligned(nbytes)
        buf = np.empty(span, dtype=np.uint8)
        mv = memoryview(buf)
        got = 0
        while got < span:
            k = os.preadv(self.fd, [mv[got:]], off + got)
            if k <= 0:
                raise OSError(f"short read of the store spill at {off + got}")
            got += k
        self.page_bytes_read += span
        return _device(buf[: int(nbytes)].view(_NP[dtype]).reshape(shape), dtype)

    # ------------------------------------------------------------------ the rest
    def io_stats(self) -> dict:
        """The engine's top-level I/O counters for the spill (the device cross-check adds
        `spill_page_bytes` to the expert, PLE and embedding bytes)."""
        return {
            "spill_dir": self.dir,
            "spill_page_bytes": int(self.page_bytes_written + self.page_bytes_read),
            "spill_page_bytes_written": int(self.page_bytes_written),
            "spill_page_bytes_read": int(self.page_bytes_read),
            "spill_bytes_written": int(self.bytes_written),
            "spill_bytes_read": int(self.bytes_read),
        }

    def stats(self) -> dict:
        return {
            "spill_dir": self.dir,
            "spill_nocache": self.nocache,
            "spill_file_bytes": int(self.file_bytes),
            "spill_writes_pending": len(self._pending),
            "spills": self.spills,
            "spill_entries_written": self.entries_written,
            "spill_write_failures": self.write_failures,
            "spill_read_backs": self.reads,
            "spill_bytes_written": int(self.bytes_written),
            "spill_bytes_read": int(self.bytes_read),
            "spill_page_bytes_written": int(self.page_bytes_written),
            "spill_page_bytes_read": int(self.page_bytes_read),
            "spill_write_seconds": round(self.write_seconds, 3),
            "spill_read_seconds": round(self.read_seconds, 3),
            "restore_failures": self.restore_failures,
        }

    def close(self) -> None:
        """Stop the writer, close the (already unlinked) file, and remove the directory if this
        spill created it and nothing else is in it."""
        if self.fd < 0:
            return
        self._jobs.put(None)
        self._writer.join(timeout=30)
        os.close(self.fd)
        self.fd = -1
        if self.created_dir:
            with contextlib.suppress(OSError):
                os.rmdir(self.dir)


def _st_dev(path: str) -> int:
    """The device of `path`, or of its nearest ancestor that exists."""
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return os.stat(p).st_dev


def _inside(path: str, top: str) -> bool:
    path, top = os.path.realpath(path), os.path.realpath(top)
    return os.path.commonpath([path, top]) == top


def open_store_spill(
    pack_path: str, model_path: str, directory: str | None = None, nocache: bool = True
):
    """The engine's spill and None, or None and why it is off.

    By default the spill goes in `<pack dir>/../jitengine-spill`: the pack's volume (when that
    directory is on the pack's device; else off), so its bytes are counted with the expert reads on
    the device that moves them, and never under the pack directory, which the engine only reads.
    `directory` (option `store_spill_dir`) names another place. A place under the pack or checkpoint
    directory, or one that cannot be written, turns the spill off: the store stays resident and
    the annex never lends, which costs speed, never a request.
    """
    pack_dir = os.path.dirname(os.path.abspath(pack_path))
    where = os.path.abspath(directory or os.path.join(os.path.dirname(pack_dir), "jitengine-spill"))
    why = None
    for top in (pack_dir, os.path.abspath(model_path)):
        if _inside(where, top):
            why = f"{where} is inside {top}, which the engine only reads"
    try:
        if why is None and directory is None and _st_dev(where) != _st_dev(pack_dir):
            why = f"{where} is not on the expert pack's device"
    except OSError as exc:
        why = f"{where}: {exc}"
    if why is None:
        try:
            return StoreSpill(where, nocache=nocache), None
        except OSError as exc:
            why = f"{where}: {exc}"
    return None, why
