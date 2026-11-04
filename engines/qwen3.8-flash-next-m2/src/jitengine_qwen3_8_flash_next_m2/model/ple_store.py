"""The PLE tier: streamed, uncached reads of hashed n-gram rows from the FNXPLE01 pack.

Layout (checkpoint/pack_format.py): a header block, then each segment's rows back to back at `row_stride`; row
g lives at `file_offset + (g - row_start) * row_stride` of its segment. Rows are deduplicated within
one call only and never kept, so `ple_bytes == ple_rows_read x row_bytes`.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import CheckpointScan
from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import (
    DTYPE_BYTES,
    PAGE,
    PLE_MAGIC,
    PackError,
    PLEScan,
    default_ple_pack_path,
    read_header_block,
    validate_ple,
)
from jitengine_qwen3_8_flash_next_m2.moe.expert_store import F_NOCACHE

WORKERS = 16

_NP_STORAGE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16, "F32": np.float32}
_MX_DTYPE = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


# ------------------------------------------------------------------ pack file
def open_ple_pack(
    model_path: str, path: str | None = None, scan: CheckpointScan | None = None, log=print
) -> tuple[str, dict, PLEScan]:
    """Validate the PLE pack for `model_path`; returns (path, header, PLEScan). Never builds one."""
    scan = scan or CheckpointScan(model_path)
    ple = PLEScan(scan.model_path, scan.ckpt, scan.prefix)
    path = os.path.expanduser(path) if path else default_ple_pack_path(ple)
    if not os.path.exists(path):
        raise PackError(f"PLE pack {path} does not exist")
    ok, why = validate_ple(path, ple)
    if not ok:
        log(f"[ple-pack] rejecting {path}: {why}")
        raise PackError(f"PLE pack {path} unusable: {why}")
    return path, read_header_block(path, PLE_MAGIC), ple


# ------------------------------------------------------------------ the reader
class PLEStore:
    """Streamed rows of the n-gram table: `fetch(ids)` -> `[N, values_per_row]` dequantised.

    Dedup is within one call only; nothing is kept between calls. `ple_page_bytes` counts the 16 KiB
    device page each row read moves.
    """

    def __init__(
        self, path: str, meta: dict, workers: int = WORKERS, nocache: bool = True, seam=None
    ):
        self.path = path
        self.meta = meta
        self.row_bytes = int(meta["row_bytes"])
        self.row_stride = int(meta["row_stride"])
        self.total_rows = int(meta["total_rows"])
        self.fields = list(meta["fields"])
        if sum(int(f["bytes"]) for f in self.fields) != self.row_bytes:
            raise PackError("PLE header: row_bytes != sum of field bytes")
        q = meta["quantization"]
        self.bits = int(q["bits"])
        self.group_size = int(q["group_size"])
        by = {f["name"]: f for f in self.fields}
        self._w, self._s, self._b = by["weight"], by["scales"], by["biases"]
        self.values_per_row = (
            int(self._w["cols"]) * (DTYPE_BYTES[self._w["dtype"]] * 8) // self.bits
        )
        if self.values_per_row != int(self._s["cols"]) * self.group_size:
            raise PackError("PLE header: packed values per row != groups x group_size")
        # segments as sorted arrays for a vectorised row -> offset map
        segs = sorted(meta["segments"], key=lambda s: s["row_start"])
        self._seg_start = np.array([s["row_start"] for s in segs], dtype=np.int64)
        self._seg_end = np.array([s["row_start"] + s["rows"] for s in segs], dtype=np.int64)
        self._seg_off = np.array([s["file_offset"] for s in segs], dtype=np.int64)
        self.rows_present = int(self._seg_end.sum() - self._seg_start.sum())
        self.fd = os.open(path, os.O_RDONLY)
        self.nocache = bool(nocache)
        if self.nocache:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="ple-io"
        )
        self._lock = threading.Lock()
        # fault-injection seam (faults.py), checked before a read is issued or counted; `owners`
        # is the request-id list of the current unit of work, set by the engine
        self.seam = seam
        self.owners: list | None = None
        self.rows_requested = 0
        self.rows_read = 0
        self.bytes = 0
        self.read_seconds = 0.0
        self.calls = 0

    # ------------------------------------------------------------------ addressing
    def offsets_of(self, rows: np.ndarray) -> np.ndarray:
        """Byte offsets of global rows via the header's segment list; raises for an absent row."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.total_rows):
            bad = rows[(rows < 0) | (rows >= self.total_rows)][0]
            raise PackError(f"PLE row {int(bad)} out of range 0..{self.total_rows - 1}")
        i = np.searchsorted(self._seg_start, rows, side="right") - 1
        ok = (i >= 0) & (rows < self._seg_end[np.maximum(i, 0)])
        if not bool(ok.all()):
            raise PackError(
                f"PLE row {int(rows[~ok][0])} is not in this pack ({len(self._seg_start)} segments)"
            )
        return self._seg_off[i] + (rows - self._seg_start[i]) * self.row_stride

    # ------------------------------------------------------------------ I/O
    def _read_rows(self, rows: np.ndarray) -> np.ndarray:
        """Raw `[n, row_bytes]` uint8 for unique global rows, one `pread` each."""
        n = rows.size
        buf = np.empty((n, self.row_bytes), dtype=np.uint8)
        if n == 0:
            return buf
        offs = self.offsets_of(rows)
        fd, rb = self.fd, self.row_bytes

        def one(i: int) -> None:
            g = int(rows[i])
            mv = memoryview(buf[i])
            got = 0
            off = int(offs[i])
            while got < rb:
                k = os.preadv(fd, [mv[got:]], off + got)
                if k <= 0:
                    raise OSError(f"short read of PLE row {g} at {off + got} on {self.path}")
                got += k

        if n == 1:
            one(0)
        else:
            futures = [self._pool.submit(one, i) for i in range(n)]
            self._wait_all(futures)
        return buf

    def _wait_all(self, futures: list) -> None:
        """Wait for every future, then re-raise the first error if any failed."""
        first: BaseException | None = None
        for f in futures:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def fetch_raw(self, ids, row_of=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(words [N, cols], scales and biases [N, groups] u16) for flat `ids`, in order."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if self.seam is not None:  # before any read or counter: a failed read counts nothing
            self.seam.check("ple_row", ids, row_of, self.owners)
        t0 = time.monotonic()
        uniq, inv = np.unique(ids, return_inverse=True)
        raw = self._read_rows(uniq)
        dt = time.monotonic() - t0
        with self._lock:
            self.calls += 1
            self.rows_requested += int(ids.size)
            self.rows_read += int(uniq.size)
            self.bytes += int(uniq.size) * self.row_bytes
            self.read_seconds += dt
        rows = raw[inv.reshape(-1)]
        w = np.ascontiguousarray(
            rows[:, self._w["offset"] : self._w["offset"] + self._w["bytes"]]
        ).view(_NP_STORAGE[self._w["dtype"]])
        s = np.ascontiguousarray(
            rows[:, self._s["offset"] : self._s["offset"] + self._s["bytes"]]
        ).view(_NP_STORAGE[self._s["dtype"]])
        b = np.ascontiguousarray(
            rows[:, self._b["offset"] : self._b["offset"] + self._b["bytes"]]
        ).view(_NP_STORAGE[self._b["dtype"]])
        return w, s, b

    def fetch(self, ids, row_of=None) -> mx.array:
        """Dequantised rows `[N, values_per_row]` in the table's scale dtype (bf16 on the target)."""
        w, s, b = self.fetch_raw(ids, row_of)
        wq = mx.array(w)
        sc = mx.array(s)
        bi = mx.array(b)
        if self._s["dtype"] != "F32":
            sc = sc.view(_MX_DTYPE[self._s["dtype"]])
            bi = bi.view(_MX_DTYPE[self._b["dtype"]])
        return mx.dequantize(wq, sc, bi, group_size=self.group_size, bits=self.bits, mode="affine")

    # ------------------------------------------------------------------ observability
    def counters(self) -> dict:
        with self._lock:
            return {
                "ple_rows_requested": self.rows_requested,
                "ple_rows_read": self.rows_read,
                "ple_bytes": self.bytes,
                "ple_read_seconds": self.read_seconds,
                "ple_page_bytes": self.rows_read * PAGE,
                "ple_calls": self.calls,
            }

    def stats(self) -> dict:
        return {
            **self.counters(),
            "ple_row_bytes": self.row_bytes,
            "ple_total_rows": self.total_rows,
            "ple_rows_present": self.rows_present,
            "ple_values_per_row": self.values_per_row,
            "ple_path": self.path,
            "ple_nocache": self.nocache,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
