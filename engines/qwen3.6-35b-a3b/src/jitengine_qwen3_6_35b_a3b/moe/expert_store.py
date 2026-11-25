"""Uncached reads of fixed-size expert records from the pack straight into pool memory.

`F_NOCACHE` (Darwin's `O_DIRECT`) does not bypass pages already in the unified buffer cache, so the
counters report requested bytes, which equal device bytes only on a purged cache. Queue depths past
`WORKERS` lengthen miss stalls without adding throughput.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
WORKERS = 8


class ExpertStore:
    """The pack file as a random-access array of fixed-size records."""

    def __init__(
        self,
        path: str,
        data_offset: int,
        record_bytes: int,
        num_records: int,
        workers: int = WORKERS,
        nocache: bool = True,
        record_stride: int | None = None,
    ):
        self.path = path
        self.data_offset = int(data_offset)
        self.record_bytes = int(record_bytes)
        # payload bytes vs disk stride: a pack may pad records to the page size (format v2)
        self.record_stride = int(record_stride or record_bytes)
        if self.record_stride < self.record_bytes:
            raise ValueError(
                f"record stride {self.record_stride} < record bytes {self.record_bytes}"
            )
        self.num_records = int(num_records)
        self.fd = os.open(path, os.O_RDONLY)
        self.nocache = bool(nocache)
        if self.nocache:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="expert-io"
        )
        self._lock = threading.Lock()
        self.reads = 0
        self.bytes_requested = 0
        self.read_seconds = 0.0

    def offset_of(self, record: int) -> int:
        if not 0 <= record < self.num_records:
            raise IndexError(f"record {record} out of range (0, {self.num_records})")
        return self.data_offset + record * self.record_stride

    def _preadv_full(self, buffers: list, offset: int) -> int:
        """`preadv` until every destination byte is filled (a short read is legal on any fd)."""
        views = [memoryview(b).cast("B") for b in buffers]
        total = sum(len(v) for v in views)
        got = 0
        while got < total:
            n = os.preadv(self.fd, views, offset + got)
            if n <= 0:
                raise OSError(
                    f"short read at {offset + got} on {self.path} (wanted {total - got} more)"
                )
            got += n
            # drop the buffers preadv has already filled, slice the one it stopped inside
            k = n
            while views and k >= len(views[0]):
                k -= len(views[0])
                views.pop(0)
            if k and views:
                views[0] = views[0][k:]
        return got

    def read_into(self, record: int, buffers: list) -> int:
        """Read record `record` into `buffers` (their total length must be `record_bytes`)."""
        t0 = time.monotonic()
        n = self._preadv_full(buffers, self.offset_of(record))
        dt = time.monotonic() - t0
        with self._lock:
            self.reads += 1
            self.bytes_requested += n
            self.read_seconds += dt
        return n

    def read_many(self, jobs: list) -> float:
        """`jobs` is [(record, [buffers]), ...]; issued at queue depth `workers`. Returns seconds."""
        if not jobs:
            return 0.0
        t0 = time.monotonic()
        if len(jobs) == 1:
            self.read_into(*jobs[0])
        else:
            futures = [self._pool.submit(self.read_into, rec, bufs) for rec, bufs in jobs]
            first = self._first_error(futures)
            if first is not None:
                raise first
        return time.monotonic() - t0

    def _first_error(self, futures: list) -> BaseException | None:
        """Wait for every read and return the first error, so no sibling read is still running when the caller rolls back."""
        # wait for every future before re-raising: the caller's rollback frees pool buffers
        # that a still-running sibling read would otherwise overwrite; the first error is raised
        first: BaseException | None = None
        for f in futures:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001
                if first is None:
                    first = exc
        return first

    def stats(self) -> dict:
        with self._lock:
            return {
                "reads": self.reads,
                "bytes_requested": self.bytes_requested,
                "read_seconds": self.read_seconds,
                "workers": self._pool._max_workers,
                "nocache": self.nocache,
                "path": self.path,
            }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
