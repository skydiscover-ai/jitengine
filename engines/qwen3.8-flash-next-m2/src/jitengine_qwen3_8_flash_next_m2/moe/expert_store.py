"""Uncached reads of expert records from the FNXEPK01 pack, straight into pool memory.

Reads are `os.preadv` calls on one `F_NOCACHE` descriptor (Darwin's stand-in for `O_DIRECT`). That
flag does not bypass pages already in the unified buffer cache, so the counters report requested
bytes, which equal device bytes only on a purged cache.
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
        record_stride: int | None = None,
        num_experts: int | None = None,
        layers: list[int] | None = None,
        workers: int = WORKERS,
        nocache: bool = True,
    ):
        self.path = path
        self.data_offset = int(data_offset)
        self.record_bytes = int(record_bytes)
        self.record_stride = int(record_stride) if record_stride is not None else self.record_bytes
        if self.record_stride < self.record_bytes:
            raise ValueError(
                f"record_stride {self.record_stride} < record_bytes {self.record_bytes}"
            )
        self.num_records = int(num_records)  # of the global id space (layers x experts)
        # layer -> position in the pack (identity for a full pack)
        self.num_experts = int(num_experts) if num_experts else 0
        self._layer_pos: dict[int, int] | None = None
        if layers is not None:
            if not self.num_experts:
                raise ValueError("num_experts is required with a layer list")
            self._layer_pos = {int(layer): i for i, layer in enumerate(layers)}
        self.fd = os.open(path, os.O_RDONLY)
        self.nocache = bool(nocache)
        if self.nocache:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="expert-io"
        )
        # two speculative readers; more crowd the device queue the demand reads share
        self._pf_workers = 2
        self._pf_pool = None
        self._lock = threading.Lock()
        self.reads = 0
        self.bytes_requested = 0
        self.read_seconds = 0.0

    @classmethod
    def from_header(cls, path: str, meta: dict, **kw) -> ExpertStore:
        """The store whose geometry the pack header describes."""
        return cls(
            path,
            meta["data_offset"],
            meta["record_bytes"],
            int(meta["num_layers"]) * int(meta["num_experts"]),
            record_stride=meta["record_stride"],
            num_experts=meta["num_experts"],
            layers=meta["layers"],
            **kw,
        )

    # ------------------------------------------------------------------ internals
    def offset_of(self, record: int) -> int:
        """Byte offset of global record id `record` (= layer * num_experts + expert)."""
        if not 0 <= record < self.num_records:
            raise IndexError(f"record {record} out of range (0, {self.num_records})")
        if self._layer_pos is not None:
            layer, expert = divmod(int(record), self.num_experts)
            pos = self._layer_pos.get(layer)
            if pos is None:
                raise IndexError(f"layer {layer} is not in this pack")
            record = pos * self.num_experts + expert
        return self.data_offset + record * self.record_stride

    def _preadv_full(self, buffers: list, offset: int) -> int:
        """`preadv` until every destination byte is filled (a short read is legal on any fd)."""
        views = [memoryview(b).cast("B") for b in buffers]
        total = sum(len(v) for v in views)
        if total != self.record_bytes:
            raise ValueError(
                f"destination buffers total {total} B, record is {self.record_bytes} B"
            )
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

    # ------------------------------------------------------------------ public
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
            self._read_concurrently(jobs)
        return time.monotonic() - t0

    def _read_concurrently(self, jobs: list) -> None:
        """Run every job on the pool, waiting for all before re-raising the first failure."""
        futures = [self._pool.submit(self.read_into, rec, bufs) for rec, bufs in jobs]
        # Wait for every future before re-raising: an abandoned sibling could still be writing
        # into pool buffers that the caller's rollback has returned to the free list.
        first: BaseException | None = None
        for f in futures:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def submit_demand(self, record: int, buffers: list, fn=None):
        """Queue `record` into `buffers` on the demand executor; returns a Future.

        Demand reads never queue behind speculative ones. Pass `fn` as the outer store's
        `read_into`: a fault-injection wrapper overrides it, and `self.read_into` would bypass it.
        """
        return self._pool.submit(fn or self.read_into, record, buffers)

    def submit(self, record: int, buffers: list):
        """Queue `record` into `buffers` for a speculative read; returns a Future.

        Runs on its own executor so prefetches and demand reads never queue behind each other.
        `buffers` are reserve-slot rows, which no kernel can be reading.
        """
        if self._pf_pool is None:
            self._pf_pool = ThreadPoolExecutor(
                max_workers=self._pf_workers, thread_name_prefix="expert-pf"
            )
        return self._pf_pool.submit(self.read_into, record, buffers)

    def stats(self) -> dict:
        with self._lock:
            return {
                "reads": self.reads,
                "bytes_requested": self.bytes_requested,
                "read_seconds": self.read_seconds,
                "workers": self._pool._max_workers,
                "nocache": self.nocache,
                "path": self.path,
                "record_bytes": self.record_bytes,
                "record_stride": self.record_stride,
                "sched": {
                    "sched": "two-executors",
                    "max_inflight": self._pool._max_workers + self._pf_workers,
                },
            }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._pf_pool is not None:
            self._pf_pool.shutdown(wait=True)
            self._pf_pool = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
