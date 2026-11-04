"""The token-embedding tier: `embed_tokens` rows read from the checkpoint instead of held resident.

A decode step gathers one row of the 0.37 GiB table per token, so the table need not sit in the
dense weights: the plan does not charge it (`embed_streamed_bytes`) and the expert pool takes its
bytes (`budget.plan_budget`). Rows come from the checkpoint's own tensors (`Checkpoint.slice_of`;
no pack is built): each field (weight, scales, biases) is one contiguous row-major tensor, so a row
is one `pread` per field, through `F_NOCACHE` descriptors as the PLE and expert tiers read.

A small LRU of rows (`cache_rows`, host memory the plan charges as its own bucket) keeps the rows
a request keeps drawing from: a prefill inserts its prompt's rows, so the tokens a decode emits are
mostly hits and the step's critical path carries no read. A row is immutable, so the cache needs no
invalidation.

Bit-identical to the resident `QEmbedding.lookup`: that gathers `weight[ids]`, `scales[ids]`,
`biases[ids]` and dequantizes them; this builds arrays of the same shape, dtype and bytes from the
shard and runs the same `mx.dequantize`. Affine dequantize is elementwise per group, so where the
bytes came from cannot enter the result. A tied checkpoint (`lm_head` is the embedding, read whole
every step) keeps the table resident.

The lookup needs the ids on the host, so a lazy id (the solo pipeline's in-flight token, the MTP
draft's verified token) is materialized first: that wait is the price of the slots.

`embed_page_bytes` counts the 16 KiB device pages the field reads move, so the device cross-check
(iostat bytes over the engine's reported reads) can account for this tier as it does for PLE.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import PAGE, PackError
from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import F_NOCACHE

EMBED = "model.embed_tokens"
FIELDS = ("weight", "scales", "biases")
WORKERS = 8
# Rows the LRU holds by default: a 2,952-token prompt has ~1,500 distinct ids, and the cache must
# also hold the rows a long decode adds; 8,192 x 1,600 B is 13.1 MB (about five compact slots).
CACHE_ROWS = 8192
# Host bytes per cached row beyond the row itself (the id -> slot map, the LRU entry), charged too.
CACHE_ROW_OVERHEAD = 256

_NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16, "F32": np.float32}
_MX_VIEW = {"BF16": mx.bfloat16, "F16": mx.float16}


def _tied(scan) -> bool:
    return (scan.prefix + "lm_head.weight") not in scan.weight_map


def stream_wanted(options: dict) -> bool:
    """The `embed_stream` option: "on" (default) or "off" (the resident table)."""
    v = str(options.get("embed_stream", "on")).strip().lower()
    if v not in ("on", "off"):
        raise ValueError(f"embed_stream {v!r}: 'on' or 'off'")
    return v == "on"


def embed_row_bytes(scan) -> int:
    """Bytes of one embedding row over its fields, from the shard headers; 0 if not row-major."""
    total = 0
    for f in FIELDS:
        name = scan.prefix + f"{EMBED}.{f}"
        if name not in scan.weight_map:
            if f == "biases":
                continue
            return 0
        _p, _s, length, _dt, shape = scan.ckpt.slice_of(name)
        if len(shape) != 2 or int(length) % int(shape[0]):
            return 0
        total += int(length) // int(shape[0])
    return total


def embed_streamed_bytes(scan, options: dict) -> int:
    """Bytes of the embedding table that are NOT resident under `options`: the table's full size
    when it is streamed, else 0. From the safetensors headers; nothing is read. The single
    predicate the plan, the loader and the model share."""
    if not stream_wanted(options) or _tied(scan) or not embed_row_bytes(scan):
        return 0
    total = 0
    for f in FIELDS:
        name = scan.prefix + f"{EMBED}.{f}"
        if name in scan.weight_map:
            total += int(scan.ckpt.slice_of(name)[2])
    return total


def resident(name: str) -> bool:
    """The loader's `keep` filter while the table streams: every dense tensor but the table's."""
    return not name.startswith(EMBED + ".")


def cache_rows_for(options: dict) -> int:
    """The `embed_cache_rows` option (default CACHE_ROWS); 0 reads every row from the shard."""
    return max(0, int(options.get("embed_cache_rows", CACHE_ROWS)))


def embed_cache_bytes(scan, options: dict) -> int:
    """What the plan charges for the row cache: its rows and their bookkeeping; 0 if not streamed."""
    if not embed_streamed_bytes(scan, options):
        return 0
    return cache_rows_for(options) * (embed_row_bytes(scan) + CACHE_ROW_OVERHEAD)


class StreamedEmbedding:
    """`lookup(ids)` -> `[*ids.shape, hidden]` dequantized rows, read from the checkpoint.

    Duck-types the parts of `QEmbedding` the engine uses: `lookup`, `out_dims` (the row count, the
    token-id bound), `in_dims`, `bits`, `group_size`. Called from the model thread; the counters are
    read from others, under the lock."""

    def __init__(
        self,
        scan,
        bits: int,
        group_size: int,
        cache_rows: int = CACHE_ROWS,
        workers: int = WORKERS,
        nocache: bool = True,
    ):
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.fields: list[tuple] = []  # (name, fd, base offset, row bytes, dtype, cols)
        self._fds: dict[str, int] = {}
        self.nocache = bool(nocache)
        rows = None
        for f in FIELDS:
            name = scan.prefix + f"{EMBED}.{f}"
            if f == "biases" and name not in scan.weight_map:
                continue
            path, start, length, dtype, shape = scan.ckpt.slice_of(name)
            if len(shape) != 2:
                raise PackError(f"{name}: shape {shape} is not 2-D")
            rows = int(shape[0]) if rows is None else rows
            if int(shape[0]) != rows:
                raise PackError(f"{name}: {shape[0]} rows != {rows}")
            per_row, rem = divmod(int(length), rows)
            item = np.dtype(_NP[dtype]).itemsize if dtype in _NP else 0
            if rem or not item or per_row != int(shape[1]) * item:
                raise PackError(f"{name}: {length} B of {dtype} {shape} is not a row-major table")
            if path not in self._fds:
                fd = os.open(path, os.O_RDONLY)
                if self.nocache:
                    fcntl.fcntl(fd, F_NOCACHE, 1)
                self._fds[path] = fd
            self.fields.append((f, self._fds[path], int(start), per_row, dtype, int(shape[1])))
        if len(self.fields) < 2 or self.fields[0][0] != "weight" or self.fields[1][0] != "scales":
            raise PackError("embed_tokens: needs weight and scales")
        self.has_biases = len(self.fields) == 3
        self.out_dims = int(rows)
        wcols = self.fields[0][5]
        self.in_dims = wcols * np.dtype(_NP[self.fields[0][4]]).itemsize * 8 // self.bits
        if self.in_dims != self.fields[1][5] * self.group_size:
            raise PackError(
                f"embed_tokens: {self.in_dims} packed values per row != "
                f"{self.fields[1][5]} groups x {self.group_size}"
            )
        self.row_bytes = sum(f[3] for f in self.fields)
        self.table_bytes = self.row_bytes * self.out_dims
        # the LRU: per-field row storage, id -> slot (most recent last), free slots
        self.cache_rows = max(0, int(cache_rows))
        self._cache = self._rows(self.cache_rows)
        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free = list(range(self.cache_rows - 1, -1, -1))
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="embed-io"
        )
        self._lock = threading.Lock()
        self.rows_requested = 0
        self.rows_read = 0
        self.cache_hits = 0
        self.bytes = 0
        self.page_bytes = 0
        self.read_seconds = 0.0
        self.calls = 0

    def _rows(self, n: int) -> list[np.ndarray]:
        """Per field, empty `[n, cols]` row storage in the field's storage dtype."""
        return [np.empty((n, cols), dtype=_NP[dt]) for (_f, _fd, _s, _pr, dt, cols) in self.fields]

    # ------------------------------------------------------------------ I/O
    def _pages(self, off: int, n: int) -> int:
        return ((off + n - 1) // PAGE - off // PAGE + 1) * PAGE

    def _read_into(self, bufs: list[np.ndarray], at: np.ndarray, ids: np.ndarray) -> int:
        """Read rows `ids` into `bufs[f][at]`, one `pread` per (row, field); returns device pages."""
        jobs = [(fi, int(a), int(t)) for fi in range(len(self.fields)) for a, t in zip(at, ids)]

        def one(job) -> None:
            fi, a, t = job
            name, fd, start, per_row, _dt, _c = self.fields[fi]
            mv = memoryview(bufs[fi][a]).cast("B")
            off = start + t * per_row
            got = 0
            while got < per_row:
                k = os.preadv(fd, [mv[got:]], off + got)
                if k <= 0:
                    raise OSError(f"short read of {EMBED}.{name} row {t} at {off + got}")
                got += k

        if len(jobs) <= 1:
            for j in jobs:
                one(j)
        else:
            first: BaseException | None = None
            for fu in [self._pool.submit(one, j) for j in jobs]:
                try:
                    fu.result()
                except BaseException as exc:  # noqa: BLE001
                    first = first or exc
            if first is not None:
                raise first
        return sum(
            self._pages(self.fields[fi][2] + t * self.fields[fi][3], self.fields[fi][3])
            for fi, _a, t in jobs
        )

    def _insert(self, ids: np.ndarray, bufs: list[np.ndarray], at: np.ndarray) -> None:
        """Put rows `bufs[f][at]` of `ids` into the LRU, evicting the least recent."""
        if not self.cache_rows:
            return
        k = min(len(ids), self.cache_rows)  # a call with more misses than slots keeps its last ones
        for t, a in zip(ids[-k:].tolist(), at[-k:].tolist()):
            if self._free:
                s = self._free.pop()
            else:
                _old, s = self._lru.popitem(last=False)
            for fi in range(len(self.fields)):
                self._cache[fi][s] = bufs[fi][a]
            self._lru[t] = s

    def fetch_raw(self, ids) -> list[np.ndarray]:
        """Per field, `[N, cols]` row storage for the flat ids, in order (cache, then shard)."""
        flat = np.asarray(ids).astype(np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.out_dims):
            bad = flat[(flat < 0) | (flat >= self.out_dims)][0]
            raise IndexError(
                f"token id {int(bad)} outside the embedding table 0..{self.out_dims - 1}"
            )
        with self._lock:
            t0 = time.monotonic()
            uniq, inv = np.unique(flat, return_inverse=True)
            bufs = self._rows(uniq.size)
            miss = []
            for j, t in enumerate(uniq.tolist()):
                s = self._lru.get(t)
                if s is None:
                    miss.append(j)
                    continue
                self._lru.move_to_end(t)
                for fi in range(len(self.fields)):
                    bufs[fi][j] = self._cache[fi][s]
            pages = 0
            if miss:
                at = np.asarray(miss, dtype=np.int64)
                pages = self._read_into(bufs, at, uniq[at])
                self._insert(uniq[at], bufs, at)
            self.calls += 1
            self.rows_requested += int(flat.size)
            self.cache_hits += int(uniq.size) - len(miss)
            self.rows_read += len(miss)
            self.bytes += len(miss) * self.row_bytes
            self.page_bytes += pages
            self.read_seconds += time.monotonic() - t0
        inv = inv.reshape(-1)
        return [b[inv] for b in bufs]

    def lookup(self, ids) -> mx.array:
        """The resident `QEmbedding.lookup`, bit for bit: same row bytes, shapes and dequantize."""
        shape = tuple(int(d) for d in ids.shape)
        arrs = []
        for (_f, _fd, _s, _pr, dt, cols), p in zip(self.fields, self.fetch_raw(ids)):
            a = mx.array(p.reshape(*shape, cols))
            arrs.append(a.view(_MX_VIEW[dt]) if dt in _MX_VIEW else a)
        return mx.dequantize(
            arrs[0],
            scales=arrs[1],
            biases=arrs[2] if self.has_biases else None,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )

    # ------------------------------------------------------------------ observability
    def stats(self) -> dict:
        with self._lock:
            return {
                "embed_streamed": True,
                "embed_rows_requested": self.rows_requested,
                "embed_rows_read": self.rows_read,
                "embed_cache_hits": self.cache_hits,
                "embed_cache_rows": self.cache_rows,
                "embed_cache_live": len(self._lru),
                "embed_bytes": self.bytes,
                "embed_page_bytes": self.page_bytes,
                "embed_read_seconds": self.read_seconds,
                "embed_calls": self.calls,
                "embed_row_bytes": self.row_bytes,
                "embed_table_bytes": self.table_bytes,
                "embed_nocache": self.nocache,
            }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass  # already closed; the rest still have to be closed
        self._fds = {}
