"""The token-embedding tier: `embed_tokens` rows streamed from the checkpoint instead of resident.

At single-row decode the embedding gathers one row of the table per token, so the table need not
sit in the dense weights. Rows are read on demand the way `ple_store` reads the n-gram table:
`pread` with `F_NOCACHE`, one read per (unique row, field), deduplicated inside one call only,
nothing kept between calls. The freed bytes go to the expert pool through `budget.plan_budget`
(`embed_streamed_bytes`).

Output is bit-identical to the resident `QEmbedding.lookup`: that gathers `weight[ids]`,
`scales[ids]`, `biases[ids]` and dequantizes them; this reads exactly those row bytes from the
shard and runs the same `mx.dequantize` on arrays of the same shape and dtype. Affine dequantize is
elementwise per group, so the rows' neighbours in the table cannot enter the result.

The source is the checkpoint's own tensors (`Checkpoint.slice_of`), so no pack has to be built;
each field is one contiguous row-major tensor, hence one read per row per field. The reads are the
tier's own counters (`embed_*` in stats) and do not enter the expert pool's `miss_bytes`.

A tied checkpoint (`lm_head` is the embedding, read in full every step) keeps the table resident.
`embed_streamed_bytes(scan)` is the single predicate the plan, the loader and the model share.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import PackError

EMBED = "model.embed_tokens"
FIELDS = ("weight", "scales", "biases")
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
WORKERS = 16

_NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16, "F32": np.float32}
_MX_VIEW = {"BF16": mx.bfloat16, "F16": mx.float16}


def _tied(scan) -> bool:
    return (scan.prefix + "lm_head.weight") not in scan.weight_map


def embed_streamed_bytes(scan) -> int:
    """Bytes of the embedding table that are NOT resident under this configuration: the table's
    full size when it is streamed, else 0. From the safetensors headers; nothing is read."""
    if _tied(scan):
        return 0
    total = 0
    for f in FIELDS:
        name = f"{EMBED}.{f}"
        if (scan.prefix + name) not in scan.weight_map:
            if f == "biases":
                continue
            return 0
        total += int(scan.ckpt.slice_of(scan.prefix + name)[2])
    return total


class StreamedEmbedding:
    """`lookup(ids)` -> `[*ids.shape, hidden]` dequantised rows, read from the checkpoint per call.

    Duck-types the parts of `QEmbedding` the engine uses: `lookup`, `out_dims` (the row count, the
    token-id bound), `in_dims`, `bits`, `group_size`."""

    def __init__(
        self, scan, bits: int, group_size: int, workers: int = WORKERS, nocache: bool = True
    ):
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.fields = []  # (name, fd, base offset, row bytes, dtype, cols)
        self._fds: dict[str, int] = {}
        rows = None
        for f in FIELDS:
            name = f"{EMBED}.{f}"
            if f == "biases" and (scan.prefix + name) not in scan.weight_map:
                continue
            path, start, length, dtype, shape = scan.ckpt.slice_of(scan.prefix + name)
            if len(shape) != 2:
                raise PackError(f"{name}: shape {shape} is not 2-D")
            if rows is None:
                rows = int(shape[0])
            if int(shape[0]) != rows:
                raise PackError(f"{name}: {shape[0]} rows != {rows}")
            per_row, rem = divmod(int(length), rows)
            if rem or dtype not in _NP:
                raise PackError(f"{name}: {length} B / dtype {dtype} is not a row-major table")
            if path not in self._fds:
                fd = os.open(path, os.O_RDONLY)
                if nocache:
                    fcntl.fcntl(fd, F_NOCACHE, 1)
                self._fds[path] = fd
            self.fields.append((f, self._fds[path], int(start), per_row, dtype, int(shape[1])))
        self.has_biases = len(self.fields) == 3
        self.out_dims = int(rows)
        wcols = self.fields[0][5]
        self.in_dims = wcols * (np.dtype(_NP[self.fields[0][4]]).itemsize * 8) // self.bits
        if self.in_dims != self.fields[1][5] * self.group_size:
            raise PackError(
                f"embed_tokens: {self.in_dims} packed values per row != "
                f"{self.fields[1][5]} groups x {self.group_size}"
            )
        self.row_bytes = sum(f[3] for f in self.fields)
        self.table_bytes = self.row_bytes * self.out_dims
        self.nocache = bool(nocache)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="embed-io"
        )
        self._lock = threading.Lock()
        self.rows_requested = 0
        self.rows_read = 0
        self.bytes = 0
        self.read_seconds = 0.0
        self.calls = 0

    def _read_rows(self, rows: np.ndarray) -> list[np.ndarray]:
        """Per field, `[n, cols]` raw storage for unique rows; one `pread` per (row, field)."""
        n = rows.size
        bufs = [np.empty((n, cols), dtype=_NP[dt]) for (_f, _fd, _s, _pr, dt, cols) in self.fields]
        jobs = [(fi, i) for fi in range(len(self.fields)) for i in range(n)]

        def one(job) -> None:
            fi, i = job
            _f, fd, start, per_row, _dt, _c = self.fields[fi]
            mv = memoryview(bufs[fi][i]).cast("B")
            off = start + int(rows[i]) * per_row
            got = 0
            while got < per_row:
                k = os.preadv(fd, [mv[got:]], off + got)
                if k <= 0:
                    raise OSError(
                        f"short read of embed_tokens.{_f} row {int(rows[i])} at {off + got}"
                    )
                got += k

        if len(jobs) <= 1:
            for j in jobs:
                one(j)
        else:
            futures = [self._pool.submit(one, j) for j in jobs]
            first: BaseException | None = None
            for fu in futures:
                try:
                    fu.result()
                except BaseException as exc:  # noqa: BLE001
                    if first is None:
                        first = exc
            if first is not None:
                raise first
        return bufs

    def fetch_raw(self, ids) -> tuple[np.ndarray, list[np.ndarray]]:
        """(flat int64 ids, per-field `[N, cols]` storage in the caller's order)."""
        flat = np.asarray(ids).astype(np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.out_dims):
            bad = flat[(flat < 0) | (flat >= self.out_dims)][0]
            raise IndexError(
                f"token id {int(bad)} outside the embedding table 0..{self.out_dims - 1}"
            )
        t0 = time.monotonic()
        uniq, inv = np.unique(flat, return_inverse=True)
        raw = self._read_rows(uniq)
        dt = time.monotonic() - t0
        with self._lock:
            self.calls += 1
            self.rows_requested += int(flat.size)
            self.rows_read += int(uniq.size)
            self.bytes += int(uniq.size) * self.row_bytes
            self.read_seconds += dt
        inv = inv.reshape(-1)
        return flat, [b[inv] for b in raw]

    def lookup(self, ids) -> mx.array:
        """The resident `QEmbedding.lookup`, bit for bit: same row bytes, same shapes, same dequant."""
        shape = tuple(int(d) for d in ids.shape)
        _flat, parts = self.fetch_raw(ids)
        arrs = []
        for (_f, _fd, _s, _pr, dt, cols), p in zip(self.fields, parts):
            a = mx.array(p.reshape(*shape, cols))
            if dt in _MX_VIEW:
                a = a.view(_MX_VIEW[dt])
            arrs.append(a)
        return mx.dequantize(
            arrs[0],
            scales=arrs[1],
            biases=arrs[2] if self.has_biases else None,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "embed_streamed": True,
                "embed_rows_requested": self.rows_requested,
                "embed_rows_read": self.rows_read,
                "embed_bytes": self.bytes,
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
