"""Write the PLE pack pack_format.py defines, for the one n-gram table geometry it is built for.

A record is the row's weight (U32 [20]), scales (BF16 [5]) and biases (BF16 [5]) copied verbatim,
so a random row read touches about one page, not three. Global row g = shard * rows_per_shard +
local row. After the header, each segment's rows sit back to back at a dense 100 B stride; all
reads go through the header's segment list, so a row missing from a partial pack raises instead of
returning another row's bytes. `scan_checkpoint` refuses a table of any other geometry than the one
below; `parse_shards` and `parse_rows` read a builder's selection.
"""

from __future__ import annotations

import os
import time

import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.expert_writer import EXPECT_QUANT, PREFIX
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    PLE_FIELDS,
    PLE_MAGIC,
    PackError,
    PLEScan,
    header_block,
    ple_pack_bytes,
    ple_pack_header,
)
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_io import write_all

# The n-gram table's geometry, which `scan_checkpoint` insists on: one 100-byte row is the record.
EXPECT_SHARDS = 128
EXPECT_ROWS_PER_SHARD = 2_500_012
EXPECT_TOTAL_ROWS = 320_001_536
EXPECT_ROW_BYTES = 100
EXPECT_FIELD_BYTES = [80, 10, 10]
EXPECT_FIELD_DTYPE = ["U32", "BF16", "BF16"]
EXPECT_FIELD_COLS = [20, 5, 5]


def scan_checkpoint(model_path: str) -> PLEScan:
    """The checkpoint's `PLEScan` under `PREFIX`, refused unless its geometry is the one above."""
    scan = PLEScan(model_path, prefix=PREFIX)
    if scan.num_shards != EXPECT_SHARDS:
        raise PackError(f"{scan.num_shards} n-gram shards, expected {EXPECT_SHARDS}")
    for shard in range(scan.num_shards):
        for field in PLE_FIELDS:
            module = scan.tensor_name(shard, field).rsplit(".", 1)[0]
            q = scan.ckpt.quant_of(module)
            if q != EXPECT_QUANT:
                raise PackError(f"{module}: quantization {q} != {EXPECT_QUANT}")
    if scan.rows_per_shard != EXPECT_ROWS_PER_SHARD:
        raise PackError(f"{scan.rows_per_shard} rows/shard, expected {EXPECT_ROWS_PER_SHARD}")
    if scan.field_bytes != EXPECT_FIELD_BYTES:
        raise PackError(f"field bytes {scan.field_bytes} != {EXPECT_FIELD_BYTES}")
    if scan.field_dtype != EXPECT_FIELD_DTYPE:
        raise PackError(f"field dtypes {scan.field_dtype} != {EXPECT_FIELD_DTYPE}")
    if scan.field_cols != EXPECT_FIELD_COLS:
        raise PackError(f"field cols {scan.field_cols} != {EXPECT_FIELD_COLS}")
    # the packed values must fill the scale groups exactly, or the row is not self-contained
    vals_per_row = scan.field_cols[0] * 32 // EXPECT_QUANT["bits"]
    if vals_per_row != scan.field_cols[1] * EXPECT_QUANT["group_size"]:
        raise PackError(
            f"{vals_per_row} packed values != {scan.field_cols[1]} groups x "
            f"{EXPECT_QUANT['group_size']}"
        )
    if scan.row_bytes != EXPECT_ROW_BYTES:
        raise PackError(f"row {scan.row_bytes} B != {EXPECT_ROW_BYTES} B")
    if scan.total_rows != EXPECT_TOTAL_ROWS:
        raise PackError(f"{scan.total_rows} rows != {EXPECT_TOTAL_ROWS}")
    return scan


# ------------------------------------------------------------------ segment selection
def parse_shards(spec: str | None, num_shards: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(num_shards))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    for s in out:
        if not 0 <= s < num_shards:
            raise PackError(f"shard {s} out of range 0..{num_shards - 1}")
    if len(set(out)) != len(out):
        raise PackError(f"duplicate shards in {spec!r}")
    return sorted(out)


def parse_rows(
    spec: str | None, shards: list[int], rows_per_shard: int
) -> dict[int, tuple[int, int]]:
    """`--rows` -> shard-local [lo, hi) per selected shard.

    Comma-separated forms: `all`, `N` (= `0:N`), `lo:hi` (every selected shard), and `S=lo:hi` or
    `S=N` (shard S only, overriding the general form).
    """
    default = (0, rows_per_shard)
    per: dict[int, tuple[int, int]] = {}
    if spec is not None and spec != "all":
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            target = None
            if "=" in part:
                s, part = part.split("=", 1)
                target = int(s)
                if target not in shards:
                    raise PackError(f"--rows names shard {target}, which --shards does not select")
            default = _record_row_range(part, target, rows_per_shard, default, per)
    return {s: per.get(s, default) for s in shards}


def _record_row_range(
    part: str,
    target: int | None,
    rows_per_shard: int,
    default: tuple[int, int],
    per: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Parse one N or lo:hi part and record it as the general range or as shard target's range."""
    lo, hi = (
        (0, int(part)) if ":" not in part else (int(part.split(":")[0]), int(part.split(":")[1]))
    )
    if not 0 <= lo < hi <= rows_per_shard:
        raise PackError(f"row range {lo}:{hi} outside 0..{rows_per_shard}")
    if target is None:
        default = (lo, hi)
    else:
        per[target] = (lo, hi)
    return default


def build_pack(
    scan: PLEScan,
    path: str,
    segments: list[dict],
    rows_per_chunk: int = 262_144,
    log=print,
    chunk_bytes: int = 32 << 20,
) -> str:
    """Write the selected rows, chunked, and publish by atomic rename.

    A chunk is up to `rows_per_chunk` rows of one shard (never crossing a shard boundary, where the
    source tensors change), so each field is one contiguous read.
    """
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    total = ple_pack_bytes(scan, segments)
    n_rows = sum(s["rows"] for s in segments)
    log(
        f"[ple-pack] building {path} ({total / 1e9:.2f} GB, {n_rows} rows of {scan.row_bytes} B in "
        f"{len(segments)} segment(s)) from {scan.model_path}"
    )

    fds: dict[str, int] = {}
    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    done = 0
    try:
        os.write(out, header_block(PLE_MAGIC, ple_pack_header(scan, segments, complete=False)))
        for seg in segments:
            g = seg["row_start"]
            left = seg["rows"]
            while left:
                shard, local = scan.shard_of(g)
                n = min(left, rows_per_chunk, scan.rows_per_shard - local)
                row = np.empty((n, scan.row_bytes), dtype=np.uint8)
                _read_chunk_fields(scan, shard, local, n, row, fds, chunk_bytes)
                write_all(out, memoryview(row).cast("B"), chunk_bytes)
                done += n
                g += n
                left -= n
                if done % (rows_per_chunk * 16) < n or left == 0:
                    el = time.monotonic() - t0
                    log(
                        f"[ple-pack] {done}/{n_rows} rows ({done * scan.row_bytes / 1e9:.2f} GB) "
                        f"in {el:.0f}s ({done * scan.row_bytes / 1e9 / max(el, 1e-9):.2f} GB/s)"
                    )
        os.fsync(out)
        os.pwrite(out, header_block(PLE_MAGIC, ple_pack_header(scan, segments, complete=True)), 0)
        os.fsync(out)
    except BaseException:
        _discard_partial_pack(out, tmp, fds)
        raise
    _close_pack(out, fds)
    os.rename(tmp, path)
    dt = time.monotonic() - t0
    log(
        f"[ple-pack] built {total / 1e9:.2f} GB in {dt:.0f}s "
        f"({total / 1e9 / max(dt, 1e-9):.2f} GB/s) -> {path}"
    )
    return path


def _read_chunk_fields(
    scan: PLEScan,
    shard: int,
    local: int,
    n: int,
    row: np.ndarray,
    fds: dict[str, int],
    chunk_bytes: int,
) -> None:
    """Fill row with a chunk's fields, one sequential read each, opening shards on first use."""
    for f_i, field in enumerate(PLE_FIELDS):
        src, start, span = scan.span_of(shard, local, n, field)
        fd = fds.get(src)
        if fd is None:
            fd = fds[src] = os.open(src, os.O_RDONLY)
        buf = np.empty(span, dtype=np.uint8)
        mv = memoryview(buf).cast("B")
        got = 0
        while got < span:
            k = os.preadv(fd, [mv[got : got + min(chunk_bytes, span - got)]], start + got)
            if k <= 0:
                raise PackError(f"short read on {src} at {start + got}")
            got += k
        off = scan.field_offsets[f_i]
        width = scan.field_bytes[f_i]
        row[:, off : off + width] = buf.reshape(n, width)


def _discard_partial_pack(out: int, tmp: str, fds: dict[str, int]) -> None:
    """Close the output and shard descriptors and remove the temp file, ignoring an unlink error."""
    os.close(out)
    try:
        os.unlink(tmp)
    except OSError:
        pass  # the temp file may already be gone; cleanup must not hide the original error
    for fd in fds.values():
        os.close(fd)


def _close_pack(out: int, fds: dict[str, int]) -> None:
    """Close the output and the shard descriptors."""
    os.close(out)
    for fd in fds.values():
        os.close(fd)
