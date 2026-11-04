"""Write the expert pack checkpoint/pack_format.py defines, for the one checkpoint geometry it is built for.

A record is that expert's 9 `FIELDS` slices, copied byte for byte from the checkpoint in order, at
`HEADER_BYTES + r * record_stride`; the stride is a page multiple so every record is aligned for
direct I/O. A partial pack (`--layers`) indexes records by the layer's position in the header's
layer list, not by layer id. The header's completion marker is written last and the file is
published by atomic rename. `scan_checkpoint` refuses a checkpoint of any other geometry than the
one below; `build_pack` writes whatever scan it is given.
"""

from __future__ import annotations

import os
import time

import numpy as np

from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import (
    FIELDS,
    HEADER_BYTES,
    MAGIC,
    ExpertScan,
    PackError,
    header_block,
    pack_bytes,
    pack_header,
)
from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_io import write_all

PREFIX = "language_model."  # the VL conversion's; the PLE pack is built from the same checkpoint

# This checkpoint's geometry, which `scan_checkpoint` insists on. The record size is the pack's ABI:
# a checkpoint of another shape needs a new format version, not a silently different pack.
EXPECT_LAYERS = 48
EXPECT_EXPERTS = 512
EXPECT_RECORD_BYTES = 3_072_000
EXPECT_LAYER_BYTES = 1_572_864_000
EXPECT_FIELD_BYTES = [819_200, 102_400, 102_400] * 3
EXPECT_FIELD_DTYPE = ["U32", "BF16", "BF16"] * 3
EXPECT_QUANT = {"bits": 4, "group_size": 32, "mode": "affine"}


def scan_checkpoint(model_path: str) -> ExpertScan:
    """The checkpoint's `ExpertScan` under `PREFIX`; refused unless its geometry is the above."""
    scan = ExpertScan(model_path, prefix=PREFIX)
    if scan.num_layers != EXPECT_LAYERS:
        raise PackError(f"{scan.num_layers} layers, expected {EXPECT_LAYERS}")
    for layer in range(scan.num_layers):
        for field in FIELDS:
            module = scan.tensor_name(layer, field).rsplit(".", 1)[0]
            q = scan.ckpt.quant_of(module)
            if q != EXPECT_QUANT:
                raise PackError(f"{module}: quantization {q} != {EXPECT_QUANT}")
    if scan.num_experts != EXPECT_EXPERTS:
        raise PackError(f"{scan.num_experts} experts, expected {EXPECT_EXPERTS}")
    if scan.field_bytes != EXPECT_FIELD_BYTES:
        raise PackError(f"field bytes {scan.field_bytes} != {EXPECT_FIELD_BYTES}")
    if scan.field_dtype != EXPECT_FIELD_DTYPE:
        raise PackError(f"field dtypes {scan.field_dtype} != {EXPECT_FIELD_DTYPE}")
    if scan.record_bytes != EXPECT_RECORD_BYTES:
        raise PackError(f"record {scan.record_bytes} B != {EXPECT_RECORD_BYTES} B")
    if scan.record_bytes * scan.num_experts != EXPECT_LAYER_BYTES:
        raise PackError(
            f"layer bytes {scan.record_bytes * scan.num_experts} != {EXPECT_LAYER_BYTES}"
        )
    return scan


def build_pack(
    scan: ExpertScan,
    path: str,
    layers: list[int] | None = None,
    experts_per_chunk: int = 64,
    log=print,
    chunk_bytes: int = 32 << 20,
) -> str:
    """Write the pack in expert chunks (9 sequential reads each) and publish it by atomic rename."""
    layers = _check_layers(scan, layers)

    t0 = time.monotonic()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    total = pack_bytes(scan, layers)
    log(
        f"[expert-pack] building {path} ({total / 1e9:.2f} GB, {len(layers)} layers x "
        f"{scan.num_experts} experts of {scan.record_bytes} B, stride {scan.record_stride}) "
        f"from {scan.model_path}"
    )

    E = scan.num_experts
    n_chunk = min(experts_per_chunk, E)
    if E % n_chunk:
        raise PackError(f"{E} experts is not a multiple of experts_per_chunk={n_chunk}")
    fds: dict[str, int] = {}
    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    written = 0
    try:
        os.write(out, header_block(MAGIC, pack_header(scan, layers, complete=False)))
        rec = np.zeros((n_chunk, scan.record_stride), dtype=np.uint8)
        for l_i, layer in enumerate(layers):
            for e0 in range(0, E, n_chunk):
                _read_chunk_fields(scan, layer, e0, fds, n_chunk, chunk_bytes, rec)
                write_all(out, memoryview(rec).cast("B"), chunk_bytes)
                written += rec.nbytes
            el = time.monotonic() - t0
            done = HEADER_BYTES + written
            log(
                f"[expert-pack] layer {l_i + 1}/{len(layers)} (index {layer}): {done / 1e9:.2f} GB "
                f"in {el:.0f}s ({done / 1e9 / max(el, 1e-9):.2f} GB/s)"
            )
        os.fsync(out)
        # completion marker last, then fsync again: a crash before this leaves an unusable temp file
        os.pwrite(out, header_block(MAGIC, pack_header(scan, layers, complete=True)), 0)
        os.fsync(out)
    except BaseException:
        _discard_partial_pack(out, tmp, fds)
        raise
    _close_pack(out, fds)
    os.rename(tmp, path)
    dt = time.monotonic() - t0
    log(
        f"[expert-pack] built {total / 1e9:.2f} GB in {dt:.0f}s "
        f"({total / 1e9 / max(dt, 1e-9):.2f} GB/s) -> {path}"
    )
    return path


def _check_layers(scan: ExpertScan, layers: list[int] | None) -> list[int]:
    """The layers to pack (all when None), or PackError if one is out of range or repeated."""
    layers = list(range(scan.num_layers)) if layers is None else list(layers)
    for layer_id in layers:
        if not 0 <= layer_id < scan.num_layers:
            raise PackError(f"layer {layer_id} out of range 0..{scan.num_layers - 1}")
    if len(set(layers)) != len(layers):
        raise PackError(f"duplicate layers in {layers}")
    return layers


def _read_chunk_fields(
    scan: ExpertScan,
    layer: int,
    e0: int,
    fds: dict[str, int],
    n_chunk: int,
    chunk_bytes: int,
    rec: np.ndarray,
) -> None:
    """Fill rec with a chunk's fields, one sequential read each, opening shards on first use."""
    for f_i, field in enumerate(FIELDS):
        src, start, length = scan.slice_of(layer, e0, field)
        fd = fds.get(src)
        if fd is None:
            fd = fds[src] = os.open(src, os.O_RDONLY)
        span = length * n_chunk
        buf = np.empty(span, dtype=np.uint8)
        mv = memoryview(buf).cast("B")
        got = 0
        while got < span:
            n = os.preadv(fd, [mv[got : got + min(chunk_bytes, span - got)]], start + got)
            if n <= 0:
                raise PackError(f"short read on {src} at {start + got}")
            got += n
        off = scan.field_offsets[f_i]
        rec[:, off : off + length] = buf.reshape(n_chunk, length)


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
