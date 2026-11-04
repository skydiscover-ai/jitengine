"""SKYP16X1: the expert pack in the pool's compact pack16 slot layout, one page-aligned record per
(layer, expert), so a read lands the final slot bytes and nothing is encoded at publish.

A record is what `kernels/pack16.encode_records` writes into a compact slot for the FNXEPK01
record of the same (layer, expert): the three weight fields verbatim, then one u16 word per
(scale, bias) group, in `Geometry.compact_fields()` order (gate weight, gate words, up weight, up
words, down weight, down words). The file starts with pack_format's `HEADER_BYTES` block (magic, u32
JSON length, the JSON, zeros) recording the checkpoint identity, the SOURCE pack's identity (the
FNXEPK01 file the records were encoded from), both field layouts, the codec's constants and the
escape table its words index, and a completion marker written last.

The engine looks the file up beside the expert pack's directory (`pack16_path_for`) and uses it only
if `validate_p16` accepts it against the scanned checkpoint, the source pack it opened and its own
resident escape table; anything else (absent, partial, stale, another table) falls back to the
source pack's 3,072,000 B records and the publish encode. The writer is `p16_writer`.
"""

from __future__ import annotations

import hashlib
import os

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    HEADER_BYTES,
    MAGIC,
    PAGE,
    PackError,
    align_up,
    header_block,
    read_header_block,
)
from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as _p16

MAGIC_P16 = b"SKYP16X1"
VERSION = 1
KIND = "flashnext-experts-pack16"
SUFFIX = ".p16x"
SUBDIR = "pack16"


def tag_of(scan) -> str:
    """`<checkpoint>-<index sha12>`, the expert pack's own default name stem."""
    return f"{os.path.basename(scan.model_path.rstrip('/'))}-{scan.ckpt.index_sha256[:12]}"


def pack16_path_for(source: str, scan) -> str:
    """Where the engine looks for the pack16 file of the expert pack at `source`:
    `<dirname(source's directory)>/pack16/<tag>.p16x`, i.e. `dirname(PACK_DIR)/pack16/...` for the
    default pack location."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(source)))
    return os.path.join(root, SUBDIR, tag_of(scan) + SUFFIX)


def source_identity(source: str) -> dict:
    """What a pack16 header records of the FNXEPK01 file it was encoded from."""
    st = os.stat(source)
    with open(source, "rb") as f:
        block = f.read(HEADER_BYTES)
    if block[:8] != MAGIC:
        raise PackError(f"{source}: not an FNXEPK01 expert pack")
    return {
        "path": os.path.abspath(source),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "header_sha256": hashlib.sha256(block).hexdigest(),
    }


def record_stride(geom: _p16.Geometry) -> int:
    return align_up(int(geom.slot_bytes), PAGE)


def codec_constants(geom: _p16.Geometry) -> dict:
    """The encoding the words assume: a reader with other constants must not use the file."""
    return {
        "name": "pack16",
        "exp_base": _p16.EXP_BASE,
        "escape_code": _p16.ESC,
        "group_size": geom.group_size,
        "bits": geom.bits,
    }


def p16_header(
    scan,
    source: dict,
    geom: _p16.Geometry,
    table: list,
    build_id: str,
    complete: bool,
    built_at: float,
    writer: dict | None = None,
) -> dict:
    """The header JSON; the writer writes it incomplete first and complete last."""
    stride = record_stride(geom)
    layers = list(range(scan.num_layers))
    return {
        "version": VERSION,
        "kind": KIND,
        "checkpoint": scan.ckpt.identity(),
        "source": dict(source),
        "data_offset": HEADER_BYTES,
        "num_layers": scan.num_layers,
        "num_experts": scan.num_experts,
        "layers": layers,
        "records": len(layers) * scan.num_experts,
        "record_bytes": int(geom.slot_bytes),
        "record_stride": stride,
        "fields": geom.compact_fields(),
        "source_record_bytes": int(scan.record_bytes),
        "source_fields": scan.fields(),
        "quantization": scan.quantization(),
        "codec": codec_constants(geom),
        "escapes": [int(v) for v in table],
        "build_id": build_id,
        "writer": dict(writer or {}),
        "complete": bool(complete),
        "built_at": built_at,
    }


def p16_header_block(meta: dict) -> bytes:
    return header_block(MAGIC_P16, meta)


def p16_bytes(meta: dict) -> int:
    """The file size a header describes: the header block, then one stride per record."""
    return int(meta["data_offset"]) + int(meta["records"]) * int(meta["record_stride"])


def validate_p16(
    path: str, scan, source: str, geom: _p16.Geometry, table: list
) -> tuple[bool, str]:
    """(ok, why) for the pack16 file at `path` against the scanned checkpoint, the source pack the
    engine opened, the pool's geometry and its resident escape table. Never raises."""
    try:
        meta = read_header_block(path, MAGIC_P16)
    except (OSError, ValueError, PackError) as exc:
        return False, str(exc)
    if meta.get("version") != VERSION or meta.get("kind") != KIND:
        return (
            False,
            f"version/kind {meta.get('version')!r}/{meta.get('kind')!r} != {VERSION}/{KIND}",
        )
    if not meta.get("complete"):
        return False, "completion marker absent (partial build)"
    if meta.get("checkpoint") != scan.ckpt.identity():
        return False, "checkpoint identity differs (shard size/mtime or index hash changed)"
    try:
        have_src = source_identity(source)
    except (OSError, PackError) as exc:
        return False, f"source pack unreadable: {exc}"
    if meta.get("source") != have_src:
        return False, f"source pack identity {meta.get('source')!r} != {have_src!r}"
    expected = {
        "num_layers": scan.num_layers,
        "num_experts": scan.num_experts,
        "layers": list(range(scan.num_layers)),
        "records": scan.num_layers * scan.num_experts,
        "data_offset": HEADER_BYTES,
        "record_bytes": int(geom.slot_bytes),
        "record_stride": record_stride(geom),
        "fields": geom.compact_fields(),
        "source_record_bytes": int(scan.record_bytes),
        "source_fields": scan.fields(),
        "codec": codec_constants(geom),
        "escapes": [int(v) for v in table],
    }
    for k, want in expected.items():
        if meta.get(k) != want:
            return False, f"{k} {meta.get(k)!r} != {want!r}"
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, str(exc)
    if size != p16_bytes(meta):
        return False, f"size {size} != {p16_bytes(meta)} (truncated)"
    return True, "ok"
