"""The on-disk formats of Flash-Next's two packs, defined once for their writers and readers.

The expert pack (FNXEPK01, the unprefixed names) holds one page-aligned record per (layer, expert):
the expert's `FIELDS` slices, copied byte for byte from the checkpoint. The PLE pack (FNXPLE01, the
`PLE_` and `ple_` names) holds one record per row of the n-gram table, segment by segment. Each file
starts with a `HEADER_BYTES` block (magic, u32 JSON length, the JSON, zero padding) recording the
checkpoint identity, the geometry and a completion marker written last.

`ExpertScan` and `PLEScan` take the geometry from the checkpoint's shard headers, whatever it is,
and `validate` and `validate_ple` check a pack's header against itself and the scan, so the engine
opens any pack built from its checkpoint. Only the writers (`expert_writer`, `ple_writer`) insist
on the real model's geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time

from jitengine_qwen3_8_flash_next_m2 import envs

# ------------------------------------------------------------------ both packs
PAGE = 16 << 10
HEADER_BYTES = 1 << 20  # a multiple of PAGE, so record offsets inherit the file's alignment
DTYPE_BYTES = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4}

# The prefix is derived, not hard-coded: a VL conversion nests the tower under `language_model.`.
_LAYER0_PROBE = "model.layers.0.mlp.switch_mlp.gate_proj.weight"


class PackError(RuntimeError):
    """A pack or checkpoint that does not match the format."""


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """(tensor metadata dict, absolute byte offset of the blob) for one shard."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    if len(raw) != n:
        raise PackError(f"{path}: truncated safetensors header")
    return json.loads(raw), 8 + n


def align_up(n: int, to: int = PAGE) -> int:
    """`n` rounded up to a multiple of `to`."""
    return -(-n // to) * to


def detect_prefix(weight_map: dict) -> str:
    """The common prefix in front of `model.layers.N....`, derived from the checkpoint index."""
    cands = {k[: -len(_LAYER0_PROBE)] for k in weight_map if k.endswith(_LAYER0_PROBE)}
    if len(cands) != 1:
        raise PackError(f"cannot derive tensor prefix: {len(cands)} candidates for {_LAYER0_PROBE}")
    return cands.pop()


class Checkpoint:
    """The shard headers of one checkpoint, read once (a few MB of I/O), plus its identity."""

    def __init__(self, model_path: str):
        self.model_path = os.path.expanduser(model_path)
        idx_path = os.path.join(self.model_path, "model.safetensors.index.json")
        with open(idx_path, "rb") as f:
            idx_raw = f.read()
        self.index_sha256 = hashlib.sha256(idx_raw).hexdigest()
        self.weight_map: dict[str, str] = json.loads(idx_raw)["weight_map"]

        self.shard_names = sorted(set(self.weight_map.values()))
        self.shards = []
        self._headers: dict[str, tuple[dict, int]] = {}
        for name in self.shard_names:
            p = os.path.join(self.model_path, name)
            st = os.stat(p)
            self.shards.append({"name": name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        self.config: dict | None = None  # config.json, read by `quant_of` (only the writers ask)

    def header(self, shard: str) -> tuple[dict, int]:
        """(tensor metadata, blob offset) of one shard, read on first use."""
        h = self._headers.get(shard)
        if h is None:
            h = self._headers[shard] = read_safetensors_header(os.path.join(self.model_path, shard))
        return h

    def slice_of(self, name: str) -> tuple[str, int, int, str, list[int]]:
        """(shard path, absolute offset, length, dtype, shape) of a whole tensor in the checkpoint."""
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackError(f"missing tensor {name}")
        hdr, data_off = self.header(shard)
        m = hdr[name]
        lo, hi = m["data_offsets"]
        return (
            os.path.join(self.model_path, shard),
            data_off + lo,
            hi - lo,
            m["dtype"],
            [int(d) for d in m["shape"]],
        )

    def quant_of(self, name: str) -> dict:
        """The effective quantization for a tensor: config.json's per-tensor override, else global."""
        if self.config is None:
            with open(os.path.join(self.model_path, "config.json")) as f:
                self.config = json.load(f)
        q = self.config["quantization"]
        entry = q.get(name)
        if isinstance(entry, dict):
            return {"bits": entry["bits"], "group_size": entry["group_size"], "mode": entry["mode"]}
        return {"bits": q["bits"], "group_size": q["group_size"], "mode": q["mode"]}

    def identity(self) -> dict:
        """What a pack header records of the checkpoint; a pack recording anything else is stale."""
        return {"index_sha256": self.index_sha256, "shards": self.shards}


def header_block(magic: bytes, meta: dict) -> bytes:
    """The `HEADER_BYTES` block a pack starts with: magic, u32 JSON length, the JSON, zeros."""
    raw = json.dumps(meta).encode()
    block = magic + struct.pack("<I", len(raw)) + raw
    if len(block) > HEADER_BYTES:
        raise PackError("pack header does not fit in the header block")
    return block + b"\x00" * (HEADER_BYTES - len(block))


def read_header_block(path: str, magic: bytes) -> dict:
    """The JSON of the header block at the start of `path`, which must begin with `magic`."""
    with open(path, "rb") as f:
        block = f.read(HEADER_BYTES)
    if len(block) < 12 or block[:8] != magic:
        raise PackError(f"{path}: bad magic (expected {magic!r})")
    n = struct.unpack("<I", block[8:12])[0]
    if 12 + n > len(block):
        raise PackError(f"{path}: header json ({n} B) exceeds the header block")
    return json.loads(block[12 : 12 + n])


# ------------------------------------------------------------------ FNXEPK01: the expert pack
MAGIC = b"FNXEPK01"
VERSION = 1
SUBMODULE = "mlp.switch_mlp"
FIELDS = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
]


class ExpertScan:
    """Where every routed-expert slice lives, from the shard headers only (a few MB of I/O).

    The geometry is the checkpoint's own (any layer and expert count, any field sizes), as long as
    every layer matches layer 0. `prefix` is derived from the index unless given.
    """

    def __init__(self, model_path: str, ckpt: Checkpoint | None = None, prefix: str | None = None):
        self.ckpt = ckpt or Checkpoint(model_path)
        self.model_path = self.ckpt.model_path
        weight_map = self.ckpt.weight_map
        self.prefix = detect_prefix(weight_map) if prefix is None else prefix

        self._count_layers(weight_map)

        self.num_experts = 0
        self.field_bytes: list[int] = []
        self.field_dtype: list[str] = []
        self.field_shape: list[list[int]] = []
        self.slices: dict[tuple[int, str], tuple[str, int, int]] = {}
        for layer in range(self.num_layers):
            for f_i, field in enumerate(FIELDS):
                name, dtype, shape, per_expert = self._read_field_slice(layer, field)
                self._check_layer_field(layer, f_i, name, dtype, shape, per_expert)

        self.field_offsets: list[int] = []
        acc = 0
        for b in self.field_bytes:
            self.field_offsets.append(acc)
            acc += b
        self.record_bytes = acc
        self.record_stride = align_up(self.record_bytes, PAGE)  # every record page-aligned

    def _count_layers(self, weight_map: dict[str, str]) -> None:
        """Set num_layers to the index's language-model layer count; PackError if there are none."""
        self.num_layers = 0
        while any(
            k.startswith(f"{self.prefix}model.layers.{self.num_layers}.") for k in weight_map
        ):
            self.num_layers += 1
        if self.num_layers == 0:
            raise PackError(f"no language-model layers under {self.prefix!r}")

    def _read_field_slice(self, layer: int, field: str) -> tuple:
        """Locate a layer's field, check its expert count, and record its per-expert slice."""
        name = self.tensor_name(layer, field)
        path, start, length, dtype, shape = self.ckpt.slice_of(name)
        if self.num_experts == 0:
            self.num_experts = int(shape[0])
        if int(shape[0]) != self.num_experts:
            raise PackError(f"{name}: {shape[0]} experts, expected {self.num_experts}")
        per_expert, rem = divmod(length, self.num_experts)
        if rem:
            raise PackError(f"{name}: {length} B does not divide into {self.num_experts}")
        self.slices[(layer, field)] = (path, start, per_expert)
        return name, dtype, shape, per_expert

    def _check_layer_field(
        self, layer: int, f_i: int, name: str, dtype: str, shape: list[int], per_expert: int
    ) -> None:
        """Take layer 0's field geometry as the pack's; PackError if a later layer's field differs."""
        if layer == 0:
            self.field_bytes.append(per_expert)
            self.field_dtype.append(dtype)
            self.field_shape.append([int(d) for d in shape[1:]])
        elif per_expert != self.field_bytes[f_i]:
            raise PackError(f"{name}: slice {per_expert} != layer-0 {self.field_bytes[f_i]}")
        elif dtype != self.field_dtype[f_i]:
            raise PackError(f"{name}: dtype {dtype} != layer-0 {self.field_dtype[f_i]}")

    def tensor_name(self, layer: int, field: str) -> str:
        """The checkpoint tensor holding one field of every expert of a layer."""
        return f"{self.prefix}model.layers.{layer}.{SUBMODULE}.{field}"

    @property
    def num_records(self) -> int:
        return self.num_layers * self.num_experts

    def fields(self) -> list[dict]:
        """The header's field table: each record field's name, dtype, shape, size and offset."""
        return [
            {"name": n, "dtype": d, "shape": s, "bytes": b, "offset": o}
            for n, d, s, b, o in zip(
                FIELDS, self.field_dtype, self.field_shape, self.field_bytes, self.field_offsets
            )
        ]

    def quantization(self) -> dict:
        """The experts' quantization in config.json (the writer checks every expert has it)."""
        return self.ckpt.quant_of(self.tensor_name(0, FIELDS[0]).rsplit(".", 1)[0])

    def geometry(self) -> dict:
        """The geometry the writer records in a pack header."""
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "record_bytes": self.record_bytes,
            "record_stride": self.record_stride,
            "tensor_prefix": self.prefix,
            "submodule": SUBMODULE,
            "quantization": self.quantization(),
            "fields": self.fields(),
        }

    def slice_of(self, layer: int, expert: int, field: str) -> tuple[str, int, int]:
        """(shard path, absolute byte offset, length) of one expert's field in the checkpoint."""
        path, start, per_expert = self.slices[(layer, field)]
        return path, start + expert * per_expert, per_expert


def _pack_dir(file_var: envs.EnvPath, what: str) -> str:
    """The pack directory, or a PackError naming the two variables that say where a pack is."""
    root = envs.PACK_DIR.get()
    if not root:
        raise PackError(
            f"set {envs.PACK_DIR.name} to the directory holding the packs (built by "
            f"`jitengine-qwen3.8-flash-next-m2 pack`), or {file_var.name} to the {what}"
        )
    return root


def default_pack_path(scan: ExpertScan, layers: list[int] | None = None) -> str:
    """`JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK`, else `<checkpoint>-<index sha12>.experts` in the pack dir.

    A builder's layer subset gets `-L<layers>` before the extension, so it never takes the whole
    pack's name; the engine always looks up the whole pack.
    """
    path = envs.EXPERT_PACK.get()
    if path:
        return path
    root = _pack_dir(envs.EXPERT_PACK, "expert pack")
    tag = f"{os.path.basename(scan.model_path.rstrip('/'))}-{scan.ckpt.index_sha256[:12]}"
    if layers is not None and len(layers) != scan.num_layers:
        tag += f"-L{'_'.join(str(layer) for layer in layers)}"
    return os.path.join(root, f"{tag}.experts")


def pack_bytes(scan: ExpertScan, layers: list[int]) -> int:
    """The size of a pack holding `layers`: the header block, then one stride per record."""
    return HEADER_BYTES + len(layers) * scan.num_experts * scan.record_stride


def record_index(meta: dict, layer: int, expert: int) -> int:
    """Record index of (layer, expert) in a pack whose header says which layers it holds."""
    layers = meta["layers"]
    if layer not in layers:
        raise PackError(f"layer {layer} is not in this pack (layers={layers})")
    if not 0 <= expert < meta["num_experts"]:
        raise PackError(f"expert {expert} out of range")
    return layers.index(layer) * meta["num_experts"] + expert


def record_offset(meta: dict, layer: int, expert: int) -> int:
    """Byte offset of (layer, expert)'s record in the pack whose header is `meta`."""
    return int(meta["data_offset"]) + record_index(meta, layer, expert) * int(meta["record_stride"])


def pack_header(scan: ExpertScan, layers: list[int], complete: bool) -> dict:
    """The expert pack's header JSON; the builder writes it incomplete first and complete last."""
    return {
        "version": VERSION,
        "kind": "flashnext-experts",
        "checkpoint": scan.ckpt.identity(),
        "model_path": scan.model_path,
        "data_offset": HEADER_BYTES,
        "layers": list(layers),
        "records": len(layers) * scan.num_experts,
        "complete": bool(complete),
        "built_at": time.time(),
        **scan.geometry(),
    }


def validate(path: str, scan: ExpertScan, layers: list[int] | None = None) -> tuple[bool, str]:  # noqa: C901
    """(ok, why). Never raises on a corrupt file: the caller decides rebuild vs refuse.

    The header must agree with itself and with the scanned checkpoint, whatever their geometry:
    that is all the engine asks. A builder passes the layers it writes; the pack must then hold
    exactly those, laid out as `pack_header` lays them out (the `HEADER_BYTES` header, the scan's
    stride), or the builder writes it again.
    """
    try:
        meta = read_header_block(path, MAGIC)
    except (OSError, ValueError, PackError) as exc:
        return False, str(exc)
    if int(meta.get("version", -1)) != VERSION:
        return False, f"version {meta.get('version')} != {VERSION}"
    if not meta.get("complete"):
        return False, "completion marker absent (partial repack)"
    if meta.get("checkpoint") != scan.ckpt.identity():
        return False, "checkpoint identity differs (shard size/mtime or index hash changed)"
    expected = {
        "num_layers": scan.num_layers,
        "num_experts": scan.num_experts,
        "record_bytes": scan.record_bytes,
        "fields": scan.fields(),
    }
    for k, want in expected.items():
        if meta.get(k) != want:
            return False, f"{k} {meta.get(k)!r} != {want!r}"
    for k in ("data_offset", "record_stride", "records"):
        if not isinstance(meta.get(k), int) or meta[k] <= 0:
            return False, f"bad {k} {meta.get(k)!r}"
    if meta["record_stride"] < meta["record_bytes"]:
        return False, f"record_stride {meta['record_stride']} < record_bytes {meta['record_bytes']}"
    have = meta.get("layers")
    if (
        not isinstance(have, list)
        or not have
        or len(set(have)) != len(have)
        or any(not isinstance(layer, int) or not 0 <= layer < scan.num_layers for layer in have)
    ):
        return False, f"bad layer list {have!r}"
    if meta["records"] != len(have) * scan.num_experts:
        return False, f"records {meta['records']} != {len(have)} layers x {scan.num_experts}"
    want = meta["data_offset"] + meta["records"] * meta["record_stride"]
    size = os.path.getsize(path)
    if size != want:
        return False, f"size {size} != {want} (truncated)"
    if layers is not None:
        if have != list(layers):
            return False, f"layers {have} != requested {list(layers)}"
        layout = (meta["data_offset"], meta["record_stride"])
        written = (HEADER_BYTES, scan.record_stride)  # what `pack_header` lays out
        if layout != written:
            return False, f"data_offset, record_stride {layout} != {written}"
    return True, "ok"


# ------------------------------------------------------------------ FNXPLE01: the PLE pack
PLE_MAGIC = b"FNXPLE01"
PLE_VERSION = 1
PLE_SUBMODULE = "ple.ple_embedding.ngram_embedding"
PLE_FIELDS = ["weight", "scales", "biases"]


class PLEScan:
    """Where every n-gram row's three field slices live, from the shard headers only.

    The geometry is the table's own (any shard and row count, any row size), as long as every shard
    matches shard 0. `prefix` is derived from the index unless given.
    """

    def __init__(self, model_path: str, ckpt: Checkpoint | None = None, prefix: str | None = None):
        self.ckpt = ckpt or Checkpoint(model_path)
        self.model_path = self.ckpt.model_path
        weight_map = self.ckpt.weight_map
        self.prefix = detect_prefix(weight_map) if prefix is None else prefix

        self._find_layer(weight_map)

        self._count_shards(weight_map)

        self.rows_per_shard = 0
        self.field_bytes: list[int] = []
        self.field_dtype: list[str] = []
        self.field_cols: list[int] = []
        self.slices: dict[tuple[int, str], tuple[str, int, int]] = {}
        for shard in range(self.num_shards):
            for f_i, field in enumerate(PLE_FIELDS):
                name, dtype, shape, per_row = self._read_field_slice(shard, field)
                self._check_shard_field(shard, f_i, name, dtype, shape, per_row)

        self.field_offsets: list[int] = []
        acc = 0
        for b in self.field_bytes:
            self.field_offsets.append(acc)
            acc += b
        self.row_bytes = acc
        self.total_rows = self.num_shards * self.rows_per_shard

    def _find_layer(self, weight_map: dict[str, str]) -> None:
        """Set layer to the one layer holding PLE shards; PackError unless there is exactly one."""
        probe = f".{PLE_SUBMODULE}.shards.0.weight"
        layers = sorted(
            {int(k.split(".layers.")[1].split(".")[0]) for k in weight_map if k.endswith(probe)}
        )
        if len(layers) != 1:
            raise PackError(f"expected exactly one PLE layer, found {layers}")
        self.layer = layers[0]

    def _count_shards(self, weight_map: dict[str, str]) -> None:
        """Set num_shards to the consecutive n-gram shards in the index; PackError if none."""
        self.num_shards = 0
        while self.tensor_name(self.num_shards, "weight") in weight_map:
            self.num_shards += 1
        if self.num_shards == 0:
            raise PackError("no PLE shards found")

    def _read_field_slice(self, shard: int, field: str) -> tuple:
        """Locate a shard's field, check it is 2-D with the common row count, record its row slice."""
        name = self.tensor_name(shard, field)
        path, start, length, dtype, shape = self.ckpt.slice_of(name)
        if len(shape) != 2:
            raise PackError(f"{name}: shape {shape} is not 2-D")
        if self.rows_per_shard == 0:
            self.rows_per_shard = int(shape[0])
        if int(shape[0]) != self.rows_per_shard:
            raise PackError(f"{name}: {shape[0]} rows, expected {self.rows_per_shard}")
        per_row, rem = divmod(length, self.rows_per_shard)
        if rem:
            raise PackError(f"{name}: {length} B does not divide into {self.rows_per_shard} rows")
        self.slices[(shard, field)] = (path, start, per_row)
        return name, dtype, shape, per_row

    def _check_shard_field(
        self, shard: int, f_i: int, name: str, dtype: str, shape: list[int], per_row: int
    ) -> None:
        """Take shard 0's field geometry as the pack's; PackError if a later shard differs."""
        if shard == 0:
            self.field_bytes.append(per_row)
            self.field_dtype.append(dtype)
            self.field_cols.append(int(shape[1]))
        elif per_row != self.field_bytes[f_i] or dtype != self.field_dtype[f_i]:
            raise PackError(
                f"{name}: {per_row} B/{dtype} != shard-0 "
                f"{self.field_bytes[f_i]} B/{self.field_dtype[f_i]}"
            )

    def tensor_name(self, shard: int, field: str) -> str:
        """The checkpoint tensor holding one field of one n-gram shard's rows."""
        return f"{self.prefix}model.layers.{self.layer}.{PLE_SUBMODULE}.shards.{shard}.{field}"

    def fields(self) -> list[dict]:
        """The header's field table: each row field's name, dtype, columns, size and offset."""
        return [
            {"name": n, "dtype": d, "cols": c, "bytes": b, "offset": o}
            for n, d, c, b, o in zip(
                PLE_FIELDS, self.field_dtype, self.field_cols, self.field_bytes, self.field_offsets
            )
        ]

    def quantization(self) -> dict:
        """The table's quantization in config.json (the writer checks every shard has it)."""
        return self.ckpt.quant_of(self.tensor_name(0, PLE_FIELDS[0]).rsplit(".", 1)[0])

    def geometry(self) -> dict:
        """The geometry the writer records in a PLE pack header (rows are stored back to back)."""
        return {
            "ple_layer": self.layer,
            "num_shards": self.num_shards,
            "rows_per_shard": self.rows_per_shard,
            "total_rows": self.total_rows,
            "row_bytes": self.row_bytes,
            "row_stride": self.row_bytes,
            "tensor_prefix": self.prefix,
            "submodule": PLE_SUBMODULE,
            "quantization": self.quantization(),
            "fields": self.fields(),
        }

    def shard_of(self, global_row: int) -> tuple[int, int]:
        """(shard, shard-local row) of a global row."""
        if not 0 <= global_row < self.total_rows:
            raise PackError(f"row {global_row} out of range 0..{self.total_rows - 1}")
        return divmod(global_row, self.rows_per_shard)

    def slice_of(self, global_row: int, field: str) -> tuple[str, int, int]:
        """(shard path, absolute byte offset, length) of one row's field in the checkpoint."""
        shard, local = self.shard_of(global_row)
        path, start, per_row = self.slices[(shard, field)]
        return path, start + local * per_row, per_row

    def span_of(self, shard: int, local_row: int, rows: int, field: str) -> tuple[str, int, int]:
        """The contiguous byte span covering `rows` consecutive rows of one shard's field."""
        path, start, per_row = self.slices[(shard, field)]
        if local_row < 0 or rows < 0 or local_row + rows > self.rows_per_shard:
            raise PackError(f"rows {local_row}..{local_row + rows} outside shard {shard}")
        return path, start + local_row * per_row, rows * per_row


def segments_for(scan: PLEScan, shards: list[int], rows: dict[int, tuple[int, int]]) -> list[dict]:
    """The header's segment list: global row ranges in ascending order, contiguous ones merged."""
    segs: list[dict] = []
    for s in shards:
        lo, hi = rows[s]
        g0 = s * scan.rows_per_shard + lo
        if segs and segs[-1]["row_start"] + segs[-1]["rows"] == g0:
            segs[-1]["rows"] += hi - lo
        else:
            segs.append({"row_start": g0, "rows": hi - lo, "file_offset": 0})
    off = HEADER_BYTES
    for seg in segs:
        seg["file_offset"] = off
        off += seg["rows"] * scan.row_bytes
    return segs


def ple_pack_bytes(scan: PLEScan, segments: list[dict]) -> int:
    """The size of a PLE pack holding `segments`: the header block, then the rows back to back."""
    return HEADER_BYTES + sum(s["rows"] for s in segments) * scan.row_bytes


def row_offset(meta: dict, global_row: int) -> int:
    """Byte offset of a global row in the pack, via the header's segment list."""
    for seg in meta["segments"]:
        if seg["row_start"] <= global_row < seg["row_start"] + seg["rows"]:
            return seg["file_offset"] + (global_row - seg["row_start"]) * meta["row_stride"]
    raise PackError(f"row {global_row} is not in this pack ({len(meta['segments'])} segments)")


def rows_present(meta: dict) -> int:
    """How many rows the PLE pack whose header is `meta` holds."""
    return sum(s["rows"] for s in meta["segments"])


def default_ple_pack_path(scan: PLEScan, segments: list[dict] | None = None) -> str:
    """`JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK`, else `<checkpoint>-<index sha12>.ple` in the pack dir.

    A builder's row selection gets `-R<rows>` before the extension; the engine always looks up the
    whole table.
    """
    path = envs.PLE_PACK.get()
    if path:
        return path
    root = _pack_dir(envs.PLE_PACK, "PLE pack")
    tag = f"{os.path.basename(scan.model_path.rstrip('/'))}-{scan.ckpt.index_sha256[:12]}"
    if segments is not None and sum(s["rows"] for s in segments) != scan.total_rows:
        tag += f"-R{sum(s['rows'] for s in segments)}"
    return os.path.join(root, f"{tag}.ple")


def ple_pack_header(scan: PLEScan, segments: list[dict], complete: bool) -> dict:
    """The PLE pack's header JSON; the builder writes it incomplete first and complete last."""
    return {
        "version": PLE_VERSION,
        "kind": "flashnext-ple-rows",
        "checkpoint": scan.ckpt.identity(),
        "model_path": scan.model_path,
        "data_offset": HEADER_BYTES,
        "segments": [dict(s) for s in segments],
        "rows_present": sum(s["rows"] for s in segments),
        "complete": bool(complete),
        "built_at": time.time(),
        **scan.geometry(),
    }


def validate_ple(path: str, scan: PLEScan, segments: list[dict] | None = None) -> tuple[bool, str]:  # noqa: C901
    """(ok, why). Never raises on a corrupt file: the caller decides rebuild vs refuse.

    As `validate`: the header must agree with itself and with the scanned table, which is all the
    engine asks; a builder passes the segments it writes, and the pack must then hold exactly
    those, laid out as `ple_pack_header` lays them out (the `HEADER_BYTES` header, rows back to
    back), or the builder writes it again.
    """
    try:
        meta = read_header_block(path, PLE_MAGIC)
    except (OSError, ValueError, PackError) as exc:
        return False, str(exc)
    if int(meta.get("version", -1)) != PLE_VERSION:
        return False, f"version {meta.get('version')} != {PLE_VERSION}"
    if not meta.get("complete"):
        return False, "completion marker absent (partial repack)"
    if meta.get("checkpoint") != scan.ckpt.identity():
        return False, "checkpoint identity differs (shard size/mtime or index hash changed)"
    fields = scan.fields()
    expected = {
        "ple_layer": scan.layer,
        "num_shards": scan.num_shards,
        "rows_per_shard": scan.rows_per_shard,
        "total_rows": scan.total_rows,
        "row_bytes": scan.row_bytes,
        "fields": fields,
    }
    for k, want in expected.items():
        if meta.get(k) != want:
            return False, f"{k} {meta.get(k)!r} != {want!r}"
    if not isinstance(meta.get("row_stride"), int) or meta["row_stride"] < scan.row_bytes:
        return False, f"bad row_stride {meta.get('row_stride')!r}"
    q = meta.get("quantization") or {}
    if not isinstance(q.get("bits"), int) or not isinstance(q.get("group_size"), int):
        return False, f"bad quantization {q!r}"
    # the packed values must fill the scale groups exactly, or a row cannot be dequantized alone
    values = fields[0]["cols"] * (DTYPE_BYTES[fields[0]["dtype"]] * 8) // q["bits"]
    if values != fields[1]["cols"] * q["group_size"]:
        return False, (
            f"{values} packed values per row != {fields[1]['cols']} groups x {q['group_size']}"
        )
    segs = meta.get("segments")
    if not isinstance(segs, list) or not segs:
        return False, f"bad segment list {segs!r}"
    off = int(meta.get("data_offset", 0))
    if off <= 0:
        return False, "bad data_offset"
    last_end = -1
    for seg in segs:
        if seg["row_start"] <= last_end:
            return False, f"segments out of order or overlapping at {seg}"
        if seg["rows"] <= 0 or seg["row_start"] + seg["rows"] > scan.total_rows:
            return False, f"segment {seg} outside the table"
        if seg["file_offset"] != off:
            return False, f"segment {seg} offset != {off}"
        off += seg["rows"] * meta["row_stride"]
        last_end = seg["row_start"] + seg["rows"] - 1
    got = os.path.getsize(path)
    if got != off:
        return False, f"size {got} != {off} (truncated)"
    if segments is not None:
        if [(s["row_start"], s["rows"]) for s in segs] != [
            (s["row_start"], s["rows"]) for s in segments
        ]:
            return False, "segments differ from the requested selection"
        layout = (int(meta["data_offset"]), meta["row_stride"])
        written = (HEADER_BYTES, scan.row_bytes)  # what `ple_pack_header` lays out
        if layout != written:
            return False, f"data_offset, row_stride {layout} != {written}"
    return True, "ok"
