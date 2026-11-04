"""Expert pack: each routed expert as one page-aligned record, so one `preadv` loads it.

Layout: a 1 MiB header block (8-byte magic, 4-byte JSON length, JSON, zero pad), then record
r = layer * num_experts + expert at 1 MiB + r * record_stride, holding the 9 `FIELDS` slices
byte for byte as they appear in the shards.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time

from jitengine_qwen3_6_35b_a3b_m2 import envs

MAGIC = b"MOEXPK01"
# Packs of any other version are rejected, never reinterpreted.
VERSION = 2
HEADER_BYTES = 1 << 20  # a multiple of PAGE, so every record offset is page-aligned
PAGE = 16 << 10
PREFIX = "language_model."

# One expert's tensors in record order, relative to `model.layers.{l}.mlp.switch_mlp.`.
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

_DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "U16": 2, "U8": 1, "I8": 1}


class PackError(RuntimeError):
    pass


# ------------------------------------------------------------------ checkpoint scan
def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), 8 + n


class CheckpointScan:
    """Where every routed-expert slice lives, plus the dense byte total, from the shard headers.

    Reads no tensor data, so the budget planner can call it before any Metal allocation.
    """

    def __init__(self, model_path: str):
        self.model_path = os.path.expanduser(model_path)
        idx_path, weight_map = self._read_index_files()

        headers = self._read_shard_headers(weight_map)

        self._count_layers(weight_map, idx_path)

        self._scan_expert_slices(weight_map, headers)

        self.field_offsets = []
        acc = 0
        for b in self.field_bytes:
            self.field_offsets.append(acc)
            acc += b
        self.record_bytes = acc
        self.record_stride = -(-acc // PAGE) * PAGE  # disk bytes per record (page aligned)

        self._sum_byte_totals(weight_map, headers)

    def _read_index_files(self) -> tuple:
        """Hash the shard index and config.json into the scan; return the index path and its weight map."""
        idx_path = os.path.join(self.model_path, "model.safetensors.index.json")
        with open(idx_path, "rb") as f:
            idx_raw = f.read()
        self.index_sha256 = hashlib.sha256(idx_raw).hexdigest()
        with open(os.path.join(self.model_path, "config.json"), "rb") as f:
            self.config_sha256 = hashlib.sha256(f.read()).hexdigest()
        weight_map = json.loads(idx_raw)["weight_map"]
        return idx_path, weight_map

    def _read_shard_headers(self, weight_map: dict) -> dict[str, tuple[dict, int]]:
        """List the shards and read every shard header; return the headers by shard name."""
        shard_names = sorted(set(weight_map.values()))
        self.shards = []
        headers: dict[str, tuple[dict, int]] = {}
        for name in shard_names:
            p = os.path.join(self.model_path, name)
            st = os.stat(p)
            self.shards.append({"name": name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
            headers[name] = _read_header(p)
        return headers

    def _count_layers(self, weight_map: dict, idx_path: str) -> None:
        """Set `num_layers` to the number of language-model layers the index names."""
        self.num_layers = 0
        while any(k.startswith(f"{PREFIX}model.layers.{self.num_layers}.") for k in weight_map):
            self.num_layers += 1
        if self.num_layers == 0:
            raise PackError(f"no language-model layers found in {idx_path}")

    def _scan_expert_slices(self, weight_map: dict, headers: dict[str, tuple[dict, int]]) -> None:
        """Locate every layer's routed-expert slices and set the per-field geometry."""
        self.field_bytes: list[int] = []
        self.field_shape: list[list[int]] = []
        self.field_dtype: list[str] = []
        self.num_experts = 0
        self.slices: dict[
            tuple[int, str], tuple[str, int, int]
        ] = {}  # (layer, field) -> (shard, start, per_expert)
        for layer in range(self.num_layers):
            for f_i, field in enumerate(FIELDS):
                name = f"{PREFIX}model.layers.{layer}.mlp.switch_mlp.{field}"
                shard = weight_map.get(name)
                if shard is None:
                    raise PackError(f"missing expert tensor {name}")
                hdr, data_off = headers[shard]
                meta = hdr[name]
                shape = meta["shape"]
                lo, hi = meta["data_offsets"]
                per_expert = self._record_slice(name, shape, lo, hi, layer, field, shard, data_off)
                self._note_field_geometry(layer, f_i, per_expert, shape, meta, name)

    def _record_slice(
        self,
        name: str,
        shape: list,
        lo: int,
        hi: int,
        layer: int,
        field: str,
        shard: str,
        data_off: int,
    ) -> int:
        """Check one expert tensor's expert count and slice size, record its slice; return the per-expert bytes."""
        if self.num_experts == 0:
            self.num_experts = int(shape[0])
        if int(shape[0]) != self.num_experts:
            raise PackError(f"{name} has {shape[0]} experts, expected {self.num_experts}")
        per_expert = (hi - lo) // self.num_experts
        if per_expert * self.num_experts != hi - lo:
            raise PackError(f"{name} does not divide into {self.num_experts} experts")
        self.slices[(layer, field)] = (shard, data_off + lo, per_expert)
        return per_expert

    def _note_field_geometry(
        self, layer: int, f_i: int, per_expert: int, shape: list, meta: dict, name: str
    ) -> None:
        """Layer 0 sets field `f_i`'s bytes, shape and dtype; later layers must match its slice size."""
        if layer == 0:
            self.field_bytes.append(per_expert)
            self.field_shape.append([int(d) for d in shape[1:]])
            self.field_dtype.append(meta["dtype"])
        elif self.field_bytes[f_i] != per_expert:
            raise PackError(f"{name} slice {per_expert} != layer-0 slice {self.field_bytes[f_i]}")

    def _sum_byte_totals(self, weight_map: dict, headers: dict[str, tuple[dict, int]]) -> None:
        """Set the dense and routed-expert byte totals (vision, MTP and expert tensors are not dense)."""
        # dense = language tensors other than routed experts and MTP (vision is never read)
        dense = 0
        for name, shard in weight_map.items():
            if not name.startswith(PREFIX):
                continue
            if ".mlp.switch_mlp." in name or ".mtp." in name or name.startswith(PREFIX + "mtp."):
                continue
            hdr, _ = headers[shard]
            lo, hi = hdr[name]["data_offsets"]
            dense += hi - lo
        self.dense_bytes = dense
        self.expert_bytes = self.record_bytes * self.num_layers * self.num_experts

    @property
    def num_records(self) -> int:
        return self.num_layers * self.num_experts

    def identity(self) -> dict:
        return {
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "shards": self.shards,
        }

    def geometry(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "record_bytes": self.record_bytes,
            "record_stride": self.record_stride,
            "fields": [
                {"name": n, "dtype": d, "shape": s, "bytes": b, "offset": o}
                for n, d, s, b, o in zip(
                    FIELDS, self.field_dtype, self.field_shape, self.field_bytes, self.field_offsets
                )
            ],
        }

    def slice_of(self, layer: int, expert: int, field: str) -> tuple[str, int, int]:
        """(shard path, absolute byte offset, length) of one expert's slice in the checkpoint."""
        shard, start, per_expert = self.slices[(layer, field)]
        return os.path.join(self.model_path, shard), start + expert * per_expert, per_expert


# ------------------------------------------------------------------ pack file
def default_pack_path(scan: CheckpointScan) -> str:
    env = envs.EXPERT_PACK.get()
    if env:
        return env
    base = envs.PACK_DIR.get() or os.path.expanduser("~/.cache/jitengine")
    tag = f"{os.path.basename(scan.model_path.rstrip('/'))}-{scan.index_sha256[:12]}"
    return os.path.join(os.path.expanduser(base), f"{tag}.v{VERSION}.experts")


def _header_block(scan: CheckpointScan, complete: bool, built_at: float) -> bytes:
    meta = {
        "version": VERSION,
        "checkpoint": scan.identity(),
        "model_path": scan.model_path,
        "data_offset": HEADER_BYTES,
        "complete": bool(complete),
        "built_at": built_at,
        **scan.geometry(),
    }
    raw = json.dumps(meta).encode()
    block = MAGIC + struct.pack("<I", len(raw)) + raw
    if len(block) > HEADER_BYTES:
        raise PackError("pack header does not fit in the header block")
    return block + b"\x00" * (HEADER_BYTES - len(block))


def read_pack_header(path: str) -> dict:
    with open(path, "rb") as f:
        block = f.read(HEADER_BYTES)
    if len(block) < 12 or block[:8] != MAGIC:
        raise PackError(f"{path}: not an expert pack (bad magic)")
    n = struct.unpack("<I", block[8:12])[0]
    return json.loads(block[12 : 12 + n])


def validate(path: str, scan: CheckpointScan) -> tuple[bool, str]:
    """(ok, why). Never raises on a corrupt file: the caller decides rebuild vs refuse."""
    try:
        meta = read_pack_header(path)
    except (OSError, ValueError, PackError) as exc:
        return False, str(exc)
    if int(meta.get("version", -1)) != VERSION:
        return False, f"version {meta.get('version')} != {VERSION}"
    if not meta.get("complete"):
        return False, "completion marker absent (partial repack)"
    if meta.get("checkpoint") != scan.identity():
        return False, "checkpoint identity differs (shard size/mtime or index hash changed)"
    geom = scan.geometry()
    for k in ("num_layers", "num_experts", "record_bytes", "record_stride"):
        if meta.get(k) != geom[k]:
            return False, f"{k} {meta.get(k)} != {geom[k]}"
    if meta.get("fields") != geom["fields"]:
        return False, "field layout differs"
    want = HEADER_BYTES + scan.num_records * scan.record_stride
    have = os.path.getsize(path)
    if have != want:
        return False, f"size {have} != {want} (truncated)"
    return True, "ok"


def build_pack(scan: CheckpointScan, path: str, log=print, chunk_bytes: int = 32 << 20) -> str:
    """Write the pack one layer of records at a time and publish it by atomic rename."""
    import numpy as np

    t0 = time.monotonic()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    total = scan.num_records * scan.record_stride
    log(
        f"[pack] building {path} ({total / 2**30:.2f} GiB, {scan.num_records} records of "
        f"{scan.record_bytes} B, stride {scan.record_stride} B) from {scan.model_path}"
    )

    fds: dict[str, int] = {}
    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(out, _header_block(scan, complete=False, built_at=time.time()))
        rec = np.zeros((scan.num_experts, scan.record_stride), dtype=np.uint8)  # padding stays zero
        for layer in range(scan.num_layers):
            for f_i, field in enumerate(FIELDS):
                src, start, length = scan.slice_of(layer, 0, field)
                fd = _open_source(fds, src)
                span = length * scan.num_experts
                off = scan.field_offsets[f_i]
                buf = np.empty(span, dtype=np.uint8)
                _read_into(buf, fd, src, start, span, chunk_bytes)
                rec[:, off : off + length] = buf.reshape(scan.num_experts, length)
            _write_all(out, rec, chunk_bytes)
            _log_progress(log, scan, layer, t0)
        os.fsync(out)
        # completion marker last, then fsync again: a crash before this leaves an unusable temp file
        os.pwrite(out, _header_block(scan, complete=True, built_at=time.time()), 0)
        os.fsync(out)
    except BaseException:
        _discard_partial(out, tmp)
        for fd in fds.values():
            os.close(fd)
        raise
    os.close(out)
    for fd in fds.values():
        os.close(fd)
    os.rename(tmp, path)
    log(f"[pack] built in {time.monotonic() - t0:.0f}s -> {path}")
    return path


def _open_source(fds: dict[str, int], src: str) -> int:
    """The read-only descriptor of shard `src`, opened once and kept in `fds`."""
    fd = fds.get(src)
    if fd is None:
        fd = fds[src] = os.open(src, os.O_RDONLY)
    return fd


def _read_into(buf: object, fd: int, src: str, start: int, span: int, chunk_bytes: int) -> None:
    """Fill `buf` with `span` bytes of `fd` from offset `start`, in chunks; a short read raises PackError."""
    got = 0
    while got < span:
        n = os.preadv(fd, [memoryview(buf)[got : got + min(chunk_bytes, span - got)]], start + got)
        if n <= 0:
            raise PackError(f"short read on {src} at {start + got}")
        got += n


def _write_all(out: int, rec: object, chunk_bytes: int) -> None:
    """Write the whole record buffer to `out` in chunks."""
    mv = memoryview(rec).cast("B")
    written = 0
    while written < len(mv):
        written += os.write(out, mv[written : written + chunk_bytes])


def _log_progress(log, scan: CheckpointScan, layer: int, t0: float) -> None:
    """Log the progress of the build every eighth layer and after the last."""
    if layer % 8 == 0 or layer == scan.num_layers - 1:
        done = (layer + 1) * scan.num_experts * scan.record_stride
        el = time.monotonic() - t0
        log(
            f"[pack] layer {layer + 1}/{scan.num_layers}: {done / 2**30:.1f} GiB in {el:.0f}s "
            f"({done / 1e9 / max(el, 1e-9):.2f} GB/s)"
        )


def _discard_partial(out: int, tmp: str) -> None:
    """Close the output descriptor and remove the temporary pack file, ignoring a failed unlink."""
    os.close(out)
    try:
        os.unlink(tmp)
    except OSError:
        pass  # the file may already be gone; a failed unlink only leaves a stray temporary file


def open_pack(
    model_path: str, build: bool = True, path: str | None = None, log=print
) -> tuple[str, dict, CheckpointScan]:
    """Validate (and if allowed build) the pack for `model_path`. Never returns a stale one."""
    scan = CheckpointScan(model_path)
    path = path or default_pack_path(scan)
    if os.path.exists(path):
        ok, why = validate(path, scan)
        if ok:
            return path, read_pack_header(path), scan
        log(f"[pack] rejecting {path}: {why}")
        if not build:
            raise PackError(
                f"expert pack {path} unusable: {why}; rebuild it with `jitengine-qwen3.6-35b-a3b-m2 pack "
                f"--model-path {scan.model_path} --output {path}`"
            )
        os.unlink(path)
    elif not build:
        raise PackError(
            f"expert pack {path} does not exist; prepare it first with "
            f"`jitengine-qwen3.6-35b-a3b-m2 pack --model-path {scan.model_path} --output {path}` "
            f"(or pass build_expert_pack=True / --build-expert-pack)"
        )
    build_pack(scan, path, log=log)
    ok, why = validate(path, scan)
    if not ok:
        raise PackError(f"freshly built pack {path} does not validate: {why}")
    return path, read_pack_header(path), scan
