"""Write the SKYP16X1 pack (p16_format.py) from the FNXEPK01 expert pack, verifying every record.

Each record is pack16's own `encode_records` output (the encode the pool runs at publish) for the
source record of the same (layer, expert), at `HEADER_BYTES + r * record_stride` of `<out>.partial`,
written with the page cache off. The build goes one LAYER per chunk (512 records, 1.42 GB on
Flash-Next), each chunk its own call, so a caller can schedule the chunks around other work; it
resumes from the last verified chunk. A chunk is recorded only once it is verified:
  (a) its bytes, read back uncached, equal the encode output record for record, stride pads zero;
  (b) its weight fields equal the checkpoint shards' 4-bit weights and its words decode (the Metal
      `p16_decode` the kernels use) to the shards' exact bf16 scale and bias bits, the shards read
      uncached (never the source pack);
  (c) every escaped word indexes the table and its group decodes to that exact pair.
The encoder fails on a pair that neither fits a code nor is in the table: no lossy word, ever.
`finalize` requires every layer verified and every table entry used, writes the completion marker,
checks the file with the engine's own `validate_p16`, re-reads it whole (its sha256, and per layer,
which must equal what the chunks recorded), writes the manifest and only then renames `<out>.partial`
to `<out>`.

The output path is given (`--out`), never derived. This module opens only its own outputs for write
(the partial, its state file, the manifest); its uncached descriptors come from pack_io.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    MAGIC,
    PackError,
    read_header_block,
    validate,
)
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_io import (
    aligned_buf,
    open_uncached,
    pread_full,
    uncached,
)
from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

GIB = 1 << 30
MIN_FREE_BYTES = 8 * GIB  # a chunk never starts unless this much stays free after the build
BATCH = 64  # records per encode / write / verify step inside a chunk
HASH_CHUNK = 256 << 20
F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)
EXIT_DONE = 10  # `--chunk next` found nothing to build


def escape_table(scan, source: str, src_meta: dict, geom: p16.Geometry) -> list:
    """The checkpoint's shipped table, else a census of the source pack (as the engine takes it)."""
    from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import ExpertStore
    from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.codec import census, known_escapes

    table = known_escapes(scan.ckpt.index_sha256)
    if table is not None:
        return table
    store = ExpertStore.from_header(source, src_meta, workers=1)
    try:
        return census(store, int(src_meta["records"]), geom)
    finally:
        store.close()


def _sha_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _writer_identity() -> dict:
    return {"writer_sha256": _sha_file(__file__), "pack16_kernel_sha256": _sha_file(p16.__file__)}


def _pwrite_all(fd: int, buf: np.ndarray, offset: int) -> None:
    mv = memoryview(buf).cast("B")
    done = 0
    while done < len(mv):
        n = os.pwrite(fd, mv[done:], offset + done)
        if n <= 0:
            raise PackError(f"short write at {offset + done}")
        done += n


def _atomic_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class P16Build:
    """One pack16 build: `out` is the final path; the work happens in `<out>.partial`."""

    def __init__(
        self,
        scan,
        source: str,
        out: str,
        table: list | None = None,
        batch: int = BATCH,
        log=print,
        min_free: int = MIN_FREE_BYTES,
    ):
        self.scan, self.source, self.log = scan, os.path.abspath(source), log
        self.out = os.path.abspath(out)
        self.partial = self.out + ".partial"
        self.state_path = self.partial + ".state.json"
        self.batch, self.min_free = int(batch), int(min_free)
        ok, why = validate(self.source, scan)
        if not ok:
            raise PackError(
                f"source pack {self.source} does not validate for this checkpoint: {why}"
            )
        self.src_meta = read_header_block(self.source, MAGIC)
        q = scan.quantization()
        self.geom = p16.Geometry(self.src_meta["fields"], int(q["group_size"]), int(q["bits"]))
        self.table = [
            int(v)
            for v in (
                table
                if table is not None
                else escape_table(scan, self.source, self.src_meta, self.geom)
            )
        ]
        if len(self.table) > p16.ECAP:
            raise PackError(f"escape table of {len(self.table)} entries exceeds {p16.ECAP}")
        self.src_ident = pf.source_identity(self.source)
        self.stride = pf.record_stride(self.geom)
        self.E = int(scan.num_experts)
        self.state: dict | None = None
        self.meta: dict | None = None
        self._exc = None

    # ------------------------------------------------------------------ the partial file
    def _expected(self, build_id: str, built_at: float, complete: bool) -> dict:
        return pf.p16_header(
            self.scan,
            self.src_ident,
            self.geom,
            self.table,
            build_id,
            complete,
            built_at,
            writer=_writer_identity(),
        )

    @staticmethod
    def _comparable(meta: dict) -> dict:
        return {
            k: v for k, v in meta.items() if k not in ("build_id", "built_at", "writer", "complete")
        }

    def open(self) -> dict:
        """Create `<out>.partial` (header, then the full size, sparse) or resume a matching one."""
        if os.path.exists(self.out):
            raise PackError(f"{self.out} exists already; a build never overwrites a finished pack")
        if os.path.exists(self.partial):
            meta = read_header_block(self.partial, pf.MAGIC_P16)
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
            except (OSError, ValueError) as exc:
                raise PackError(
                    f"{self.partial} has no readable state file ({exc}); remove it to rebuild"
                ) from exc
            want = self._expected(meta.get("build_id", ""), meta.get("built_at", 0.0), False)
            if (
                meta.get("complete")
                or state.get("build_id") != meta.get("build_id")
                or self._comparable(meta) != self._comparable(want)
            ):
                raise PackError(
                    f"{self.partial} is from another build (checkpoint, source, table or "
                    "layout differ); remove it to rebuild"
                )
            if os.path.getsize(self.partial) != pf.p16_bytes(meta):
                raise PackError(
                    f"{self.partial} has size {os.path.getsize(self.partial)}, not {pf.p16_bytes(meta)}"
                )
            self.meta, self.state = meta, state
            self.log(
                f"[pack16] resuming {self.partial}: {len(state['layers'])}/{self.scan.num_layers} layers verified"
            )
            return state
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        meta = self._expected(uuid.uuid4().hex, time.time(), False)
        need = pf.p16_bytes(meta)
        free = shutil.disk_usage(os.path.dirname(self.out)).free
        if free - need < self.min_free:
            raise PackError(
                f"{free / GIB:.1f} GiB free, the pack needs {need / GIB:.1f} GiB and "
                f"{self.min_free / GIB:.0f} GiB must stay free"
            )
        state = {
            "build_id": meta["build_id"],
            "partial": self.partial,
            "layers": {},
            "table": self.table,
        }
        _atomic_json(self.state_path, state)
        fd = os.open(self.partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            uncached(fd)
            _pwrite_all(fd, np.frombuffer(pf.p16_header_block(meta), dtype=np.uint8), 0)
            os.ftruncate(fd, need)  # sparse: blocks are allocated as the chunks write them
            os.fsync(fd)
        finally:
            os.close(fd)
        self.meta, self.state = meta, state
        self.log(
            f"[pack16] created {self.partial}: {meta['records']} records of {meta['record_bytes']} B, "
            f"stride {meta['record_stride']}, {need / 1e9:.2f} GB, {len(self.table)} escape(s)"
        )
        return state

    def peek(self) -> dict | None:
        """The state of a build in progress, or None; creates nothing."""
        try:
            with open(self.state_path) as f:
                return json.load(f) if os.path.exists(self.partial) else None
        except OSError, ValueError:
            return None

    def todo(self, state: dict | None = None) -> list[int]:
        state = self.state if state is None else state
        done = {int(k) for k in (state or {"layers": {}})["layers"]}
        return [layer for layer in range(self.scan.num_layers) if layer not in done]

    # ------------------------------------------------------------------ one chunk
    def _exc_array(self):
        if self._exc is None:
            self._exc = p16.exc_array(self.table)
        return self._exc

    def _encode(self, recs: np.ndarray) -> np.ndarray:
        """Encode `recs` (u32 [n, source record words]) with the pool's kernel: [n, stride] u8."""
        n = recs.shape[0]
        cf = self.geom.compact_fields()
        sets = [
            mx.zeros((n, *f["shape"]), dtype=mx.uint32 if f["dtype"] == "U32" else mx.uint16)
            for f in cf
        ]
        staging = mx.array(recs)
        jobs = np.stack([np.arange(n), np.arange(n)], axis=1)
        dep = mx.zeros((1,), dtype=mx.uint32)
        err = p16.encode_records(
            self.geom, staging, sets, None, n, self._exc_array(), len(self.table), jobs, dep
        )
        mx.eval(err, *sets)
        stale, lossy = p16.stale_and_lossy(int(err.item()))
        if stale or lossy:
            raise PackError(
                f"pack16 encode: {lossy} group(s) outside the codec and the escape table "
                f"({len(self.table)} entries), {stale} stale: no lossy word is ever written"
            )
        out = aligned_buf(n * self.stride).reshape(n, self.stride)
        out[:] = 0
        for f, a in zip(cf, sets, strict=True):
            out[:, f["offset"] : f["offset"] + f["bytes"]] = (
                np.asarray(a).reshape(n, -1).view(np.uint8)
            )
        return out

    def _shard_slab(self, fds: dict, layer: int, e0: int, n: int, field: str) -> np.ndarray:
        path, start, length = self.scan.slice_of(layer, e0, field)
        fd = fds.get(path)
        if fd is None:
            fd = fds[path] = open_uncached(path)
        buf = np.empty(n * length, dtype=np.uint8)
        pread_full(fd, buf, start, path)
        return buf.reshape(n, length)

    def _verify(
        self, got: np.ndarray, want: np.ndarray, layer: int, e0: int, fds: dict, esc_found: list
    ) -> dict:
        """(a), (b) and (c) for records e0.. of `layer`; raises PackError on the first mismatch."""
        n = got.shape[0]
        if not np.array_equal(got, want):
            bad = int(np.flatnonzero((got != want).any(axis=1))[0])
            raise PackError(
                f"layer {layer} expert {e0 + bad}: the bytes read back differ from the encode output"
            )
        cf = self.geom.compact_fields()
        by = {f["name"]: f for f in cf}
        exc = self._exc_array()
        groups = 0
        for proj in p16.PROJ:
            w = by[f"{proj}.weight"]
            if not np.array_equal(
                got[:, w["offset"] : w["offset"] + w["bytes"]],
                self._shard_slab(fds, layer, e0, n, f"{proj}.weight"),
            ):
                raise PackError(
                    f"layer {layer} experts {e0}..{e0 + n - 1}: {proj}.weight differs from the checkpoint"
                )
            wd = by[f"{proj}.words"]
            flat = (
                np.ascontiguousarray(got[:, wd["offset"] : wd["offset"] + wd["bytes"]])
                .view(np.uint16)
                .reshape(-1)
            )
            sb = np.asarray(p16.decode_words(mx.array(flat), exc)).reshape(n, -1, 2)
            s_want = self._shard_slab(fds, layer, e0, n, f"{proj}.scales").view(np.uint16)
            b_want = self._shard_slab(fds, layer, e0, n, f"{proj}.biases").view(np.uint16)
            if not (np.array_equal(sb[:, :, 0], s_want) and np.array_equal(sb[:, :, 1], b_want)):
                raise PackError(
                    f"layer {layer} experts {e0}..{e0 + n - 1}: {proj} words do not decode to the "
                    "checkpoint's scale/bias bits"
                )
            groups += flat.size
            wv = flat.reshape(n, -1).astype(np.uint32)
            for r, g in zip(*np.nonzero((wv & 31) == p16.ESC), strict=True):
                idx = int(wv[r, g] >> 5)
                if idx >= len(self.table):
                    raise PackError(
                        f"layer {layer} expert {e0 + r} {proj} group {g}: escape index {idx} "
                        f"outside the {len(self.table)}-entry table"
                    )
                e = self.table[idx]
                if (int(sb[r, g, 0]), int(sb[r, g, 1])) != (e & 0xFFFF, e >> 16):
                    raise PackError(
                        f"layer {layer} expert {e0 + r} {proj} group {g}: escaped group does not "
                        "decode to its table pair"
                    )
                esc_found.append(
                    [layer, int(e0 + r), proj, int(g), idx, int(sb[r, g, 0]), int(sb[r, g, 1])]
                )
        return {"groups_decoded": groups}

    def build_layer(self, layer: int) -> dict:
        """Encode, write and verify every record of `layer`; record it in the state file."""
        if str(layer) in self.state["layers"]:
            return self.state["layers"][str(layer)]
        remaining = len(self.todo()) * self.E * self.stride
        free = shutil.disk_usage(os.path.dirname(self.out)).free
        if free - remaining < self.min_free:
            raise PackError(
                f"{free / GIB:.1f} GiB free with {remaining / GIB:.1f} GiB still to write: "
                f"below the {self.min_free / GIB:.0f} GiB floor, stopping"
            )
        t0 = time.monotonic()
        sm = self.src_meta
        src_rw, src_stride = int(sm["record_bytes"]) // 4, int(sm["record_stride"])
        sfd = open_uncached(self.source)
        rfd = open_uncached(self.partial)
        wfd = uncached(os.open(self.partial, os.O_WRONLY))
        fds: dict = {}
        h = hashlib.sha256()
        esc_found: list = []
        groups = 0
        try:
            for e0 in range(0, self.E, self.batch):
                n = min(self.batch, self.E - e0)
                r0 = layer * self.E + e0
                raw = aligned_buf(n * src_stride)
                pread_full(sfd, raw, int(sm["data_offset"]) + r0 * src_stride, self.source)
                recs = np.ascontiguousarray(raw.reshape(n, src_stride)[:, : src_rw * 4]).view(
                    np.uint32
                )
                want = self._encode(recs)
                off = int(self.meta["data_offset"]) + r0 * self.stride
                _pwrite_all(wfd, want, off)
                os.fsync(wfd)
                got = aligned_buf(n * self.stride).reshape(n, self.stride)
                pread_full(rfd, got, off, self.partial)
                groups += self._verify(got, want, layer, e0, fds, esc_found)["groups_decoded"]
                h.update(memoryview(got).cast("B"))
        finally:
            for fd in (sfd, rfd, wfd, *fds.values()):
                os.close(fd)
        rec = {
            "sha256": h.hexdigest(),
            "records": self.E,
            "bytes": self.E * self.stride,
            "groups_decoded": groups,
            "escaped": esc_found,
            "seconds": round(time.monotonic() - t0, 2),
        }
        self.state["layers"][str(layer)] = rec
        _atomic_json(self.state_path, self.state)
        self.log(
            f"[pack16] layer {layer} verified: {self.E} records, {groups} groups decoded, "
            f"{len(esc_found)} escaped, {rec['seconds']}s, sha256 {rec['sha256'][:16]}"
        )
        return rec

    # ------------------------------------------------------------------ the end
    def finalize(self, manifest_extra: dict | None = None) -> dict:
        """Completion marker, the engine's validation, whole-file hashes, the manifest, the rename."""
        todo = self.todo()
        if todo:
            raise PackError(f"{len(todo)} layer(s) not built yet: {todo[:8]}")
        uses = [0] * len(self.table)
        escaped = []
        for layer in range(self.scan.num_layers):
            for row in self.state["layers"][str(layer)]["escaped"]:
                uses[row[4]] += 1
                escaped.append(row)
        if any(u == 0 for u in uses):
            raise PackError(
                f"escape table entries never used {[i for i, u in enumerate(uses) if not u]}: "
                "the table is not exactly this checkpoint's"
            )
        meta = dict(self.meta, complete=True)
        fd = uncached(os.open(self.partial, os.O_WRONLY))
        try:
            _pwrite_all(fd, np.frombuffer(pf.p16_header_block(meta), dtype=np.uint8), 0)
            os.fsync(fd)
            fcntl.fcntl(fd, F_FULLFSYNC)
        finally:
            os.close(fd)
        ok, why = pf.validate_p16(self.partial, self.scan, self.source, self.geom, self.table)
        if not ok:
            raise PackError(f"the finished partial does not validate: {why}")
        whole, per_layer = self._hash_file(meta)
        for layer, got in enumerate(per_layer):
            want = self.state["layers"][str(layer)]["sha256"]
            if got != want:
                raise PackError(
                    f"layer {layer} re-read sha256 {got} != the verified chunk's {want}"
                )
        records = sum(r["records"] for r in self.state["layers"].values())
        manifest = {
            "path": self.out,
            "size": os.path.getsize(self.partial),
            "sha256": whole,
            "per_layer_sha256": per_layer,
            "header": meta,
            "escapes": self.table,
            "escape_uses": uses,
            "escaped_groups": escaped,
            "source": self.src_ident,
            "verification": {
                "records": records,
                "bytes_compared": records * self.stride,
                "groups_decoded": sum(r["groups_decoded"] for r in self.state["layers"].values()),
                "checks": "(a) read back uncached == encode_records output incl. zero stride pads; "
                "(b) weight fields == shard bytes, words through the Metal p16_decode == "
                "shard scale/bias bits (shards read uncached); (c) escaped words index the "
                "table and decode to its exact pair; whole-file re-read per-layer sha256 == "
                "the chunks'",
            },
            "chunks": {
                k: {"seconds": v["seconds"], "sha256": v["sha256"]}
                for k, v in self.state["layers"].items()
            },
            "finished_at": time.time(),
            **(manifest_extra or {}),
        }
        man_tmp = self.out + ".manifest.json.tmp"
        _atomic_json(man_tmp, manifest)
        os.rename(self.partial, self.out)
        os.replace(man_tmp, self.out + ".manifest.json")
        os.replace(self.state_path, self.out + ".state.json")
        self.log(f"[pack16] finished {self.out}: {manifest['size']} B, sha256 {whole}")
        return manifest

    def _hash_file(self, meta: dict) -> tuple[str, list]:
        whole = hashlib.sha256()
        layer_bytes = self.E * self.stride
        per = []
        fd = open_uncached(self.partial)
        try:
            buf = aligned_buf(HASH_CHUNK)
            pread_full(fd, buf[: int(meta["data_offset"])], 0, self.partial)
            whole.update(memoryview(buf[: int(meta["data_offset"])]))
            off = int(meta["data_offset"])
            for _layer in range(self.scan.num_layers):
                h = hashlib.sha256()
                left = layer_bytes
                while left:
                    n = min(left, HASH_CHUNK)
                    pread_full(fd, buf[:n], off, self.partial)
                    mv = memoryview(buf[:n])
                    h.update(mv)
                    whole.update(mv)
                    off += n
                    left -= n
                per.append(h.hexdigest())
            if off != os.path.getsize(self.partial):
                raise PackError(f"hashed {off} B of {os.path.getsize(self.partial)}")
        finally:
            os.close(fd)
        return whole.hexdigest(), per
