"""Flash-Next's two pack formats, pinned byte for byte on small synthetic checkpoints.

The builders (`build_experts.main`, `build_ple.main`) write an expert pack (whole, and one of two
layers) and a PLE pack (whole, and two segments) from a 2-layer, 2-expert checkpoint whose
per-expert and per-row field sizes are the real ones; the writers' expected counts are patched down
to it. Each file must equal the one assembled here from the layout. The readers (`open_pack`,
`open_ple_pack`, `PLEStore`) must accept it, parse the same header and find every record and row
where the header puts it, and open packs of a toy geometry as well; builder and reader must agree
on default paths; both must refuse a damaged copy. Only the PLE row reader needs MLX.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jitengine_qwen3_8_flash_next_m2.checkpoint import (
    build_experts,
    build_ple,
    expert_writer,
    ple_writer,
)
from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import PackError as ReaderPackError
from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import open_pack
from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_io import PackError, Reader

HEADER = 1 << 20
QUANT = {"bits": 4, "group_size": 32, "mode": "affine"}
WIDTH = {"U32": 4, "BF16": 2}
LAYERS, EXPERTS = 2, 2
# the nine record fields at the real checkpoint's per-expert shapes: 3,072,000 B per record
RECORD = [
    ("gate_proj.weight", "U32", [640, 320]),
    ("gate_proj.scales", "BF16", [640, 80]),
    ("gate_proj.biases", "BF16", [640, 80]),
    ("up_proj.weight", "U32", [640, 320]),
    ("up_proj.scales", "BF16", [640, 80]),
    ("up_proj.biases", "BF16", [640, 80]),
    ("down_proj.weight", "U32", [2560, 80]),
    ("down_proj.scales", "BF16", [2560, 20]),
    ("down_proj.biases", "BF16", [2560, 20]),
]
RECORD_BYTES, RECORD_STRIDE = 3_072_000, 3_080_192  # the stride is 188 x 16 KiB
SHARDS, ROWS = 2, 8  # PLE n-gram shards, rows per shard
ROW = [("weight", "U32", 20), ("scales", "BF16", 5), ("biases", "BF16", 5)]  # 100 B per row
# a toy geometry (hidden 64, expert width 32, 3,840 B records; 20 B rows) the writers accept only
# with their expected sizes patched, and the readers open as they are
TOY_RECORD = [
    ("gate_proj.weight", "U32", [32, 8]),
    ("gate_proj.scales", "BF16", [32, 2]),
    ("gate_proj.biases", "BF16", [32, 2]),
    ("up_proj.weight", "U32", [32, 8]),
    ("up_proj.scales", "BF16", [32, 2]),
    ("up_proj.biases", "BF16", [32, 2]),
    ("down_proj.weight", "U32", [64, 4]),
    ("down_proj.scales", "BF16", [64, 1]),
    ("down_proj.biases", "BF16", [64, 1]),
]
TOY_ROW = [("weight", "U32", 4), ("scales", "BF16", 1), ("biases", "BF16", 1)]
MTIME_NS = 1_700_000_000_123_456_789
BUILT_AT = 1_700_000_000.5
WHOLE = [{"row_start": 0, "rows": 16, "file_offset": HEADER}]
TWO_SEGMENTS = [
    {"row_start": 2, "rows": 3, "file_offset": HEADER},
    {"row_start": 8, "rows": 3, "file_offset": HEADER + 300},
]
# each pack: its builder, the selection it holds, and the builder's chunking
BUILD = {
    "experts": (build_experts, [], []),
    "experts-L1": (build_experts, ["--layers", "1"], ["--experts-per-chunk", "1"]),
    "ple": (build_ple, [], []),
    "ple-R6": (build_ple, ["--shards", "0,1", "--rows", "0=2:5,1=0:3"], ["--rows-per-chunk", "2"]),
}
COUNTS = {  # the writers' expected counts, patched down to the synthetic checkpoints
    expert_writer: {"EXPECT_LAYERS": 2, "EXPECT_EXPERTS": 2, "EXPECT_LAYER_BYTES": 6_144_000},
    ple_writer: {"EXPECT_SHARDS": 2, "EXPECT_ROWS_PER_SHARD": 8, "EXPECT_TOTAL_ROWS": 16},
}
TOY_SIZES = {  # and the field sizes, for the toy geometry
    expert_writer: {
        "EXPECT_RECORD_BYTES": 3840,
        "EXPECT_LAYER_BYTES": 7680,
        "EXPECT_FIELD_BYTES": [1024, 128, 128] * 3,
    },
    ple_writer: {
        "EXPECT_ROW_BYTES": 20,
        "EXPECT_FIELD_BYTES": [16, 2, 2],
        "EXPECT_FIELD_COLS": [4, 1, 1],
    },
}


def _quiet(*_args, **_kw) -> None:
    pass


def _page_cached(path: str) -> Reader:
    """`Reader` without F_NOCACHE, which only Darwin has."""
    return Reader(path, nocache=False)


def _payload(name: str, n: int) -> bytes:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def _write_shard(path: str, tensors: dict) -> None:
    header, blobs, at = {}, [], 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    os.utime(path, ns=(MTIME_NS, MTIME_NS))


def _make_checkpoint(model: str, record=RECORD, row=ROW) -> SimpleNamespace:
    """Two shards: layer 0's experts, PLE shard 0 and two dense tensors; layer 1's and PLE shard 1."""
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shards: list[dict] = [{}, {}]
    data: dict[str, bytes] = {}
    p = "language_model.model.layers"
    for layer in range(LAYERS):
        for field, dtype, shape in record:
            name = f"{p}.{layer}.mlp.switch_mlp.{field}"
            data[name] = _payload(name, EXPERTS * int(np.prod(shape)) * WIDTH[dtype])
            shards[layer][name] = (dtype, [EXPERTS, *shape], data[name])
    for shard in range(SHARDS):
        for field, dtype, cols in row:
            name = f"{p}.0.ple.ple_embedding.ngram_embedding.shards.{shard}.{field}"
            data[name] = _payload(name, ROWS * cols * WIDTH[dtype])
            shards[shard][name] = (dtype, [ROWS, cols], data[name])
    shards[0]["language_model.model.embed_tokens.weight"] = ("BF16", [16, 8], bytes(256))
    shards[0]["vision_tower.patch_embed.weight"] = ("BF16", [4], bytes(8))
    os.makedirs(model)
    with open(os.path.join(model, "config.json"), "w") as f:
        json.dump({"model_type": "qwen4_exp", "quantization": QUANT}, f)
    weight_map = {n: names[i] for i, s in enumerate(shards) for n in s}
    for name, tensors in zip(names, shards):
        _write_shard(os.path.join(model, name), tensors)
    index = json.dumps(
        {"metadata": {}, "weight_map": weight_map}, indent=1, sort_keys=True
    ).encode()
    with open(os.path.join(model, "model.safetensors.index.json"), "wb") as f:
        f.write(index)
    identity = {
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "shards": [
            {"name": n, "size": os.path.getsize(os.path.join(model, n)), "mtime_ns": MTIME_NS}
            for n in names
        ],
    }
    return SimpleNamespace(
        model=model,
        names=names,
        data=data,
        identity=identity,
        weight_map=weight_map,
        record=record,
        row=row,
    )


def _record(ck, layer: int, expert: int) -> bytes:
    out = b""
    for field, dtype, shape in ck.record:
        n = int(np.prod(shape)) * WIDTH[dtype]
        out += ck.data[f"language_model.model.layers.{layer}.mlp.switch_mlp.{field}"][
            expert * n : (expert + 1) * n
        ]
    return out


def _row(ck, row: int) -> bytes:
    shard, local = divmod(row, ROWS)
    out = b""
    for field, dtype, cols in ck.row:
        n = cols * WIDTH[dtype]
        name = f"language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.{shard}.{field}"
        out += ck.data[name][local * n : (local + 1) * n]
    return out


def _fields(layout, extent: str) -> list[dict]:
    out, at = [], 0
    for name, dtype, size in layout:
        n = (int(np.prod(size)) if extent == "shape" else size) * WIDTH[dtype]
        out.append({"name": name, "dtype": dtype, extent: size, "bytes": n, "offset": at})
        at += n
    return out


def _expert_header(ck, layers: list[int]) -> dict:
    return {
        "version": 1,
        "kind": "flashnext-experts",
        "checkpoint": ck.identity,
        "model_path": ck.model,
        "data_offset": HEADER,
        "layers": layers,
        "records": len(layers) * EXPERTS,
        "complete": True,
        "built_at": BUILT_AT,
        "num_layers": LAYERS,
        "num_experts": EXPERTS,
        "record_bytes": RECORD_BYTES,
        "record_stride": RECORD_STRIDE,
        "tensor_prefix": "language_model.",
        "submodule": "mlp.switch_mlp",
        "quantization": QUANT,
        "fields": _fields(RECORD, "shape"),
    }


def _ple_header(ck, segments: list[dict]) -> dict:
    return {
        "version": 1,
        "kind": "flashnext-ple-rows",
        "checkpoint": ck.identity,
        "model_path": ck.model,
        "data_offset": HEADER,
        "segments": segments,
        "rows_present": sum(s["rows"] for s in segments),
        "complete": True,
        "built_at": BUILT_AT,
        "ple_layer": 0,
        "num_shards": SHARDS,
        "rows_per_shard": ROWS,
        "total_rows": SHARDS * ROWS,
        "row_bytes": 100,
        "row_stride": 100,
        "tensor_prefix": "language_model.",
        "submodule": "ple.ple_embedding.ngram_embedding",
        "quantization": QUANT,
        "fields": _fields(ROW, "cols"),
    }


def _header_block(magic: bytes, meta: dict) -> bytes:
    raw = json.dumps(meta).encode()
    return magic + struct.pack("<I", len(raw)) + raw + bytes(HEADER - 12 - len(raw))


def _expected_expert_pack(ck, layers: list[int]) -> bytes:
    out = [_header_block(b"FNXEPK01", _expert_header(ck, layers))]
    for layer in layers:
        for expert in range(EXPERTS):
            out.append(_record(ck, layer, expert) + bytes(RECORD_STRIDE - RECORD_BYTES))
    return b"".join(out)


def _expected_ple_pack(ck, segments: list[dict]) -> bytes:
    out = [_header_block(b"FNXPLE01", _ple_header(ck, segments))]
    for seg in segments:
        out += [_row(ck, g) for g in range(seg["row_start"], seg["row_start"] + seg["rows"])]
    return b"".join(out)


def _read_header(path: str, magic: bytes) -> dict:
    with open(path, "rb") as f:
        block = f.read(HEADER)
    assert block[:8] == magic
    (n,) = struct.unpack("<I", block[8:12])
    assert not any(block[12 + n :]), "the header block is zero-padded"
    return json.loads(block[12 : 12 + n])


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _build(ck, key: str, path: str | None = None, *extra: str) -> int:
    builder, selection, chunking = BUILD[key]
    pack = ["--pack", path] if path else []
    return builder.main(["--model", ck.model, *pack, *selection, *chunking, *extra])


def _patched(mp, *tables) -> None:
    for table in tables:
        for module, sizes in table.items():
            for name, value in sizes.items():
                mp.setattr(module, name, value)


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    """The checkpoint and the four packs, built with the writers' expected counts patched to it."""
    root = str(tmp_path_factory.mktemp("flashnext-pack"))
    ck = _make_checkpoint(os.path.join(root, "ckpt"))
    ck.packs = {k: os.path.join(root, k) for k in BUILD}
    with pytest.MonkeyPatch.context() as mp:
        _patched(mp, COUNTS)
        with pytest.MonkeyPatch.context() as clock:
            clock.setattr(time, "time", lambda: BUILT_AT)
            for key in BUILD:
                assert _build(ck, key, ck.packs[key]) == 0
        yield ck


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A checkpoint of toy field sizes and its two whole packs, written with every size patched."""
    root = str(tmp_path_factory.mktemp("flashnext-toy"))
    ck = _make_checkpoint(os.path.join(root, "ckpt"), TOY_RECORD, TOY_ROW)
    ck.packs = {k: os.path.join(root, k) for k in ("experts", "ple")}
    with pytest.MonkeyPatch.context() as mp:
        _patched(mp, COUNTS, TOY_SIZES)
        for key in ck.packs:
            assert _build(ck, key, ck.packs[key]) == 0
    return ck


@pytest.mark.parametrize("key, layers", [("experts", [0, 1]), ("experts-L1", [1])])
def test_the_expert_pack_is_the_documented_layout(ck, key, layers):
    path = ck.packs[key]
    assert _read_header(path, b"FNXEPK01") == _expert_header(ck, layers)
    assert os.path.getsize(path) == HEADER + len(layers) * EXPERTS * RECORD_STRIDE
    assert _sha256(path) == hashlib.sha256(_expected_expert_pack(ck, layers)).hexdigest()


@pytest.mark.parametrize("key, segments", [("ple", WHOLE), ("ple-R6", TWO_SEGMENTS)])
def test_the_ple_pack_is_the_documented_layout(ck, key, segments):
    path = ck.packs[key]
    assert _read_header(path, b"FNXPLE01") == _ple_header(ck, segments)
    assert _sha256(path) == hashlib.sha256(_expected_ple_pack(ck, segments)).hexdigest()


# a selection other than the one each pack holds
OTHER = {"experts": ["--layers", "0"], "experts-L1": [], "ple": ["--rows", "0:4"], "ple-R6": []}


@pytest.mark.parametrize("key", list(BUILD))
def test_the_builder_reuses_its_pack_and_refuses_another_selection(ck, key):
    builder, selection, _ = BUILD[key]
    path, before = ck.packs[key], _sha256(ck.packs[key])
    assert builder.main(["--model", ck.model, "--pack", path, *selection, "--no-build"]) == 0
    assert builder.main(["--model", ck.model, "--pack", path, *OTHER[key], "--no-build"]) == 2
    assert _sha256(path) == before


def test_the_expert_reader_parses_and_finds_every_record(ck):
    for key, layers in (("experts", [0, 1]), ("experts-L1", [1])):
        path, meta, scan = open_pack(ck.model, path=ck.packs[key], log=_quiet)
        assert path == ck.packs[key] and meta == _expert_header(ck, layers)
        with open(path, "rb") as f:
            for i, layer in enumerate(layers):
                for expert in range(EXPERTS):
                    f.seek(meta["data_offset"] + (i * EXPERTS + expert) * meta["record_stride"])
                    assert f.read(RECORD_BYTES) == _record(ck, layer, expert)
    # what the budget planner reads from the reader's scan
    assert scan.prefix == "language_model." and scan.weight_map == ck.weight_map
    assert (scan.num_layers, scan.num_experts, scan.num_records) == (LAYERS, EXPERTS, 4)
    assert scan.record_bytes == RECORD_BYTES
    assert scan.field_bytes == [f["bytes"] for f in _fields(RECORD, "shape")]
    assert scan.field_offsets == [f["offset"] for f in _fields(RECORD, "shape")]
    assert scan.field_dtype == [d for _, d, _ in RECORD]
    assert scan.field_shape == [s for _, _, s in RECORD]
    assert scan.dense_bytes == 256 and scan.ple_table_bytes == SHARDS * ROWS * 100
    assert scan.expert_bytes == LAYERS * EXPERTS * RECORD_BYTES
    assert sorted(scan.headers) == ck.names
    for name in ck.names:
        with open(os.path.join(ck.model, name), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            assert scan.headers[name] == (json.loads(f.read(n)), 8 + n)


def _check_rows(ck, path: str, segments: list[dict]) -> None:
    from jitengine_qwen3_8_flash_next_m2.model.ple_store import PLEStore, open_ple_pack

    got, meta, _ple = open_ple_pack(ck.model, path=path, log=_quiet)
    assert got == path and meta["segments"] == segments
    store = PLEStore(path, meta, workers=2, nocache=False)
    try:
        present = [g for s in segments for g in range(s["row_start"], s["row_start"] + s["rows"])]
        ids = present[::-1] + present[:2]  # out of order, with repeats
        w, s, b = store.fetch_raw(np.array(ids))
        for i, g in enumerate(ids):
            assert w[i].tobytes() + s[i].tobytes() + b[i].tobytes() == _row(ck, g)
        absent = sorted(set(range(SHARDS * ROWS)) - set(present))
        if absent:
            with pytest.raises(ReaderPackError):
                store.fetch_raw(np.array(absent[:1]))
    finally:
        store.close()


def test_the_ple_reader_parses_and_finds_every_row(ck):
    pytest.importorskip("mlx.core")
    from jitengine_qwen3_8_flash_next_m2.model.ple_store import open_ple_pack

    for key, segments in (("ple", WHOLE), ("ple-R6", TWO_SEGMENTS)):
        assert open_ple_pack(ck.model, path=ck.packs[key], log=_quiet)[1] == _ple_header(
            ck, segments
        )
        _check_rows(ck, ck.packs[key], segments)


def test_the_readers_open_packs_of_a_toy_geometry(toy):
    path, meta, scan = open_pack(toy.model, path=toy.packs["experts"], log=_quiet)
    assert (meta["record_bytes"], scan.record_bytes, scan.num_records) == (3840, 3840, 4)
    assert meta["record_stride"] >= 3840 and meta["fields"] == _fields(TOY_RECORD, "shape")
    with open(path, "rb") as f:
        for layer in range(LAYERS):
            for expert in range(EXPERTS):
                f.seek(meta["data_offset"] + (layer * EXPERTS + expert) * meta["record_stride"])
                assert f.read(3840) == _record(toy, layer, expert)
    pytest.importorskip("mlx.core")
    _check_rows(toy, toy.packs["ple"], WHOLE)


def test_builder_and_reader_agree_on_default_paths(ck, monkeypatch, tmp_path):
    for var in (
        "JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK",
        "JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK",
        "JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    _patched(monkeypatch, COUNTS)
    for key in ("experts", "ple"):
        with pytest.raises(PackError):
            _build(ck, key, None, "--no-build")
    with pytest.raises(ReaderPackError):
        open_pack(ck.model, log=_quiet)
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR", str(tmp_path))
    tag = f"ckpt-{ck.identity['index_sha256'][:12]}"
    assert _build(ck, "experts-L1") == 0
    with pytest.raises(ReaderPackError):
        open_pack(ck.model, log=_quiet)  # the engine looks up the whole pack only
    for key in ("experts", "ple-R6", "ple"):
        assert _build(ck, key) == 0
    names = [f"{tag}-L1.experts", f"{tag}.experts", f"{tag}-R6.ple", f"{tag}.ple"]
    assert sorted(os.listdir(tmp_path)) == sorted(names)
    assert open_pack(ck.model, log=_quiet)[0] == str(tmp_path / f"{tag}.experts")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK", ck.packs["experts-L1"])
    assert open_pack(ck.model, log=_quiet)[0] == ck.packs["experts-L1"]
    pytest.importorskip("mlx.core")
    from jitengine_qwen3_8_flash_next_m2.model.ple_store import open_ple_pack

    assert open_ple_pack(ck.model, log=_quiet)[0] == str(tmp_path / f"{tag}.ple")
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK", ck.packs["ple-R6"])
    assert open_ple_pack(ck.model, log=_quiet)[0] == ck.packs["ple-R6"]


def _damaged(src: str, dst: str, magic: bytes, how: str) -> str:
    shutil.copyfile(src, dst)
    if how == "truncated":
        os.truncate(dst, os.path.getsize(dst) - 1)
    elif how == "magic":
        with open(dst, "r+b") as f:
            f.write(b"FNXBAD00")
    else:
        meta = _read_header(dst, magic)
        experts = magic == b"FNXEPK01"
        if how == "layout" and not experts:
            meta["segments"][1]["file_offset"] += 100
        meta.update(
            {
                "incomplete": {"complete": False},
                "version": {"version": 2},
                "geometry": {"num_experts": 3} if experts else {"num_shards": 3},
                "layout": {"record_stride": RECORD_STRIDE + (16 << 10)} if experts else {},
            }[how]
        )
        with open(dst, "r+b") as f:
            f.write(_header_block(magic, meta))
    return dst


# "layout": a wrong record stride in an expert pack, a misplaced segment in a PLE pack
DAMAGE = ["truncated", "magic", "incomplete", "version", "geometry", "layout", "stale checkpoint"]


@pytest.mark.parametrize("how", DAMAGE)
def test_both_sides_refuse_a_damaged_expert_pack(ck, tmp_path, how):
    path = ck.packs["experts"]
    shard = os.path.join(ck.model, ck.names[1])
    if how == "stale checkpoint":
        os.utime(shard, ns=(MTIME_NS + 1, MTIME_NS + 1))
    else:
        path = _damaged(path, str(tmp_path / "x.experts"), b"FNXEPK01", how)
    try:
        assert build_experts.main(["--model", ck.model, "--pack", path, "--no-build"]) == 2
        with pytest.raises(ReaderPackError):
            open_pack(ck.model, path=path, log=_quiet)
    finally:
        os.utime(shard, ns=(MTIME_NS, MTIME_NS))


@pytest.mark.parametrize("how", DAMAGE)
def test_both_sides_refuse_a_damaged_ple_pack(ck, tmp_path, how):
    path = ck.packs["ple-R6"]
    shard = os.path.join(ck.model, ck.names[1])
    if how == "stale checkpoint":
        os.utime(shard, ns=(MTIME_NS + 1, MTIME_NS + 1))
    else:
        path = _damaged(path, str(tmp_path / "x.ple"), b"FNXPLE01", how)
    try:
        assert _build(ck, "ple-R6", path, "--no-build") == 2
        pytest.importorskip("mlx.core")  # the PLE reader imports MLX
        from jitengine_qwen3_8_flash_next_m2.model.ple_store import open_ple_pack

        with pytest.raises(ReaderPackError):
            open_ple_pack(ck.model, path=path, log=_quiet)
    finally:
        os.utime(shard, ns=(MTIME_NS, MTIME_NS))


def test_the_builders_verify_their_packs(ck, monkeypatch):
    for module in (build_experts, build_ple):
        monkeypatch.setattr(module, "Reader", _page_cached)
    for key, n in (("experts", 4), ("experts-L1", 2)):  # every record the pack holds
        scan = open_pack(ck.model, path=ck.packs[key], log=_quiet)[2]
        rep = build_experts.verify(scan, ck.packs[key], n)
        assert rep["ok"] and rep["records_verified"] == n and not rep["mismatches"]
    pytest.importorskip("mlx.core")
    from jitengine_qwen3_8_flash_next_m2.model.ple_store import open_ple_pack

    for key in ("ple", "ple-R6"):
        ple = open_ple_pack(ck.model, path=ck.packs[key], log=_quiet)[2]
        rep = build_ple.verify(ple, ck.packs[key], 6)
        assert rep["ok"] and rep["rows_verified"] == 6 and not rep["mismatches"]
