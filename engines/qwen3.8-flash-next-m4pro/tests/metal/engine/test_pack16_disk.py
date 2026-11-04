"""The engine reading pack16 records from disk (SKYP16X1, checkpoint/p16_format.py) on the tiny model.

The writer (`build_p16`) encodes every expert record with the pool's own publish encode, verifies it
against the checkpoint and finishes the file; the engine finds it beside the expert pack's directory
and, when its header validates, reads compact records straight into the slots. Claims checked here:
the file is what the writer says (manifest hashes, the engine's validation, each record equal to the
host encoder's words for the shards' bits); every token and digested logits row equals the same
engine's with the file unused (`pack16_disk` off), through misses, the extension, the annex and the
MTP verify pass; the expert counters keep the io_accounting identities with `expert_io` naming the
file, its 16-bit record size and F_NOCACHE (the e2e fixtures have no pack16 file, so they see
only the fallback); an absent, partial or mismatched file falls back; and a (scale, bias) pair
outside the escape table fails the build.
"""

import hashlib
import json
import os
import shutil
import struct

import numpy as np
import pytest


def _metal() -> bool:
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001
        return False


# every test here runs a custom Metal kernel; a mark, since the module-scoped fixture would
# otherwise build its pack16 file before a function-scoped skip could run
pytestmark = pytest.mark.skipif(not _metal(), reason="no MLX/Metal")

N = 24
OPTS = dict(
    prefix_grid_tokens=64,
    prefix_min_divergence_tokens=16,
    landing_slots=2,
    prefetch_reserve=2,
    pool_ext_slots=4,
    ext_scratch_tokens=48,
)
ANNEX = 6
WRONG_EVERY = 3
MIN_FREE = 1 << 28  # the volume floor for these KB-sized files (the real build keeps 8 GiB)


class _HeadSlot:
    keys = None

    def __init__(self):
        self.offset, self.idx_bk = 0, {}


class ScriptedHead:
    """test_mtp_verify's stand-in for the checkpoint's head: drafts the reference token at the next
    position, wrong at every `WRONG_EVERY`-th."""

    def __init__(self, ref):
        self.ref = list(ref)

    def new_state(self, capacity):
        return {"slot": _HeadSlot(), "n": 0}

    def draft(self, st, h_pre, tokens):
        import mlx.core as mx

        st["n"] += len(tokens)
        k = st["n"]
        tok = self.ref[k] if k < len(self.ref) else 0
        if k % WRONG_EVERY == 0:
            tok += 1
        return mx.array([tok], dtype=mx.int32)


def _layout(tmp_path, model, source):
    """tmp/pack/<name>.experts (a copy of `source`) and where the engine looks for its pack16 file."""
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import ExpertScan

    os.makedirs(tmp_path / "pack", exist_ok=True)
    pack = str(tmp_path / "pack" / os.path.basename(source))
    shutil.copyfile(source, pack)
    return pack, pf.pack16_path_for(pack, ExpertScan(model))


def _build(model, pack, out, *extra):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_p16

    return build_p16.main(
        ["--model", model, "--source", pack, "--out", out, "--min-free-gib", "0.25", *extra]
    )


@pytest.fixture(scope="module")
def p16_tiny(tiny_flashnext, tmp_path_factory):
    """(model, expert pack copy, pack16 file) built once for the module by the package's writer."""
    model, source = tiny_flashnext()
    tmp = tmp_path_factory.mktemp("p16")
    pack, out = _layout(tmp, model, source)
    assert _build(model, pack, out, "--chunk", "next") == 0  # one chunk, then the rest: it resumes
    assert os.path.exists(out + ".partial") and not os.path.exists(out)
    assert _build(model, pack, out, "--all") == 0
    return model, pack, out


def _engine(model, pack, monkeypatch, annex=0, **opts):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 16  # of 32 records, so reads happen all the way through
        plan.store_spill_slots = annex
        return plan

    with monkeypatch.context() as mp:
        mp.setattr(fne, "plan_budget", capped)
        return fne.create_engine(
            model, expert_pack_path=pack, log_requests=False, **dict(OPTS, **opts)
        )


def _req(prompt, n=N):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=prompt, max_tokens=n, stop_at_eos=False)


def _go(e, prompts, digest, monkeypatch, mtp_ref=None):
    holder = {}

    def load():
        e.mtp, e.mtp_gen = ScriptedHead(holder["ref"]), e.mtp_gen + 1
        e.mtp_loads += 1

    e._mtp_load = load
    e.mtp_on, e.mtp = mtp_ref is not None, None
    toks, rows = [], []
    for i, p in enumerate(prompts):
        if mtp_ref is not None:
            holder["ref"] = mtp_ref[i]
        path = f"{digest}.{i}"
        if os.path.exists(path):
            os.remove(path)
        monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", path)
        toks.append(e.generate(_req(p)).token_ids)
        monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST")
        rows.append(
            [ln for ln in open(path).read().split("\n") if ln] if os.path.exists(path) else []
        )
    e.mtp_on, e.mtp = False, None
    return toks, rows


PROMPTS = [
    list(range(1000, 1100)) + [5, 6, 7],
    list(range(2000, 2090)) + [1, 2],
    list(range(1000, 1100)) + [5, 6, 7, 8, 9],
]


def test_the_writer_leaves_a_verified_file_the_engine_accepts(p16_tiny, requires_mlx):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.expert_reader import CheckpointScan
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import read_header_block
    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16
    from jitengine_qwen3_8_flash_next_m4pro.moe.expert_store import ExpertStore
    from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.codec import census

    model, pack, out = p16_tiny
    assert not os.path.exists(out + ".partial")
    man = json.load(open(out + ".manifest.json"))
    raw = open(out, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == man["sha256"] and len(raw) == man["size"]
    meta = read_header_block(out, pf.MAGIC_P16)
    scan = CheckpointScan(model)
    geom = p16.Geometry(scan.fields(), 32, 4)
    src = ExpertStore.from_header(
        pack, read_header_block(pack, b"FNXEPK01"), workers=1, nocache=False
    )
    table = census(src, scan.num_records, geom)
    src.close()
    assert pf.validate_p16(out, scan, pack, geom, table) == (True, "ok")
    assert meta["record_bytes"] == geom.slot_bytes < scan.record_bytes and meta["escapes"] == table
    assert man["verification"]["records"] == scan.num_records == meta["records"]
    stride, d0, layer_bytes = (
        meta["record_stride"],
        meta["data_offset"],
        scan.num_experts * meta["record_stride"],
    )
    assert man["per_layer_sha256"] == [
        hashlib.sha256(raw[d0 + i * layer_bytes : d0 + (i + 1) * layer_bytes]).hexdigest()
        for i in range(scan.num_layers)
    ]
    # every record is the host encoder's words for the shards' own bits, weights verbatim
    shards: dict = {}

    def piece(path, off, n):
        if path not in shards:
            shards[path] = open(path, "rb").read()
        return shards[path][off : off + n]

    for layer in range(scan.num_layers):
        for e in range(scan.num_experts):
            rec = np.frombuffer(
                raw,
                dtype=np.uint8,
                count=stride,
                offset=d0 + (layer * scan.num_experts + e) * stride,
            )
            at = 0
            for p in p16.PROJ:
                w = [scan.slice_of(layer, e, f"{p}.{t}") for t in ("weight", "scales", "biases")]
                got = [piece(*x) for x in w]
                assert rec[at : at + len(got[0])].tobytes() == got[0]
                at += len(got[0])
                words, _ = p16.encode_np(
                    np.frombuffer(got[1], np.uint16), np.frombuffer(got[2], np.uint16), list(table)
                )
                assert rec[at : at + words.nbytes].tobytes() == words.tobytes(), (layer, e, p)
                at += words.nbytes
            assert at == meta["record_bytes"] and not rec[at:].any()


def test_reading_pack16_from_disk_changes_no_token_and_no_row(
    p16_tiny, requires_mlx, tmp_path, monkeypatch
):
    model, pack, out = p16_tiny
    ref_e = _engine(model, pack, monkeypatch, annex=ANNEX, pack16_disk="off")
    cand = _engine(model, pack, monkeypatch, annex=ANNEX)
    rs, cs = ref_e.stats(), cand.stats()
    assert rs["pack16_disk"] is False and "pack16_disk=off" in rs["pack16_disk_why"]
    assert cs["pack16_disk"] is True and cs["expert_io"]["path"] == out
    meta_bytes = json.load(open(out + ".manifest.json"))["header"]["record_bytes"]
    assert cs["expert_io"]["record_bytes"] == meta_bytes == cand.expert_pool.slot_bytes
    assert cs["expert_io"]["nocache"] is True and cs["expert_pool_staging_bytes"] == 0
    assert rs["expert_io"]["record_bytes"] == ref_e.expert_pool.geom.record_bytes > meta_bytes
    assert cand.expert_pool.ann_slots == ANNEX and cand.expert_pool.ext_slots > 0

    ref, ref_rows = _go(ref_e, PROMPTS, str(tmp_path / "ref"), monkeypatch)
    s0 = cand.stats()
    got, got_rows = _go(cand, PROMPTS, str(tmp_path / "cand"), monkeypatch)
    s1 = cand.stats()
    assert got == ref and all(len(r) == N for r in got_rows) and got_rows == ref_rows
    # the io_accounting identities over the window, in the file's own record size
    d = {
        k: s1[k] - s0[k]
        for k in (
            "expert_lookups",
            "expert_hits",
            "expert_misses",
            "expert_dedup_saved",
            "expert_miss_bytes",
            "expert_prefetch_bytes",
        )
    }
    dio = s1["expert_io"]["bytes_requested"] - s0["expert_io"]["bytes_requested"]
    dretry = s1.get("expert_retry_bytes", 0) - s0.get("expert_retry_bytes", 0)
    assert s0["expert_counter_identity_ok"] and s1["expert_counter_identity_ok"]
    assert d["expert_hits"] + d["expert_misses"] + d["expert_dedup_saved"] == d["expert_lookups"]
    assert d["expert_misses"] > 0 and d["expert_miss_bytes"] == d["expert_misses"] * meta_bytes
    assert dio == d["expert_miss_bytes"] + d["expert_prefetch_bytes"] + dretry
    assert s1["expert_pool_encodes"] == 0 and ref_e.stats()["expert_pool_encodes"] > 0
    assert s1["expert_pool_ann_regrows"] > s0["expert_pool_ann_regrows"]

    # the MTP verify pass, through the same requests
    ref_m, ref_m_rows = _go(ref_e, PROMPTS, str(tmp_path / "refm"), monkeypatch, mtp_ref=ref)
    v0 = cand.mtp_verifies
    got_m, got_m_rows = _go(cand, PROMPTS, str(tmp_path / "candm"), monkeypatch, mtp_ref=ref)
    assert got_m == ref_m == ref and got_m_rows == ref_m_rows == ref_rows and cand.mtp_verifies > v0


def _rewrite_header(path, change):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import read_header_block

    meta = change(read_header_block(path, pf.MAGIC_P16))
    with open(path, "r+b") as f:
        f.write(pf.p16_header_block(meta))


@pytest.mark.parametrize("case", ["absent", "partial", "other-table", "other-source", "truncated"])
def test_a_missing_or_mismatched_file_falls_back(
    p16_tiny, requires_mlx, tmp_path, monkeypatch, case
):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf

    model, pack, out = p16_tiny
    pack2, out2 = _layout(tmp_path, model, pack)
    if case != "absent":
        os.makedirs(os.path.dirname(out2), exist_ok=True)
        shutil.copyfile(out, out2)
        # the copy names its own source pack, so it is valid until the case's one change below
        _rewrite_header(out2, lambda m: dict(m, source=pf.source_identity(pack2)))
        e0 = _engine(model, pack2, monkeypatch)
        assert e0.stats()["pack16_disk"] is True, e0.stats()["pack16_disk_why"]
    if case == "partial":
        _rewrite_header(out2, lambda m: dict(m, complete=False))
    elif case == "other-table":
        _rewrite_header(out2, lambda m: dict(m, escapes=m["escapes"] + [0x3F803F80]))
    elif case == "other-source":
        _rewrite_header(out2, lambda m: dict(m, source=dict(m["source"], header_sha256="0" * 64)))
    elif case == "truncated":
        os.truncate(out2, os.path.getsize(out2) - 4096)
    e = _engine(model, pack2, monkeypatch)
    s = e.stats()
    assert s["pack16_disk"] is False and s["expert_io"]["path"] == pack2, s["pack16_disk_why"]
    assert s["expert_io"]["record_bytes"] == e.expert_pool.geom.record_bytes
    want = {
        "absent": "absent",
        "partial": "completion marker",
        "other-table": "escapes",
        "other-source": "source pack identity",
        "truncated": "truncated",
    }[case]
    assert want in s["pack16_disk_why"]


def _patched_checkpoint(tmp_path, model):
    """A copy of the tiny checkpoint with one expert group's bias set to a value no pack16 code holds."""
    dst = str(tmp_path / "model-esc")
    shutil.copytree(model, dst)
    name = "language_model.model.layers.1.mlp.switch_mlp.up_proj.biases"
    shard = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"][name]
    path = os.path.join(dst, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    lo, hi = hdr[name]["data_offsets"]
    per_expert = (hi - lo) // hdr[name]["shape"][0]
    with open(path, "r+b") as f:
        f.seek(8 + n + lo + 3 * per_expert + 2 * 5)  # expert 3, group 5
        f.write(struct.pack("<H", 0x3F80))  # bias 1.0
    return dst


def test_a_pair_outside_the_escape_table_fails_the_build(
    tiny_flashnext, requires_mlx, tmp_path, monkeypatch
):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import expert_writer
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.p16_writer import P16Build
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import ExpertScan, PackError

    model, _ = tiny_flashnext()
    esc_model = _patched_checkpoint(tmp_path, model)
    os.makedirs(tmp_path / "pack")
    pack = str(tmp_path / "pack" / "esc.experts")
    expert_writer.build_pack(ExpertScan(esc_model), pack, log=lambda *_: None)
    scan = ExpertScan(esc_model)
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import p16_format as pf

    out = pf.pack16_path_for(pack, scan)
    with pytest.raises(PackError, match="outside the codec and the escape table"):
        b = P16Build(scan, pack, out, table=[], log=lambda *_: None, min_free=MIN_FREE)
        b.open()
        for layer in b.todo():
            b.build_layer(layer)
    shutil.rmtree(os.path.dirname(out))  # the failed build's partial
    b = P16Build(
        scan, pack, out, log=lambda *_: None, min_free=MIN_FREE
    )  # the census table holds the pair
    assert len(b.table) == 1
    b.open()
    for layer in b.todo():
        b.build_layer(layer)
    man = b.finalize()
    assert man["escape_uses"] == [1] and man["escaped_groups"][0][:5] == [1, 3, "up_proj", 5, 0]
    # the escaped group, served from the file, gives every token the encode path gives
    ref = _engine(esc_model, pack, monkeypatch, pack16_disk="off")
    cand = _engine(esc_model, pack, monkeypatch)
    assert cand.stats()["pack16_disk"] is True and cand.stats()["expert_pool_escapes"] == 1
    prompts = [list(range(100, 160)), list(range(3000, 3050)) + [4]]
    a, ra = _go(ref, prompts, str(tmp_path / "r"), monkeypatch)
    c, rc = _go(cand, prompts, str(tmp_path / "c"), monkeypatch)
    assert a == c and ra == rc
