"""The pool reading compact records from disk (`disk16`): every publication path leaves exactly the
bytes the staged encode would have written, with no staging, no encode and whole compact reads.

`codec.py` claims a `disk16` pool's reads land in the slots' six arrays by one read each and are
published by the same swap or remap as bf16 reads, so a published slot holds the compact record the
store served; and that this is byte for byte what the staged pool's publish encode writes for the
3,072,000 B-shaped record of the same id. Records are 4-bit affine quantized as the checkpoint is,
in a small Flash-Next-like geometry, one of them with an escaped (scale, bias) group.
"""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import OffloadMoeCache  # noqa: E402

SHAPES = ((64, 256), (64, 256), (256, 64))  # gate, up, down (N, K)
ESC_ID, ESC_GROUP = 7, 5  # this record's up_proj group 5 gets a bias no code represents


def _fields():
    out, off = [], 0
    for p, (n, k) in zip(("gate_proj", "up_proj", "down_proj"), SHAPES, strict=True):
        for t, dt, shape, nb in (
            ("weight", "U32", [n, k // 8], n * k // 2),
            ("scales", "BF16", [n, k // 32], n * k // 16),
            ("biases", "BF16", [n, k // 32], n * k // 16),
        ):
            out.append(
                {"name": f"{p}.{t}", "dtype": dt, "shape": shape, "bytes": nb, "offset": off}
            )
            off += nb
    return out


def _records(n, geom, seed=0):
    """`n` source records (u32 [n, record words]) with one escaped group, and the table they need."""
    parts = []
    for i, (rows, k) in enumerate(SHAPES):
        w = mx.random.normal((n, rows, k), key=mx.random.key(seed + i)) * k**-0.5
        q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=32, bits=4)
        s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        parts += [
            np.asarray(q).reshape(n, -1),
            np.asarray(s.view(mx.uint16)).reshape(n, -1).view(np.uint32),
            np.asarray(b.view(mx.uint16)).reshape(n, -1).view(np.uint32),
        ]
    flat = np.ascontiguousarray(np.concatenate(parts, axis=1))
    _w, _so, bo = geom.offsets[1]
    flat.view(np.uint16)[ESC_ID, 2 * bo + ESC_GROUP] = (
        0x3F80  # bias 1.0: q * scale is never near it
    )
    table: list = []
    f16 = flat.view(np.uint16)
    for (_w, so, bo), g in zip(geom.offsets, geom.groups, strict=True):
        p16.encode_np(f16[:, 2 * so : 2 * so + g], f16[:, 2 * bo : 2 * bo + g], table)
    assert len(table) == 1
    return flat, table


def _compact(flat, geom, table):
    """The staged pool's publish encode of every record, as the compact records a disk pack holds."""
    n = flat.shape[0]
    cf = geom.compact_fields()
    sets = [
        mx.zeros((n, *f["shape"]), dtype=mx.uint32 if f["dtype"] == "U32" else mx.uint16)
        for f in cf
    ]
    jobs = np.stack([np.arange(n), np.arange(n)], axis=1)
    err = p16.encode_records(
        geom,
        mx.array(flat),
        sets,
        None,
        n,
        p16.exc_array(table),
        len(table),
        jobs,
        mx.zeros((1,), dtype=mx.uint32),
    )
    mx.eval(err, *sets)
    assert int(err.item()) == 0
    return np.concatenate([np.asarray(a).reshape(n, -1).view(np.uint8) for a in sets], axis=1)


class _Store:
    """Serves whole records into the destination buffers; futures complete before they return."""

    def __init__(self, recs: np.ndarray):
        self.recs, self.reads, self.nbufs = recs.view(np.uint8), 0, set()

    def read_into(self, gid, bufs):
        self.nbufs.add(len(bufs))
        at = 0
        for b in bufs:
            v = np.asarray(b).view(np.uint8).reshape(-1)
            v[:] = self.recs[gid, at : at + v.size]
            at += v.size
        assert at == self.recs.shape[1], "a read fills exactly one whole record"
        self.reads += 1
        return at

    def read_many(self, jobs):
        for gid, bufs in jobs:
            self.read_into(gid, bufs)
        return 0.0

    def submit_demand(self, gid, bufs, read):
        fut = Future()
        read(gid, bufs)
        fut.set_result(None)
        return fut

    def submit(self, gid, bufs):
        fut = Future()
        self.read_into(gid, bufs)
        fut.set_result(None)
        return fut

    def stats(self):
        return {}


def _pools(slots, layers=2, experts=40, **opts):
    """(disk16 pool, staged pool, source records, geom, table) over the same ids."""
    fields = _fields()
    geom = p16.Geometry(fields, 32, 4)
    flat, table = _records(layers * experts, geom)
    disk = OffloadMoeCache(
        _Store(_compact(flat, geom, table)),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=32,
        codec=(geom, table),
        disk16=True,
        **opts,
    )
    staged = OffloadMoeCache(
        _Store(flat),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=32,
        codec=(geom, table),
        **opts,
    )
    return disk, staged, flat, geom, table


def _slot_bytes(pool, gid):
    slot = int(pool.slot_for_id[gid])
    if slot >= pool.ann0:
        fs, r = pool.fields_ann, slot - pool.ann0
    elif slot >= pool.slots_main:
        fs, r = pool.fields_ext, slot - pool.slots_main
    else:
        fs, r = pool.fields, slot
    return b"".join(np.asarray(f.tensor[r]).tobytes() for f in fs)


def _check_resident(pool, flat, geom, table):
    """Every published slot holds its record exactly (weights verbatim, words decode to the bits)."""
    mx.eval(pool.encode_dep)
    gids = np.flatnonzero(pool.slot_for_id >= 0)
    assert gids.size
    for g in gids.tolist():
        by = dict(zip([f["name"] for f in geom.compact_fields()], [None] * 6, strict=True))
        raw = np.frombuffer(_slot_bytes(pool, g), dtype=np.uint8)
        at = 0
        for f in geom.compact_fields():
            by[f["name"]] = raw[at : at + f["bytes"]]
            at += f["bytes"]
        rec, r16 = flat[g], flat[g].view(np.uint16)
        for p, name in enumerate(("gate_proj", "up_proj", "down_proj")):
            ow, so, bo = geom.offsets[p]
            n = geom.groups[p]
            assert np.array_equal(
                by[f"{name}.weight"].view(np.uint32), rec[ow : ow + n * geom.wpg]
            ), (g, p)
            s, b = p16.decode_np(by[f"{name}.words"].view(np.uint16), table)
            assert np.array_equal(s, r16[2 * so : 2 * so + n]), (g, p)
            assert np.array_equal(b, r16[2 * bo : 2 * bo + n]), (g, p)


def _drive(pool):
    """Demand fetches through the landing region, a synchronous wave and a harvested prefetch."""
    for layer, ids in ((0, [1, 2]), (1, [3]), (0, [5, 6, ESC_ID]), (1, [7, 8]), (0, [1, 9])):
        sp = pool.segments_begin(layer, np.asarray(ids))
        pool.segments_finish(sp)
        pool.reader_barrier()
    pool.ensure(np.arange(10, 35, dtype=np.int64))  # 25 misses: past the landing region
    pool.reader_barrier()
    if pool.reserve_n:
        assert pool.prefetch(np.asarray([60, 61, 62], dtype=np.int64)) == 3
        pool.harvest_ready()


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_compact_records_land_in_the_slots_with_no_staging_and_no_encode():
    disk, _staged, flat, geom, table = _pools(60, landing_slots=2, reserve_slots=4)
    assert disk.codec == "pack16" and disk.disk16 and disk.landing == "direct"
    assert disk.staging_n == 0 and disk.staging is None
    assert disk.record_bytes == disk.slot_bytes == geom.slot_bytes < geom.record_bytes
    _drive(disk)
    assert (
        disk.landing_fetches > 0
        and disk.landing_fallbacks > 0
        and disk.prefetch_used + disk.prefetched > 0
    )
    _check_resident(disk, flat, geom, table)
    assert disk.encodes == 0 and disk.encoded_records == 0
    assert disk.store.nbufs == {6}  # one read per record, into the slot's six arrays
    s = disk.stats()
    assert s["pack16_disk"] is True and s["expert_pool_staging_bytes"] == 0
    assert s["expert_miss_bytes"] == s["expert_misses"] * geom.slot_bytes
    assert disk.prefetch_bytes == disk.prefetched * geom.slot_bytes


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_every_slot_holds_what_the_staged_publish_encode_writes():
    disk, staged, _flat, _geom, _table = _pools(60, landing_slots=2, reserve_slots=4)
    _drive(disk)
    _drive(staged)
    assert staged.encodes > 0 and not staged.stats()["pack16_disk"]
    both = np.flatnonzero((disk.slot_for_id >= 0) & (staged.slot_for_id >= 0))
    assert ESC_ID in both.tolist() and both.size >= 20
    for g in both.tolist():
        assert _slot_bytes(disk, g) == _slot_bytes(staged, g), g


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_the_extension_and_the_annex_take_direct_reads_and_settle_them_before_they_go():
    disk, _s, flat, geom, table = _pools(
        60, landing_slots=0, reserve_slots=4, ext_slots=12, ann_slots=6
    )
    assert all(f.host is not None for f in disk.fields + disk.fields_ext)
    assert disk.regrow_ann() == 6 and all(f.host is not None for f in disk.fields_ann)
    # an extension slot in the reserve (a harvest's victim can be one): the next prefetch reads into it
    ext_slot = next(s for s in disk._free if disk.slots_main <= s < disk.ann0)
    disk._free.remove(ext_slot)
    disk._is_reserve[ext_slot] = True
    disk._reserve_free.append(ext_slot)
    disk.ensure(np.arange(0, 40, dtype=np.int64))
    disk.reader_barrier()
    disk.ensure(np.arange(40, 73, dtype=np.int64))  # 73 records: every free slot, the annex's too
    disk.reader_barrier()
    _check_resident(disk, flat, geom, table)
    assert (disk.slot_for_id[np.arange(73)] >= disk.ann0).sum() == 6
    assert disk.prefetch(np.asarray([75], dtype=np.int64)) == 1
    assert disk._inflight[75][0] == ext_slot
    disk.release_ext()  # the annex goes first, then the read bound for the extension is settled
    assert not disk._inflight and not disk.ann_active and disk.prefetch_wasted == 1
    assert disk.encodes == 0


def test_disk16_needs_the_codec():
    fields = _fields()
    with pytest.raises(ValueError, match="pack16 codec"):
        OffloadMoeCache(
            _Store(np.zeros((80, 1), np.uint8)),
            num_layers=2,
            num_experts=40,
            slots=60,
            fields=fields,
            group_size=32,
            disk16=True,
        )
