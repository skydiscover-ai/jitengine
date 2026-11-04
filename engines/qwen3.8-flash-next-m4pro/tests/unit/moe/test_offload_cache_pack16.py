"""The pool with pack16 compact slots: every publication path leaves the record's exact bytes.

`codec.py` claims reads still land whole (in staging) and each publication (pointer swap, prefetch
remap, synchronous fill) encodes the staged record into the slot it publishes, so a published
slot's weights are the record's and its words decode to the record's (scale, bias) bits; and that
a read that is not published gives its staging back. Records are 4-bit affine quantized as the
checkpoint is, in a small Flash-Next-like geometry.
"""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import OffloadMoeCache  # noqa: E402

SHAPES = ((64, 256), (64, 256), (256, 64))  # gate, up, down (N, K)


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


class _Records:
    """`n` quantized records in the pack layout, u32 [n, record_bytes / 4]."""

    def __init__(self, n, seed=0):
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
        self.flat = np.ascontiguousarray(np.concatenate(parts, axis=1))


class _Store:
    """Serves the records; demand and speculative reads complete before their future returns."""

    def __init__(self, recs, fail_demand=None):
        self.recs, self.fail_demand, self.reads = recs, fail_demand, 0

    def read_into(self, gid, bufs):
        dst = (
            np.concatenate([np.asarray(b).view(np.uint8).reshape(-1) for b in bufs])
            if len(bufs) > 1
            else None
        )
        assert dst is None, "a pack16 read lands in one staging record"
        bufs[0].view(np.uint32).reshape(-1)[:] = self.recs.flat[gid]
        self.reads += 1

    def read_many(self, jobs):
        for gid, bufs in jobs:
            self.read_into(gid, bufs)
        return 0.0

    def submit_demand(self, gid, bufs, read):
        fut = Future()
        if self.fail_demand is not None:
            fut.set_exception(self.fail_demand)
            return fut
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


def _pool(slots, layers=2, experts=40, store=None, **opts):
    fields = _fields()
    geom = p16.Geometry(fields, 32, 4)
    recs = _Records(layers * experts)
    table: list = []
    flat16 = recs.flat.view(np.uint16)
    for (_w, so, bo), n in zip(geom.offsets, geom.groups, strict=True):
        p16.encode_np(flat16[:, 2 * so : 2 * so + n], flat16[:, 2 * bo : 2 * bo + n], table)
    pool = OffloadMoeCache(
        store or _Store(recs),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=32,
        codec=(geom, table),
        **opts,
    )
    return pool, recs, geom, table


def _check_resident(pool, recs, geom, table):
    """Every published slot holds its record exactly (weights verbatim, words decode to the bits)."""
    mx.eval(pool.encode_dep)
    gids = np.flatnonzero(pool.slot_for_id >= 0)
    assert gids.size
    for g in gids.tolist():
        slot = int(pool.slot_for_id[g])
        if slot >= getattr(pool, "ann0", 1 << 30):  # the annex (a third array set)
            fs, r = pool.fields_ann, slot - pool.ann0
        else:
            fs = pool.fields if slot < pool.slots_main else pool.fields_ext
            r = slot if slot < pool.slots_main else slot - pool.slots_main
        by = {f.name: np.asarray(f.tensor[r]) for f in fs}
        rec = recs.flat[g]
        r16 = rec.view(np.uint16)
        for p, name in enumerate(("gate_proj", "up_proj", "down_proj")):
            ow, so, bo = geom.offsets[p]
            n = geom.groups[p]
            assert np.array_equal(by[f"{name}.weight"].reshape(-1), rec[ow : ow + n * geom.wpg]), (
                g,
                p,
            )
            s, b = p16.decode_np(by[f"{name}.words"].reshape(-1), table)
            assert np.array_equal(s, r16[2 * so : 2 * so + n]), (g, p)
            assert np.array_equal(b, r16[2 * bo : 2 * bo + n]), (g, p)


def _staging_idle(pool):
    return not pool._staged and len(pool._stage_free) == pool.staging_n


def test_compact_slots_are_smaller_and_the_staging_set_is_whole_records():
    pool, _r, geom, _t = _pool(60, landing_slots=2)
    assert pool.codec == "pack16" and pool.slot_bytes == geom.slot_bytes < pool.record_bytes
    assert {f.name for f in pool.fields} == {
        f"{p}.{t}" for p in ("gate_proj", "up_proj", "down_proj") for t in ("weight", "words")
    }
    assert pool.staging.shape == (pool.staging_n, pool.record_bytes // 4)
    assert not hasattr(pool.gate_proj, "scales") and not hasattr(pool.gate_proj, "biases")
    assert pool.stats()["expert_pool_bytes"] == pool.slots * geom.slot_bytes


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_demand_reads_published_by_swap_hold_their_records():
    pool, recs, geom, table = _pool(60, landing_slots=2)
    assert pool.landing == "direct" and pool.landing_n == 2
    for layer, ids in ((0, [1, 2]), (1, [3]), (0, [5, 6]), (1, [7, 8]), (0, [1, 9])):
        sp = pool.segments_begin(layer, np.asarray(ids))
        pool.segments_finish(sp)
        pool.reader_barrier()
    assert pool.landing_fetches > 0
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_a_wave_too_large_for_the_landing_region_fills_synchronously_in_chunks():
    pool, recs, geom, table = _pool(60, landing_slots=2)
    pool.ensure(np.arange(25, dtype=np.int64))  # 25 misses: past the region and the sync staging
    pool.reader_barrier()
    assert pool.landing_fallbacks == 1 and pool.encoded_records == 25
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_prefetches_are_encoded_when_harvested():
    pool, recs, geom, table = _pool(60, landing_slots=0, reserve_slots=4)
    assert pool.reserve_n == 4
    assert pool.prefetch(np.asarray([40, 41, 42], dtype=np.int64)) == 3
    assert len(pool._staged) == 3
    pool.harvest_ready()
    assert np.all(pool.slot_for_id[[40, 41, 42]] >= 0)
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)


def test_a_failed_landing_read_returns_its_staging():
    recs = _Records(80)
    pool, _r, _geom, _table = _pool(
        60, landing_slots=2, store=_Store(recs, fail_demand=ValueError("bad read"))
    )
    with pytest.raises(ValueError):
        pool.ensure(np.asarray([4, 5], dtype=np.int64))
    assert _staging_idle(pool)
    assert np.all(pool.slot_for_id < 0)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_releasing_the_extension_settles_reads_bound_for_it():
    pool, recs, geom, table = _pool(60, landing_slots=0, reserve_slots=4, ext_slots=12)
    pool.ensure(np.arange(40, 52, dtype=np.int64))
    pool.reader_barrier()
    # a harvest's victim joins the reserve, so a reserve slot can be an extension slot: make one
    ext_slot = next(s for s in pool._free if s >= pool.slots_main)
    pool._free.remove(ext_slot)
    pool._is_reserve[ext_slot] = True
    pool._reserve_free.append(ext_slot)
    assert pool.prefetch(np.asarray([60], dtype=np.int64)) == 1
    assert pool._inflight[60][0] == ext_slot
    pool.release_ext()
    assert 60 not in pool._inflight and pool.slot_for_id[60] < 0
    assert _staging_idle(pool)
    pool.regrow_ext()
    pool.ensure(np.arange(0, 30, dtype=np.int64))
    pool.reader_barrier()
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_releasing_the_annex_settles_reads_bound_for_it():
    """A prefetch in flight into an annex reserve slot and
    an annex landing slot (a pointer swap or a harvest's victim can put either there) are settled at
    release, never published into the released set; the reserve and landing keep their sizes from
    main slots; the main pool and the extension keep their records; a regrown annex serves exact
    bytes."""
    pool, recs, geom, table = _pool(44, landing_slots=2, reserve_slots=4, ext_slots=6, ann_slots=6)
    assert pool.ann0 == pool.slots_main + pool.ext_slots and pool.slots == pool.ann0 + 6
    # /stats reports the plan's pool (main + extension), never the annex's borrowed id range
    st = pool.stats()
    assert st["expert_pool_slots"] == st["pool_slots"] == pool.ann0 == 50
    assert st["expert_pool_bytes"] == 50 * pool.slot_bytes and st["expert_pool_ann_slots"] == 6
    assert st["expert_pool_ann_live"] == 0 and st["expert_pool_ann_live_bytes"] == 0
    assert not pool.ann_active and pool._disabled[pool.ann0 :].all()
    assert pool.regrow_ann() == 6 and pool.ann_active and not pool._disabled[pool.ann0 :].any()
    st = pool.stats()
    assert st["expert_pool_slots"] == 50 and st["expert_pool_ann_live_bytes"] == 6 * pool.slot_bytes
    n_res, n_land = len(pool._reserve_free), len(pool._landing_free)
    for lst in (pool._reserve_free, pool._landing_free):  # one annex slot into each
        s = next(x for x in pool._free if x >= pool.ann0)
        pool._free.remove(s)
        pool._is_reserve[s] = True
        lst.append(s)
    ann_res = pool._reserve_free[-1]
    assert pool.reserve_n > 0 and pool.landing_n > 0
    for lo, hi in ((0, 20), (20, 40), (40, 49)):  # past what main and extension hold
        pool.ensure(np.arange(lo, hi, dtype=np.int64))
        pool.reader_barrier()
    assert (pool.slot_for_id >= pool.ann0).any()  # the annex took records
    _check_resident(pool, recs, geom, table)
    assert pool.prefetch(np.asarray([75], dtype=np.int64)) == 1
    assert pool._inflight[75][0] == ann_res
    kept = {
        g: int(pool.slot_for_id[g])
        for g in np.flatnonzero((pool.slot_for_id >= 0) & (pool.slot_for_id < pool.ann0)).tolist()
    }
    dropped = pool.release_ann()
    assert dropped > 0 and not pool.ann_active and pool._disabled[pool.ann0 :].all()
    assert 75 not in pool._inflight and pool.slot_for_id[75] < 0
    assert all(s < pool.ann0 for s in pool._reserve_free + pool._landing_free + pool._free)
    assert len(pool._reserve_free) == n_res + 1 and len(pool._landing_free) == n_land + 1
    assert np.all(pool.slot_for_id[pool.slot_for_id >= 0] < pool.ann0)
    # main and extension records stay where they were, but for the two the hand-back evicted to
    # refill the reserve and landing region (as `release_ext` does)
    moved = {g for g, sl in kept.items() if int(pool.slot_for_id[g]) != sl}
    assert len(moved) <= 2 and all(pool.slot_for_id[g] < 0 for g in moved)
    assert _staging_idle(pool)
    _check_resident(pool, recs, geom, table)
    pool.regrow_ann()
    for lo, hi in (
        (49, 60),
        (60, 70),
        (70, 80),
    ):  # records not resident: the regrown annex takes some
        pool.ensure(np.arange(lo, hi, dtype=np.int64))
        pool.reader_barrier()
    assert (pool.slot_for_id >= pool.ann0).any()
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)
    # the extension's release takes the annex first (it is lent only beside a live extension)
    pool.release_ext()
    assert not pool.ann_active and not pool.ext_active
    assert np.all(pool.slot_for_id[pool.slot_for_id >= 0] < pool.slots_main)
    pool.regrow_ext()
    assert (
        pool._disabled[pool.ann0 :].all() and not pool.ann_active
    )  # regrowing the extension lends nothing
    pool.ensure(np.arange(40, 80, dtype=np.int64))
    pool.reader_barrier()
    assert np.all(pool.slot_for_id[pool.slot_for_id >= 0] < pool.ann0)
    _check_resident(pool, recs, geom, table)


class _SlowDemand(_Store):
    """`_Store` whose demand reads land 50 ms after they are submitted, on a worker thread."""

    def __init__(self, recs):
        super().__init__(recs)
        from concurrent.futures import ThreadPoolExecutor

        self.ex = ThreadPoolExecutor(max_workers=2)

    def submit_demand(self, gid, bufs, read):
        import time

        def late():
            time.sleep(0.05)
            read(gid, bufs)

        return self.ex.submit(late)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_a_correction_built_on_a_prefetch_and_a_landing_read_encodes_each_once_landed():
    """`_split_inflight`: one layer needs a record still in flight from the miss chain and one
    new read. The correction is built on both before either lands (`pred_dep`); the harvest
    publishes the prefetch while the landing read is still in flight, so the two must be separate
    encodes, or the landing slot would be encoded before its bytes arrive."""
    recs = _Records(80)
    pool, _r, geom, table = _pool(60, landing_slots=2, reserve_slots=4, store=_SlowDemand(recs))
    pool.ensure(np.arange(0, 10, dtype=np.int64))
    pool.reader_barrier()
    assert pool.prefetch(np.asarray([45], dtype=np.int64)) == 1  # layer 1, expert 5: in flight
    sp = pool.segments_begin(1, np.asarray([5, 6]))  # 45 in flight, 46 a new landing read
    assert sp["pend"].get("split_inflight")
    dep = pool.pred_dep(sp["pred_slots"])  # what `_correction` builds on
    pool.segments_finish(sp)
    mx.eval(dep)
    assert int(dep.item()) == 0  # each encode found its read's stamp
    assert np.all(pool.slot_for_id[[45, 46]] >= 0)
    _check_resident(pool, recs, geom, table)
    assert _staging_idle(pool)


@pytest.mark.usefixtures("requires_mlx")  # the pack16 encode is a Metal kernel
def test_an_encode_evaluated_before_its_read_is_published_raises():
    """The bad order: a `pred_dep` graph (what a correction is built on) evaluated before
    `segments_finish`. Without the read stamp the encode would copy whatever the staging record
    held before, a valid earlier record, with an error count of 0, and the slot would keep that
    record's bytes under the new expert id. With it, the encode writes nothing and counts
    `STALE_UNIT`, and the pool's next check raises."""
    recs = _Records(80)
    pool, _r, _geom, _table = _pool(60, landing_slots=2, store=_SlowDemand(recs))
    pool.ensure(np.arange(0, 10, dtype=np.int64))  # every staging record now holds an older read
    pool.reader_barrier()
    sp = pool.segments_begin(1, np.asarray([5, 6]))  # two new landing reads, in flight
    dep = pool.pred_dep(sp["pred_slots"])
    mx.eval(dep)  # the bad order
    stale, lossy = p16.stale_and_lossy(int(dep.item()))
    assert stale >= 1 and lossy == 0
    pool.segments_finish(sp)
    with pytest.raises(InvariantError, match="before their read was published"):
        for _ in range(2):  # the check reads the count as of its previous call
            pool._check_encodes()
