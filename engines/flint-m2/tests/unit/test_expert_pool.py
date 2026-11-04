"""The expert pool (offload mode) against the resident banks, on a small synthetic checkpoint: the MoE layer's output
from pooled experts is bit-identical to the one from stacked expert tensors, through both gathers (decode's unsorted
one and prefill's sorted one), both landings, and evictions. Runs on whatever device MLX defaults to (the CPU on the
Linux CI, the GPU on a Mac)."""

from types import SimpleNamespace

import numpy as np
import pytest

from jitengine_flint_m2 import expert_pool
from jitengine_flint_m2.expert_pool import ExpertPool
from jitengine_flint_m2.loader import QLinear
from jitengine_flint_m2.model import model as graph

H, INTER, E, LAYERS = 256, 128, 16, (2, 3)
PROJ = (("gate_proj", INTER, H), ("up_proj", INTER, H), ("down_proj", H, INTER))


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def make_checkpoint(tmp_path, seed=0):
    """A raw file holding each routed layer's nine [E, ...] tensors, its tensor index (config.tensor_index's form) and
    the same tensors as numpy arrays."""
    rng = np.random.default_rng(seed)
    path = tmp_path / "experts.bin"
    index, arrays, blobs, off = {}, {}, [], 0
    for layer in LAYERS:
        for proj, rows, cols in PROJ:
            parts = {
                "weight": ("U8", rng.integers(0, 256, size=(E, rows, cols // 2), dtype=np.uint8)),
                "scales": ("BF16", _bf16_bits(rng.normal(0, 0.02, (E, rows, cols // 64)))),
                "biases": ("BF16", _bf16_bits(rng.normal(0, 0.01, (E, rows, cols // 64)))),
            }
            for suffix, (dt, arr) in parts.items():
                name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                raw = arr.tobytes()
                index[name] = (str(path), dt, list(arr.shape), off, len(raw))
                arrays[name] = arr
                blobs.append(raw)
                off += len(raw)
    path.write_bytes(b"".join(blobs))
    cfg = SimpleNamespace(
        moe_layers=LAYERS, n_experts=E, n_layers=4, bits=4, group_size=64, top_k=6, rms_eps=1e-6
    )
    return cfg, index, arrays


def stacked(arrays, layer):
    """The resident form: one [E, ...] QLinear per projection."""
    import mlx.core as mx

    out = []
    for proj, _, _ in PROJ:
        p = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        w = mx.array(arrays[p + ".weight"].view(np.uint32))
        s, b = (mx.array(arrays[f"{p}.{k}"]).view(mx.bfloat16) for k in ("scales", "biases"))
        out.append(QLinear(w, s, b, 4, 64))
    return tuple(out)


def layer_weights(arrays, layer, seed=1):
    import mlx.core as mx

    rng = np.random.default_rng(seed + layer)
    router = mx.array(_bf16_bits(rng.normal(0, 0.05, (E, H)))).view(mx.bfloat16)
    shared = tuple(QLinear(q.w[0], q.s[0], q.b[0], 4, 64) for q in stacked(arrays, layer))
    return SimpleNamespace(
        index=layer, dense=None, router=router, experts=stacked(arrays, layer), shared=shared
    )


def bits(a) -> np.ndarray:
    import mlx.core as mx

    return np.array(a.view(mx.uint16))


@pytest.mark.parametrize("slots", [E, 3 * E])
@pytest.mark.parametrize("landing", ["auto", "scatter"])
def test_pooled_moe_is_bit_identical(tmp_path, monkeypatch, landing, slots):
    import mlx.core as mx

    if landing == "scatter":
        monkeypatch.setattr(expert_pool, "_writable_host_view", lambda a: None)
    cfg, index, arrays = make_checkpoint(tmp_path)
    # slots = E: every switch of layer evicts; 3 E: more slots than experts, so a gather over the pool's arrays
    # would see other rows per expert than the banks' (MLX picks the sorted gather's kernel from that, on Metal)
    pool = ExpertPool(cfg, index, slots=slots, log=lambda m: None)
    assert landing == "auto" or pool.landing == "scatter"
    pool.selfcheck()
    resident = SimpleNamespace(cfg=cfg, pool=None)
    pooled = SimpleNamespace(cfg=cfg, pool=pool)
    rng = np.random.default_rng(7)
    try:
        for step in range(6):
            layer = LAYERS[step % 2]
            lw = layer_weights(arrays, layer)
            for n in (1, 16, 64):  # 6 rows: the unsorted gather; 96 and 384 rows: the sorted one
                x = mx.array(_bf16_bits(rng.normal(0, 1, (1, n, H)))).view(mx.bfloat16)
                want = graph._mlp(resident, lw, x)
                got = graph._mlp(pooled, SimpleNamespace(**{**vars(lw), "experts": None}), x)
                mx.eval(want, got)
                assert np.array_equal(bits(want), bits(got)), (step, layer, n)
        st = pool.stats()
        assert st["misses"] > 0 and st["hits"] > 0 and (st["evictions"] > 0) == (slots == E), st
        assert st["read_bytes"] == st["misses"] * pool.record_bytes
        assert st["landing"] == pool.landing and pool.resident() <= min(slots, len(LAYERS) * E)
    finally:
        pool.close()


def test_pool_bookkeeping(tmp_path):
    """Slots map both ways, protected slots are never victims, a failed read publishes nothing."""
    cfg, index, _ = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="at least"):
        ExpertPool(cfg, index, slots=E - 1, log=lambda m: None)
    pool = ExpertPool(cfg, index, slots=2 * E, log=lambda m: None)
    try:
        a = np.arange(E, dtype=np.int64) + LAYERS[0] * E
        b = np.arange(E, dtype=np.int64) + LAYERS[1] * E
        sa, sb = pool.ensure(a), pool.ensure(b)
        assert len(set(sa) | set(sb)) == 2 * E and pool.resident() == 2 * E
        assert np.array_equal(pool.ensure(a), sa) and pool.stats()["hits"] == E
        assert all(pool.id_of_slot[s] == g for s, g in zip(sa, a))
        broken = SimpleNamespace(**{**vars(cfg)})
        pool2 = ExpertPool(broken, index, slots=E, log=lambda m: None)
        pool2.ensure(a[:4])

        def fail(fd, dst, off):
            raise OSError("injected")

        pool2._read_piece = fail
        with pytest.raises(OSError, match="injected"):
            pool2.ensure(b[:4])
        assert (pool2.slot_for_id[b[:4]] < 0).all() and pool2.resident() == 4
        pool2.close()
    finally:
        pool.close()


def test_fill_reads_each_slice_once(tmp_path):
    """A miss layer is read as its records' nine slices, each one job, the bytes a record read would give."""
    cfg, index, _ = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=2 * E, log=lambda m: None)
    try:
        calls = []
        real = pool._read_piece
        pool._read_piece = lambda fd, dst, off: (calls.append((off, len(dst))), real(fd, dst, off))
        g = np.array([LAYERS[0] * E + 3, LAYERS[1] * E + 5, LAYERS[1] * E + 9], dtype=np.int64)
        slots = pool.ensure(g)
        assert len(calls) == 9 * len(g) and len(set(calls)) == len(calls)
        assert sum(n for _, n in calls) == len(g) * pool.record_bytes == pool.stats()["read_bytes"]
        for gid, s in zip(g, slots):
            want = [np.empty(f.nbytes, dtype=np.uint8) for f in pool.fields]
            pool._read_record(int(gid), [memoryview(w) for w in want])
            for f, w in zip(pool.fields, want):
                assert np.array_equal(
                    np.array(f.host[s] if f.host is not None else f.tensor[int(s)])
                    .view(np.uint8)
                    .reshape(-1),
                    w,
                )
    finally:
        pool.close()


@pytest.mark.parametrize(
    "slots", [E, E + 5, 2 * E - 3]
)  # 2 * E would hold every record and never evict
def test_heap_victims_follow_use_last_slot(tmp_path, monkeypatch, slots):
    """Victims are the free slots first, then the unprotected published slots in (use, last, slot) order, the seed's
    lexsort order, through decays (DECAY_LAYERS = 1: a halving every n_layers lookups) and a rebase."""
    monkeypatch.setattr(expert_pool, "DECAY_LAYERS", 1)
    monkeypatch.setattr(expert_pool, "REBASE_EPOCHS", 7)
    cfg, index, _ = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=slots, log=lambda m: None)
    pool._fill = lambda gids, s: None
    shadow_use = np.zeros(slots)  # the seed's float counts, halved in place
    checked = [0]
    real = pool._victims

    def victims(k, protected):
        free = list(pool.free)
        want = [free.pop() for _ in range(min(k, len(free)))]
        if k > len(want):
            pub = [s for s in range(slots) if pool.id_of_slot[s] >= 0 and s not in set(protected)]
            pub.sort(key=lambda s: (shadow_use[s], pool.last[s], s))
            want += pub[: k - len(want)]
            checked[0] += 1
        got = real(k, protected)
        assert got == want
        return got

    pool._victims = victims
    rng = np.random.default_rng(3)
    ep = 0
    try:
        for step in range(400):
            if (
                pool.clock + 1
            ) % cfg.n_layers == 0:  # ensure() decays (and rebases) before it picks victims
                shadow_use *= 0.5
                ep += 1
                if ep >= 7:  # rebased: the seed's counts below 2**-REBASE_EPOCHS drop to 0 here too
                    ep = 0
                    shadow_use = np.floor(shadow_use)
            layer = LAYERS[step % 2]
            e = np.unique(
                rng.choice(
                    E, int(rng.integers(1, 9)), p=np.linspace(2, 1, E) / np.linspace(2, 1, E).sum()
                )
            )
            g = e.astype(np.int64) + layer * E
            before = pool.slot_for_id[g].copy()
            out = pool.ensure(g)
            shadow_use[out[before < 0]] = 0.0
            shadow_use[out] += 1.0
        assert checked[0] > 50 and len(pool.heap) <= 4 * slots + 8
    finally:
        pool.close()


def test_prefill_tables_add_no_use_counts(tmp_path):
    """layer_table (prefill's sorted gather) refreshes last use but counts nothing; slots_for (decode) counts.
    So after a prompt sweeps a layer, the experts decode keeps using outrank the sweep's one-off experts."""
    import mlx.core as mx

    cfg, index, _ = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=E + 8, log=lambda m: None)
    try:
        L0, L1 = LAYERS
        pool.slots_for(
            L0, mx.array(np.array([[[1, 2, 3, 4, 5, 6]]], dtype=np.uint32))
        )  # decode counts 1..6
        hot = np.arange(1, 7) + L0 * E
        assert all(pool.use[s] == 1 << pool.epoch for s in pool.slot_for_id[hot])
        pool.layer_table(
            L0, mx.array(np.arange(12, dtype=np.uint32).reshape(1, 2, 6))
        )  # prefill sweep 0..11
        assert all(pool.use[s] == 1 << pool.epoch for s in pool.slot_for_id[hot]), (
            "a prefill table added counts"
        )
        cold = np.array([0, 7, 8, 9, 10, 11]) + L0 * E
        assert all(pool.use[s] == 0 and pool.last[s] == pool.clock for s in pool.slot_for_id[cold])
        pool.ensure(np.arange(E, dtype=np.int64) + L1 * E)  # 16 misses, 12 free: 4 victims
        assert (pool.slot_for_id[hot] >= 0).all(), "a decode-hot expert lost to a prefill one-off"
        assert int((pool.slot_for_id[cold] < 0).sum()) == 4
    finally:
        pool.close()


@pytest.mark.parametrize("n", [1, 2, 8])
def test_annex_moe_is_bit_identical(tmp_path, n):
    """With the host annex open and the base pool one layer wide, lookups alternate layers, so records move base ->
    annex -> base: the MoE output keeps the resident layer's bits, and the drive is read only for records in neither
    (read_bytes == misses * record_bytes, annex hits counted as hits)."""
    import mlx.core as mx

    cfg, index, arrays = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=E, log=lambda m: None, annex_max=2 * E)
    assert pool.open_annex(10**12) == 2 * E
    resident = SimpleNamespace(cfg=cfg, pool=None)
    pooled = SimpleNamespace(cfg=cfg, pool=pool)
    rng = np.random.default_rng(11)
    try:
        for step in range(24):
            layer = LAYERS[step % 2]
            lw = layer_weights(arrays, layer)
            x = mx.array(_bf16_bits(rng.normal(0, 1, (1, n, H)))).view(mx.bfloat16)
            want = graph._mlp(resident, lw, x)
            got = graph._mlp(pooled, SimpleNamespace(**{**vars(lw), "experts": None}), x)
            mx.eval(want, got)
            assert np.array_equal(bits(want), bits(got)), (step, layer, n)
        st = pool.stats()
        assert st["annex_hits"] > 0 and st["annex_saves"] > 0, st
        assert st["read_bytes"] == st["misses"] * pool.record_bytes
        assert st["hits"] + st["misses"] >= st["annex_hits"] + st["misses"]
    finally:
        pool.close()


@pytest.mark.parametrize("annex", [0, 2 * E])
@pytest.mark.parametrize("slots", [E, 3 * E])
def test_speculated_moe_is_bit_identical(tmp_path, slots, annex):
    """Decode (one row) with speculation: the graph built on device_slots is kept when every record was resident;
    otherwise only the missing records' rows are recomputed on their real slots and spliced in (all of them when
    every record missed). All keep the resident layer's bits, all three paths run, a kept graph's slots and a partial
    rebuild's resident slots equal slots_for's, and the drive is read only for misses."""
    import mlx.core as mx

    cfg, index, arrays = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=slots, log=lambda m: None, annex_max=annex, speculate=True)
    if annex:
        assert pool.open_annex(10**12) == annex
    resident = SimpleNamespace(cfg=cfg, pool=None)
    pooled = SimpleNamespace(cfg=cfg, pool=pool)
    kept = partial = full = 0
    check = pool.check_slots

    def counted(layer, ids):
        nonlocal kept, partial, full
        spec = np.array(pool.device_slots(layer, ids)).reshape(-1)
        out = check(layer, ids)
        e = np.array(ids).astype(np.int64).reshape(-1) + layer * pool.n_experts
        if out is None:
            kept += 1
            assert np.array_equal(spec, pool.slot_for_id[e]), "a kept graph ran on stale slots"
        else:
            slots, missing = out
            assert np.array_equal(slots.reshape(-1), pool.slot_for_id[e])
            hit = ~missing.reshape(-1)
            assert np.array_equal(spec[hit], slots.reshape(-1)[hit]), "a resident row moved"
            if hit.any():
                partial += 1
            else:
                full += 1
        return out

    pool.check_slots = counted
    rng = np.random.default_rng(12)
    try:
        for step in range(64):
            layer = LAYERS[
                (step // 16) % 2
            ]  # runs of one layer, long enough to make it resident when slots == E
            lw = layer_weights(arrays, layer)
            x = mx.array(_bf16_bits(rng.normal(0, 1, (1, 1, H)))).view(mx.bfloat16)
            want = graph._mlp(resident, lw, x)
            got = graph._mlp(pooled, SimpleNamespace(**{**vars(lw), "experts": None}), x)
            mx.eval(want, got)
            assert np.array_equal(bits(want), bits(got)), (step, layer)
        assert kept > 0 and partial > 0 and full > 0, (kept, partial, full)
        st = pool.stats()
        assert st["read_bytes"] == st["misses"] * pool.record_bytes
    finally:
        pool.close()


def test_annex_bookkeeping(tmp_path):  # noqa: PLR0915  (one scenario, step by step)
    """The annex opens only with a row to spare, never twice; a victim's bytes go to the annex, come back exact, and
    leave its row free (exclusive); a full annex reuses its least recently spilled row; a failed drive read publishes
    nothing and loses no annex record not being copied back; prefill tables refuse an open annex; close drops every
    record, unmaps, and the pool reads from the drive again; scatter landing never opens one."""
    import mlx.core as mx

    cfg, index, _arrays = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=E, log=lambda m: None, annex_max=4)
    a = np.arange(E, dtype=np.int64) + LAYERS[0] * E
    b = np.arange(E, dtype=np.int64) + LAYERS[1] * E

    def record(g):
        want = [np.empty(f.nbytes, dtype=np.uint8) for f in pool.fields]
        pool._read_record(int(g), [memoryview(w) for w in want])
        return b"".join(w.tobytes() for w in want)

    def slot_bytes(s):
        return b"".join(v.tobytes() for v in pool._record_views(int(s)))

    try:
        assert pool.open_annex(pool.record_bytes - 1) == 0 and pool.annex_slots == 0
        assert pool.open_annex(10**12) == 4 and pool.open_annex(10**12) == 0, (
            "a second open while open"
        )
        pool.ensure(a)  # base full with layer 0
        m0, rb0 = pool.misses, pool.read_bytes
        pool.ensure(b[:3])  # 3 victims of layer 0 spill
        assert pool.annex_saves == 3 and len(pool._row_of) == 3
        spilled = list(pool._row_of)
        for g in spilled:
            assert pool.slot_for_id[g] < 0 and bytes(pool._annex[pool._row_of[g]]) == record(g)
        pool.ensure(
            b[3:5]
        )  # 2 more victims: the annex (4 rows) is full, so the oldest spilled row is reused
        assert len(pool._row_of) == 4 and spilled[0] not in pool._row_of and pool.annex_saves == 5
        assert pool.misses == m0 + 5 and pool.read_bytes == rb0 + 5 * pool.record_bytes
        back = list(pool._row_of)[:2]
        h0, m1, rb1 = pool.annex_hits, pool.misses, pool.read_bytes
        s = pool.ensure(
            np.array(sorted(back), dtype=np.int64)
        )  # both from the annex, no drive read
        assert pool.annex_hits == h0 + 2 and pool.misses == m1 and pool.read_bytes == rb1
        for g, sl in zip(sorted(back), s):
            assert pool.slot_for_id[g] == sl and slot_bytes(sl) == record(g)
            assert g not in pool._row_of, "an annex hit kept its row (the annex is exclusive)"
        # a failed drive read: the misses' slots go back to the free list, nothing is published
        pool._read_piece = lambda fd, dst, off: (_ for _ in ()).throw(OSError("injected"))
        fresh = np.array(
            [g for g in a if pool.slot_for_id[g] < 0 and g not in pool._row_of][:2], dtype=np.int64
        )
        held = dict(pool._row_of)
        with pytest.raises(OSError, match="injected"):
            pool.ensure(fresh)
        assert (pool.slot_for_id[fresh] < 0).all() and len(pool.free) == 2
        del pool._read_piece
        for g in pool._row_of:  # every record still in the annex is that record's bytes
            assert bytes(pool._annex[pool._row_of[g]]) == record(g)
        assert set(held) <= set(pool._row_of), "a failed read lost an annex record"
        with pytest.raises(RuntimeError, match="annex open"):
            pool.layer_table(LAYERS[0], mx.array(np.arange(6, dtype=np.uint32).reshape(1, 1, 6)))
        n_held = len(pool._row_of)
        assert pool.close_annex() == n_held and pool.annex_slots == 0 and pool._annex is None
        assert pool.close_annex() == 0
        gone = np.array([g for g in a if pool.slot_for_id[g] < 0][:1], dtype=np.int64)
        m2 = pool.misses
        pool.ensure(gone)
        assert pool.misses == m2 + 1, "closed: a former annex record is read from the drive"
        st = pool.stats()
        assert st["annex_opens"] == 1 and st["annex_closes"] == 1 and st["annex_slots"] == 0
        assert st["read_bytes"] == st["misses"] * pool.record_bytes
    finally:
        pool.close()


def test_annex_needs_direct_landing(tmp_path, monkeypatch):
    monkeypatch.setattr(expert_pool, "_writable_host_view", lambda a: None)
    cfg, index, _ = make_checkpoint(tmp_path)
    pool = ExpertPool(cfg, index, slots=E, log=lambda m: None, annex_max=E)
    try:
        assert pool.landing == "scatter" and pool.open_annex(10**12) == 0
    finally:
        pool.close()
