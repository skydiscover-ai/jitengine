"""The cycle-aware victim rule (single-row decode): its lazily scaled EMA is the dense EMA, its key is
the cycle distance plus L (1 - p) / p, its victims are the full sort's, and it changes which slot
holds which record, never a record's bytes, while every protection still holds."""

from concurrent.futures import Future

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import OffloadMoeCache  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import victims as V  # noqa: E402

NAMES = [
    f"{p}.{t}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for t in ("weight", "scales", "biases")
]


class _Store:
    """Fills every byte of a record with a marker naming its id; demand and prefetch reads land
    at once."""

    def read_many(self, jobs):
        for gid, bufs in jobs:
            for b in bufs:
                b.view(np.uint8).fill(gid % 251 + 1)
        return 0.0

    def read_into(self, gid, bufs):
        self.read_many([(gid, bufs)])

    def submit_demand(self, gid, bufs, read):
        fut = Future()
        read(gid, bufs)
        fut.set_result(None)
        return fut

    def submit(self, gid, bufs):
        return self.submit_demand(gid, bufs, self.read_into)

    def stats(self):
        return {}


def _pool(slots, experts, layers, store=None, **options):
    fields, off = [], 0
    for n in NAMES:
        w = n.endswith("weight")
        fields.append(
            {
                "name": n,
                "dtype": "U32",
                "shape": [2, 8] if w else [2, 1],
                "bytes": 64 if w else 8,
                "offset": off,
            }
        )
        off += fields[-1]["bytes"]
    return OffloadMoeCache(
        store or _Store(),
        num_layers=layers,
        num_experts=experts,
        slots=slots,
        fields=fields,
        group_size=64,
        **options,
    )


def _holds(pool, slot):
    return int(np.array(pool.gate_proj.weight[int(slot)]).view(np.uint8).reshape(-1)[0]) - 1


def _p(pool):
    """p of every id from the pool's scaled state."""
    E = pool.num_experts
    lay = np.arange(pool.num_ids) // E
    return pool._cyc_w[: pool.num_ids] * pool._cyc_s[lay]


def _visits(rng, layers, experts, k, n):
    """`n` single-row decode visits: layers in turn, k distinct sorted ids each, skewed so the EMA
    has structure."""
    w = 1.0 / np.arange(1, experts + 1) ** 0.8
    w /= w.sum()
    out = []
    for t in range(n):
        lay = t % layers
        e = np.sort(rng.choice(experts, size=k, replace=False, p=w))
        out.append(e.astype(np.int64) + lay * experts)
    return out


@pytest.mark.parametrize("rescale", [V.CYC_RESCALE, 0.5])
def test_the_scaled_ema_is_the_dense_ema(monkeypatch, rescale):
    """Against the replay's dense update (p *= 1 - a over the layer, p[routed] += a), including
    the fold of a layer's scale into its values (forced often with `rescale` 0.5)."""
    monkeypatch.setattr(V, "CYC_RESCALE", rescale)
    L, E, k, N = 4, 16, 3, 64
    pool = _pool(slots=40, experts=E, layers=L)
    pool.cyc_tokens = N
    ref = np.zeros(L * E)
    a = 1.0 / N
    for g in _visits(np.random.default_rng(0), L, E, k, 600):
        pool._cyc_observe(g)
        lay = int(g[0]) // E
        ref[lay * E : (lay + 1) * E] *= 1.0 - a
        ref[g] += a
        np.testing.assert_allclose(_p(pool), ref, rtol=1e-12, atol=1e-300)
    assert pool.cyc_observations == 600
    assert pool._cyc_w[pool.num_ids] == 0.0  # the empty slot's sentinel never moves


def test_a_wave_or_a_repeated_visit_is_not_an_observation():
    pool = _pool(slots=40, experts=8, layers=4)
    pool.cyc_tokens = 64
    pool._cyc_observe(np.array([1, 2]))
    pool._cyc_observe(np.array([3]))  # the same layer again: a redo or split of one visit
    pool._cyc_observe(np.array([7, 9]))  # spans layers 0 and 1: a wave
    pool._cyc_observe(np.array([], dtype=np.int64))
    assert pool.cyc_observations == 1 and pool.cyc_layer == 0
    pool._cyc_observe(np.array([9]))
    assert pool.cyc_observations == 2 and pool.cyc_layer == 1


def test_the_key_is_the_cycle_distance_plus_the_tail():
    L, E = 4, 8
    pool = _pool(slots=32, experts=E, layers=L)
    pool.cyc_tokens = 4  # alpha 1/4
    for g in ([0, 1], [8], [16, 17], [24], [0]):  # layers 0, 1, 2, 3, 0: last observed is 0
        pool._cyc_observe(np.array(g, dtype=np.int64))
    p = _p(pool)
    np.testing.assert_allclose(
        p[[0, 1, 8, 16, 24]], [0.25 + 0.75 * 0.25, 0.75 * 0.25, 0.25, 0.25, 0.25]
    )
    ids = np.array([0, 1, 2, 8, 16, 24, -1], dtype=np.int64)
    far = pool._cyc_far_of(ids)
    d = np.array([4, 4, 4, 1, 2, 3])  # from a visit of layer 0: 0 comes round in 4, 1 in 1, ...
    with np.errstate(divide="ignore"):
        want = d + L * (1 - p[ids[:-1]]) / p[ids[:-1]]
    np.testing.assert_allclose(far[[0, 1, 3, 4, 5]], want[[0, 1, 3, 4, 5]])
    assert np.isinf(far[2]) and np.isinf(far[6])  # never routed, and an empty slot


@pytest.mark.parametrize("seed", range(6))
def test_victims_are_the_full_sorts(seed):
    """`_cyc_victims` (one head per layer) against the full sort over every slot
    (`cand[lexsort((cand, last, -far))][:k]`), with ties in the key (never routed ids are all
    +inf) and in `last`, and empty slots outside the free list protected (the full sort would
    take them first; the head rule never does)."""
    rng = np.random.default_rng(seed)
    L, E, S = 4, 32, 96
    pool = _pool(slots=S, experts=E, layers=L, landing_slots=0)
    pool.cyc_tokens = 64
    for g in _visits(rng, L, E, 4, 200):
        pool._cyc_observe(g)
    ids = rng.choice(L * E, size=S, replace=False).astype(np.int64)
    ids[rng.random(S) < 0.05] = -1
    pool.id_of_slot[:] = ids
    pool.slot_for_id[:] = -1
    live = np.flatnonzero(ids >= 0)
    pool.slot_for_id[ids[live]] = live.astype(np.int32)
    pool.last[:] = rng.integers(0, 6, size=S)
    pool.cyc_invalidate()  # residency was written by hand
    for k in (1, 2, 7, 40):
        prot = (rng.random(S) < 0.3) | (ids < 0)
        want = pool._cyc_victims_full(prot.copy(), k)
        got = pool._cyc_victims(prot.copy(), k)
        assert got.tolist() == want.tolist(), (k, got, want)


def test_the_heads_follow_every_publish_and_eviction():
    """Through `ensure` (observe, publish, evict) with the landing region and harvested prefetches,
    every pick equals the full sort's, so no publish, eviction or visit leaves a head stale."""
    L, E = 4, 60
    pool = _pool(slots=160, experts=E, layers=L, decay_tokens=32, reserve_slots=8, landing_slots=8)
    pool.cyc_tokens = 64
    visits = _visits(np.random.default_rng(3), L, E, 6, 300)
    for t, g in enumerate(visits):
        lay = int(g[0]) // E
        sp = pool.segments_begin(lay, g - lay * E)
        pool.segments_finish(sp)
        pool.reader_barrier()
        if t % 3 == 0 and t + 1 < len(visits):
            pool.prefetch(visits[t + 1])
        prot = pool._is_reserve.copy() | (pool.id_of_slot < 0)
        for k in (1, 3):
            got = pool._cyc_victims(prot.copy(), k).tolist()
            assert got == pool._cyc_victims_full(prot.copy(), k).tolist(), (t, k)
    assert pool.evictions_cyc > 0 and pool.prefetched > 0


def _drive(pool, visits, rule_n):
    """Single-row decode visits through `segments_begin` / `segments_finish` with the rule at
    `rule_n` (0 = decayed use), and a prefetch of the next visit's ids every third visit (harvested
    at the next `ensure`); returns the misses, after checking every returned slot holds its record."""
    pool.cyc_tokens = rule_n
    for t, g in enumerate(visits):
        E = pool.num_experts
        lay = int(g[0]) // E
        sp = pool.segments_begin(lay, g - lay * E)
        ((_pos, slots),) = pool.segments_finish(sp)
        assert [_holds(pool, s) for s in slots] == g.tolist()
        pool.reader_barrier()
        if t % 3 == 0 and t + 1 < len(visits):
            pool.prefetch(visits[t + 1])
    return pool.misses


def test_either_rule_serves_every_record_but_they_keep_different_ones():
    """The same visit sequence under decayed use and under the cycle-aware rule, with the landing
    region and the prefetch reserve: every slot an `ensure` returns holds the requested record
    under both (eviction never changes a value), and the two rules really evict differently."""
    L, E = 4, 60  # 240 ids: `_holds`'s marker is unique per id
    visits = _visits(np.random.default_rng(11), L, E, 6, 800)
    res = {}
    for n in (0, 64):
        pool = _pool(
            slots=160, experts=E, layers=L, decay_tokens=32, reserve_slots=8, landing_slots=8
        )
        assert pool.landing_n == 8 and pool.reserve_n == 8
        m = _drive(pool, visits, n)
        res[n] = (m, pool.slot_for_id >= 0, pool.evictions_cyc, pool.prefetch_used)
    assert res[0][2] == 0 and res[64][2] > 0  # the cyc rule chose victims only when armed
    assert res[64][3] > 0  # harvest remaps happened under the cyc rule
    assert res[0][0] != res[64][0] or (res[0][1] != res[64][1]).any()


def test_unarmed_the_pool_is_the_decayed_use_rule():
    """`cyc_tokens` 0: the decay rule picks, nothing is observed, and the cyc counters stay 0."""
    L, E = 4, 60
    visits = _visits(np.random.default_rng(5), L, E, 6, 300)
    a = _pool(slots=160, experts=E, layers=L, decay_tokens=32, landing_slots=8)
    _drive(a, visits, 0)
    assert a.cyc_observations == 0 and a.evictions_cyc == 0
    assert a.stats()["expert_pool_cyc_evict_solo"] == 0


def test_a_rule_that_picks_a_pending_reader_trips_the_reader_barrier_assertion(monkeypatch):
    """Negative control: a cyc rule that ignored the pending-reader protection is caught by the
    existing `_evict_victims` assertion before any byte moves."""
    pool = _pool(slots=16, experts=8, layers=4, landing_slots=0)
    pool.cyc_tokens = 64
    pool.ensure(np.arange(8, dtype=np.int64), overlap=(mx.zeros(1),))  # pending readers
    pool.ensure(np.arange(8, 16, dtype=np.int64))
    pool.reader_barrier()
    s = pool.ensure(np.arange(16, 24, dtype=np.int64), overlap=(mx.zeros(1),))
    bad = int(s[0])
    monkeypatch.setattr(type(pool), "_cyc_victims", lambda self, prot, k: np.full(k, bad))
    with pytest.raises(RuntimeError, match="pending reader"):
        pool.ensure(np.arange(24, 26, dtype=np.int64))


def test_a_harvest_rule_that_picks_a_protected_slot_raises(monkeypatch):
    """Negative control on the harvest path: a victim inside the harvest's protection set (a slot
    of the layer being fetched) raises instead of remapping it."""
    pool = _pool(slots=40, experts=8, layers=8, reserve_slots=4, landing_slots=0)
    assert pool.reserve_n == 4 and len(pool._free) == 36
    pool.cyc_tokens = 64
    for lay in range(5):  # 36 records fill every pool slot
        pool.ensure(np.arange(lay * 8, min(lay * 8 + 8, 36), dtype=np.int64))
    assert not pool._free
    pool.clock += 10 * pool.num_layers  # nothing is "used within the last token"
    assert pool.prefetch(np.array([40])) == 1
    want = np.array([0, 1], dtype=np.int64)
    keep = int(pool.slot_for_id[0])
    monkeypatch.setattr(type(pool), "_cyc_victims", lambda self, prot, k: np.full(k, keep))
    with pytest.raises(InvariantError, match="protected slot"):
        pool._harvest(want)


def test_the_rule_names_parse():
    assert V.evict_period("cyc64") == 64 and V.evict_period("CYC16") == 16
    assert V.evict_period("decay32") == 0
    for bad in ("cyc0", "cyc", "lru", "decay16", ""):
        with pytest.raises(ValueError):
            V.evict_period(bad)


def test_the_engine_arms_the_rule_around_single_row_decode_only(cpu_flashnext_engine):
    """The served default: `cyc_solo` 64, armed (`cyc_tokens`) only inside a b == 1 step, so a
    prompt's prefill never observes and nothing stays armed after a request; `decay32` turns it off,
    and an unknown rule is refused when the engine is built."""
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    e = cpu_flashnext_engine(pool_ext_slots=0, prefix_store_entries=0)
    pool = e.expert_pool
    assert pool.cyc_solo == 64 and pool.cyc_tokens == 0
    assert "solo decode cyc-64" in e.stats()["expert_pool_victim_policy"]
    L = pool.num_layers

    def counts():
        return pool.cyc_observations, e.stats()["solo_decode_steps"]

    o0, s0 = counts()
    e.generate(GenRequest(prompt_ids=list(range(40, 80)), max_tokens=1, stop_at_eos=False))
    o1, s1 = counts()
    e.generate(GenRequest(prompt_ids=list(range(90, 130)), max_tokens=6, stop_at_eos=False))
    o2, s2 = counts()
    assert pool.cyc_tokens == 0
    # one visit per layer per b == 1 step and none for the 40-token prefills (an observed prefill
    # would add its L layer visits on top)
    assert (o1 - o0, o2 - o1) == ((s1 - s0) * L, (s2 - s1) * L), (o0, o1, o2, s0, s1, s2, L)
    assert s2 - s1 >= 5
    off = cpu_flashnext_engine(
        pool_ext_slots=0, prefix_store_entries=0, expert_pool_evict="decay32"
    )
    off.generate(GenRequest(prompt_ids=list(range(90, 130)), max_tokens=6, stop_at_eos=False))
    assert off.expert_pool.cyc_solo == 0 and off.expert_pool.cyc_observations == 0
    with pytest.raises(ValueError, match="eviction rule"):
        cpu_flashnext_engine(expert_pool_evict="lru")
