"""The MTP verify pass on the tiny model, greedy: every emitted token and every digested logits row
is the solo path's, with accepted and rejected drafts, with misses and evictions, and with the PLE
layer.

`Qwen4ExpForCausalLM.verify2` promises each of its two rows bit-identical to a solo step's at that
position and `rollback1` the state after the first; `DecodingMixin._mtp_round` draws from those rows
with the row's own sampler and logs exactly the rows it draws from, at the solo step numbers. A draft
only decides how many positions a pass carries, so a scripted head (right on most positions, wrong
on every `WRONG_EVERY`-th) stands in for the checkpoint's, which the tiny model does not have. The
digest rows are sha256 of the full fp32 rows, so equal digests are full-vector equality.
"""

import os

import pytest

N = 24  # tokens per request
WRONG_EVERY = 3  # the scripted head drafts a wrong token at every third position


def _req(prompt, n=N):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=prompt, max_tokens=n, stop_at_eos=False)


class _HeadSlot:
    """What `Qwen4ExpForCausalLM._snap` reads of the head's KV slot."""

    keys = None

    def __init__(self):
        self.offset, self.idx_bk = 0, {}


class ScriptedHead:
    """`MTPHead`'s interface: after consuming its entries, drafts the reference token at the next
    position (generated token `n`, n = entries consumed), wrong at every `WRONG_EVERY`-th."""

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


def _rows(path):
    return [ln for ln in open(path).read().split("\n") if ln] if os.path.exists(path) else []


def _run(e, prompt, digest, monkeypatch, n=N):
    if os.path.exists(digest):
        os.remove(digest)
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", digest)
    out = e.generate(_req(prompt, n))
    monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST")
    return out.token_ids, _rows(digest)


def _check_against_solo(e, prompts, tmp_path, monkeypatch, n=N):
    """Each prompt solo (no head), then with the scripted head: tokens and digest rows equal."""
    assert e.model.verify_ok(), "the tiny model's solo step is not the lookahead step"
    assert e.mtp is None
    verifies = rejected = 0
    for i, p in enumerate(prompts):
        e.mtp = None
        ref, ref_rows = _run(e, p, str(tmp_path / f"solo{i}.digest"), monkeypatch, n)
        assert [int(r.split()[0]) for r in ref_rows] == list(range(n))
        v0, a0 = e.mtp_verifies, e.mtp_accepted
        e.mtp, e.mtp_gen = ScriptedHead(ref), e.mtp_gen + 1
        got, got_rows = _run(e, p, str(tmp_path / f"mtp{i}.digest"), monkeypatch, n)
        e.mtp = None
        assert got == ref, (i, got, ref)
        bad = next((j for j, (x, y) in enumerate(zip(got_rows, ref_rows)) if x != y), None)
        assert len(got_rows) == len(ref_rows) and bad is None, (i, bad, len(got_rows))
        verifies += e.mtp_verifies - v0
        rejected += (e.mtp_verifies - v0) - (e.mtp_accepted - a0)
    assert verifies > 0 and rejected > 0, (verifies, rejected)
    return verifies, rejected


@pytest.fixture
def mtp_engine(flashnext_engine):
    """A tiny engine without the extension pool, so `_ext_policy` never loads or drops a head and
    the test's scripted head stays in place."""

    def make(**options):
        options.setdefault("pool_ext_slots", 0)
        options.setdefault("prefix_store_entries", 0)
        return flashnext_engine(**options)

    return make


def test_verified_tokens_and_rows_are_the_solo_paths(
    mtp_engine, requires_mlx, tmp_path, monkeypatch
):
    e = mtp_engine()
    _check_against_solo(e, [[5, 6, 7], list(range(100, 140))], tmp_path, monkeypatch)


def test_with_misses_and_evictions(mtp_engine, requires_mlx, tmp_path, monkeypatch):
    """A pool of 12 of the 32 records, so verify passes miss, evict and redo layers (the plan's pool
    is capped, as in test_engine_consistency)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 12
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)
    e = mtp_engine()
    assert e.expert_pool.slots == 12
    m0 = e.stats()["expert_misses"]
    _check_against_solo(e, [list(range(100, 130)), list(range(2000, 2070))], tmp_path, monkeypatch)
    assert e.stats()["expert_misses"] > m0


def test_a_one_token_request_and_the_last_token_are_solo_steps(
    mtp_engine, requires_mlx, tmp_path, monkeypatch
):
    """max_tokens 1 and 2 (the client's warmups): a pass never draws past max_tokens."""
    e = mtp_engine()
    for n in (1, 2, 3):
        e.mtp = None
        ref, ref_rows = _run(e, [9, 10, 11], str(tmp_path / f"s{n}.digest"), monkeypatch, n)
        e.mtp, e.mtp_gen = ScriptedHead(ref), e.mtp_gen + 1
        got, got_rows = _run(e, [9, 10, 11], str(tmp_path / f"m{n}.digest"), monkeypatch, n)
        e.mtp = None
        assert (got, got_rows) == (ref, ref_rows) and len(got_rows) == n


def test_with_the_ple_layer_each_position_is_read_once(
    tiny_flashnext_ple, requires_mlx, tmp_path, monkeypatch
):
    """The PLE fixture: tokens and rows as solo, and the PLE store reads one row set per kept
    position plus one per rejected draft (rebuilt layers reuse the two-entry memo)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine

    model, pack, ple = tiny_flashnext_ple()
    e = create_engine(
        model,
        expert_pack_path=pack,
        ple_pack_path=ple,
        ple_io_nocache=False,
        log_requests=False,
        pool_ext_slots=0,
        prefix_store_entries=0,
    )
    assert e.model.verify_ok()
    p = list(range(300, 340))
    e.mtp = None
    c0 = e.ple_store.counters()["ple_calls"]
    ref, ref_rows = _run(e, p, str(tmp_path / "solo.digest"), monkeypatch)
    c1 = e.ple_store.counters()["ple_calls"]
    v0, a0 = e.mtp_verifies, e.mtp_accepted
    e.mtp, e.mtp_gen = ScriptedHead(ref), e.mtp_gen + 1
    got, got_rows = _run(e, p, str(tmp_path / "mtp.digest"), monkeypatch)
    c2 = e.ple_store.counters()["ple_calls"]
    e.mtp = None
    assert (got, got_rows) == (ref, ref_rows)
    rejected = (e.mtp_verifies - v0) - (e.mtp_accepted - a0)
    assert rejected > 0
    n_ple = len(e.model.ple_layers)
    assert (c2 - c1) - (c1 - c0) == n_ple * rejected, (c1 - c0, c2 - c1, rejected)


def test_the_verify_pass_issues_its_ple_reads_at_round_start_and_reads_the_same_rows(
    tiny_flashnext_ple, requires_mlx, tmp_path, monkeypatch
):
    """The PLE fixture with the scripted head, the PLE switches off (options `ple_prefetch`, `ple_host_context`
    off: the base's PLE path) and on: the same tokens, digest rows and PLE reads (calls, rows, bytes),
    and with it on the verify passes' rows were issued at round start (`ple_prefetched_calls`)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine

    model, pack, ple = tiny_flashnext_ple()
    e = create_engine(
        model,
        expert_pack_path=pack,
        ple_pack_path=ple,
        ple_io_nocache=False,
        log_requests=False,
        pool_ext_slots=0,
        prefix_store_entries=0,
    )
    p = list(range(300, 340))
    e.mtp = None
    ref, ref_rows = _run(e, p, str(tmp_path / "solo.digest"), monkeypatch)
    keys = ("ple_calls", "ple_rows_requested", "ple_rows_read", "ple_bytes", "ple_prefetched_calls")
    out = {}
    for on in (False, True):
        e.model.set_ple_switches(on, on)
        assert {k: e.stats()[k] for k in ("ple_prefetch", "ple_host_context")} == {
            "ple_prefetch": on,
            "ple_host_context": on,
        }
        c0, v0 = e.ple_store.counters(), e.mtp_verifies
        e.mtp, e.mtp_gen = ScriptedHead(ref), e.mtp_gen + 1
        got, got_rows = _run(e, p, str(tmp_path / f"mtp{int(on)}.digest"), monkeypatch)
        e.mtp = None
        c1 = e.ple_store.counters()
        out[on] = (got, got_rows, {k: c1[k] - c0[k] for k in keys}, e.mtp_verifies - v0)
    for on in (False, True):
        assert (out[on][0], out[on][1]) == (ref, ref_rows), on
    off, on_ = out[False][2], out[True][2]
    assert {k: off[k] for k in keys[:4]} == {k: on_[k] for k in keys[:4]}, (off, on_)
    verifies = out[True][3]
    assert verifies > 0 and off["ple_prefetched_calls"] == 0
    # a verify pass issues each position's rows early unless the memo already holds that history
    assert 0 < on_["ple_prefetched_calls"] <= 2 * verifies, (on_, verifies)


def test_rollback_restores_every_layer_to_the_first_position(flashnext_engine, requires_mlx):
    """verify2 with a wrong draft, then rollback1: every layer's state (DeltaNet conv and ssm, the
    KV offset, keys and block-key cache) equals a solo step's after the first position, and the
    pass's first row equals that solo step's logits."""
    import mlx.core as mx
    import numpy as np

    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState

    e = flashnext_engine(pool_ext_slots=0, prefix_store_entries=0)
    m = e.model
    prompt = mx.array([list(range(50, 90))], dtype=mx.int32)

    def fresh():
        st = RequestState(m.cfg, prompt.shape[1] + 1, 8, dtype=m.dtype)
        m.forward(prompt, st, logits="none")
        st.ensure_room(2)
        return st

    a, b = fresh(), fresh()
    t = 7
    ref = m.forward(mx.array([[t]], dtype=mx.int32), a, logits="last")
    lg, _hid, snaps = m.verify2(t, 12345, b)
    m.rollback1(b, snaps)
    mx.eval(ref, lg)
    assert np.asarray(ref).tobytes() == np.asarray(lg[0:1]).tobytes()
    assert a.tokens == b.tokens
    for sa, sb in zip(a.layers, b.layers):
        if hasattr(sa, "keys"):
            assert sa.offset == sb.offset and sorted(sa.idx_bk) == sorted(sb.idx_bk)
            n = sa.offset
            for f in ("keys", "values", "idx"):
                x, y = getattr(sa, f)[:, :, :n], getattr(sb, f)[:, :, :n]
                assert mx.array_equal(x, y).item(), f
        else:
            assert mx.array_equal(sa.conv, sb.conv).item() and mx.array_equal(sa.ssm, sb.ssm).item()
    # and the next position decodes as solo from there
    t2 = int(mx.argmax(ref, axis=-1).item())
    r2 = m.forward(mx.array([[t2]], dtype=mx.int32), a, logits="last")
    g2 = m.forward(mx.array([[t2]], dtype=mx.int32), b, logits="last")
    mx.eval(r2, g2)
    assert np.asarray(r2).tobytes() == np.asarray(g2).tobytes()


@pytest.mark.parametrize("ext", [0, 4])
def test_with_the_landing_region_the_chain_reserve_and_the_extension(
    flashnext_engine, requires_mlx, tmp_path, monkeypatch, ext
):
    """The served machinery on a small pool: demand reads through the landing region (pointer
    swaps), the miss chain's prefetches into the reserve, and (ext) the single-row extension,
    released for each uncovered prefill and regrown after it, the head released and reloaded with
    it (`_mtp_load` installs the scripted head)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 24  # of 32 records: a region and a reserve need 8x their size in slots
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)
    e = flashnext_engine(
        landing_slots=2,
        prefetch_reserve=2,
        pool_ext_slots=ext,
        ext_scratch_tokens=16,
        prefix_store_entries=0,
    )
    assert e.model.verify_ok() and e.expert_pool.landing_n == 2 and e.expert_pool.reserve_n == 2
    holder = {}

    def load():
        e.mtp, e.mtp_gen = ScriptedHead(holder["ref"]), e.mtp_gen + 1
        e.mtp_loads += 1

    e._mtp_load = load
    verifies = rejected = 0
    n = 48
    for i, p in enumerate([list(range(100, 140)), list(range(2000, 2070)), list(range(7, 60))]):
        e.mtp_on, e.mtp = False, None
        ref, ref_rows = _run(e, p, str(tmp_path / f"solo{i}.digest"), monkeypatch, n)
        holder["ref"] = ref
        e.mtp_on = True
        if not ext:  # no extension, so no `_ext_policy` load: install it here
            load()
        v0, a0, l0 = e.mtp_verifies, e.mtp_accepted, e.mtp_loads
        got, got_rows = _run(e, p, str(tmp_path / f"mtp{i}.digest"), monkeypatch, n)
        assert got == ref, (i, got, ref)
        bad = next((j for j, (x, y) in enumerate(zip(got_rows, ref_rows)) if x != y), None)
        assert len(got_rows) == len(ref_rows) and bad is None, (i, bad, len(got_rows))
        if ext:
            assert e.mtp_loads > l0  # the head was released for the prefill and loaded again
        verifies += e.mtp_verifies - v0
        rejected += (e.mtp_verifies - v0) - (e.mtp_accepted - a0)
    s = e.stats()
    assert verifies > 0 and rejected > 0 and s["expert_misses"] > 0
    assert e.expert_pool.landing_fetches > 0


def _state_fields(st) -> list:
    """Every field of a request's decode state, as comparable bytes: tokens; per KV slot the offset,
    the valid keys, values and indexer keys and the whole block-key cache (keys and values); per
    DeltaNet slot conv and ssm; per PLE slot context and conv. (A PLE slot's `cached` memo and a KV
    slot's buffers past `offset` are not state: the memo is keyed by the full n-gram, and nothing
    reads past `offset`.)"""
    import mlx.core as mx
    import numpy as np

    def b(a):
        return np.asarray(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes()

    def cache(d):
        out = []
        for k in sorted(d, key=str):
            v = d[k]
            parts = v[1] if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], tuple) else v
            out.append(
                (
                    str(k),
                    repr(
                        [
                            b(x) if isinstance(x, mx.array) else repr(x)
                            for x in (parts if isinstance(parts, tuple) else (parts,))
                        ]
                    ),
                )
            )
        return out

    f = [("tokens", st.tokens)]
    for i, s in enumerate(st.layers):
        if hasattr(s, "keys"):
            n = s.offset
            f.append(
                (
                    i,
                    "kv",
                    n,
                    b(s.keys[:, :, :n]),
                    b(s.values[:, :, :n]),
                    b(s.idx[:, :, :n]),
                    cache(s.idx_bk),
                )
            )
        else:
            f.append((i, "gdn", b(s.conv), b(s.ssm)))
    for i, p in sorted(st.ple.items()):
        f.append((i, "ple", b(p.context), b(p.conv)))
    return f


def test_rollback_after_p1_only_rebuilds_restores_every_state_field(
    tiny_flashnext_ple, requires_mlx, monkeypatch
):
    """Snapshot, mutate the live state through a rebuild, roll back, compare with a pristine solo
    run, every field at once. Every layer of the pass takes the p+1-only rebuild (position 1 marked
    corrected wherever position 0 is not; rebuilding from the same h3 is exact), so each restored
    snapshot is built on again before `rollback1` restores it. The pass starts where p + 1 completes
    a QSA block past `prune_from`, so the attention layer's block-key cache grows during the
    rebuild. The PLE fixture covers PLE, DeltaNet, attention and KV state."""
    import mlx.core as mx
    import numpy as np

    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine
    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState
    from jitengine_qwen3_8_flash_next_m4pro.model.moe import speculative as sp

    begin = sp.SpeculativeGatherMixin.spec_settle_pair_begin

    def p1_only(self, recs, chains=None):
        pc = begin(self, recs, chains)
        if not pc["outs"][0][1] and not pc["outs"][1][1]:
            pc["outs"][1] = (pc["outs"][1][0], True)
        return pc

    monkeypatch.setattr(sp.SpeculativeGatherMixin, "spec_settle_pair_begin", p1_only)
    model, pack, ple = tiny_flashnext_ple()
    e = create_engine(
        model,
        expert_pack_path=pack,
        ple_pack_path=ple,
        ple_io_nocache=False,
        log_requests=False,
        pool_ext_slots=0,
        prefix_store_entries=0,
    )
    m = e.model
    ix = next(layer.block.indexer for layer in m.layers if layer.is_attention)
    L = 42  # the pass covers positions 42 and 43: 43 + 1 completes a block
    assert (L + 2) % ix.cr == 0 and L + 1 >= ix.prune_from
    prompt = mx.array([list(range(500, 500 + L))], dtype=mx.int32)

    def fresh():
        st = RequestState(m.cfg, L + 1, 8, dtype=m.dtype)
        m.forward(prompt, st, logits="none")
        st.ensure_room(2)
        return st

    a, b = fresh(), fresh()
    t = 7
    ref = m.forward(mx.array([[t]], dtype=mx.int32), a, logits="last")
    lg, _hid, snaps = m.verify2(t, 12345, b)  # a draft the solo row does not pick
    m.rollback1(b, snaps)
    mx.eval(ref, lg)
    assert np.asarray(ref).tobytes() == np.asarray(lg[0:1]).tobytes()
    fa, fb = _state_fields(a), _state_fields(b)
    bad = [x[0] for x, y in zip(fa, fb) if x != y]
    assert not bad, f"state fields differ after rollback: {bad}"
    t2 = int(mx.argmax(ref, axis=-1).item())
    r2 = m.forward(mx.array([[t2]], dtype=mx.int32), a, logits="last")
    g2 = m.forward(mx.array([[t2]], dtype=mx.int32), b, logits="last")
    mx.eval(r2, g2)
    assert np.asarray(r2).tobytes() == np.asarray(g2).tobytes()


def _count_compiled_pair_builds(monkeypatch) -> list:
    """Counts calls into the compiled pair build (`_build_spec2_compiled`): [all, those taking the
    previous layer's two-row h3 whole]."""
    from jitengine_qwen3_8_flash_next_m4pro.model import model as M

    n = [0, 0]
    f0 = M.Qwen4ExpTextDecoderLayer._build_spec2_compiled

    def counted(self, *a, **k):
        n[0] += 1
        n[1] += a[-1] is not None  # `pair`
        return f0(self, *a, **k)

    monkeypatch.setattr(M.Qwen4ExpTextDecoderLayer, "_build_spec2_compiled", counted)
    return n


def test_the_verify_tests_reach_the_compiled_pair_build(
    mtp_engine, requires_mlx, tmp_path, monkeypatch
):
    """Every layer of the tiny model takes the compiled pair build (DeltaNet and attention alike),
    so the row checks above cover it: rows as solo with it on, and it ran on every layer of every
    pass, every layer after the first taking the previous one's two-row h3 whole on most passes."""
    n = _count_compiled_pair_builds(monkeypatch)
    e = mtp_engine()
    m = e.model
    modes = [L._pair_mode(m.dtype) for L in m.layers]
    assert "gdn" in modes and "attention" in modes, modes
    verifies, _ = _check_against_solo(e, [[5, 6, 7], list(range(100, 140))], tmp_path, monkeypatch)
    assert n[0] >= verifies * sum(md != "solo" for md in modes), (n[0], verifies, modes)
    assert n[1] >= verifies, (n, verifies)


def _layer_state(st, i) -> list:
    """`_state_fields` for one layer (and its PLE slot)."""
    f = _state_fields(st)
    return [x for x in f[1:] if x[0] == i]


@pytest.mark.parametrize("ple", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_compiled_pair_build_is_the_eager_one(  # noqa: PLR0915
    flashnext_engine, tiny_flashnext_ple, requires_mlx, dtype, ple
):
    """`build_spec2` compiled and eager (`PAIR_COMPILED` off) from one state on distinct residual
    pairs, on every layer (DeltaNet, attention, and with `ple` the PLE layer): both records, the
    row-0 rollback state and the layer's whole state after, byte for byte. The compiled DeltaNet
    build runs lazy (`GDN_LAZY`, the default), so its rollback state is the one `LazyRow0` makes."""
    import mlx.core as mx
    import numpy as np

    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine
    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState
    from jitengine_qwen3_8_flash_next_m4pro.model import model as M

    opts = dict(pool_ext_slots=0, prefix_store_entries=0, compute_dtype=dtype)
    if ple:
        model, pack, plep = tiny_flashnext_ple()
        e = create_engine(
            model,
            expert_pack_path=pack,
            ple_pack_path=plep,
            ple_io_nocache=False,
            log_requests=False,
            **opts,
        )
    else:
        e = flashnext_engine(**opts)
    m = e.model
    layers = [i for i, L in enumerate(m.layers) if L._pair_mode(m.dtype) != "solo"]
    assert any(m.layers[i].is_attention for i in layers) and any(
        not m.layers[i].is_attention for i in layers
    )
    if ple:
        assert any(m.layers[i].ple is not None for i in layers)
    K = m.cfg.hc_count * m.cfg.hidden_size

    def b(a):
        mx.eval(a)
        return np.asarray(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes()

    def build(i, st, h0, h1, ta, da, compiled):
        L, slot, pst = m.layers[i], st.layers[i], st.ple.get(i)
        got = {}

        def on_row0(pl):
            sn, _ = m._snap(slot, pst)
            # a lazy rollback point (`GDN_LAZY`, compiled only) compared as the state it makes
            ssm = sn[2].state() if sn[0] == "delta_lazy" else sn[2]
            got["row0"] = (
                repr(sn[:2]) if sn[0] == "kv" else (b(sn[1]), b(ssm)),
                None if pl is None else (b(pl[0]), b(pl[1])),
            )

        old = M.PAIR_COMPILED
        M.PAIR_COMPILED = compiled
        try:
            recs = L.build_spec2(h0, h1, slot, st.tokens, ta, da, pst, on_row0)
        finally:
            M.PAIR_COMPILED = old
        f = [got["row0"], _layer_state(st, i)]
        for r in recs:
            f += [
                b(r[k])
                for k in ("x2", "inds", "scores", "sh", "shg", "h2", "inject", "y_spec", "h3")
            ]
            f += [r["snap"][0].tobytes(), r["snap"][1].tobytes()]
        return f

    rng = np.random.default_rng(7)
    for t in range(3):
        n = int(rng.integers(3, 30))
        st = RequestState(m.cfg, n + 4, 8, dtype=m.dtype)
        m.forward(
            mx.array([rng.integers(5, 4000, size=n).tolist()], dtype=mx.int32), st, logits="none"
        )
        st.ensure_room(2)
        for i in layers:
            slot, pst = st.layers[i], st.ple.get(i)
            pre = m._snap(slot, pst)
            h0, h1 = (
                mx.array(rng.standard_normal((1, 1, K)).astype(np.float32)).astype(m.residual_dtype)
                for _ in range(2)
            )
            ta, da = (mx.array([[int(rng.integers(5, 4000))]], dtype=mx.int32) for _ in range(2))
            eager = build(i, st, h0, h1, ta, da, False)
            m._restore(slot, pst, pre)
            comp = build(i, st, h0, h1, ta, da, True)
            m._restore(slot, pst, pre)
            assert eager == comp, (
                t,
                i,
                next(j for j, (x, y) in enumerate(zip(eager, comp)) if x != y),
            )


@pytest.mark.parametrize("slots", [12, 24])
def test_rows_do_not_depend_on_the_victim_rule(
    flashnext_engine, requires_mlx, tmp_path, monkeypatch, slots
):
    """Eviction decides which slot holds which record, never a value: under misses (12 or 24 of the
    32 records; at 24 the landing region and the chain reserve too), solo and with the scripted
    head's accepted and rejected drafts, the cycle-aware rule (`cyc64`, single-row decode's
    default) and decayed use (`decay32`) emit the same tokens and the same digest rows, while
    really evicting differently."""
    import numpy as np

    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = slots
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)
    n = 40
    prompts = [list(range(100, 140)), list(range(2000, 2070)), list(range(7, 60))]
    seen = {}
    for rule in ("decay32", "cyc64"):
        opts = dict(pool_ext_slots=0, prefix_store_entries=0, expert_pool_evict=rule)
        if slots == 24:
            opts.update(landing_slots=2, prefetch_reserve=2)
        e = flashnext_engine(**opts)
        pool = e.expert_pool
        assert pool.slots == slots and pool.cyc_solo == (64 if rule == "cyc64" else 0)
        if slots == 24:
            assert pool.landing_n == 2 and pool.reserve_n == 2
        out, resident = [], []
        for i, p in enumerate(prompts):
            e.mtp = None
            ref, ref_rows = _run(e, p, str(tmp_path / f"{rule}-solo{i}.digest"), monkeypatch, n)
            resident.append(np.flatnonzero(pool.slot_for_id >= 0).tolist())
            v0, a0 = e.mtp_verifies, e.mtp_accepted
            e.mtp, e.mtp_gen = ScriptedHead(ref), e.mtp_gen + 1
            got, got_rows = _run(e, p, str(tmp_path / f"{rule}-mtp{i}.digest"), monkeypatch, n)
            e.mtp = None
            resident.append(np.flatnonzero(pool.slot_for_id >= 0).tolist())
            assert (got, got_rows) == (ref, ref_rows), (rule, i)
            assert e.mtp_verifies > v0 and e.mtp_accepted - a0 < e.mtp_verifies - v0
            out.append((ref, ref_rows))
        seen[rule] = (out, resident, pool.stats())
    assert seen["cyc64"][0] == seen["decay32"][0]
    cyc, dec = seen["cyc64"][2], seen["decay32"][2]
    assert cyc["expert_pool_evictions_cyc"] > 0 and dec["expert_pool_evictions_cyc"] == 0
    assert cyc["expert_misses"] > 0 and dec["expert_misses"] > 0
    assert seen["cyc64"][1] != seen["decay32"][1] or cyc["expert_misses"] != dec["expert_misses"]


# --------------------------------------------------------- the lazy rollback state (GDN_LAZY)
# Each fixture compares drawn rows and states byte for byte with the stored
# path (`GDN_LAZY` off), and each mutant (m1 stale, m2 ahead, m3 rebuild without row 0)
# is first shown to compute something different, then shown caught.
def _gdn_layers(m) -> list:
    """The DeltaNet layers whose compiled pair build runs lazy."""
    return [
        i
        for i, L in enumerate(m.layers)
        if L._pair_mode(m.residual_dtype) == "gdn" and L._lazy_row0()
    ]


def _bytes(a) -> bytes:
    import mlx.core as mx
    import numpy as np

    mx.eval(a)
    return np.asarray(a).tobytes()


def _snap_bytes(sn) -> tuple:
    """A verify pass's rollback snapshot as bytes: a DeltaNet one as (conv, ssm), a lazy one's ssm
    made here by the row-0 launch `LazyRow0.state` runs, the snapshot itself left as the pass left
    it; a KV one as its offset and block keys; and the PLE pair."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import gdn_recurrence_step

    (s, pl) = sn
    if s[0] == "kv":
        out = ("kv", s[1], repr(sorted(s[2], key=str)))
    elif s[0] == "delta_lazy":
        lz = s[2]
        ssm = gdn_recurrence_step(*lz.row0, lz.pre, *lz._dims())[1] if lz.pending else lz.ssm
        out = ("delta", _bytes(s[1]), _bytes(ssm))
    else:
        out = ("delta", _bytes(s[1]), _bytes(s[2]))
    return out + (None if pl is None else (_bytes(pl[0]), _bytes(pl[1])),)


def _pass(m, st, t, d, lazy, monkeypatch):
    """verify2 under `GDN_LAZY` = lazy: (logits bytes, hidden bytes, snaps)."""
    from jitengine_qwen3_8_flash_next_m4pro.model import gdn

    monkeypatch.setattr(gdn, "GDN_LAZY", lazy)
    lg, hid, snaps = m.verify2(t, d, st)
    return _bytes(lg), _bytes(hid), snaps


def _fresh_states(m, prompt, n):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState

    out = []
    for _ in range(n):
        st = RequestState(m.cfg, len(prompt) + 1, 8, dtype=m.dtype)
        m.forward(mx.array([prompt], dtype=mx.int32), st, logits="none")
        st.ensure_room(2)
        out.append(st)
    return out


def _capped(monkeypatch, slots):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = slots
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)


def _ple_engine(tiny_flashnext_ple):
    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine

    model, pack, ple = tiny_flashnext_ple()
    return create_engine(
        model,
        expert_pack_path=pack,
        ple_pack_path=ple,
        ple_io_nocache=False,
        log_requests=False,
        pool_ext_slots=0,
        prefix_store_entries=0,
    )


def _force(monkeypatch, which):
    """Every layer of a pass takes a rebuild: "p1" the p + 1-only one (position 1 marked
    corrected wherever position 0 is not), "both" the both-rows one (position 0 marked
    corrected). Rebuilding from the same h3 is exact."""
    from jitengine_qwen3_8_flash_next_m4pro.model.moe import speculative as sp

    begin = sp.SpeculativeGatherMixin.spec_settle_pair_begin

    def forced(self, recs, chains=None):
        pc = begin(self, recs, chains)
        if not pc["outs"][0][1] and not pc["outs"][1][1]:
            r = 1 if which == "p1" else 0
            pc["outs"][r] = (pc["outs"][r][0], True)
        return pc

    monkeypatch.setattr(sp.SpeculativeGatherMixin, "spec_settle_pair_begin", forced)


PROMPT = list(range(500, 542))


def _round(m, lazy, d, monkeypatch, then=None):
    """One verify pass from a fresh state under `GDN_LAZY` = lazy (draft d; 12345 is wrong), a
    rollback if the draft is wrong, then `then` ("solo": the next position as a solo step; "save":
    a prefix-store snapshot of the state). Every observable as bytes, in one list."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.model.gdn import LazyRow0

    (st,) = _fresh_states(m, PROMPT, 1)
    lg, hid, snaps = _pass(m, st, 7, d, lazy, monkeypatch)
    out = [("logits", lg), ("hidden", hid), ("after_pass", _state_fields(st))]
    out.append(("snaps", [_snap_bytes(sn) for sn in snaps]))
    if d == 12345:
        m.rollback1(st, snaps)
        out.append(("after_rollback", _state_fields(st)))
        assert not any(isinstance(getattr(x, "ssm", None), LazyRow0) for x in st.layers)
    if then == "solo":
        out.append(("solo", _bytes(m.forward(mx.array([[11]], dtype=mx.int32), st, logits="last"))))
    elif then == "save":
        e = st.snapshot(list(range(st.tokens)))
        out.append(("save", [_bytes(a) for a in e.arrays()]))
    return out


def _diff(a, b) -> list:
    return [x[0] for x, y in zip(a, b) if x != y]


@pytest.mark.parametrize("slots", [None, 12])
def test_the_lazy_rollback_states_are_the_stored_ones(
    flashnext_engine, requires_mlx, monkeypatch, slots
):
    """Real verify passes from one state, `GDN_LAZY` off (the stored rollback states) and on: both
    logits rows and the hidden rows equal byte for byte, every layer's state after the pass equal,
    and every layer's rollback snapshot equal, a DeltaNet layer's lazy one as the (conv, ssm) its
    `LazyRow0` makes. Accepted and rejected drafts; with 12 of 32 records the pass misses."""
    if slots:
        _capped(monkeypatch, slots)
    e = flashnext_engine(pool_ext_slots=0, prefix_store_entries=0)
    m = e.model
    lazy_layers = _gdn_layers(m)
    assert lazy_layers, "no DeltaNet layer runs the lazy pair build"
    for k, prompt in enumerate([list(range(50, 90)), list(range(2000, 2033))]):
        for d in (5, 12345):
            a, b = _fresh_states(m, prompt, 2)
            ra = _pass(m, a, 7 + k, d, False, monkeypatch)
            rb = _pass(m, b, 7 + k, d, True, monkeypatch)
            assert ra[:2] == rb[:2], (k, d)
            assert _state_fields(a) == _state_fields(b), (k, d)
            for i in range(len(m.layers)):
                assert rb[2][i][0][0] == ("delta_lazy" if i in lazy_layers else ra[2][i][0][0])
                assert _snap_bytes(ra[2][i]) == _snap_bytes(rb[2][i]), (k, d, i)


@pytest.mark.parametrize("rebuild", [None, "p1", "both"])
@pytest.mark.parametrize("then", ["solo", "save"])
def test_rounds_are_the_stored_ones(tiny_flashnext_ple, requires_mlx, monkeypatch, rebuild, then):
    """On the PLE fixture (DeltaNet, attention and PLE layers), `GDN_LAZY` on against off (the
    stored path): an accepted and a rejected round, each plain, with every layer's p + 1-only
    rebuild forced, or with every layer's both-rows rebuild forced; after the round a solo step at
    the next position or a prefix-store save. Logits and hidden rows, the state after the pass,
    the rollback snapshots, the state after `rollback1` and the solo step's logits or the saved
    entry: all byte for byte. Counters (lazy): a rejected round makes each lazy layer's state once
    (`gdn_lazy_materialized`), an accepted one none; a p + 1-only rebuild of a lazy layer is
    advanced by the pair launch (`gdn_lazy_consumed` = `gdn_row1_rebuilds`), never made."""
    if rebuild:
        _force(monkeypatch, rebuild)
    e = _ple_engine(tiny_flashnext_ple)
    m = e.model
    n_lazy = len(_gdn_layers(m))
    # a forced rebuild reaches every layer after the first (layer i + 1 at layer i's settle)
    n_rebuilt = sum(not L.is_attention for L in m.layers[1:]) if rebuild == "p1" else 0
    assert n_lazy > 0 and any(m.layers[i].ple is not None for i in _gdn_layers(m))
    for d in (5, 12345):
        ref = _round(m, False, d, monkeypatch, then)
        s0 = m.gdn_lazy_stats()
        got = _round(m, True, d, monkeypatch, then)
        s1 = m.gdn_lazy_stats()
        assert got == ref, (d, _diff(got, ref))
        made = s1["gdn_lazy_materialized"] - s0["gdn_lazy_materialized"]
        assert made == (n_lazy if d == 12345 else 0), (d, made)
        used = s1["gdn_lazy_consumed"] - s0["gdn_lazy_consumed"]
        rebuilt = s1["gdn_row1_rebuilds"] - s0["gdn_row1_rebuilds"]
        assert used == rebuilt and (rebuilt == n_rebuilt or not rebuild), (used, rebuilt, n_rebuilt)
    assert e.stats()["mtp"]["gdn_lazy"] is True


def _mutant(monkeypatch, which):
    """m1: a rollback makes the state from before the pass (one step stale); m2: a restore leaves
    the slot's final state in place of the lazy point (one step ahead); m3: a p + 1-only rebuild
    advances from the state before the pass without row 0's step."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import gdn_recurrence_step
    from jitengine_qwen3_8_flash_next_m4pro.model import gdn
    from jitengine_qwen3_8_flash_next_m4pro.model.model import Qwen4ExpForCausalLM as M

    if which == "m1":

        def state(self):
            if self.ssm is None:
                self.ssm = self.pre
            return self.ssm

        monkeypatch.setattr(gdn.LazyRow0, "state", state)
    elif which == "m2":
        restore = M._restore

        def _restore(slot, ple_state, snap):
            ssm = getattr(slot, "ssm", None)
            restore(slot, ple_state, snap)
            if snap[0][0] == "delta_lazy":
                slot.ssm = ssm

        monkeypatch.setattr(M, "_restore", staticmethod(_restore))
    else:

        def advance(self, row1):
            return gdn_recurrence_step(*row1, self.pre, *self._dims())

        monkeypatch.setattr(gdn.LazyRow0, "advance", advance)


@pytest.mark.parametrize("which", ["m1", "m2", "m3"])
def test_each_mutant_computes_otherwise_and_is_caught(
    tiny_flashnext_ple, requires_mlx, monkeypatch, which
):
    """Each mutant of the lazy point, first on one layer's pass (the mutant's state or row-1
    output differs from the reference's bytes, so a catch is not of an inert change), then through
    the round fixtures: m1 and m2 fail the rejected round, m3 the p + 1-only rebuild."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.model import gdn

    e = _ple_engine(tiny_flashnext_ple)
    m = e.model
    i = _gdn_layers(m)[0]
    (st,) = _fresh_states(m, PROMPT, 1)
    _lg, _hid, snaps = _pass(m, st, 7, 12345, True, monkeypatch)
    lz, final = snaps[i][0][2], st.layers[i].ssm
    row1 = tuple(mx.random.normal(a.shape).astype(a.dtype) if a.ndim == 2 else a for a in lz.row0)
    twin = gdn.LazyRow0(lz.block, lz.pre, lz.row0)
    ref_state, ref_y1 = _bytes(twin.state()), _bytes(twin.advance(row1)[0])
    ref_adv = _bytes(gdn.LazyRow0(lz.block, lz.pre, lz.row0).advance(row1)[0])
    assert ref_adv == ref_y1  # the pair launch from the state before the pass = the row-1 launch
    _mutant(monkeypatch, which)
    if which == "m1":
        assert _bytes(gdn.LazyRow0(lz.block, lz.pre, lz.row0).state()) != ref_state
    elif which == "m2":
        assert _bytes(final) != ref_state
    else:
        assert _bytes(gdn.LazyRow0(lz.block, lz.pre, lz.row0).advance(row1)[0]) != ref_y1
    if which == "m3":
        _force(monkeypatch, "p1")
    ref = _round(m, False, 12345, monkeypatch, "solo")
    got = _round(m, True, 12345, monkeypatch, "solo")
    bad = _diff(got, ref)
    assert bad, f"mutant {which} not caught"
    want = ("logits", "after_pass") if which == "m3" else ("after_rollback",)
    assert set(want) <= set(bad), (which, bad)


def test_served_rounds_count_the_lazy_states_they_make(
    mtp_engine, requires_mlx, tmp_path, monkeypatch
):
    """The scripted head through the engine (`_mtp_round`) on 12 of 32 records, `GDN_LAZY` off then
    on: tokens and digest rows are the solo path's both ways, `/stats` reports the switch, every
    rejected pass makes each lazy layer's state exactly once and an accepted pass none
    (made = rejected x lazy layers), and every p + 1-only rebuild of a DeltaNet layer advances its
    lazy point by the pair launch (consumed = rebuilds)."""
    from jitengine_qwen3_8_flash_next_m4pro.model import gdn

    _capped(monkeypatch, 12)
    e = mtp_engine()
    n_lazy = len(_gdn_layers(e.model))
    prompts = [list(range(100, 130)), list(range(2000, 2070))]
    for lazy in (False, True):
        monkeypatch.setattr(gdn, "GDN_LAZY", lazy)
        s0 = e.stats()["mtp"]
        _verifies, rejected = _check_against_solo(e, prompts, tmp_path, monkeypatch)
        s1 = e.stats()["mtp"]
        assert s1["gdn_lazy"] is lazy and s1["gdn_lazy_layers"] == (n_lazy if lazy else 0)
        made = s1["gdn_lazy_materialized"] - s0["gdn_lazy_materialized"]
        used = s1["gdn_lazy_consumed"] - s0["gdn_lazy_consumed"]
        rebuilt = s1["gdn_row1_rebuilds"] - s0["gdn_row1_rebuilds"]
        assert made == (rejected * n_lazy if lazy else 0), (made, rejected)
        assert used == (rebuilt if lazy else 0), (used, rebuilt)


def test_without_the_pair_source_lazy_is_off(mtp_engine, requires_mlx, tmp_path, monkeypatch):
    """A recurrence source the pair launch cannot be cut from (an anchor missing): no pair launch,
    its check fails, every block keeps `pair_ok` off, `/stats` says `gdn_lazy` false, and the
    verify pass runs the stored path with the solo path's rows."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_recurrence as gr

    assert gr._pair_source(gr._SOURCE.replace(gr._LOOP_STEP, "")) is None
    monkeypatch.setattr(gr, "_SOURCE_PAIR", None)
    assert not gr.pair_available() and gr.check_pair(2, 6, 32, 32, None) is False
    e = mtp_engine()
    assert not any(getattr(L.block, "pair_ok", False) for L in e.model.layers)
    assert e.stats()["mtp"]["gdn_lazy"] is False
    _check_against_solo(e, [list(range(100, 130))], tmp_path, monkeypatch)
    snaps_tags = set()
    st = _fresh_states(e.model, PROMPT, 1)[0]
    for sn in e.model.verify2(7, 12345, st)[2]:
        snaps_tags.add(sn[0][0])
    assert "delta_lazy" not in snaps_tags and "delta" in snaps_tags
