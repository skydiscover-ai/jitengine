"""The prefix store's memory lent to the expert pool's annex while one row decodes, on the tiny model.

`SchedulingMixin._annex_policy` spills the store and regrows the annex (a third pool array set) once a
lone row decodes, and releases the annex and reads the store back before any admission. The annex
only holds records, so every token and every digested logits row must be what the same engine
gives with the store resident (`store_spill` off), through a cold request, a store hit, a request
sharing no prefix with the others, and a short follow-up turn, which must release the annex alone
(never the extension). The store's writes run behind the decode, so the annex opens a few steps in. The MTP verify pass reads
the annex too (the scripted head of test_mtp_verify stands in for the checkpoint's).
"""

import os

WRONG_EVERY = 3  # the scripted head drafts a wrong token at every third position


class _HeadSlot:
    """What `Qwen4ExpForCausalLM._snap` reads of the head's KV slot."""

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


def _rows(path):
    return [ln for ln in open(path).read().split("\n") if ln] if os.path.exists(path) else []


N = 32
OPTS = dict(
    prefix_grid_tokens=64,
    prefix_min_divergence_tokens=16,
    landing_slots=2,
    prefetch_reserve=2,
    pool_ext_slots=4,
    ext_scratch_tokens=48,
)
ANNEX = 6


def _req(prompt, n=N):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=prompt, max_tokens=n, stop_at_eos=False)


def _engine(flashnext_engine, monkeypatch, annex: int):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 16  # of 32 records, so the annex's slots fill with misses
        plan.store_spill_slots = annex
        return plan

    with monkeypatch.context() as mp:
        mp.setattr(fne, "plan_budget", capped)
        e = flashnext_engine(**OPTS)  # the spill's default place: beside the pack directory
    return e


def _go(e, prompts, digest, monkeypatch, mtp_ref=None):
    """Each prompt in turn; the scripted head (built from `mtp_ref`) when given."""
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
        rows.append(_rows(path))
    e.mtp_on, e.mtp = False, None
    return toks, rows


def test_the_annex_changes_no_token_and_no_row(
    flashnext_engine, requires_mlx, tmp_path, monkeypatch
):
    shared = list(range(1000, 1100))
    other = list(range(2000, 2090))  # no prefix in common with `shared`
    prompts = [
        shared + [5, 6, 7],
        shared + [5, 6, 7],
        other + [1, 2],
        shared + [5, 6, 7, 8, 9, 10, 11],
    ]
    ref_e = _engine(flashnext_engine, monkeypatch, 0)
    cand = _engine(flashnext_engine, monkeypatch, ANNEX)
    pool = cand.expert_pool
    assert ref_e.expert_pool.ann_slots == 0 and pool.ann_slots == ANNEX
    assert pool.ann0 == pool.slots_main + pool.ext_slots and pool.slots == pool.ann0 + ANNEX
    assert cand.plan.as_dict()["borrowed"][0]["sources"] == {
        "prefix store cap": ANNEX * pool.slot_bytes
    }
    # the spill's default place: on the pack's volume, beside (never in) the pack directory
    pack_dir = os.path.dirname(cand.expert_pack_path)
    assert cand.store_spiller.dir == os.path.join(os.path.dirname(pack_dir), "jitengine-spill")

    ref, ref_rows = _go(ref_e, prompts, str(tmp_path / "ref"), monkeypatch)
    s0 = cand.stats()
    got, got_rows = _go(cand, prompts, str(tmp_path / "cand"), monkeypatch)
    assert got == ref
    for i, (x, y) in enumerate(zip(got_rows, ref_rows)):
        bad = next((j for j, (a, b) in enumerate(zip(x, y)) if a != b), None)
        assert len(x) == len(y) == N and bad is None, (i, bad)
    s = cand.stats()
    ps = s["prefix_store"]
    assert s["expert_pool_ann_regrows"] - s0["expert_pool_ann_regrows"] >= len(
        prompts
    )  # each request
    assert s["expert_pool_ann_dropped"] > s0["expert_pool_ann_dropped"]  # the annex held records
    assert (
        ps["spill_entries_written"] >= 2
        and ps["spill_read_backs"] >= 2
        and ps["spill_bytes_read"] > 0
    )
    assert ps["hits"] >= 2  # requests 2 and 4 restored their prefix from a read-back entry
    assert ps["spill_write_failures"] == 0 and ps["restore_failures"] == 0
    # the engine's I/O counters carry the spill's device bytes (whole pages, both directions)
    assert s["spill_page_bytes"] == ps["spill_page_bytes_written"] + ps["spill_page_bytes_read"] > 0
    assert ps["spill_page_bytes_written"] >= ps["spill_bytes_written"]

    # MTP on, through the same requests: the verify pass gathers annex slots too
    ref_m, ref_m_rows = _go(ref_e, prompts, str(tmp_path / "refm"), monkeypatch, mtp_ref=ref)
    v0 = cand.mtp_verifies
    got_m, got_m_rows = _go(cand, prompts, str(tmp_path / "candm"), monkeypatch, mtp_ref=ref)
    assert got_m == ref_m == ref and got_m_rows == ref_m_rows == ref_rows
    assert cand.mtp_verifies > v0


def test_a_short_follow_up_releases_the_annex_not_the_extension(
    flashnext_engine, requires_mlx, tmp_path, monkeypatch
):
    """A follow-up whose uncovered prompt is below `ext_scratch_tokens` keeps the extension (and its
    records); only the annex goes back, so the turn costs at most the annex's refill."""
    shared = list(range(3000, 3100))
    e = _engine(flashnext_engine, monkeypatch, ANNEX)
    _go(e, [shared + [1, 2, 3]], str(tmp_path / "a"), monkeypatch)
    s0 = e.stats()
    _go(e, [shared + [1, 2, 3, 4, 5]], str(tmp_path / "b"), monkeypatch)  # 2 uncovered tokens
    s = e.stats()
    assert s["expert_pool_ext_releases"] == s0["expert_pool_ext_releases"]
    assert s["expert_pool_ann_releases"] > s0["expert_pool_ann_releases"]
    assert s["expert_pool_ann_regrows"] > s0["expert_pool_ann_regrows"]
    assert s["prefix_store"]["spill_read_backs"] > s0["prefix_store"]["spill_read_backs"]
