"""The draft's vocabulary changes acceptance, never a token or a logged row (model.draft_vocab).

On the tiny model with a pool of 12 of its 32 expert records (verify passes miss, evict and redo
layers), each prompt runs three ways: solo (no head), then with a head stand-in whose drafts are read
out of the tiny model's lm_head through `draft_vocab.draft_id`, over every row (`full`) or through a
`SubsetReadout` (`subset`: a base of 1 in 16 ids, plus the prompt's own ids through the engine's
per-request hook). The tiny model has no MTP head, so the stand-in aims each draft at the reference
token: its activation is that token's dequantized lm_head row. `full` then drafts the reference token
wherever its row wins the argmax, `subset` only where the token is also in S.
- The enabling condition: `subset` and `full` accept different counts, and `subset` rejects; otherwise
  the comparison would be vacuous.
- The check: every digest row and every token equal across the three runs.
- The negative control: `_mtp_round` mutated to draw the second row after a rejection. The mutation is
  certified to change the program (its clause is in the reference exactly once and the mutant's source
  differs), and the check must catch it on a subset run.
"""

import inspect
import os
import textwrap

import pytest

# every test here runs a custom Metal kernel
pytestmark = pytest.mark.usefixtures("requires_mlx")

N = 24  # tokens per request
P_MAX = 64
PROMPTS = ([5, 6, 7, 300, 301, 302], list(range(1000, 1040)))


class _HeadSlot:
    """What `Qwen4ExpForCausalLM._snap` reads of the head's KV slot."""

    keys = None

    def __init__(self):
        self.offset, self.idx_bk = 0, {}


class AimedHead:
    """`MTPHead`'s interface: after consuming k entries, reads out a draft for generated token k from
    the reference token's own lm_head row, through `readout` (None: every row)."""

    def __init__(self, lm_head, ref, readout=None):
        import mlx.core as mx

        self.lm_head, self.readout, self.n = lm_head, readout, len(ref)
        ids = mx.array([int(t) for t in ref] + [0], dtype=mx.int32)
        lm = lm_head
        self.aim = mx.dequantize(
            mx.take(lm.weight, ids, axis=0),
            mx.take(lm.scales, ids, axis=0),
            None if lm.biases is None else mx.take(lm.biases, ids, axis=0),
            group_size=lm.group_size,
            bits=lm.bits,
        ).astype(mx.float32)
        mx.eval(self.aim)

    def new_state(self, capacity):
        return {"slot": _HeadSlot(), "n": 0}

    def draft(self, st, h_pre, tokens):
        from jitengine_qwen3_8_flash_next_m4pro.model.draft_vocab import draft_id

        st["n"] += len(tokens)
        k = min(st["n"], self.n)
        return draft_id(self.lm_head, self.readout, self.aim[k : k + 1].reshape(1, 1, -1))


def _req(prompt, n=N):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=prompt, max_tokens=n, stop_at_eos=False)


def _run(e, prompt, digest, monkeypatch, n=N):
    if os.path.exists(digest):
        os.remove(digest)
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", digest)
    out = e.generate(_req(prompt, n))
    monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST")
    rows = [ln for ln in open(digest).read().split("\n") if ln] if os.path.exists(digest) else []
    return out.token_ids, rows


@pytest.fixture
def engine(flashnext_engine, monkeypatch):
    """The tiny engine with a 12-slot pool, no extension (so `_ext_policy` never loads or drops a
    head and the test's stand-in stays in place) and no prefix store."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 12
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)
    # no rope-table isolation here: the memo is per thread, so the entries the main
    # thread left from earlier tests cannot reach this engine's worker
    e = flashnext_engine(pool_ext_slots=0, prefix_store_entries=0, mtp_min_slots=0)
    assert e.expert_pool.slots == 12 and e.model.verify_ok()
    return e


def _base(e):
    import numpy as np

    return np.arange(0, int(e.model.lm_head.out_dims), 16, dtype=np.int32)


def _three_ways(e, prompt, tag, tmp_path, monkeypatch):
    """(solo tokens, solo rows, {mode: (tokens, rows, verifies, accepted)}) for one prompt."""
    from jitengine_qwen3_8_flash_next_m4pro.model.draft_vocab import SubsetReadout

    e.mtp = None
    ref, ref_rows = _run(e, prompt, str(tmp_path / f"{tag}.solo"), monkeypatch)
    assert [int(r.split()[0]) for r in ref_rows] == list(range(N))
    out = {}
    for mode in ("full", "subset"):
        ro = None if mode == "full" else SubsetReadout(e.model.lm_head, _base(e), p_max=P_MAX)
        v0, a0 = e.mtp_verifies, e.mtp_accepted
        e.mtp, e.mtp_gen = AimedHead(e.model.lm_head, ref, ro), e.mtp_gen + 1
        got, rows = _run(e, prompt, str(tmp_path / f"{tag}.{mode}"), monkeypatch)
        e.mtp = None
        out[mode] = (got, rows, e.mtp_verifies - v0, e.mtp_accepted - a0)
    return ref, ref_rows, out


def test_drafts_change_acceptance_never_a_row(engine, requires_mlx, tmp_path, monkeypatch):
    e = engine
    acc = {"full": 0, "subset": 0}
    rej = {"full": 0, "subset": 0}
    for i, p in enumerate(PROMPTS):
        rows0 = e.stats()["draft_prompt_rows"]
        ref, ref_rows, out = _three_ways(e, p, f"p{i}", tmp_path, monkeypatch)
        for mode, (got, rows, ver, accepted) in out.items():
            assert got == ref, (i, mode, got, ref)
            bad = next((j for j, (x, y) in enumerate(zip(rows, ref_rows)) if x != y), None)
            assert len(rows) == len(ref_rows) and bad is None, (i, mode, bad, len(rows))
            assert ver > 0, (i, mode)
            acc[mode] += accepted
            rej[mode] += ver - accepted
        st = e.stats()
        base = set(_base(e).tolist())
        want_rows = len({t for t in p if t not in base})
        assert st["draft_vocab"] == "subset" and st["draft_subset_rows"] == len(base) + P_MAX
        assert st["draft_prompt_rows"] - rows0 == want_rows and want_rows > 0
    # the enabling condition: the subset's drafts differ from the full readout's where it matters
    assert acc["full"] != acc["subset"] and rej["subset"] > 0, (acc, rej)
    assert acc["full"] > acc["subset"], (acc, rej)


def _mutant():
    """`_mtp_round` with the acceptance test removed: the second row is drawn after a rejection."""
    from jitengine_qwen3_8_flash_next_m4pro.scheduler import decode

    src = textwrap.dedent(inspect.getsource(decode.DecodingMixin._mtp_round))
    clause = "if y[0] == dv:"
    assert src.count(clause) == 1, "the clause the mutant removes is not in the reference once"
    mut = src.replace(clause, "if True:  # MUTANT: draw row 1 after a rejection")
    assert mut != src
    ns = dict(vars(decode))
    exec(compile(mut, "<mutant _mtp_round>", "exec"), ns)
    return ns["_mtp_round"]


def test_a_mutant_that_draws_after_a_rejection_is_caught(
    engine, requires_mlx, tmp_path, monkeypatch
):
    from jitengine_qwen3_8_flash_next_m4pro.model.draft_vocab import SubsetReadout
    from jitengine_qwen3_8_flash_next_m4pro.scheduler.decode import DecodingMixin

    e = engine
    p = PROMPTS[1]
    e.mtp = None
    ref, ref_rows = _run(e, p, str(tmp_path / "ref.solo"), monkeypatch)
    mutant = _mutant()
    monkeypatch.setattr(DecodingMixin, "_mtp_round", mutant)
    assert e._mtp_round.__func__ is mutant
    ro = SubsetReadout(e.model.lm_head, _base(e), p_max=P_MAX)
    e.mtp, e.mtp_gen = AimedHead(e.model.lm_head, ref, ro), e.mtp_gen + 1
    got, rows = _run(e, p, str(tmp_path / "mutant.subset"), monkeypatch)
    e.mtp = None
    assert len(got) == N  # it ran to the end: caught on its rows, not on a crash
    assert (got, rows) != (ref, ref_rows), "the planted mutant passed the check"
