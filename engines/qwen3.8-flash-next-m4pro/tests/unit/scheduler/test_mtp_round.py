"""The MTP round's bookkeeping on stand-ins (MLX on any device, no custom kernel): which rows it
digests and at which step numbers, what it emits, when it is a solo step, how a failure fails its
row; and the capacity gate that sizes the single-row extension around the head."""

import hashlib
import os
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

import numpy as np  # noqa: E402

from jitengine_qwen3_8_flash_next_m4pro.model.mtp import MTP_MIN_SLOTS  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.sampler import Sampler  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.scheduler.decode import DecodingMixin  # noqa: E402
from jitengine_qwen3_8_flash_next_m4pro.scheduler.scheduler import SchedulingMixin  # noqa: E402

V = 16


def _logits(tok: int, salt: float) -> mx.array:
    """A [1, V] row whose argmax is `tok`."""
    row = np.full((1, V), salt, dtype=np.float32)
    row[0, tok] = 10.0 + salt
    return mx.array(row)


class Model:
    """`forward_hidden` / `verify2` / `rollback1` stand-ins: position p's argmax is `script[p]`."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def forward_hidden(self, tokens, st, telem=None):
        p = st.tokens
        st.tokens += 1
        self.calls.append(("solo", p))
        return _logits(self.script[p], 0.25 * p), mx.zeros((1, 8))

    def verify2(self, t, d, st, telem=None):
        p = st.tokens
        st.tokens += 2
        self.calls.append(("verify", p, d))
        lg = mx.concatenate(
            [_logits(self.script[p], 0.25 * p), _logits(self.script[p + 1], 0.25 * (p + 1))]
        )
        return lg, mx.zeros((2, 8)), ["snaps"]

    def rollback1(self, st, snaps):
        assert snaps == ["snaps"]
        st.tokens -= 1
        self.calls.append(("rollback",))

    @staticmethod
    def _snap(slot, ple_state):
        return ("kv", slot.offset, {}), None

    @staticmethod
    def _restore(slot, ple_state, snap):
        slot.offset = snap[0][1]


class Head:
    """Drafts `drafts[k]` after consuming k entries in total."""

    def __init__(self, drafts):
        self.drafts = drafts

    def new_state(self, capacity):
        return {"slot": SimpleNamespace(offset=0, idx_bk={}, keys=None), "n": 0}

    def draft(self, st, h_pre, tokens):
        st["n"] += len(tokens)
        return mx.array([self.drafts[st["n"]]], dtype=mx.int32)


def _engine(model, head, consumed):
    stats = SimpleNamespace(record_step=lambda b: None, record_moe=lambda t: None)

    def consume(rows, toks, reads):
        for r, t in zip(rows, toks):
            consumed.append(t)
            r.n += 1
            r.next_tok = t
            if r.n >= r.max_tokens:
                r.retired = True

    return SimpleNamespace(
        model=model,
        mtp=head,
        mtp_gen=1,
        mtp_verifies=0,
        mtp_accepted=0,
        row_faults=0,
        _telem_tokens=0,
        expert_pool=SimpleNamespace(decay_tokens=32),
        stats_=stats,
        _set_owners=lambda owners: None,
        _ensure_room=lambda row, n=1: None,
        _arm_step_telemetry=lambda b: (None, []),
        _consume=consume,
        _fail=lambda row, exc: (_ for _ in ()).throw(exc),
    )


def _row(max_tokens, prompt_len=5, next_tok=1):
    st = SimpleNamespace(tokens=prompt_len - 1, layers=[], ple={})
    return SimpleNamespace(
        rid="r",
        state=st,
        next_tok=next_tok,
        n=0,
        max_tokens=max_tokens,
        mtp=None,
        logprobs_k=0,
        sampler=Sampler(0.0, 1.0, 0, None, V),
        retired=False,
    )


def _sha(lg):
    return hashlib.sha256(np.asarray(lg.astype(mx.float32)).reshape(-1).tobytes()).hexdigest()


def _round_all(eng, row):
    while not row.retired:
        assert DecodingMixin._mtp_round(eng, row) > 0


def test_rows_are_digested_at_the_solo_step_numbers(tmp_path, monkeypatch):
    """Accepted passes log two rows, rejected passes one, at steps 0..n-1, each the row drawn from."""
    dig = tmp_path / "d"
    monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", str(dig))
    script = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # position p's token (p from 4)
    script = [0] * 4 + script
    model = Model(script)
    # drafts after k entries: right except k == 3 (wrong), so one rejection
    right = {k: script[4 + k] for k in range(12)}
    drafts = {**right, 3: 15}
    eng = _engine(model, Head(drafts), consumed := [])
    row = _row(max_tokens=8)
    _round_all(eng, row)
    assert consumed == script[4:12]
    rows = dig.read_text().split()
    steps, shas = rows[0::2], rows[1::2]
    assert steps == [str(i) for i in range(8)]
    assert shas == [_sha(_logits(script[4 + i], 0.25 * (4 + i))) for i in range(8)]
    kinds = [c[0] for c in model.calls]
    assert kinds.count("rollback") == 1 and kinds[0] == "solo"
    assert eng.mtp_verifies - eng.mtp_accepted == 1


def test_a_pass_never_draws_past_max_tokens(tmp_path, monkeypatch):
    """With one token left the round is a solo step: max_tokens 1 and 2 log exactly 1 and 2 rows."""
    for n in (1, 2, 3):
        dig = tmp_path / f"d{n}"
        monkeypatch.setenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", str(dig))
        script = [0] * 4 + list(range(1, 10))
        model = Model(script)
        eng = _engine(model, Head({k: script[4 + k] for k in range(9)}), consumed := [])
        row = _row(max_tokens=n)
        _round_all(eng, row)
        assert consumed == script[4 : 4 + n]
        assert dig.read_text().split()[0::2] == [str(i) for i in range(n)]
        assert all(c[0] != "verify" or c[1] + 1 < 4 + n for c in model.calls)


def test_a_stale_draft_is_not_verified(monkeypatch):
    """A draft made for another position, token or head generation is dropped for a solo step."""
    monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", raising=False)
    script = [0] * 4 + list(range(1, 10))
    for bad in ({"pos": 99}, {"t": 99}, {"gen": 0}):
        model = Model(script)
        eng = _engine(model, Head({k: script[4 + k] for k in range(9)}), [])
        row = _row(max_tokens=6)
        m = {
            "st": Head({}).new_state(8),
            "d": mx.array([2]),
            "pos": row.state.tokens,
            "t": 1,
            "gen": 1,
        }
        row.mtp = dict(m, **bad)
        DecodingMixin._mtp_round(eng, row)
        assert model.calls[0][0] == "solo"


class RaisingModel(Model):
    def forward_hidden(self, *args):
        raise RuntimeError("first round failed")

    def verify2(self, *args):
        raise RuntimeError("verify failed")


@pytest.mark.parametrize("which", ["solo", "verify"])
def test_a_raising_round_fails_its_row_and_drops_its_draft(which, monkeypatch):
    monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", raising=False)
    failed = []
    eng = _engine(RaisingModel([0] * 16), Head({}), [])
    eng._fail = lambda row, exc: failed.append((row.rid, type(exc).__name__))
    row = _row(max_tokens=8)
    if which == "verify":
        row.mtp = {
            "st": Head({}).new_state(8),
            "d": mx.array([5]),
            "pos": row.state.tokens,
            "t": 1,
            "gen": 1,
        }
    assert DecodingMixin._mtp_round(eng, row) == 0
    assert failed == [("r", "RuntimeError")] and row.mtp is None and eng.row_faults == 0


# ------------------------------------------------------------------ the capacity gate
def _sched(pool_slots, mtp_bytes):
    return SimpleNamespace(
        plan=SimpleNamespace(pool_slots=pool_slots), mtp_bytes=mtp_bytes, mtp_on=True
    )


def test_the_head_is_charged_inside_the_extension():
    rec = 3_072_000
    head = 1_470_955_171 + (256 << 20)
    s = _sched(9150, head)
    left = SchedulingMixin._charge_mtp_head(s, 2197, rec)
    assert left == 2197 - (-(-head // rec)) and s.mtp_on and s.mtp_bytes == head
    assert (2197 - left) * rec >= head  # the extension plus the head never exceed the extension


def test_below_the_slot_threshold_mtp_is_off_and_the_extension_untouched():
    rec = 3_072_000
    s = _sched(4800, 1_700_000_000)  # 24 GB-like: 4,800 + 1,339 = 6,139 slots
    assert SchedulingMixin._charge_mtp_head(s, 1339, rec) == 1339
    assert s.mtp_on is False and s.mtp_bytes == 0
    s = _sched(MTP_MIN_SLOTS - 2000, 1_700_000_000)
    assert SchedulingMixin._charge_mtp_head(s, 1999, rec) == 1999 and s.mtp_on is False


def test_an_extension_too_small_for_the_head_keeps_mtp_off():
    s = _sched(12000, 1_700_000_000)
    assert SchedulingMixin._charge_mtp_head(s, 500, 3_072_000) == 500 and s.mtp_on is False


def test_without_a_head_the_extension_is_untouched():
    s = _sched(12000, 0)
    assert SchedulingMixin._charge_mtp_head(s, 2000, 3_072_000) == 2000 and s.mtp_on is True


def test_the_digest_hook_is_off_without_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_LOGITS_DIGEST", raising=False)
    script = [0] * 4 + list(range(1, 10))
    eng = _engine(Model(script), Head({k: script[4 + k] for k in range(9)}), [])
    row = _row(max_tokens=4)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _round_all(eng, row)
    finally:
        os.chdir(cwd)
    assert os.listdir(tmp_path) == []


def test_the_mtp_min_slots_option_moves_the_threshold():
    """`mtp_min_slots` 0 turns the head on at 24 GB's slot count (the miss-path proof's setting)."""
    rec = 3_072_000
    s = _sched(4800, 1_700_000_000)
    s.options = {"mtp_min_slots": 0}
    assert SchedulingMixin._charge_mtp_head(s, 1339, rec) == 1339 - 554 and s.mtp_on
