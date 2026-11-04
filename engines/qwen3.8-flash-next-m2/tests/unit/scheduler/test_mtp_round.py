"""An MTP round that raises fails its own row, so the worker loop does not run it again."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.scheduler.decode import DecodingMixin


class RaisingModel:
    def forward_hidden(self, *args):
        raise RuntimeError("first round failed")

    def verify2(self, *args):
        raise RuntimeError("verify failed")


def engine(failed):
    return SimpleNamespace(
        model=RaisingModel(),
        mtp=object(),
        expert_pool=SimpleNamespace(decay_tokens=32),
        mtp_gen=0,
        row_faults=0,
        _step_no=0,
        _telem_tokens=0,
        _set_owners=lambda owners: None,
        _ensure_room=lambda row, n: None,
        _fail=lambda row, exc: failed.append((row.rid, type(exc).__name__)),
    )


def lone_row(mtp=None):
    state = SimpleNamespace(layers=[], ple={}, tokens=[1, 2])
    return SimpleNamespace(rid="r1", state=state, next_tok=3, mtp=mtp, max_tokens=8, logprobs_k=0)


@pytest.mark.parametrize("round_", ["first", "verify"])
def test_a_raising_round_fails_its_row_and_drops_its_draft(round_):
    failed = []
    eng = engine(failed)
    row = lone_row()
    if round_ == "verify":
        row.mtp = {"st": {}, "d": mx.array([5]), "pos": row.state.tokens, "t": 3, "gen": 0}
    assert DecodingMixin._mtp_round(eng, row) == 0
    assert failed == [("r1", "RuntimeError")]
    assert row.mtp is None
    assert (eng.row_faults, eng._step_no) == (0, 0)
