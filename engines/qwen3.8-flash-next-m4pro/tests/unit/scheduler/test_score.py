"""Teacher-forced scoring, on MLX's CPU build: the forward partition, the logit readout, and
`Engine.score` against a plain forward pass of the model on the same tiny checkpoint (the engine
runs on the stock ops, as in tests/unit/engine/test_engine.py)."""

from itertools import pairwise

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.scheduler import score as sc

PROMPT, CONT = [3, 4, 5], [6, 7, 8]


# ---------------------------------------------------------------- the partition
def test_score_chunks_end_on_the_grid_and_finish_with_the_last_token_alone():
    assert sc.score_chunks(1, 4) == [(0, 1)]
    assert sc.score_chunks(2, 4) == [(0, 1), (1, 2)]
    assert sc.score_chunks(10, 4) == [(0, 4), (4, 8), (8, 9), (9, 10)]
    assert sc.score_chunks(9, 4) == [(0, 4), (4, 8), (8, 9)]
    assert sc.score_chunks(5, 100) == [(0, 4), (4, 5)]


@pytest.mark.parametrize("n", [1, 2, 7, 64, 129])
@pytest.mark.parametrize("grid", [1, 3, 64])
def test_score_chunks_cover_every_position_once_in_order(n, grid):
    chunks = sc.score_chunks(n, grid)
    assert chunks[0][0] == 0 and chunks[-1] == (n - 1, n)
    assert all(a < b for a, b in chunks)
    assert all(b == c for (_, b), (c, _) in pairwise(chunks))
    # every chunk but the last two ends on the absolute grid
    assert all(b % grid == 0 for _, b in chunks[:-2])
    assert sc.chunk_calls(n, grid) == chunks


def test_score_chunks_clamp_the_grid_and_refuse_an_empty_sequence():
    assert sc.score_chunks(3, 0) == sc.score_chunks(3, 1)
    assert sc.score_chunks(3, -5) == sc.score_chunks(3, 1)
    with pytest.raises(ValueError, match="at least one token"):
        sc.score_chunks(0, 4)


# ---------------------------------------------------------------- the readout
def _logits(rows):
    return mx.array(np.array(rows, dtype=np.float32))


def test_readout_gives_normalized_logprobs_sorted_by_value_then_token_id():
    logits = _logits([[1.0, 3.0, 3.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    r = sc.Readout(logits, 3, targets=[4, 2])
    mx.eval(*r.arrays())
    first, second = r.entries()
    lp = np.array(logits[0]) - np.log(np.exp(np.array(logits[0])).sum())
    # a tie of 3.0 and 3.0 is broken by the lower id first
    assert [t for t, _ in first["top"]] == [1, 2, 4]
    assert [v for _, v in first["top"]] == pytest.approx([lp[1], lp[2], lp[4]], abs=1e-6)
    assert first["logprob"] == pytest.approx(lp[4], abs=1e-6)
    # a uniform row: every logprob is -log V, and the top-3 are the three lowest ids
    assert [t for t, _ in second["top"]] == [0, 1, 2]
    assert second["logprob"] == pytest.approx(-np.log(5), abs=1e-6)
    assert all(v == pytest.approx(-np.log(5), abs=1e-6) for _, v in second["top"])


def test_readout_without_targets_has_no_logprob_and_fewer_targets_leave_the_rest_none():
    logits = _logits([[0.0, 1.0, 2.0]] * 3)
    r = sc.Readout(logits, 2)
    mx.eval(*r.arrays())
    assert [e["logprob"] for e in r.entries()] == [None, None, None]
    r = sc.Readout(logits, 2, targets=mx.array([2, 0], dtype=mx.int32))
    mx.eval(*r.arrays())
    got = [e["logprob"] for e in r.entries()]
    assert got[2] is None and got[0] > got[1]


@pytest.mark.parametrize("k, want", [(0, 1), (-3, 1), (2, 2), (3, 3), (99, 3)])
def test_readout_clamps_k_between_one_and_the_vocabulary(k, want):
    r = sc.Readout(_logits([[0.5, 0.1, 0.9]]), k)
    mx.eval(*r.arrays())
    (entry,) = r.entries()
    assert len(entry["top"]) == want
    assert [v for _, v in entry["top"]] == sorted((v for _, v in entry["top"]), reverse=True)


def test_decode_entry_scores_the_drawn_token_from_an_int_or_an_array():
    logits = _logits([[0.0, 1.0, 2.0, 3.0]])
    for drawn in (1, mx.array([1], dtype=mx.int32)):
        r = sc.decode_entry(logits, 2, drawn)
        mx.eval(*r.arrays())
        (entry,) = r.entries()
        want = 1.0 - np.log(np.exp([0.0, 1.0, 2.0, 3.0]).sum())
        assert entry["logprob"] == pytest.approx(want, abs=1e-6)
        assert [t for t, _ in entry["top"]] == [3, 2]
    r = sc.decode_entry(logits, 2)
    mx.eval(*r.arrays())
    assert r.entries()[0]["logprob"] is None


# ---------------------------------------------------------------- Engine.score against the model
@pytest.fixture(scope="module")
def engine(cpu_flashnext_engine):
    return cpu_flashnext_engine()


def _reference(e, seq):
    """[n, V] fp32 log-softmax of a plain forward pass of the model over `seq`: the head in one
    forward and the last token in another (the partition a cold generate row makes), through
    `head` without the row-exact option."""
    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState

    n = len(seq)
    state = RequestState(e.cfg, n, 0, reserve=0, dtype=e.compute_dtype)
    parts = []
    for a, b in sc.score_chunks(n, e.wave_chunk):
        toks = mx.array(seq[a:b], dtype=mx.int32)[None]
        parts.append(e.model.forward(toks, state, logits="hidden")[0])
    logits = e.model.head(mx.concatenate(parts, axis=0)).astype(mx.float32)
    out = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(out)
    return np.array(out)


def test_each_continuation_token_has_the_logprob_a_plain_forward_gives(engine):
    seq = PROMPT + CONT
    entries = engine.score(PROMPT, CONT, top_k=4)
    ref = _reference(engine, seq)
    assert len(entries) == len(seq)
    for i, entry in enumerate(entries[:-1]):
        assert entry["logprob"] == pytest.approx(ref[i, seq[i + 1]], abs=2e-3), i
    assert entries[-1]["logprob"] is None


def test_top_k_entries_are_the_reference_top_k_sorted_descending_then_by_id(engine):
    seq = PROMPT + CONT
    entries = engine.score(PROMPT, CONT, top_k=5)
    ref = _reference(engine, seq)
    for i, entry in enumerate(entries):
        top = entry["top"]
        assert len(top) == 5
        keys = [(-v, t) for t, v in top]
        assert keys == sorted(keys)
        for t, v in top:
            assert v == pytest.approx(ref[i, t], abs=2e-3)
        # the best entry is the reference's argmax (the head of a random model is not flat)
        assert top[0][0] == int(ref[i].argmax())
        assert top[0][1] >= float(np.sort(ref[i])[-1]) - 2e-3


def test_a_prompt_with_no_continuation_still_scores_every_position(engine):
    entries = engine.score(PROMPT, top_k=2)
    assert len(entries) == len(PROMPT)
    assert [e["logprob"] is None for e in entries] == [False, False, True]
    assert all(len(e["top"]) == 2 for e in entries)


def test_the_prompt_and_continuation_split_does_not_change_the_numbers(engine):
    a = engine.score(PROMPT, CONT, top_k=3)
    b = engine.score(PROMPT + CONT, [], top_k=3)
    c = engine.score([], PROMPT + CONT, top_k=3)
    assert a == b == c


def test_score_is_repeatable_and_leaves_generation_unchanged(engine):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    req = GenRequest(prompt_ids=[3, 4, 5, 6], max_tokens=3, stop_at_eos=False)
    before = engine.generate(req).token_ids
    first = engine.score(PROMPT, CONT, top_k=3)
    assert engine.score(PROMPT, CONT, top_k=3) == first
    assert engine.generate(req).token_ids == before


def test_top_k_below_one_is_one(engine):
    entries = engine.score(PROMPT, CONT, top_k=0)
    assert all(len(e["top"]) == 1 for e in entries)


def test_score_is_booked_as_its_own_phase(engine):
    before = engine.stats()["phases"]["score"]
    engine.score(PROMPT, CONT)
    after = engine.stats()["phases"]["score"]
    assert after["tokens"] - before["tokens"] == len(PROMPT) + len(CONT)
    assert after["calls"] - before["calls"] == 2  # the head, then the last token


# ---------------------------------------------------------------- bad inputs
def test_an_empty_sequence_is_a_value_error_and_is_counted(engine):
    before = engine.stats()["error_types"].get("ValueError", 0)
    with pytest.raises(ValueError, match="empty prompt"):
        engine.score([], [])
    assert engine.stats()["error_types"]["ValueError"] == before + 1


@pytest.mark.parametrize("which", ["prompt", "continuation"])
def test_a_token_id_outside_the_embedding_is_refused_before_any_forward(engine, which):
    limit = engine.token_id_limit
    for bad in (-1, limit):
        args = ([3, bad], [4]) if which == "prompt" else ([3], [4, bad])
        with pytest.raises(ValueError, match=f"token id {bad} out of range"):
            engine.score(*args)
    assert engine.score([limit - 1], [0], top_k=1)  # the last valid id is scored


def test_the_engine_keeps_serving_after_a_refused_score(engine):
    with pytest.raises(ValueError):
        engine.score([-5], [])
    assert engine.health()["status"] == "ok"
    assert len(engine.score(PROMPT, CONT, top_k=1)) == len(PROMPT) + len(CONT)


# ---------------------------------------------------------------- the pieces on a state
def _state(e, n):
    from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState

    return RequestState(e.cfg, n, 0, reserve=0, dtype=e.compute_dtype)


def test_score_sequence_runs_its_hooks_around_every_chunk_and_matches_the_engine(engine):
    seq = list(range(10, 20))
    calls = []
    got = sc.score_sequence(
        engine.model,
        _state(engine, len(seq)),
        seq,
        4,
        3,
        before_chunk=lambda s0, e: calls.append(("before", s0, e)),
        on_chunk=lambda s0, e: calls.append(("after", s0, e)),
    )
    chunks = sc.score_chunks(len(seq), 4)
    assert calls == [(w, a, b) for a, b in chunks for w in ("before", "after")]
    assert len(got) == len(seq) and got[-1]["logprob"] is None
    # the engine's grid differs, so only the next-token predictions are compared, loosely
    want = engine.score(seq[:5], seq[5:], top_k=3)
    assert [e["top"][0][0] for e in got] == [e["top"][0][0] for e in want]


def test_readout_rows_do_not_depend_on_how_many_rows_are_read_at_once(engine):
    seq = list(range(30, 41))
    state = _state(engine, len(seq))
    toks = mx.array(seq[:-1], dtype=mx.int32)[None]
    h = engine.model.forward(toks, state, logits="hidden")
    mx.eval(h)
    whole = sc.readout_rows(engine.model, h, 0, seq, 3)
    split = sc.readout_rows(engine.model, h, 0, seq, 3, rows_per_group=3)
    single = sc.readout_rows(engine.model, h, 0, seq, 3, rows_per_group=0)  # clamped to 1
    assert len(whole) == len(seq) - 1
    assert whole == split == single
    # the targets are the next tokens of `seq`
    assert all(e["logprob"] is not None for e in whole)


def test_served_entry_reads_a_single_token_forward(engine):
    entry = sc.served_entry(engine.model, _state(engine, 2), 7, 3)
    assert len(entry) == 1 and entry[0]["logprob"] is None and len(entry[0]["top"]) == 3
