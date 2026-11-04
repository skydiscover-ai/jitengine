"""Faults in the scheduler's memory policy, on the tiny model.

The policy runs every round before admission and decoding: it regrows the elastic extension, loads
the MTP head into it, and lends the prefix store to the annex. Each of these is a pure speed-up, so
one that fails (a head file gone or truncated, an allocation failure) is turned off for the engine's
life and requests are served without it, with the same tokens; /stats and /health list it. A policy
fault nothing contains stops the engine after `STUCK_FAULT_ROUNDS` rounds: every request gets an
error and /health is not ok, instead of requests hanging while /health said ok.
"""

import threading

import pytest

OPTS = dict(
    prefix_grid_tokens=64,
    prefix_min_divergence_tokens=16,
    landing_slots=2,
    prefetch_reserve=2,
    pool_ext_slots=4,
    ext_scratch_tokens=48,
    log_requests=False,
)
SHORT = list(range(1000, 1030))  # under ext_scratch_tokens: the extension stays live
LONG = list(range(2000, 2100))  # over it: the extension is released for the prefill, then regrown
WAIT_S = 30.0


def _engine(flashnext_engine, monkeypatch, annex: int = 0, **opts):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 16  # of 32 records: the extension is live (test_store_spill_annex's plan)
        plan.store_spill_slots = annex
        return plan

    with monkeypatch.context() as mp:
        mp.setattr(fne, "plan_budget", capped)
        e = flashnext_engine(**OPTS, **opts)
    e.mtp_on, e.mtp = False, None  # the tiny checkpoint has no head; the tests stand one in
    return e


def _run(fn, timeout=WAIT_S):
    """fn() on a client thread: its result, the exception it raised, or TimeoutError (a hang)."""
    out = {}

    def go():
        try:
            out["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            out["v"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    return out.get("v", TimeoutError(f"no result in {timeout} s"))


def _gen(e, prompt, n=32):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return _run(lambda: e.generate(GenRequest(prompt_ids=prompt, max_tokens=n, stop_at_eos=False)))


def _tokens(res):
    assert not isinstance(res, BaseException), repr(res)
    assert res.finish_reason == "length", res
    return res.token_ids


@pytest.fixture
def reference(flashnext_engine, monkeypatch, requires_mlx):
    e = _engine(flashnext_engine, monkeypatch, annex=6)
    return {tuple(p): _tokens(_gen(e, p)) for p in (SHORT, LONG)}


def test_an_mtp_head_that_cannot_load_turns_mtp_off_and_serves(
    flashnext_engine, monkeypatch, reference
):
    e = _engine(flashnext_engine, monkeypatch)
    calls = []

    def unloadable():
        calls.append(1)
        raise FileNotFoundError(2, "No such file or directory", "mtp.safetensors")

    e._mtp_load, e.mtp_on = unloadable, True
    try:
        for p in (SHORT, LONG, SHORT):
            assert _tokens(_gen(e, p)) == reference[tuple(p)]
        s = e.stats()
        assert len(calls) == 1 and not e.mtp_on and s["mtp"]["enabled"] is False
        assert "mtp.safetensors" in s["features_off"]["mtp"] and s["scheduler_faults"] == 0
        h = e.health()
        assert h["status"] == "ok" and "mtp" in h["features_off"]
    finally:
        e._mtp_load = lambda: None  # a tree that retries the load every round recovers here


def test_an_extension_that_cannot_regrow_is_off_and_requests_are_served(
    flashnext_engine, monkeypatch, reference
):
    e = _engine(flashnext_engine, monkeypatch)
    pool = e.expert_pool
    assert pool.ext_active  # the single-row operating point holds the extension
    real = pool.regrow_ext

    def fails(n=None):
        raise RuntimeError("[metal::malloc] simulated allocation failure")

    pool.regrow_ext = fails
    try:
        for p in (LONG, SHORT, LONG):  # LONG releases the extension; the regrow after it fails
            assert _tokens(_gen(e, p)) == reference[tuple(p)]
        s = e.stats()
        assert "simulated" in s["features_off"]["extension"] and s["scheduler_faults"] == 0
        assert not pool.ext_active and e.health()["status"] == "ok"
    finally:
        pool.regrow_ext = real


def test_an_annex_that_cannot_regrow_is_off_and_the_store_comes_back(
    flashnext_engine, monkeypatch, reference
):
    e = _engine(flashnext_engine, monkeypatch, annex=6)
    pool = e.expert_pool
    real = pool.regrow_ann

    def fails(n=None):
        raise RuntimeError("[metal::malloc] simulated allocation failure")

    pool.regrow_ann = fails
    try:
        for p in (LONG, LONG, SHORT):  # the second LONG resumes from the store's entry
            assert _tokens(_gen(e, p)) == reference[tuple(p)]
        s = e.stats()
        assert "simulated" in s["features_off"]["annex"] and s["scheduler_faults"] == 0
        assert not e.store.spilled and s["prefix_store"]["hits"] >= 1
        assert s["expert_pool_ann_live"] == 0 and e.health()["status"] == "ok"
    finally:
        pool.regrow_ann = real


def test_a_policy_fault_nothing_contains_fails_every_request_and_health(
    flashnext_engine, monkeypatch, requires_mlx
):
    e = _engine(flashnext_engine, monkeypatch)
    assert e.health()["status"] == "ok"

    def broken():
        raise RuntimeError("policy fault")

    e._ext_policy = broken
    try:
        r1 = _gen(e, SHORT)
        assert isinstance(r1, RuntimeError) and "policy fault" in str(r1), repr(r1)
        h = e.health()
        assert h["status"] == "error" and "policy fault" in h["error"]
        r2 = _gen(e, LONG)
        assert isinstance(r2, RuntimeError) and "policy fault" in str(r2), repr(r2)
        sc = _run(lambda: e.score(SHORT, (), 2))
        assert isinstance(sc, RuntimeError), repr(sc)
        s = e.stats()
        assert (
            s["scheduler_faults"] <= 2
            and s["batcher"]["live_rows"] == 0
            and s["batcher"]["queued"] == 0
        )
    finally:
        e._ext_policy = lambda: None  # a tree that retries the fault every round recovers here
