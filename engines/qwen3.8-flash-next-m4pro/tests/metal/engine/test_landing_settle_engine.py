"""A fault on the model thread while a layer's landing reads are in flight loses no landing slot, on the
tiny model.

Plan: 24 of the 32 records resident, a landing region of 3, a reserve of 3, no extension, no annex.
Two fault sites inside the window a decode layer's fetch is open:
  during      the miss chain's prefetch (run from `during`, inside `ensure`, reads in flight) raises
              an MLX-allocation-style error on every 2nd call;
  correction  the layer's correction (built after the split `ensure` returned, before its finish)
              raises on every 2nd call that has a demand miss.
Each faulting request fails, as a step fault does. Afterwards every region slot is free or parked
(`expert_landing_live` == `expert_landing_slots` on /stats), and a clean pass takes the landing path
again with the fault-free engine's tokens. Before the fix the region drained for good: 0 landing
fetches and a synchronous read for every demand miss, while /stats still said 3."""

import threading
import time

import pytest

PROMPTS = [list(range(1000 + 37 * i, 1000 + 37 * i + 30)) for i in range(6)]
N = 32
OPTS = dict(
    prefix_grid_tokens=64,
    prefix_min_divergence_tokens=16,
    landing_slots=3,
    prefetch_reserve=3,
    pool_ext_slots=0,
    log_requests=False,
)


def _engine(flashnext_engine, monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 24
        plan.store_spill_slots = 0
        return plan

    with monkeypatch.context() as mp:
        mp.setattr(fne, "plan_budget", capped)
        e = flashnext_engine(**OPTS)
    assert e.expert_pool.landing_n == 3, "precondition: the plan carries a landing region of 3"
    return e


def _gen(e, prompt, timeout=60.0):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    out = {}

    def go():
        try:
            out["v"] = e.generate(
                GenRequest(prompt_ids=prompt, max_tokens=N, stop_at_eos=False)
            ).token_ids
        except BaseException as exc:  # noqa: BLE001
            out["v"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    return out.get("v", TimeoutError("no result"))


def _census(e):
    """(region slots free or parked, region size), once the engine is idle."""
    time.sleep(0.3)
    p = e.expert_pool
    return len(p._landing_free) + len(p._orphans), p.landing_n


def _fetches(e):
    s = e.stats()
    return s["expert_landing_fetches"], s["expert_landing_fallbacks"]


@pytest.fixture
def reference(flashnext_engine, monkeypatch, requires_mlx):
    e = _engine(flashnext_engine, monkeypatch)
    toks = [_gen(e, p) for p in PROMPTS]
    f0, b0 = _fetches(e)
    assert [_gen(e, p) for p in PROMPTS] == toks
    f1, b1 = _fetches(e)
    return {"tokens": toks, "clean_fetches": f1 - f0, "clean_fallbacks": b1 - b0}


def _arm(e, site, monkeypatch):
    st = {"calls": 0, "fired": 0, "armed": True}
    lk = threading.Lock()

    def due() -> bool:
        with lk:
            st["calls"] += 1
            fire = st["armed"] and st["calls"] % 2 == 0
            st["fired"] += int(fire)
        return fire

    if site == "during":
        pool = e.expert_pool
        real = pool.prefetch

        def prefetch(gids):  # only the miss chain's `during` calls it
            if due():
                raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
            return real(gids)

        monkeypatch.setattr(pool, "prefetch", prefetch)
    else:
        cls = type(e.model.layers[0].moe)
        real = cls._correction

        def correction(self, rec, flat, slots_all, missing):
            if missing.any() and due():
                raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
            return real(self, rec, flat, slots_all, missing)

        monkeypatch.setattr(cls, "_correction", correction)
    return st


@pytest.mark.parametrize("site", ["during", "correction"])
def test_a_fault_while_landing_reads_are_in_flight_loses_no_landing_slot(
    flashnext_engine, monkeypatch, reference, site
):
    e = _engine(flashnext_engine, monkeypatch)
    st = _arm(e, site, monkeypatch)
    outcomes = [_gen(e, p) for _ in range(2) for p in PROMPTS]
    st["armed"] = False
    failed = [o for o in outcomes if isinstance(o, BaseException)]
    assert st["fired"] > 0 and failed, f"precondition: the fault fired inside the window ({st})"
    assert all("Resource limit" in str(o) for o in failed), failed[:2]
    live, n = _census(e)
    s = e.stats()
    assert live == n and s["expert_landing_live"] == s["expert_landing_slots"] == n, (live, n)
    assert e.expert_pool.open_fetches() == 0 and e.health()["status"] == "ok"
    f0, b0 = _fetches(e)
    assert [_gen(e, p) for p in PROMPTS] == reference["tokens"]
    f1, b1 = _fetches(e)
    assert f1 - f0 >= max(1, reference["clean_fetches"] // 2), (f1 - f0, b1 - b0, reference)
    assert _census(e) == (n, n)
