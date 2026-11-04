"""The megakernel's load check: its stock-MLX statistics, and stages chain, stock and verify on the fake GPU."""

import hashlib

import numpy as np

from jitengine_flint_m2 import loadcheck as LC
from jitengine_flint_m2.kernels import plan as P
from jitengine_flint_m2.loadcheck_text import P0_TEXT
from jitengine_flint_m2.logs import level_of
from jitengine_flint_m2.model.config import StartupError, load_config, tensor_index
from jitengine_flint_m2.weight_banks import BankPlan

WARN = "megakernel unavailable on this Mac ("


def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise {exc.__name__}")


def _words(refs) -> list:
    """Result words of a kernel that reproduces the reference rows."""
    out = []
    for ref in refs:
        w = np.zeros(P.RES_WORDS, dtype=np.uint32)
        w[P.R_ARGMAX] = ref["argmax"]
        w[P.R_TOPI : P.R_TOPI + 20] = ref["ids"]
        w[P.R_TOPV : P.R_TOPV + 20] = ref["lp"].astype(np.float32).view(np.uint32)
        out.append(w)
    return out


def _shift_lp(words, delta) -> list:
    out = [w.copy() for w in words]
    for w in out:
        w[P.R_TOPV : P.R_TOPV + 20] = (w[P.R_TOPV : P.R_TOPV + 20].view(np.float32) + delta).view(
            np.uint32
        )
    return out


def test_reference_rows_and_d20():
    """ref_rows: stock MLX's argmax over the bf16 logits (lowest id on a tie) and its top-20 by fp64 log-probability
    (ties: lowest id first); position_d20: an id missing from the kernel's top-20 is bounded by its 20th log-prob."""
    rng = np.random.default_rng(19)
    V = 4096
    lg = rng.normal(size=(6, V)).astype(np.float32)
    lg[1, [7, 9, 3000]] = 9.0  # a three-way tie at the top
    lg[2, :40] = 5.0  # ties across the top-20 boundary
    lg = lg.astype(np.float16).astype(np.float32)  # coarse values: many exact ties
    refs = LC.ref_rows(lg)
    for row, ref in zip(lg, refs):
        x = row.astype(np.float64)
        lp = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
        order = np.argsort(-lp, kind="stable")[:20]
        assert ref["argmax"] == int(np.argmax(row)) and ref["ids"].tolist() == order.tolist()
        assert np.allclose(ref["lp"], lp[order], rtol=0, atol=1e-12)
    assert refs[1]["argmax"] == 7 and refs[1]["ids"][:3].tolist() == [7, 9, 3000]
    assert refs[2]["ids"].tolist() == list(range(20))

    def brute_d20(cids, clps, ref):
        cm = dict(zip(cids, clps))
        lowest = min(clps)
        pairs = zip(ref["ids"], ref["lp"])
        return max(abs(cm[t] - lr) if t in cm else max(0.0, lr - lowest) for t, lr in pairs)

    for ref in refs:
        ids = ref["ids"].tolist()
        for case in range(4):
            cids = list(ids)
            if case >= 1:
                cids[19 - case] = V - 1 - case  # an id the reference ranks lower
            if case == 3:
                cids = cids[::-1]  # order does not matter
            clps = [
                float(ref["lp"][ids.index(t)]) + 0.01 * k if t in ids else -30.0
                for k, t in enumerate(cids)
            ]
            assert abs(LC.position_d20(cids, clps, ref) - brute_d20(cids, clps, ref)) < 1e-12


def test_stock_verdict():
    """Top-1 agreement >= 22 of 32 tokens; the median d20 is reported, not checked (STOCK_D None), and checked only when a
    bound is given; a non-finite log-probability is an infinite d20."""
    rng = np.random.default_rng(19)
    lg = rng.normal(size=(6, 4096)).astype(np.float16).astype(np.float32)
    words = _words(LC.ref_rows(lg))
    st = LC.stock_stats(words, lg)
    assert st["n"] == 6 and st["agree"] == 1.0 and st["d20_max"] < 1e-5, st
    assert LC.stock_verdict(st) == (True, "")
    bad = _shift_lp(words[:3], -2.0) + words[3:]  # three positions 2 nat off, their top-1 moved
    for w in bad[:3]:
        w[P.R_ARGMAX] += 1
    st = LC.stock_stats(bad, lg)
    assert st["agree"] == 0.5 and 0.99 < st["d20_median"] < 1.01 and 1.99 < st["d20_max"] < 2.01, st
    ok, why = LC.stock_verdict(st)
    assert not ok and f"top-1 agreement 0.500 (needs >= {LC.STOCK_A})" in why, why
    assert LC.stock_verdict(st, A=0.5, D=2.1)[0] and not LC.stock_verdict(st, A=0.5, D=0.9)[0]
    assert LC.STOCK_A == 0.6875 and LC.STOCK_D is None
    far = _shift_lp(words, -5.0)
    st = LC.stock_stats(far, lg)
    assert st["agree"] == 1.0 and st["d20_median"] > 4.9 and LC.stock_verdict(st) == (True, "")
    ok, why = LC.stock_verdict(st, D=1.2)
    assert not ok and "median d20" in why
    nanw = [w.copy() for w in words]
    nanw[0][P.R_TOPV] = np.array([np.nan], dtype=np.float32).view(np.uint32)[0]
    assert LC.stock_stats(nanw, lg)["d20_max"] == float("inf")


def test_mutant_table_and_text(flint_headers):
    """wrong_expert moves exactly one routed layer's (bank, base) to the next routed layer's; the fixed text is
    p0-v1, byte for byte, and repeats when a tokenizer yields too few ids."""
    cfg = load_config(flint_headers)
    plan = BankPlan(cfg, tensor_index(flint_headers, cfg))
    tab0 = P.pack_descriptors(plan)
    tab, layer, nxt = LC.wrong_expert_table(plan)
    diff = np.nonzero(tab != tab0)[0].tolist()
    assert diff and set(diff) <= {layer * P.NF + P.D_EBANK, layer * P.NF + P.D_EBASE}, diff
    assert tab[layer * P.NF + P.D_EBASE] == tab0[nxt * P.NF + P.D_EBASE] and nxt in plan.expert_home
    assert hashlib.sha256(P0_TEXT.encode()).hexdigest()[:16] == "4e8dfbd5b2a23a69"
    info = LC.text_info(512, 32)
    assert info == {
        "version": "p0-v1",
        "sha256": "4e8dfbd5b2a23a69",
        "prompt_tokens": 512,
        "forced_tokens": 32,
    }

    class Few:
        def encode(self, text):
            return [1, 2, 3]

    assert LC.text_ids(Few(), 7) == [1, 2, 3, 1, 2, 3, 1]
    expect_raises(ValueError, LC.text_ids, type("Empty", (), {"encode": lambda self, t: []})(), 4)


def test_stages_pass(fake_gpu):
    """With nothing injected every stage passes and /health mk.loadcheck reports each; with speculation off the verify
    stage is skipped and the others run."""
    logs = []
    e = fake_gpu.engine(logs)
    h = e.health()
    lc = h["mk"]["loadcheck"]
    assert h["decode_backend"] == "mk" and h["multi_token"] and "mk_unavailable" not in h, h
    assert all(lc[s]["ok"] for s in ("compile", "probe", "chain", "stock", "verify")), lc
    assert lc["chain"]["chain_equal"] and lc["chain"]["repeat_equal"] and lc["chain"]["passes"] == 8
    assert (
        lc["stock"]["agree"] == 1.0 and lc["stock"]["d20_median"] < 1e-5 and lc["stock"]["n"] == 32
    )
    assert (lc["stock"]["A"], lc["stock"]["D"]) == (LC.STOCK_A, LC.STOCK_D)
    assert lc["verify"]["rows"] == list(range(2, 9))
    assert not any(c["rows_bad"] for c in lc["verify"]["cases"])
    assert lc["text"]["version"] == "p0-v1" and lc["bound_s"] == LC.BOUND_S and lc["s"] > 0, lc
    assert any("load check" in m and "chain ok" in m for m in logs)
    assert not [m for m in logs if "WARNING" in m]
    e = fake_gpu.engine([], speculation=False)
    lc = e.health()["mk"]["loadcheck"]
    assert (
        lc["verify"]["ok"] and "skipped" in lc["verify"] and lc["stock"]["ok"] and lc["chain"]["ok"]
    )


def test_each_mutant_caught_by_its_stage(fake_gpu):
    """chain_bitflip is caught by the chain stage; wrong_expert, pos_shift and stale_history by the stock stage (the
    fake's mutants move every top-1, so the agreement gate refuses them; on the real kernel they are subtle and only the
    reported median d20 shows them). Each is a fallback point: "auto" + main decodes with stock MLX ops, with one
    WARNING line and /health mk_unavailable {stage, reason, loadcheck}."""
    for mutant, stage in (
        ("chain_bitflip", "chain"),
        ("wrong_expert", "stock"),
        ("pos_shift", "stock"),
        ("stale_history", "stock"),
    ):
        fake_gpu.MK.reset()
        logs = []
        e = fake_gpu.engine(logs, sim={"check_mutant": mutant})
        h = e.health()
        assert h["simulated"]["check_mutant"] == mutant, h["simulated"]
        assert h["status"] == "ready" and h["decode_backend"] == "graph" and h["mk"] is None, (
            mutant,
            h,
        )
        mu = h["mk_unavailable"]
        assert mu["stage"] == stage and mu["loadcheck"]["mutant"] == mutant, mu
        assert not mu["loadcheck"][stage]["ok"], mu
        warnings = [m for m in logs if "WARNING" in m]
        assert len(warnings) == 1 and WARN in warnings[0], logs
        assert level_of(warnings[0]) == "warning"
        assert fake_gpu.Hist.released == 1 and e.mk is None and not h["multi_token"]
        if stage == "stock":
            assert mu["loadcheck"]["chain"]["ok"] and mu["loadcheck"]["stock"]["agree"] < LC.STOCK_A
            assert "top-1 agreement" in mu["reason"] and "median d20" in mu["reason"], mu["reason"]
        else:
            assert "differs from its one-dispatch-per-phase chain at pass 4: LOGITS" in mu["reason"]
        r = e.generate(_greedy(6))
        # the stub model, through the graph backend
        assert r.token_ids == fake_gpu.stream_of(32, 6), r


def _greedy(n):
    from jitengine_flint_m2.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=list(range(1, 33)), max_tokens=n, temperature=0.0, ignore_eos=True)


def test_verify_rows_turn_speculation_off(fake_gpu):
    """verify_row: the verify stage finds the row that differs; speculation goes off and the megakernel stays."""
    logs = []
    e = fake_gpu.engine(logs, sim={"check_mutant": "verify_row"})
    h = e.health()
    vf = h["mk"]["loadcheck"]["verify"]
    assert (
        h["decode_backend"] == "mk" and not h["multi_token"] and e.mk.vrows == () and not vf["ok"]
    )
    assert (
        vf["cases"][-1]["M"] == 8
        and vf["cases"][-1]["rows_bad"] == [7]
        and "differ" in vf["reason"]
    )
    assert all(not c["rows_bad"] for c in vf["cases"][:-1]) and h["mk"]["loadcheck"]["stock"]["ok"]
    warnings = [m for m in logs if "WARNING" in m]
    assert len(warnings) == 1 and "verify passes DISABLED by the load check" in warnings[0], logs
    assert level_of(warnings[0]) == "warning"


def test_failed_stage_refuses_kv8_mk_and_a_pinned_grid(fake_gpu):
    """kv8, decode_backend "mk" or a pinned grid: a failed chain or stock stage refuses, never falls back."""
    for opts, needle in (
        (dict(profile="kv8"), "profile kv8 needs the megakernel, unavailable on this Mac (stock: "),
        (dict(decode_backend="mk"), "(stock: "),
        (dict(mk_grid=40), "mk_grid=40 is pinned"),
    ):
        fake_gpu.MK.reset()
        h = fake_gpu.engine([], sim={"check_mutant": "wrong_expert"}, **opts).health()
        assert h["status"] == "unhealthy" and needle in h["fatal"] and "mk_unavailable" not in h, h
    fake_gpu.MK.reset()
    h = fake_gpu.engine([], sim={"check_mutant": "chain_bitflip"}, profile="kv8").health()
    assert h["status"] == "unhealthy" and "(chain: " in h["fatal"]


def test_mutants_are_a_test_hook(fake_gpu):
    """A mutant needs allow_fault_injection, and only known names are taken."""
    from jitengine_flint_m2.engine.engine import Engine

    kw = dict(start=False, context_limit=4096)
    expect_raises(StartupError, Engine, fake_gpu.model, sim={"check_mutant": "wrong_expert"}, **kw)
    expect_raises(
        StartupError,
        Engine,
        fake_gpu.model,
        sim={"check_mutant": "flip_everything"},
        allow_fault_injection=True,
        **kw,
    )


def test_bound(fake_gpu, monkeypatch):
    """A check that ends over its bound fails the stage it is in (here chain), and "auto" + main falls back."""
    monkeypatch.setattr(LC, "BOUND_S", 0.0)
    mu = fake_gpu.engine([]).health()["mk_unavailable"]
    assert mu["stage"] == "chain" and "over its 0 s bound" in mu["reason"], mu
