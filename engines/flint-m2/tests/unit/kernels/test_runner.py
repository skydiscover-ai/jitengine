"""The residency probe's bounded-wait checks, on a stub GPU that replays each regime."""

import time


def test_residency_probe_timing():  # noqa: PLR0915
    """The residency probe refuses a grid whose threadgroups
    are not co-resident by a short bounded wait and by time, not by an error word of a long wait. With the M = 1 pass
    constants read from threadgroup memory a non-co-resident G = 64 grid completes its real passes (~24 ms) and on some
    tries its flag ring with no error word. Its barrier-only passes time out at SPIN_BAR (its late threadgroups arrive
    after the others' bounded wait), or, time-sliced, take >= 12 ms per dispatch. A stub GPU replays each regime: G = 64
    is refused at the chained barrier-only check on every try, before any later check runs; G = 40 passes, also with one
    slow pass in a try or a failed first try (a transient co-tenant); warm_verify applies the same check to every verify
    variant."""
    from types import SimpleNamespace

    import mlx.core as mx
    import numpy as np

    from jitengine_flint_m2.kernels import plan as P
    from jitengine_flint_m2.kernels import runner as RN

    NB = RN.BAR_PASSES
    G64 = {
        "first": 0.0,
        "bar": lambda i: (0.4, 7),
        "ring": 5.0,
        "real": 24.0,
    }  # barrier 7 times out at SPIN_BAR
    G64t = {**G64, "bar": lambda i: 0.3 if i % NB == 0 else 13.5}  # time-sliced, no spin accrued
    G40 = {"first": 0.0, "bar": lambda i: 0.2, "ring": 0.3, "real": 7.0}
    spins = []

    def stub(ms):
        k = object.__new__(RN.MegaKernel)
        k.G, k.nbar, k.cfg, k.probe_trace, k.vrows = (
            64,
            150,
            SimpleNamespace(rope_theta=1e4),
            [],
            (),
        )
        seen = []

        def run(prm, words, err_at):
            kind = (
                "ring"
                if prm[P.P_PROBE] > 1
                else "first"
                if prm[P.P_PROBE] == 1
                else "bar"
                if prm[P.P_MASK] == 0
                else "real"
            )
            if kind == "bar":
                spins.append(int(prm[P.P_SPIN]))
            t = ms[kind](seen.count(kind)) if callable(ms[kind]) else ms[kind]
            t, err = t if isinstance(t, tuple) else (t, 0)
            seen.append(kind)
            end = time.perf_counter() + t / 1e3
            while time.perf_counter() < end:  # a spin: time.sleep overshoots by ~50%
                pass
            res = np.zeros(words, dtype=np.uint32)
            res[err_at] = err
            return mx.array(res)

        k.run_pass = lambda hist, prm, prev=None, dep=None: run(prm, P.RES_WORDS, P.R_ERR)
        k.run_vpass = lambda hist, prm, prev=None: run(prm, P.VRES_WORDS, P.VR_ERR)
        k.ensure_vscratch = lambda: mx.zeros((1,))
        return k, seen

    # measured margins (runner.py): G = 40 waits ~1 us against a ~0.3 ms bound; its chained passes <= 0.4 ms each
    # against the time bound; a time-sliced G = 64 dispatch >= 12 ms
    assert NB >= 4 and RN.SPIN_BAR <= 1 << 12 and 0.4 * 5 <= RN.BAR_MS_MAX <= 12.0 / 3
    att = P.attn_params()
    for regime, why in ((G64, "grid barrier 7 timed out"), (G64t, "over its")):
        k, seen = stub(regime)
        try:
            k.probe([], att, 8)
            raise AssertionError("the G = 64 signature was admitted")
        except RN.BarrierProbeError as exc:
            assert "barrier-only passes" in str(exc) and why in str(exc), exc
        assert seen == ["first"] + ["bar"] * (NB * RN.PROBE_TRIES), seen
        assert len(k.probe_trace) == RN.PROBE_TRIES, k.probe_trace
    assert set(spins) == {RN.SPIN_BAR}, set(spins)
    slow_pass = {**G40, "bar": lambda i: 6.8 if i == 1 else 0.4}  # one delayed pass: amortised
    failed_try = {
        **G40,
        "bar": lambda i: (0.4, 3) if i < NB else 0.2,
    }  # a transient timeout: the next try passes
    for regime, bar_calls in ((G40, NB), (slow_pass, NB), (failed_try, 2 * NB)):
        k, seen = stub(regime)
        r = k.probe([], att, 8)
        assert set(r) == {"bar_ms", "ring_ms", "real_ms"} and r["bar_ms"] <= RN.BAR_MS_MAX, r
        assert (
            seen.count("bar") == bar_calls
            and seen.count("ring") == 1
            and seen.count("real") == RN.REAL_PASSES
        )
    k, seen = stub(G64)
    try:
        k.warm_verify([], att, 8, rows=(2,))
        raise AssertionError("the G = 64 signature was admitted for verify")
    except RN.BarrierProbeError as exc:
        assert "verify pass M=2: " in str(exc) and "barrier-only passes" in str(exc), exc
    assert "ring" not in seen and "real" not in seen
    k, seen = stub(G40)
    assert set(k.warm_verify([], att, 8, rows=(2, 3))) == {2, 3} and k.vrows == (2, 3)
