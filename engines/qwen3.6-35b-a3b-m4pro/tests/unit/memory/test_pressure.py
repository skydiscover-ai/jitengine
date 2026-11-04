"""Memory-pressure policy unit tests on synthetic observations."""

from __future__ import annotations

from jitengine_qwen3_6_35b_a3b_m4pro.memory.monitor import MemorySample
from jitengine_qwen3_6_35b_a3b_m4pro.memory.pressure import GIB, MIB, PressurePolicy

LIMIT = 18 << 30
MINB = 12 << 30


def S(t, fp=10 * GIB, avail=8 * GIB, swap=100 * MIB, err=None):
    return MemorySample(float(t), fp, avail, swap, err)


def run(policy, samples, idle=True, limit=16 << 30):
    return [policy.observe(s, idle=idle, current_limit_bytes=limit) for s in samples]


def test_missing_telemetry_never_grows_pool():
    policy = PressurePolicy(process_limit_bytes=18 << 30, minimum_bytes=12 << 30, mode="offload")
    for second in range(65):
        decision = policy.observe(
            MemorySample(float(second), None, None, None, "probe denied"),
            idle=True,
            current_limit_bytes=16 << 30,
        )
        assert decision.action != "grow"
    assert policy.degraded


def test_warning_needs_three_low_samples_then_trims_once():
    p = PressurePolicy(LIMIT, MINB, "offload")
    d = run(p, [S(0, avail=int(1.5 * GIB)), S(1, avail=int(1.5 * GIB))])
    assert [x.action for x in d] == ["keep", "keep"]
    d = run(p, [S(2, avail=int(1.5 * GIB)), S(3, avail=int(1.5 * GIB))])
    assert [x.action for x in d] == ["trim", "keep"] and p.state == "warning"


def test_noise_single_dip_does_not_trigger():
    p = PressurePolicy(LIMIT, MINB, "offload")
    d = run(p, [S(0), S(1, avail=int(0.5 * GIB)), S(2), S(3, avail=int(1.5 * GIB)), S(4)])
    assert all(x.action == "keep" for x in d)


def test_critical_pauses_trims_then_shrinks_when_idle():
    p = PressurePolicy(LIMIT, MINB, "offload")
    low = int(0.5 * GIB)
    acts = [x.action for x in run(p, [S(t, avail=low) for t in range(0, 40)], idle=True)]
    assert acts[:4] == ["keep", "pause", "trim", "shrink"]
    assert acts.count("shrink") == 2  # spaced 30 s from the proposal: t = 3, 33
    p2 = PressurePolicy(LIMIT, MINB, "offload")
    shrinks = 0
    for t in range(0, 70):
        d = p2.observe(S(t, avail=low), idle=True, current_limit_bytes=16 << 30)
        if d.action == "shrink":
            shrinks += 1
            assert d.target_bytes == (16 << 30) - 512 * MIB
            p2.note_result(d, True, float(t))
    assert shrinks == 3  # t = 3, 33, 63


def test_shrink_never_below_minimum():
    p = PressurePolicy(LIMIT, MINB, "offload")
    ds = [
        p.observe(S(t, avail=int(0.5 * GIB)), idle=True, current_limit_bytes=MINB + 100 * MIB)
        for t in range(0, 5)
    ]
    d = next(x for x in ds if x.action == "shrink")
    assert d.target_bytes == MINB
    d = p.observe(S(40, avail=int(0.5 * GIB)), idle=True, current_limit_bytes=MINB)
    assert d.action == "keep"


def test_pressure_while_busy_pauses_but_does_not_resize():
    p = PressurePolicy(LIMIT, MINB, "offload")
    acts = [x.action for x in run(p, [S(t, avail=int(0.5 * GIB)) for t in range(10)], idle=False)]
    assert "shrink" not in acts and "pause" in acts and "abort" not in acts


def test_over_ceiling_while_busy_aborts():
    p = PressurePolicy(LIMIT, MINB, "offload")
    d = p.observe(S(0, fp=LIMIT + 1), idle=False, current_limit_bytes=LIMIT)
    assert d.action == "abort"


def test_swap_growth_is_critical():
    p = PressurePolicy(LIMIT, MINB, "offload")
    d = run(p, [S(0, swap=100 * MIB), S(10, swap=200 * MIB), S(20, swap=400 * MIB)])
    assert d[-1].action == "pause" and "swap" in d[-1].reason


def test_recovery_requires_30s_then_resumes_then_grows_slowly():
    p = PressurePolicy(LIMIT, MINB, "offload")
    run(p, [S(t, avail=int(0.5 * GIB)) for t in range(3)])
    assert p.paused
    acts = [
        (t, p.observe(S(t, avail=4 * GIB), idle=True, current_limit_bytes=15 << 30).action)
        for t in range(3, 100)
    ]
    first_resume = next(t for t, a in acts if a == "resume")
    assert first_resume >= 33
    grows = [t for t, a in acts if a == "grow"]
    assert grows and grows[0] > first_resume


def test_failed_growth_enters_cooldown():
    p = PressurePolicy(LIMIT, MINB, "offload")
    t0 = 0
    grow_t = None
    for t in range(t0, 200):
        d = p.observe(S(t, avail=5 * GIB), idle=True, current_limit_bytes=15 << 30)
        if d.action == "grow":
            if grow_t is None:
                grow_t = t
                p.note_result(d, False, float(t))
            else:
                assert t - grow_t >= 60
                break
    assert grow_t is not None


def test_growth_never_exceeds_ceiling():
    p = PressurePolicy(LIMIT, MINB, "offload")
    for t in range(200):
        d = p.observe(S(t, avail=6 * GIB), idle=True, current_limit_bytes=LIMIT - 100 * MIB)
        if d.action == "grow":
            assert d.target_bytes == LIMIT
            return
    raise AssertionError("never grew")


def test_resident_mode_never_resizes():
    p = PressurePolicy(LIMIT, MINB, "resident")
    acts = {
        p.observe(S(t, avail=int(0.5 * GIB)), idle=True, current_limit_bytes=LIMIT).action
        for t in range(60)
    }
    assert "shrink" not in acts and "grow" not in acts
    assert "resident floor" in p.last_decision.reason
