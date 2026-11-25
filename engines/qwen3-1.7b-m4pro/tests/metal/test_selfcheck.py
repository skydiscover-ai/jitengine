"""The startup self-check passes on the real kernel and switches off a deliberately broken one."""

import mlx.core as mx
import pytest

from jitengine_qwen3_1_7b_m4pro.kernels import qmm_exact as q
from jitengine_qwen3_1_7b_m4pro.kernels import selfcheck


@pytest.fixture(autouse=True)
def _clean():
    q.DISABLED.clear()
    q._plans.clear()
    yield
    q.DISABLED.clear()
    q._plans.clear()


def test_real_kernel_passes_on_every_shape():
    res = selfcheck.run()
    assert res["exact_shapes_ok"] and res["fusion_ok"] and res["disabled"] == []
    assert {c["shape"] for c in res["checks"]} >= {"qkv", "o", "gate_up", "down", "lm_head"}


def test_a_kernel_one_ulp_off_is_disabled(monkeypatch):
    real = q.qmm_exact

    def off_by_one_ulp(x, w, s, b, segments=None):
        out = real(x, w, s, b, segments)
        return None if out is None else (out.view(mx.uint16) + 1).view(mx.bfloat16)

    monkeypatch.setattr(q, "qmm_exact", off_by_one_ulp)
    res = selfcheck.run()
    assert not res["exact_shapes_ok"] and res["disabled"]
    assert q.plan(16, 2048, (2048,)) is None  # now left to stock


def test_a_kernel_that_raises_is_disabled(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("does not compile")

    monkeypatch.setattr(q, "qmm_exact", broken)
    res = selfcheck.run()
    assert not res["exact_shapes_ok"]
    assert any("does not compile" in (c.get("error") or "") for c in res["checks"])
    assert res["disabled"]


def test_a_disabled_kernel_produces_a_startup_warning(monkeypatch):
    real = q.qmm_exact

    def off(x, w, s, b, segments=None):
        out = real(x, w, s, b, segments)
        return None if out is None else (out.view(mx.uint16) + 1).view(mx.bfloat16)

    monkeypatch.setattr(q, "qmm_exact", off)
    texts = selfcheck.warnings(selfcheck.run())
    assert texts and "slower than its published speed" in texts[0]


def test_a_clean_check_has_no_warning():
    assert selfcheck.warnings(selfcheck.run()) == []


def test_a_model_whose_layers_cannot_be_read_is_checked_on_generated_weights():
    result = selfcheck.run_for(object())  # a LoRA adapter wraps the layers; this has none at all
    assert result["exact_shapes_ok"] and result["fusion_ok"]
    assert "generated weights" in result["note"]


def _ulp_off(monkeypatch):
    real = q.qmm_exact

    def off(x, w, s, b, segments=None):
        out = real(x, w, s, b, segments)
        return None if out is None else (out.view(mx.uint16) + 1).view(mx.bfloat16)

    monkeypatch.setattr(q, "qmm_exact", off)


def test_the_fused_fallback_is_checked_at_13_to_16_rows_when_the_kernel_is_off(monkeypatch):
    from jitengine_qwen3_1_7b_m4pro.engine import fusion

    _ulp_off(monkeypatch)
    res = selfcheck.run()
    fused = {(c["shape"], c["rows"]) for c in res["checks"] if c["path"] == "fused-stock"}
    assert {("qkv", r) for r in range(1, 17)} <= fused
    bad = [c["rows"] for c in res["checks"] if c["path"] == "fused-stock" and not c["ok"]]
    # joined stock really differs from the unfused calls at some of these row counts; fused decode
    # must then stop below the first one (the fused fallback is not exact there)
    assert fusion.FUSED_MAX_ROWS == (min(bad) - 1 if bad else 16)
    assert res["fusion_ok"] == (not bad)
    fusion.FUSED_MAX_ROWS = 16


def test_fused_decode_stops_below_the_first_row_count_that_is_not_exact(monkeypatch):
    from jitengine_qwen3_1_7b_m4pro.engine import fusion

    _ulp_off(monkeypatch)
    real_same = selfcheck._same
    monkeypatch.setattr(selfcheck, "_same", lambda a, b: a.shape[0] < 6 and real_same(a, b))
    try:
        res = selfcheck.run()
        assert not res["fusion_ok"] and res["fused_max_rows"] == fusion.FUSED_MAX_ROWS == 5
        assert "above 5 rows" in " ".join(selfcheck.warnings(res))
    finally:
        fusion.FUSED_MAX_ROWS = 16


def test_a_kernel_wrong_only_on_inputs_with_outliers_or_zeros_is_disabled(monkeypatch):
    real = q.qmm_exact

    def wrong_on_outliers(x, w, s, b, segments=None):
        out = real(x, w, s, b, segments)
        if out is None or float(mx.abs(x).max()) < 10:
            return out
        return (out.view(mx.uint16) + 1).view(mx.bfloat16)

    monkeypatch.setattr(q, "qmm_exact", wrong_on_outliers)
    assert selfcheck.run()["disabled"]
    q.DISABLED.clear()
    q._plans.clear()

    def wrong_on_zeros(x, w, s, b, segments=None):
        out = real(x, w, s, b, segments)
        if out is None or float(mx.abs(x).max()) > 0:
            return out
        return (out.view(mx.uint16) + 1).view(mx.bfloat16)

    monkeypatch.setattr(q, "qmm_exact", wrong_on_zeros)
    assert selfcheck.run()["disabled"]
