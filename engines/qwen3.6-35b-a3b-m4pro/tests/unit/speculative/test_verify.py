"""verify_step builds the relaxed profile's pass inside `relaxed_rows()`, and the exact one outside."""

from __future__ import annotations

from types import SimpleNamespace


def test_verify_step_is_relaxed_only_under_the_relaxed_strategy(requires_mlx_any, monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro import linear
    from jitengine_qwen3_6_35b_a3b_m4pro.speculative import verify

    seen = []
    monkeypatch.setattr(
        verify, "_verify_step", lambda *args: seen.append(linear._RELAX_ROWS.get()) or "pass"
    )
    for relaxed in (True, False):
        model = SimpleNamespace(strategies=SimpleNamespace(verify_relaxed=relaxed))
        assert verify.verify_step(model, None, None) == "pass"
    assert seen == [True, False]
    assert linear._RELAX_ROWS.get() is False
