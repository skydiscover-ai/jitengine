"""What a failed decode step does to its request, to /stats and to the engine."""

import pytest


# Which one-token forward fails: the first decode step (nothing in flight), or the one issued while
# the first is in flight.
@pytest.mark.parametrize("failing_call", [1, 2], ids=["first_step", "pipelined_step"])
def test_a_failed_decode_step_fails_its_request_and_the_engine_keeps_serving(
    flashnext_engine, requires_mlx, failing_call
):
    """A decode step that raises fails that step's request, `/stats` counts it by type, and the
    next request is served."""
    import threading

    from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenRequest

    e = flashnext_engine()
    real = e.model.forward
    calls = {"n": 0}

    def broken(tokens, state, logits="last", trace=None, telem=None):
        if tokens.shape[1] == 1:
            calls["n"] += 1
            if calls["n"] == failing_call:
                raise RuntimeError("injected")
        return real(tokens, state, logits=logits, trace=trace, telem=telem)

    e.model.forward = broken
    box = {}

    def run():
        try:
            box["r"] = e.generate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=4))
        except Exception as exc:  # noqa: BLE001  a terminal error is the other correct answer
            box["exc"] = exc

    th = threading.Thread(target=run)
    th.start()
    th.join(60)
    assert not th.is_alive(), "request hung"
    assert calls["n"] == failing_call
    assert isinstance(box.get("exc"), RuntimeError) or box["r"].finish_reason == "error"
    assert e.stats()["error_types"] == {"RuntimeError": 1}
    e.model.forward = real
    ok = e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2, stop_at_eos=False))
    assert ok.finish_reason == "length"
