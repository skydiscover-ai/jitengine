"""Admission contract: context bound, bounded queue, pauses, aborts and faults."""

from __future__ import annotations

import threading


def test_context_limit_enforced_in_process(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine(max_context_tokens=64)
    try:
        assert e.max_kv_positions == 64
        ok = e.generate(GenRequest(prompt_ids=list(range(1, 33)), max_tokens=32))
        assert ok.finish_reason == "length" and len(ok.token_ids) == 32
        bad = e.generate(GenRequest(prompt_ids=list(range(1, 34)), max_tokens=32))
        assert bad.finish_reason == "error" and "context limit" in bad.extra["error"]
        try:
            e.score(list(range(1, 60)), list(range(1, 10)))
        except ValueError as exc:
            assert "context limit" in str(exc)
        else:
            raise AssertionError("score beyond the context limit was accepted")
    finally:
        e.close()


def test_queue_bound_rejects_excess_with_retryable_error(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine(max_queued_requests=2, max_slots=1)
    try:
        results = [None] * 6
        started = threading.Barrier(7)

        def run(i):
            started.wait()
            results[i] = e.generate(GenRequest(prompt_ids=[1, 2, 3 + i], max_tokens=64))

        ths = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in ths:
            t.start()
        started.wait()
        for t in ths:
            t.join(120)
        full = [r for r in results if r.finish_reason == "error"]
        done = [r for r in results if r.finish_reason != "error"]
        assert len(done) >= 3 and len(full) >= 1, [r.finish_reason for r in results]
        assert all(
            r.extra.get("error_type") == "QueueFullError" and r.extra.get("retryable") for r in full
        )
        assert e.stats()["batcher"]["rejected_queue_full"] == len(full)
        assert e._outstanding == 0
        # after the burst the queue admits again
        assert e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2)).finish_reason == "length"
    finally:
        e.close()


def test_single_slot_serializes_but_completes_eight(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine(max_slots=1)
    try:
        solo = e.generate(GenRequest(prompt_ids=[9, 8, 7], max_tokens=12)).token_ids
        res = e.generate_batch([GenRequest(prompt_ids=[9, 8, 7], max_tokens=12) for _ in range(8)])
        assert all(r.token_ids == solo for r in res)
        assert e.stats()["batcher"]["peak_rows"] == 1
    finally:
        e.close()


def test_admission_pause_is_retryable_and_reopens(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine()
    try:
        e.set_admission("memory_pressure")
        r = e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2))
        assert r.finish_reason == "error" and r.extra["error_type"] == "AdmissionPausedError"
        assert e.health()["admission"] == "memory_pressure"
        e.set_admission(None)
        assert e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2)).finish_reason == "length"
    finally:
        e.close()


def test_abort_drains_and_fails_running_rows(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine()
    try:
        box = {}
        got_token = threading.Event()

        def run():
            try:
                for _kind, _ in e.events(GenRequest(prompt_ids=[1, 2, 3], max_tokens=4000 - 3)):
                    got_token.set()
            except Exception as exc:  # noqa: BLE001
                box["exc"] = exc

        th = threading.Thread(target=run)
        th.start()
        assert got_token.wait(30)
        res = e.maintenance_command("abort", "test pressure")
        th.join(30)
        assert res["status"] == "applied" and res["aborted"] >= 1
        assert type(box.get("exc")).__name__ == "MemoryPressureError"
        assert e.idle() and e.pool.kv_live == 0
        assert e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2)).finish_reason == "length"
    finally:
        e.close()


def test_decode_build_fault_fails_the_request_instead_of_hanging(tiny_engine, requires_mlx):
    import threading

    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine()
    try:
        real = e.model.forward
        calls = {"n": 0}

        def broken(tokens, state, logits="last"):
            if tokens.shape[1] == 1:
                calls["n"] += 1
                raise RuntimeError("injected build fault")
            return real(tokens, state, logits=logits)

        e.model.forward = broken
        box = {}

        def run():
            try:
                box["r"] = e.generate(GenRequest(prompt_ids=[1, 2, 3], max_tokens=4))
            except Exception as exc:  # noqa: BLE001  a terminal error is the other correct answer
                box["exc"] = exc

        th = threading.Thread(target=run)
        th.start()
        th.join(30)
        assert not th.is_alive(), "request hung"
        assert ("exc" in box and "injected" in str(box["exc"])) or box["r"].finish_reason == "error"
        assert calls["n"] == 1
        assert e.stats()["error_types"] == {"RuntimeError": 1}
        e.model.forward = real
        assert e.generate(GenRequest(prompt_ids=[1, 2], max_tokens=2)).finish_reason == "length"
    finally:
        e.close()


def test_default_slots_batch_eight_clients_row_exactly(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine()
    try:
        assert e.max_slots == 8, e.max_slots
        solo = e.generate(GenRequest(prompt_ids=[9, 8, 7], max_tokens=12)).token_ids
        res = e.generate_batch([GenRequest(prompt_ids=[9, 8, 7], max_tokens=12) for _ in range(8)])
        assert all(r.token_ids == solo for r in res)
        assert e.stats()["batcher"]["peak_rows"] > 1, e.stats()["batcher"]
    finally:
        e.close()
