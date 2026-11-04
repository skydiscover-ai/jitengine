"""The stats record a generate request starts with, shared by `Req` and the max_tokens == 0 reply."""

from types import SimpleNamespace

from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.batch import new_record

KEYS = [
    "id",
    "kind",
    "prompt_tokens",
    "cached_tokens",
    "prefill_tokens",
    "prefill_ms",
    "first_token_ms",
    "decode_steps",
    "decode_ms",
    "completion_tokens",
    "finish_reason",
]


def test_a_new_record_has_zero_counters_a_length_finish_and_queue_ms_last():
    rec = new_record(SimpleNamespace(request_id="r1"), 7, 12.5)
    assert list(rec) == [*KEYS, "queue_ms"]
    assert rec["id"] == "r1" and rec["prompt_tokens"] == 7 and rec["queue_ms"] == 12.5
    assert rec["finish_reason"] == "length" and rec["kind"] == "generate"
    assert all(rec[k] in (0, 0.0) for k in KEYS[3:10])


def test_extra_fields_sit_before_queue_ms_and_a_missing_request_id_is_empty():
    rec = new_record(SimpleNamespace(), 3, 1.0, slot=2)
    assert list(rec) == [*KEYS, "slot", "queue_ms"]
    assert rec["id"] == "" and rec["slot"] == 2
