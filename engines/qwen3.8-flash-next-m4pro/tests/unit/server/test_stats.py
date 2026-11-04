"""The request counters /stats reports."""

from jitengine_qwen3_8_flash_next_m4pro.server.stats import Stats


def test_failed_requests_are_counted_by_error_type():
    """Completions count through their records, scores through `count_error`."""
    s = Stats()
    for rec in (
        {"finish_reason": "length"},
        {"finish_reason": "error", "error": "InvariantError"},
        {"finish_reason": "error", "error": "MemoryPressure"},
        {"finish_reason": "error", "error": "InvariantError"},
    ):
        s.record(rec, log=False)
    s.count_error("ValueError")
    snap = s.snapshot()
    assert snap["requests"] == 4 and snap["errors"] == 3
    assert snap["error_types"] == {"InvariantError": 2, "MemoryPressure": 1, "ValueError": 1}
