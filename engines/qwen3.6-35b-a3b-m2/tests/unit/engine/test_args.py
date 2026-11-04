"""Engine options resolve to byte-denominated values and refuse malformed ones, without MLX."""


def test_option_validation_without_model():
    # engine re-exports resolve_options from engine_options, which is MLX-free
    from jitengine_qwen3_6_35b_a3b_m2.engine.args import resolve_options

    o = resolve_options({})
    assert (
        o["process_budget_bytes"] == 18 << 30 and o["max_slots"] == 8 and o["kv_budget_rows"] == 4
    )
    assert o["max_context_tokens"] == 8192 and o["prefill_chunk"] == 256
    assert resolve_options({"memory_budget_gb": 12})["process_budget_bytes"] == 12 * 10**9
    for bad in (
        {"memory_budget_gb": 12, "process_budget_bytes": 1},
        {"process_budget_bytes": 1.5},
        {"max_slots": 0},
        {"mode": "hybrid"},
        {"memory_policy": "adaptive", "mode": "resident"},
        {"kv_budget_mb": 640},
        {"memory_budget_gb": float("nan")},
    ):
        try:
            resolve_options(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
