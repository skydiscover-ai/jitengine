"""The speculative pooled gather must be bit-identical to the synchronous pooled path."""

from __future__ import annotations


def _run(e, prompts, n=24):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    return [e.generate(GenRequest(prompt_ids=p, max_tokens=n)).token_ids for p in prompts]


def _score(e):
    return e.score(list(range(3, 20)), list(range(40, 52)), top_k=5)


def _pair(tiny_engine, slots_fraction):
    import mlx.core as mx  # noqa: F401

    base = tiny_engine(mode="offload", experts=64, speculative_gather=False)
    cap = base.capacity
    target = (
        cap.fixed_mlx_bytes
        + cap.host_side_bytes
        + int(slots_fraction * cap.total_slots) * cap.expert_slot_bytes
    )
    base.close()
    ref = tiny_engine(
        mode="offload",
        experts=64,
        speculative_gather=False,
        process_budget_bytes=int(target),
    )
    spec = tiny_engine(
        mode="offload",
        experts=64,
        speculative_gather=True,
        process_budget_bytes=int(target),
    )
    return ref, spec


PROMPTS = [[5, 6, 7, 8], [100, 200, 300, 400, 500, 600], [9] * 3, list(range(1000, 1040))]


def test_speculative_bit_identical_full_pool(tiny_engine, requires_mlx):
    ref, spec = _pair(tiny_engine, 1.0)
    try:
        assert spec.speculative_gather and not ref.speculative_gather
        assert _run(ref, PROMPTS) == _run(spec, PROMPTS)
        assert _score(ref) == _score(spec)
        st = spec.stats()
        assert st["expert_spec_calls"] > 0
    finally:
        ref.close()
        spec.close()


def test_speculative_bit_identical_with_misses(tiny_engine, requires_mlx):
    ref, spec = _pair(tiny_engine, 0.2)  # a fifth of the experts: misses on most layers
    try:
        a, b = _run(ref, PROMPTS), _run(spec, PROMPTS)
        assert a == b
        assert _score(ref) == _score(spec)
        st = spec.stats()
        assert (
            st["expert_spec_misses"] > 0 and st["expert_spec_calls"] > st["expert_spec_misses"] // 2
        )
        # identical accounting: the speculative path changes when the host looks, not what it counts
        sr = ref.stats()
        assert (sr["expert_cache_hits"], sr["expert_cache_misses"]) == (
            st["expert_cache_hits"],
            st["expert_cache_misses"],
        )
    finally:
        ref.close()
        spec.close()
