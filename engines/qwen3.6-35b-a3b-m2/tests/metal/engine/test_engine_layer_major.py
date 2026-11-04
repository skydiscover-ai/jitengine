"""Layer-major prefill is bit-identical to chunked prefill, including the entries it stores."""

from __future__ import annotations


def _engines(tiny_engine, frac):
    probe = tiny_engine(mode="offload", experts=64)
    cap = probe.capacity
    target = (
        cap.fixed_mlx_bytes
        + cap.host_side_bytes
        + int(frac * cap.total_slots) * cap.expert_slot_bytes
    )
    probe.close()
    kw = dict(
        mode="offload",
        experts=64,
        process_budget_bytes=int(target),
        prefill_chunk=16,
        speculative_gather=False,
    )
    return tiny_engine(layer_major_prefill=False, **kw), tiny_engine(layer_major_prefill=True, **kw)


def test_layer_major_bit_identical(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    for frac in (1.0, 0.25):
        ref, lm = _engines(tiny_engine, frac)
        try:
            assert lm.layer_major_prefill and not ref.layer_major_prefill
            shared = list(range(500, 800))  # 300 tokens: a shared prefix the store snapshots
            prompts = [
                shared + [7, 8, 9],
                shared + [11, 12],
                list(range(2000, 2077)),
                [3, 4],
                shared + [13],
            ]
            for p in prompts:
                a = ref.generate(GenRequest(prompt_ids=p, max_tokens=12))
                b = lm.generate(GenRequest(prompt_ids=p, max_tokens=12))
                assert a.token_ids == b.token_ids, (frac, p[:3])
                assert a.cached_prompt_tokens == b.cached_prompt_tokens
            assert ref.score(list(range(9, 60)), list(range(70, 90))) == lm.score(
                list(range(9, 60)), list(range(70, 90))
            )
            ra, rb = ref.stats(), lm.stats()
            assert ra["prefix_store"]["entry_tokens"] == rb["prefix_store"]["entry_tokens"]
            if frac < 1.0:  # fewer expert reads is the point
                assert rb["expert_cache_misses"] < ra["expert_cache_misses"], (
                    ra["expert_cache_misses"],
                    rb["expert_cache_misses"],
                )
        finally:
            ref.close()
            lm.close()


def test_layer_major_snapshot_restores_exactly(tiny_engine, requires_mlx):
    """A shared-prefix entry written by the layer-major pass restores to the cold-run output."""
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    e = tiny_engine(mode="offload", experts=64, prefill_chunk=16, prefix_cache_bytes=64 << 20)
    cold = tiny_engine(mode="offload", experts=64, prefill_chunk=16, prefix_cache_bytes=0)
    try:
        shared = list(range(600, 900))  # 300 tokens: above the store's 256-token minimum
        outs = [
            e.generate(GenRequest(prompt_ids=shared + [i, i + 1], max_tokens=10)) for i in range(3)
        ]
        assert outs[2].cached_prompt_tokens == 288, outs[2].cached_prompt_tokens  # the grid point
        for i, o in enumerate(outs):
            c = cold.generate(GenRequest(prompt_ids=shared + [i, i + 1], max_tokens=10))
            assert c.token_ids == o.token_ids
    finally:
        e.close()
        cold.close()
