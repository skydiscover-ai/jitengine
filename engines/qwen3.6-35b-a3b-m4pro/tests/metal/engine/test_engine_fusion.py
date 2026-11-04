"""Fused same-input projections must reproduce the unfused engine bit for bit."""

from __future__ import annotations


def _pair(tiny_engine, **kw):
    return (
        tiny_engine(fuse_projections=False, **kw),
        tiny_engine(fuse_projections=True, **kw),
    )


def test_fused_bit_identical_resident_and_batched(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    ref, fz = _pair(tiny_engine, experts=64, max_slots=4, prefill_chunk=16)
    try:
        assert fz.fuse_projections and fz.model.layers[0].moe.gate_up is not None
        prompts = [list(range(100, 160)), [5, 6, 7], list(range(900, 941)), [42] * 9]
        for p in prompts:
            assert (
                ref.generate(GenRequest(prompt_ids=p, max_tokens=16)).token_ids
                == fz.generate(GenRequest(prompt_ids=p, max_tokens=16)).token_ids
            )
        reqs = [GenRequest(prompt_ids=p, max_tokens=12) for p in prompts]
        a = [r.token_ids for r in ref.generate_batch(reqs)]
        b = [r.token_ids for r in fz.generate_batch(reqs)]
        assert a == b
        assert ref.score(list(range(3, 40)), list(range(50, 66))) == fz.score(
            list(range(3, 40)), list(range(50, 66))
        )
    finally:
        ref.close()
        fz.close()


def test_fused_bit_identical_offload(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    ref, fz = _pair(tiny_engine, mode="offload", experts=64)
    try:
        for p in ([1, 2, 3, 4, 5], list(range(300, 370))):
            assert (
                ref.generate(GenRequest(prompt_ids=p, max_tokens=12)).token_ids
                == fz.generate(GenRequest(prompt_ids=p, max_tokens=12)).token_ids
            )
    finally:
        ref.close()
        fz.close()
