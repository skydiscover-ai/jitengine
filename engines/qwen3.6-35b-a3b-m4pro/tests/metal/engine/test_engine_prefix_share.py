"""A restore from a child snapshot (shared-prefix KV) is bitwise the full-snapshot path."""

from __future__ import annotations


def test_share_kv_outputs_equal_full_snapshots(tiny_engine, requires_mlx):
    import jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill as PF
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    shared = list(range(600, 900))
    prompts = [shared + list(range(1000 + 40 * i, 1000 + 40 * i + 20 + 7 * i)) for i in range(3)]
    out, sizes = {}, {}
    for share in (False, True):
        PF.SHARE_KV = share
        e = tiny_engine(prefill_chunk=16, layer_major_prefill=True, prefix_cache_bytes=64 << 20)
        try:
            runs = []
            for _ in range(2):  # the second pass restores every prompt from the store
                runs.append([e.generate(GenRequest(prompt_ids=p, max_tokens=8)) for p in prompts])
            # a repeat resumes at the head's last grid point or later (scheduler/prefill.py,
            # _plan_tail_point): at most one 16-token chunk is left to recompute
            assert all(
                (len(p) - 1) // 16 * 16 <= r.cached_prompt_tokens <= len(p) - 1
                for r, p in zip(runs[1], prompts)
            ), [r.cached_prompt_tokens for r in runs[1]]
            out[share] = [[r.token_ids for r in rs] for rs in runs]
            # prompt 1 creates the shared entry (its divergence from prompt 0); prompt 2 restores it
            ends = [e.store._entries[tuple(prompts[2][:-1])]]
            assert all((x.parent is not None) == share for x in ends)
            sizes[share] = sum(x.nbytes for x in ends)
        finally:
            e.close()
            PF.SHARE_KV = True
    assert out[True] == out[False]
    assert out[True][0] == out[True][1]
    assert sizes[True] < sizes[False]


def test_child_restore_is_bitwise(tiny_engine, requires_mlx):
    """A state restored from a child (parent KV + own suffix KV) equals one restored from the full
    snapshot of the same positions, array for array."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.state import RequestState
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    e = tiny_engine(prefill_chunk=16, prefix_cache_bytes=0)
    try:
        head = list(range(300, 380))
        st = RequestState(e.cfg, len(head), 8)
        PrefillTask(e.model, e.store, 16, head[:48], st, use_store=False).run()
        parent = st.snapshot(tuple(head[:48]))
        st2 = RequestState(e.cfg, len(head), 8)
        st2.restore(parent)
        t = PrefillTask(e.model, e.store, 16, head, st2, use_store=False)
        t.pos = 48
        t.run()
        full = st2.snapshot(tuple(head))
        child = st2.snapshot(tuple(head), parent=parent)
        assert child.parent is parent and child.nbytes < full.nbytes
        a, b = RequestState(e.cfg, len(head), 8), RequestState(e.cfg, len(head), 8)
        a.restore(full)
        b.restore(child)
        assert a.tokens == b.tokens == len(head)
        for x, y in zip(a.arrays(), b.arrays()):
            assert x.shape == y.shape and bool(mx.array_equal(x, y)), (x.shape, y.shape)
    finally:
        e.close()
