"""Resizing a real tiny engine's expert pool."""

from __future__ import annotations


def _gen(engine, ids, n=8):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    return engine.generate(GenRequest(prompt_ids=ids, max_tokens=n)).token_ids


def test_real_pool_shrink_grow_keeps_outputs_and_frees_memory(tiny_engine, requires_mlx):
    import gc

    import mlx.core as mx

    e = tiny_engine(mode="offload", experts=64)
    try:
        full_slots = e.plan.pool_slots
        assert full_slots == e.capacity.total_slots
        ref = _gen(e, [5, 6, 7, 8, 9, 10])
        ceiling = e.ceiling_bytes
        floor = e.capacity.fixed_mlx_bytes + e.capacity.host_side_bytes
        slot = e.capacity.expert_slot_bytes
        small_target = floor + 40 * slot  # 40 of 256 experts, >= 4 layers x 2
        gc.collect()
        mx.clear_cache()
        before_active = mx.get_active_memory()  # process-wide: other engines are idle
        r = e.reconfigure_memory(int(small_target))
        assert r["status"] == "applied", r
        assert e.plan.pool_slots == 40 and e.expert_pool.slots == 40
        assert all(layer.moe.pool is e.expert_pool for layer in e.model.layers)
        gc.collect()
        mx.clear_cache()
        freed = before_active - mx.get_active_memory()
        assert freed >= 0.9 * (full_slots - 40) * slot, (freed, (full_slots - 40) * slot)
        assert _gen(e, [5, 6, 7, 8, 9, 10]) == ref  # smaller pool, same numbers
        for _ in range(5):  # alternate, no cumulative growth
            assert e.reconfigure_memory(int(ceiling))["status"] == "applied"
            assert e.reconfigure_memory(int(small_target))["status"] == "applied"
        gc.collect()
        mx.clear_cache()
        drift = mx.get_active_memory() - (before_active - freed)
        assert abs(drift) <= 4 * slot, drift  # no retained old pools
        assert e.pool_generation == 11
        assert _gen(e, [5, 6, 7, 8, 9, 10]) == ref
        st = e.stats()
        assert st["expert_pool_slots"] == 40 and st["memory"]["last_resize"]["status"] == "applied"
        assert st["expert_miss_bytes"] == st["expert_cache_misses"] * e.expert_pool.record_bytes
    finally:
        e.close()


def test_real_resize_is_busy_while_generating(tiny_engine, requires_mlx):
    import threading

    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    e = tiny_engine(mode="offload")
    try:
        started = threading.Event()
        out = {}

        def run():
            it = e.events(GenRequest(prompt_ids=[1, 2, 3], max_tokens=400))
            for kind, _ in it:
                if kind == "token":
                    started.set()
            out["done"] = True

        th = threading.Thread(target=run)
        th.start()
        assert started.wait(30)
        r = e.reconfigure_memory(e.ceiling_bytes // 2)
        th.join(60)
        assert r["status"] == "busy", r
        assert out.get("done")
    finally:
        e.close()
