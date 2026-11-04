"""Group prefill gives each request its solo output and falls back when the group rule declines."""

from __future__ import annotations

import threading


class _counting:
    """Counts group passes (and their sizes) while active."""

    def __enter__(self):
        from jitengine_qwen3_6_35b_a3b_m2.scheduler.prefill import PrefillTask

        self.calls, self.orig = [], PrefillTask.__dict__["run_group_layer_major"]
        orig = self.orig.__func__

        def wrapped(tasks):
            self.calls.append(len(tasks))
            return orig(tasks)

        PrefillTask.run_group_layer_major = staticmethod(wrapped)
        return self.calls

    def __exit__(self, *exc):
        from jitengine_qwen3_6_35b_a3b_m2.scheduler.prefill import PrefillTask

        PrefillTask.run_group_layer_major = self.orig


def _batch(e, reqs, timeout=300):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", e.generate_batch(reqs)), daemon=True)
    th.start()
    th.join(timeout)
    assert not th.is_alive(), "group prefill stalled"
    return out["r"]


def test_group_prefill_rows_equal_solo(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    kw = dict(prefill_chunk=16, layer_major_prefill=True)
    ref = tiny_engine(group_prefill=False, **kw)
    grp = tiny_engine(group_prefill=True, **kw)
    try:
        assert grp.group_prefill and not ref.group_prefill
        prompts = [
            list(range(100 + 7 * i, 100 + 7 * i + n)) for i, n in enumerate((37, 64, 90, 21))
        ]
        solo = [ref.generate(GenRequest(prompt_ids=p, max_tokens=10)).token_ids for p in prompts]
        with _counting() as calls:
            res = _batch(grp, [GenRequest(prompt_ids=p, max_tokens=10) for p in prompts])
        assert [r.token_ids for r in res] == solo
        assert calls and max(calls) >= 2, calls
    finally:
        ref.close()
        grp.close()


def test_group_prefill_declines_without_stalling(tiny_engine, requires_mlx):
    """A declined group must leave its rows to the sequential path instead of stalling them."""
    from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenRequest

    kw = dict(prefill_chunk=16, layer_major_prefill=True, prefix_cache_bytes=64 << 20)
    ref = tiny_engine(group_prefill=False, **kw)
    grp = tiny_engine(group_prefill=True, group_prefill_max_tokens=200, **kw)
    try:
        shared = list(range(600, 900))  # 300 shared tokens: an arrival group with a leader
        prompts = [shared + [i, i + 1, i + 2] for i in range(4)]
        solo = [ref.generate(GenRequest(prompt_ids=p, max_tokens=8)).token_ids for p in prompts]
        res = _batch(grp, [GenRequest(prompt_ids=p, max_tokens=8) for p in prompts])
        assert [r.token_ids for r in res] == solo
        # the leader runs alone (a group of all four is above the bound); the followers, once the
        # shared prefix is stored, are small enough to group and must still hit it
        assert sum(r.cached_prompt_tokens >= 288 for r in res) >= 3, [
            r.cached_prompt_tokens for r in res
        ]
    finally:
        ref.close()
        grp.close()
