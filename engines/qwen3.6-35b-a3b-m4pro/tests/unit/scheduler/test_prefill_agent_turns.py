"""A conversation that grows by appending reuses the earlier turn's snapshot.

The lookup resumes only at grid-aligned lengths (or a prompt's own whole head), so the snapshot a
turn leaves behind must sit on the grid for the next turn to find it. A fake model and state keep
this off the device: only the keys the task admits and restores are under test.
"""

from typing import ClassVar

import pytest

CHUNK = 64


class _State:
    parents: ClassVar[dict] = {}

    def __init__(self):
        self.tokens = 0

    def arrays(self):
        return []

    def snapshot(self, key, parent=None):
        from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import Entry

        assert len(key) == self.tokens
        self.parents[key] = parent
        return Entry(key, [("kv", _Arr(), _Arr())], parent)

    def restore(self, entry):
        self.tokens = entry.tokens


class _Arr:
    nbytes = 10


class _Model:
    def forward(self, ids, state, logits="none", **kw):
        state.tokens += ids.shape[1]


def _turn(store, head):
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    t = PrefillTask(_Model(), store, CHUNK, head, _State(), use_store=True)
    t.run()
    return t


def test_appended_turn_restores_the_previous_turns_last_grid_point(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
    p1 = list(range(1000, 1000 + 300))  # off the grid: 300 = 4 * 64 + 44
    t1 = _turn(store, p1)
    assert t1.hit_len == 0
    assert store.keys() == [tuple(p1[:256])]
    p2 = p1 + list(range(5000, 5000 + 200))  # reply and the next user message
    t2 = _turn(store, p2)
    assert t2.hit_len == 256, "the appended turn must restore the earlier turn's grid snapshot"
    assert t2.computed == len(p2) - 256
    assert (len(p2) // CHUNK) * CHUNK == 448
    assert tuple(p2[:448]) in store
    t3 = _turn(store, p2 + list(range(9000, 9100)))
    assert t3.hit_len == 448


def test_tail_snapshot_is_offered_unshared(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
    t = _turn(store, list(range(300)))
    key = tuple(range(256))
    assert t.tail_at == 256 and store._meta[key]["shared"] is False


def test_prompt_on_the_grid_or_inside_one_chunk_keeps_the_end_of_head_snapshot(requires_mlx_any):
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
    on_grid = list(range(2 * CHUNK))
    assert _turn(store, on_grid).tail_at == 0
    assert tuple(on_grid) in store
    short = list(range(500, 500 + CHUNK - 5))  # no grid point past the minimum inside it
    _turn(store, short)
    assert tuple(short) in store


@pytest.mark.parametrize("n", [65, 127, 129, 300])
def test_grid_snapshot_is_the_last_grid_point_inside_the_head(requires_mlx_any, n):
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
    t = _turn(store, list(range(n)))
    assert t.tail_at == (n // CHUNK) * CHUNK


def test_a_shared_snapshot_is_not_a_child_of_the_restored_entry(requires_mlx_any):
    """A group-boundary snapshot taken after a restore is stored whole: as a child it would die
    with its unprotected parent while protected (the store counts only the parent's own bytes)."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
    head1 = list(range(1000, 1300))
    _turn(store, head1)  # tail snapshot at 256
    head2 = head1[:256] + list(range(5000, 5700))
    _State.parents = {}
    PrefillTask(_Model(), store, CHUNK, head2, _State(), use_store=True, group_point=512).run()
    assert _State.parents[tuple(head2[:512])] is None  # shared group snapshot: whole
    assert _State.parents[tuple(head2[:896])] is not None  # tail snapshot: shares the parent's KV
    assert store._entries[tuple(head2[:512])].parent is None


def test_one_snapshot_per_turn_when_the_grid_point_is_already_planned_or_restored(requires_mlx_any):
    """A grid snapshot at g (planned by a group point, or the restored entry itself) stands in for
    the off-grid end-of-head entry: no second snapshot at n."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=16)
    head1 = list(range(1000, 1300))
    _turn(store, head1)  # tail snapshot at 256
    head2 = head1[:256] + list(range(5000, 5100))  # n = 356, g = 320
    PrefillTask(_Model(), store, CHUNK, head2, _State(), use_store=True, group_point=320).run()
    assert tuple(head2[:320]) in store and tuple(head2) not in store
    head3 = head1[:256] + list(range(7000, 7014))  # n = 270: g = 256 is the restored entry
    t = PrefillTask(_Model(), store, CHUNK, head3, _State(), use_store=True)
    t.run()
    assert t.hit_len == 256 and tuple(head3) not in store


class _LayerMajorModel(_Model):
    """forward_layer_major: the state advances to the end of the head and each captured point comes
    back as full-KV layers, as the real layer-major pass returns them."""

    def forward_layer_major(self, head, bounds, state, capture=()):
        import mlx.core as mx

        state.tokens = len(head)
        return {
            a: [
                ("kv", mx.zeros((1, 1, a, 4)), mx.zeros((1, 1, a, 4))),
                ("delta", mx.zeros((2,)), mx.zeros((2,))),
            ]
            for a in capture
        }


def test_layer_major_tail_snapshot_shares_the_restored_entrys_kv(requires_mlx_any):
    """The layer-major pass stores the tail snapshot as a child of the restored entry (own positions
    only), like step() does, and a shared snapshot of the same pass whole."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    store = PrefixCache(cap_bytes=1 << 30, min_divergence=CHUNK)
    head1 = list(range(1000, 1300))
    _turn(store, head1)  # tail snapshot at 256
    parent = store._entries[tuple(head1[:256])]
    head2 = head1[:256] + list(range(5000, 5700))  # n = 956, hit 256, group 512, tail 896
    t = PrefillTask(
        _LayerMajorModel(), store, CHUNK, head2, _State(), use_store=True, group_point=512
    )
    assert t.hit_len == 256 and t.tail_at == 896
    t.run_layer_major()
    tail, shared = store._entries[tuple(head2[:896])], store._entries[tuple(head2[:512])]
    assert tail.parent is parent
    assert [rec[0] for rec in tail.layers] == ["kvs", "delta"]
    assert tail.layers[0][1].shape[2] == 896 - 256
    assert shared.parent is None and shared.layers[0][0] == "kv"
    assert tuple(head2) not in store


def test_a_planned_shared_boundary_at_the_head_end_is_stored_whole_and_shared(requires_mlx_any):
    """When the head ends exactly on a grid point that is the arrival group's boundary or a
    divergence point, the end-of-head admission is that shared boundary: shared, and not a child."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import Entry, PrefixCache
    from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PrefillTask

    for kind in ("group", "divergence"):
        store = PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)
        head1 = list(range(1000, 1300))
        _turn(store, head1)  # tail snapshot at 256
        head2 = head1[:256] + list(range(5000, 5256))  # n = 512, on the grid
        if kind == "divergence":
            store.admit(Entry(tuple(head2) + (7,) * 70, [("kv", _Arr(), _Arr())]))
        _State.parents = {}
        PrefillTask(
            _Model(),
            store,
            CHUNK,
            head2,
            _State(),
            use_store=True,
            group_point=512 if kind == "group" else 0,
        ).run()
        key = tuple(head2)
        assert key in store, kind
        assert _State.parents[key] is None, kind
        assert store._meta[key]["shared"] is True, kind
