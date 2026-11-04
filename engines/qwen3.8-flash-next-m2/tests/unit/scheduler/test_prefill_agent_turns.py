"""A conversation that grows by appending reuses the earlier turn's snapshot.

The lookup resumes only at grid-aligned lengths (or a prompt's own whole head), so the snapshot a
turn leaves behind must sit on the grid for the next turn to find it. A fake model and state keep
this off the device: only the keys the task admits and restores are under test.
"""

CHUNK = 64


class _Arr:
    nbytes = 10


class _State:
    def __init__(self):
        self.tokens = 0

    def arrays(self):
        return []

    def snapshot(self, key):
        from jitengine_qwen3_8_flash_next_m2.kvcache.prefix_cache import Entry

        assert len(key) == self.tokens
        return Entry(key, [("delta", _Arr(), _Arr())])

    def restore(self, entry):
        self.tokens = entry.tokens


class _Model:
    def forward(self, ids, state, logits="none", **kw):
        state.tokens += ids.shape[1]


def _turn(store, head, snap_grid=True):
    from jitengine_qwen3_8_flash_next_m2.scheduler.prefill import PrefillTask

    t = PrefillTask(_Model(), store, CHUNK, head, _State(), use_store=True, snap_grid=snap_grid)
    t.run()
    return t


def _store():
    from jitengine_qwen3_8_flash_next_m2.kvcache.prefix_cache import PrefixCache

    return PrefixCache(cap_bytes=1 << 20, min_divergence=CHUNK)


def test_appended_turn_restores_the_previous_turns_last_grid_point(requires_mlx_any):
    store = _store()
    p1 = list(range(1000, 1300))  # off the grid: 300 = 4 * 64 + 44
    t1 = _turn(store, p1, snap_grid=False)
    assert t1.hit_len == 0 and t1.tail_at == 256
    assert store.keys() == [tuple(p1[:256])]
    p2 = p1 + list(range(5000, 5200))  # the reply and the next user message
    t2 = _turn(store, p2, snap_grid=False)
    assert t2.hit_len == 256, "the appended turn must restore the earlier turn's grid snapshot"
    assert t2.computed == len(p2) - 256
    assert tuple(p2[:448]) in store
    t3 = _turn(store, p2 + list(range(9000, 9100)), snap_grid=False)
    assert t3.hit_len == 448


def test_tail_snapshot_is_unpinned(requires_mlx_any):
    store = _store()
    t = _turn(store, list(range(300)), snap_grid=False)
    assert t.tail_at == 256 and not store._pinned


def test_prompt_on_the_grid_keeps_the_end_of_head_snapshot(requires_mlx_any):
    store = _store()
    on_grid = list(range(2 * CHUNK))
    assert _turn(store, on_grid, snap_grid=False).tail_at == 0
    assert tuple(on_grid) in store


def test_a_gain_under_the_minimum_stores_nothing(requires_mlx_any):
    store = _store()
    p = list(range(300))
    _turn(store, p, snap_grid=False)
    t = _turn(store, p + list(range(900, 910)), snap_grid=False)  # restores 256, gains 54
    assert t.hit_len == 256 and t.tail_at == 0
    assert len(store) == 1


def test_one_snapshot_per_turn_when_the_grid_point_is_already_planned_or_restored(requires_mlx_any):
    """Production (snap_grid=True): a turn adding one to two chunks plans g as its grid point, and a
    turn adding less than a chunk restores g itself. Either way g stands in for the off-grid
    end-of-head entry, so no second snapshot lands at n."""
    from jitengine_qwen3_8_flash_next_m2.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=1 << 20, min_divergence=16)
    head1 = list(range(1000, 1300))
    _turn(store, head1)  # snap_grid: pinned 64, tail 256
    head2 = head1[:256] + list(range(5000, 5094))  # n = 350, hit 256, grid point = g = 320
    t2 = _turn(store, head2)
    assert t2.hit_len == 256 and t2.grid_point == 320
    assert tuple(head2[:320]) in store and tuple(head2) not in store
    head3 = head1[:256] + list(range(7000, 7014))  # n = 270, g = 256 = hit_len
    t3 = _turn(store, head3)
    assert t3.hit_len == 256 and tuple(head3) not in store


def test_a_planned_shared_boundary_at_the_head_end_is_pinned(requires_mlx_any):
    """A head that ends exactly on the arrival group's boundary stores it as the pinned shared
    snapshot, not as an unpinned end-of-head entry."""
    from jitengine_qwen3_8_flash_next_m2.scheduler.prefill import PrefillTask

    store = _store()
    head = list(range(2 * CHUNK))
    PrefillTask(_Model(), store, CHUNK, head, _State(), use_store=True, group_point=2 * CHUNK).run()
    assert tuple(head) in store and tuple(head) in store._pinned
