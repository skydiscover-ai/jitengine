"""The prefix store: eviction, the protection of shared prefixes, and children that share their parent's KV."""

from __future__ import annotations


def test_store_drops_children_with_their_parent():
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    st = PrefixCache(cap_bytes=100)
    parent = E(key=(1, 2), nbytes=50, parent=None)
    assert st.admit(parent, shared=True)
    assert st.admit(E(key=(1, 2, 3), nbytes=10, parent=parent))
    assert st.admit(E(key=(1, 2, 4), nbytes=10, parent=parent))
    assert st.admit(E(key=(7,), nbytes=10, parent=None))
    assert st.bytes == 80
    assert st.resize(40) == 70  # the LRU parent takes both children with it
    assert len(st) == 1 and (7,) in st and st.bytes == 10
    # a child whose parent is no longer stored is refused (it would hold the parent's arrays uncounted)
    assert not st.admit(E(key=(1, 2, 5), nbytes=10, parent=parent))
    assert st.bytes == 10


def test_prefix_resize_to_zero_removes_snapshots():
    from types import SimpleNamespace

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=100)
    assert store.admit(SimpleNamespace(key=(1, 2), nbytes=60))
    assert store.resize(0) == 60
    assert store.bytes == 0
    assert len(store) == 0


def test_prefix_resize_evicts_lru_first():
    from types import SimpleNamespace

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    store = PrefixCache(cap_bytes=100)
    store.admit(SimpleNamespace(key=(1,), nbytes=40))
    store.admit(SimpleNamespace(key=(2,), nbytes=40))
    store.lookup([1, 9])  # (1,) becomes most recent
    assert store.resize(50) == 40
    assert (1,) in store and (2,) not in store
    for bad in (-1, 1.5, True):
        try:
            store.resize(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_prefix_store_protects_shared_entries_from_end_of_prompt_snapshots():
    """A one-entry store keeps a burst's shared prefix over the end-of-prompt snapshots after it."""
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache, protect_limit

    st = PrefixCache(cap_bytes=128)
    assert st.admit(E(key=(1, 2, 3, 4), nbytes=120))  # request 1, end of prompt
    assert st.admit(E(key=(1, 2), nbytes=110), shared=True)  # request 2 divergence point
    assert (1, 2) in st and (1, 2, 3, 4) not in st
    assert not st.admit(E(key=(1, 2, 5, 6), nbytes=120))  # request 2 end: refused
    assert (1, 2) in st and st.stats()["rejected_protected"] == 1
    assert st.lookup([1, 2, 7]) is not None  # request 3 hits the shared entry
    for i in range(protect_limit(1)):  # survives a burst's tail ...
        assert not st.admit(E(key=(9, i), nbytes=120))
    assert (1, 2) in st
    st.admit(E(key=(9, 99), nbytes=120))  # ... then a stale entry lapses
    assert (1, 2) not in st and len(st) == 1


def test_prefix_store_protection_grows_with_reuse():
    """A shared prefix hit 7 times survives the 8 end-of-prompt refusals of an agent burst."""
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    st = PrefixCache(cap_bytes=128)
    assert st.admit(E(key=(1, 2), nbytes=110), shared=True)
    for i in range(7):
        assert st.lookup([1, 2, 50 + i]) is not None
    for i in range(8):
        assert not st.admit(E(key=(1, 2, 50 + i), nbytes=120))
    assert (1, 2) in st


def test_prefix_store_repeated_prompt_still_cached():
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    st = PrefixCache(cap_bytes=128)
    assert st.admit(E(key=(1, 2, 3), nbytes=100))
    assert st.lookup([1, 2, 3, 4]) is not None
    assert st.admit(E(key=(1, 2, 3), nbytes=100))  # re-admission refreshes
    assert st.stats()["protected_entries"] == 1


def test_prefix_store_admits_a_repeated_end_of_prompt_key():
    """A re-sent prompt's end-of-prompt snapshot is refused once, then admitted on the repeat."""
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    st = PrefixCache(cap_bytes=128)
    assert st.admit(E(key=(1, 2), nbytes=110), shared=True)
    # each request: lookup(head) at admission, then admit(entry keyed by head) after its prefill
    st.lookup([1, 2, 3, 4])
    assert not st.admit(E(key=(1, 2, 3, 4), nbytes=120))  # first offer: refused
    st.lookup([1, 2, 3, 4])
    assert st.admit(E(key=(1, 2, 3, 4), nbytes=120))  # the prompt repeated: admitted
    assert (1, 2, 3, 4) in st and st.stats()["promoted"] == 1
    # an agent's distinct suffixes never repeat, so they never displace the shared entry
    st2 = PrefixCache(cap_bytes=128)
    st2.admit(E(key=(1, 2), nbytes=110), shared=True)
    for i in (0, 1, 2, 0, 1, 2):
        st2.lookup([1, 2, 50 + i])
        assert not st2.admit(E(key=(1, 2, 50 + i), nbytes=120))
    assert (1, 2) in st2


def test_prefix_store_repeating_agent_cycle_never_evicts_the_shared_entry():
    """Eight distinct suffixes repeated three times are never promoted over the shared prefix."""
    from types import SimpleNamespace as E

    from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache

    st = PrefixCache(cap_bytes=128)
    st.admit(E(key=(1, 2), nbytes=110), shared=True)
    order = list(range(8))
    for rep in range(3):
        seq = (
            order if rep != 1 else [7] + order[:7]
        )  # a burst boundary serves task 7 twice in a row
        for j, i in enumerate(seq):
            waiting = seq[j + 1 :]  # the rest of the burst is queued
            assert st.lookup([1, 2, 50 + i]) is not None
            assert not st.admit(
                E(key=(1, 2, 50 + i), nbytes=120),
                pending_uses=lambda k, w=waiting: bool(w) and k == (1, 2),
            ), (rep, i)
    assert (1, 2) in st and st.stats()["promoted"] == 0
