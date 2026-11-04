"""The context-copy draft index and the cost-fitted depth policy."""

from itertools import pairwise


def test_draft_index():
    from jitengine_flint_m4pro import copy_draft as D

    prompt = [5, 6, 7, 8, 9, 5, 6, 7, 1, 2]
    dx = D.DraftIndex(D.PromptIndex(prompt))
    dx.append(5)
    dx.append(6)
    # history ...1, 2, 5, 6: longest earlier match of the suffix is [5, 6] (most recent: ends at 7, next token 7)
    n, e = dx.lookup()
    assert (n, e) == (2, 7) and dx.draft(e, 3) == [7, 1, 2]
    dx.append(7)
    n, e = dx.lookup()  # [5, 6, 7] ends at 8 (most recent) -> drafts 1, 2, 5, 6, 7, then periodic
    assert (n, e) == (3, 8) and dx.draft(e, 8) == [1, 2, 5, 6, 7, 1, 2, 5]
    dx2 = D.DraftIndex(D.PromptIndex([1, 2, 3]))
    dx2.append(4)
    assert dx2.lookup() == (0, -1)
    # the shared prompt part is never modified by a request
    base = D.PromptIndex(prompt)
    before = {n: dict(v) for n, v in base.ends.items()}
    d3 = D.DraftIndex(base)
    for x in (5, 6, 7, 1, 2, 3):
        d3.append(x)
        d3.lookup()
    assert base.ends == before
    tab = D.depth_table("greedy", 1024)
    assert [D.bucket(n) for n in (0, 1, 3, 5, 8, 12, 24, 48, 64)] == [0, 1, 3, 4, 5, 6, 7, 8, 9]
    assert D.certain_no_draft(0, tab) >= 1 and D.certain_no_draft(64, tab) == 0
    for n in D.NSET:
        if tab[D.bucket(n)]:
            assert D.certain_no_draft(n, tab) == 0


def test_depth_policy():
    """The prior table is the Dinkelbach optimum of the measured costs; the online guard only backs off."""
    from jitengine_flint_m4pro import copy_draft as D

    for mode in ("greedy", "sampled"):
        for L in D.BAND_L:
            r = D.cost_ratios(L)
            assert r[0] == 1.0 and all(b > a for a, b in pairwise(r)) and len(r) == D.KMAX + 1
            tab = D.depth_table(mode, L)
            assert tab[0] == 0 and all(0 <= k <= D.KMAX for k in tab)
            assert list(tab) == sorted(
                tab
            )  # a longer match never drafts less (prior acceptance rises)
        assert D.band(1024) == 0 and D.band(10**6) == len(D.BAND_L) - 1
        p = D.DepthPolicy(mode)
        assert p.table(1500) == D.depth_table(mode, 1500)
        b = next(b for b in range(len(D.BUCKETS)) if p.table(1500)[b])
        n = D.BUCKETS[b][0]
        k0 = p.table(1500)[b]
        for _ in range(200):  # drafts from this bucket always rejected: back off to 0
            p.observe(n, p.table(1500)[b] or k0, 0)
        assert p.table(1500)[b] == 0 and p.backoffs >= 1
        assert all(
            x == y
            for i, (x, y) in enumerate(zip(p.table(1500), D.depth_table(mode, 1500)))
            if i != b
        )
        q = D.DepthPolicy(mode)
        for _ in range(200):  # always fully accepted: never deeper than the prior
            q.observe(n, k0, k0)
        assert q.table(1500) == D.depth_table(mode, 1500)
        off = D.DepthPolicy(mode, guard=False)
        off.observe(n, k0, 0)
        assert off.table(1500) == D.depth_table(mode, 1500)
    assert D.verify_ms(1, 1536) == D.VERIFY_MS[0][1][0] and D.verify_ms(8, 258560) > D.verify_ms(
        8, 131584
    )
