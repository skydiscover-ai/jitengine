"""Context-copy drafts for mode `code` speculation (host side).

Draft semantics (the facts the depth policy was fitted on): for a history H of length L
(prompt + committed tokens), find the LONGEST n in NSET whose last-n-token suffix of H occurred earlier, at its MOST
RECENT earlier occurrence ending at e < L (an n-gram "ends at e" when it covers H[e - n:e]); the draft is
H[e], H[e + 1], ..., continued periodically past the end of H (the copy of a copy, which is how a loop goes on).
Lookups verify the tokens, so a hash collision can only shorten a match, never fabricate one.

Index: one dict per n mapping the n-gram's hash to its last end position. The prompt part (ends 1..P) is built
once per prompt with numpy and kept beside the prefix-cache snapshot; a request adds its own ends on top in a
separate layer, so the shared prompt part is never modified.

Depth policy: DepthPolicy, one per request. k drafts per pass (verify pass of M = k + 1 rows) maximise the
expected committed tokens per unit of measured pass cost, from a prior acceptance model per match-length bucket and the
acceptance this request has realised so far (the online guard backs a losing bucket off, down to no drafting); capped by
the tokens left and by the compiled row counts. Committed tokens never depend on the policy (every row is checked
against the target's own selection), only the speed does.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

NSET = (64, 48, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1)  # searched longest first
BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (10, 12), (16, 24), (32, 48), (64, 64))
_BUCKET_OF = {}
for _b, (_lo, _hi) in enumerate(BUCKETS):
    for _n in range(_lo, _hi + 1):
        _BUCKET_OF[_n] = _b
MASK = (1 << 64) - 1
BASE = 0x9E3779B97F4A7C15 | 1  # odd multiplier, hashes mod 2^64
_PW = [1]
for _ in range(max(NSET)):
    _PW.append(_PW[-1] * BASE & MASK)

# ---------------------------------------------------------------- depth policy
# Per pass: k = the depth maximising expected committed tokens minus RHO x cost over k = 0 .. KMAX (Dinkelbach: RHO is
# the long-run committed tokens per M = 1-pass cost of the prior policy in the band), from
#   - the cost ratio r(M) = (T(M, L) + SYNC_MS) / T(1, L) of a verify pass of M = k + 1 rows (every verify pass is a
#     host round trip the pipelined loop cannot hide; M = 1 passes are queued on the device and cost T(1)), and
#   - the survival S(k) = q_1 ... q_k of the copy draft in the pass's match-length bucket, q_j = P(the j-th draft is
#     accepted | the first j - 1 were).
# The prior q comes from PRIOR_Q (fitted offline, below); the online guard blends in what this request realised, per
# (bucket, j), with GUARD_W pseudo-counts of prior: a bucket whose drafts keep failing backs off its depth, down to no
# drafting. Committed tokens never depend on the policy (every row is checked against the target's own selection),
# only the speed does. Nothing keys on prompt content: the inputs are the match length, the KV length and the
# request's own acceptance history.
KMAX = 7
# Measured verify-pass ms of the kernel, M = 1 .. 8 at KV length L (benchmarks/kernels/megakernel.py --verify on the
# M4 Pro, 2 interleaved rounds each, at 1,536 / 8,704 / 33,280; at 131,584 only M = 1/2/3/4/8 were measured, with
# M = 5-7 placed between M = 4 and 8 at 33K's fractions). Linear in L between points and beyond the last (the
# attention part is linear in L); 258K extrapolates to 36.75 ms at M = 1.
VERIFY_MS = (
    (1536, (7.533, 12.602, 18.382, 22.527, 30.182, 36.582, 40.982, 44.205)),
    (8704, (8.258, 13.725, 20.105, 24.475, 32.719, 38.963, 44.003, 47.901)),
    (33280, (11.043, 17.767, 25.543, 31.724, 41.704, 49.290, 55.939, 61.732)),
    (131584, (22.26, 33.34, 47.04, 58.38, 77.30, 91.65, 104.28, 115.25)),
)
# GPU idle per host round trip (measured on the M4 Pro: a drained-queue gap costs ~25% over its length)
SYNC_MS = 0.15
BAND_L = (
    1536,
    4096,
    8704,
    17020,
    33280,
    66048,
    131584,
    258560,
)  # policy evaluated at these KV lengths ...
BAND_EDGES = tuple(int((a * b) ** 0.5) for a, b in pairwise(BAND_L))  # ... nearest in log L
GUARD_W = 32.0  # prior pseudo-count per (bucket, draft index) in the online guard (replay:
# 2-64 all within +-0.3% of no guard on the 42 fit/check streams; 32 is neutral)
# PRIOR_Q[mode][bucket][j - 1] and the bucket frequencies BUCKET_W[mode] (Dinkelbach weights): pooled over rec's 9
# census code streams and this engine's own 9 code streams per mode; the held-out corpus streams are a check only.
PRIOR_Q = {
    "greedy": (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.2186, 0.4417, 0.7128, 0.7662, 0.8075, 0.8314, 0.8881),
        (0.4061, 0.6043, 0.7235, 0.7902, 0.8291, 0.8553, 0.8564),
        (0.5714, 0.6923, 0.7685, 0.8193, 0.8346, 0.8282, 0.8723),
        (0.6915, 0.7748, 0.8223, 0.8208, 0.8429, 0.8668, 0.8827),
        (0.8039, 0.8221, 0.8445, 0.8748, 0.8927, 0.9183, 0.9077),
        (0.8805, 0.9052, 0.9195, 0.9173, 0.9124, 0.9011, 0.9173),
        (0.931, 0.9329, 0.9331, 0.9381, 0.9474, 0.9566, 0.9599),
        (0.9832, 0.9829, 0.9834, 0.9831, 0.9828, 0.9825, 0.9822),
        (0.9889, 0.9877, 0.9876, 0.9874, 0.9876, 0.9875, 0.9877),
    ),
    "sampled": (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1951, 0.4451, 0.6234, 0.7918, 0.8454, 0.8659, 0.8099),
        (0.3857, 0.5803, 0.7618, 0.8451, 0.85, 0.8, 0.8186),
        (0.5455, 0.724, 0.8212, 0.8281, 0.7958, 0.813, 0.8342),
        (0.7178, 0.7798, 0.8085, 0.799, 0.8188, 0.875, 0.881),
        (0.7919, 0.8248, 0.8455, 0.8858, 0.9126, 0.9137, 0.9089),
        (0.8834, 0.8925, 0.914, 0.9072, 0.9046, 0.9006, 0.9219),
        (0.931, 0.9377, 0.9425, 0.9466, 0.9466, 0.9511, 0.9564),
        (0.9811, 0.9824, 0.982, 0.985, 0.9848, 0.9863, 0.9852),
        (0.9864, 0.9852, 0.985, 0.9848, 0.9846, 0.9843, 0.9846),
    ),
}
BUCKET_W = {
    "greedy": (1869, 4039, 1960, 1092, 1355, 1489, 1138, 1376, 1310, 2804),
    "sampled": (2175, 4525, 2082, 1076, 1297, 1427, 1106, 1275, 1271, 2198),
}


def bucket(n: int) -> int:
    return _BUCKET_OF[n]


def verify_ms(M: int, L: int) -> float:
    pts = VERIFY_MS
    if L <= pts[0][0]:
        lo, hi = pts[0], pts[1]
    else:
        lo, hi = next(((a, b) for a, b in pairwise(pts) if L <= b[0]), (pts[-2], pts[-1]))
    t = (L - lo[0]) / (hi[0] - lo[0])
    return lo[1][M - 1] + t * (hi[1][M - 1] - lo[1][M - 1])


def cost_ratios(L: int) -> tuple:
    """r(M) for M = 1 .. KMAX + 1 at KV length L (r(1) = 1: a queued M = 1 pass)."""
    t1 = verify_ms(1, L)
    return (1.0,) + tuple((verify_ms(M, L) + SYNC_MS) / t1 for M in range(2, KMAX + 2))


def _best_depth(q, r, rho, kcap=KMAX) -> int:
    best, val, e, s = 0, 1.0 - rho, 1.0, 1.0
    for k in range(1, kcap + 1):
        s *= q[k - 1]
        e += s
        v = e - rho * r[k]
        if v > val + 1e-12:
            best, val = k, v
    return best


def _fit_band(mode: str, r) -> tuple:
    """(depth per bucket, RHO) of the prior at cost ratios r: Dinkelbach on the bucket frequencies."""
    q, w = PRIOR_Q[mode], BUCKET_W[mode]
    tot = sum(w) or 1.0
    rho, ch = 1.0, (0,) * len(BUCKETS)
    for _ in range(100):
        ch = (0,) + tuple(_best_depth(q[b], r, rho) for b in range(1, len(BUCKETS)))
        num = den = 0.0
        for b, k in enumerate(ch):
            s, e = 1.0, 1.0
            for j in range(k):
                s *= q[b][j]
                e += s
            num += w[b] / tot * e
            den += w[b] / tot * r[k]
        new = num / den
        if abs(new - rho) < 1e-12:
            break
        rho = new
    return ch, rho


def band(ctx: int) -> int:
    for i, e in enumerate(BAND_EDGES):
        if ctx <= e:
            return i
    return len(BAND_L) - 1


_BANDS = {
    mode: tuple((cost_ratios(L),) + _fit_band(mode, cost_ratios(L)) for L in BAND_L)
    for mode in ("greedy", "sampled")
}


def depth_table(mode: str, ctx: int) -> tuple:
    """The prior depth per match bucket at KV length ctx (what DepthPolicy uses before it has observed anything)."""
    return _BANDS[mode][band(ctx)][1]


class DepthPolicy:
    """One request's depth policy: the prior table, corrected online by the acceptance this request realises.

    observe(n, k, a) after every verify pass drafted at match length n with depth k that accepted a <= k drafts:
    draft j (1 <= j <= min(a + 1, k)) was tested, and accepted iff j <= a. Per (bucket, j) the estimate is
    q = (accepted + GUARD_W q_prior) / (tested + GUARD_W); the bucket's depth is re-chosen against the band's cost
    ratios and RHO, capped at the prior depth (the guard backs off; it never drafts deeper than the fit)."""

    def __init__(self, mode: str, guard: bool = True):
        self.mode = mode
        self.guard = guard
        self.prior = PRIOR_Q[mode]
        nb = len(BUCKETS)
        self.tested = [[0] * KMAX for _ in range(nb)]
        self.accepted = [[0] * KMAX for _ in range(nb)]
        self.tabs = {}  # band -> effective depth per bucket (after observations)
        self.backoffs = 0  # bucket depth reductions made by the guard

    def table(self, ctx: int) -> tuple:
        i = band(ctx)
        t = self.tabs.get(i)
        if t is None:
            t = self.tabs[i] = self._table(i)
        return t

    def _q(self, b: int) -> list:
        tb, ab, pb = self.tested[b], self.accepted[b], self.prior[b]
        return [(ab[j] + GUARD_W * pb[j]) / (tb[j] + GUARD_W) for j in range(KMAX)]

    def _table(self, i: int) -> tuple:
        r, prior_tab, rho = _BANDS[self.mode][i]
        if not self.guard:
            return prior_tab
        return tuple(
            0 if b == 0 or not prior_tab[b] else _best_depth(self._q(b), r, rho, prior_tab[b])
            for b in range(len(BUCKETS))
        )

    def observe(self, n: int, k: int, a: int) -> None:
        if not self.guard or k <= 0:
            return
        b = bucket(n)
        for j in range(min(a + 1, k)):  # draft j + 1 was tested (all before it accepted)
            self.tested[b][j] += 1
            if j < a:
                self.accepted[b][j] += 1
        for i, t in list(self.tabs.items()):
            new = self._table(i)
            if new[b] < t[b]:
                self.backoffs += 1
            self.tabs[i] = new


def _prefix_hashes(tokens) -> list:
    h = [0] * (len(tokens) + 1)
    acc = 0
    for i, x in enumerate(tokens):
        acc = (acc * BASE + x + 1) & MASK
        h[i + 1] = acc
    return h


class PromptIndex:
    """The prompt's n-gram ends 1..P (immutable once built; shared by every request on this prompt)."""

    def __init__(self, prompt):
        self.prompt = tuple(int(x) for x in prompt)
        P = len(self.prompt)
        self.h = _prefix_hashes(self.prompt)
        hv = np.array(self.h, dtype=np.uint64)
        self.ends = {}
        with np.errstate(over="ignore"):
            for n in NSET:
                if P < n:
                    self.ends[n] = {}
                    continue
                g = hv[n:] - hv[:-n] * np.uint64(_PW[n])  # wraps mod 2^64: the rolling hash
                e = np.arange(n, P + 1)
                self.ends[n] = dict(
                    zip(g.tolist(), e.tolist())
                )  # a later end overwrites: most recent


class DraftIndex:
    """Prompt + committed tokens of one request; lookup() and draft() at the current history length."""

    def __init__(self, base: PromptIndex):
        self.base = base
        self.t = list(base.prompt)
        self.h = list(base.h)
        self.top = {n: {} for n in NSET}
        self.inserted = len(self.t)  # ends 1..inserted are indexed (the prompt part lives in base)

    def __len__(self) -> int:
        return len(self.t)

    def append(self, x: int) -> None:
        self.t.append(int(x))
        self.h.append((self.h[-1] * BASE + int(x) + 1) & MASK)

    def _gram(self, end: int, n: int) -> int:
        return (self.h[end] - self.h[end - n] * _PW[n]) & MASK

    def _catch_up(self) -> None:
        """Index every end < len(H) (the suffix ending AT len(H) is the query, never its own occurrence)."""
        L = len(self.t)
        for end in range(self.inserted + 1, L):
            for n in NSET:
                if end >= n:
                    self.top[n][self._gram(end, n)] = end
        self.inserted = max(self.inserted, L - 1)

    def lookup(self) -> tuple:
        """(n, e): the longest earlier match of H's suffix and the end of its most recent occurrence (0, -1 if
        not even the last token recurs)."""
        self._catch_up()
        L = len(self.t)
        t = self.t
        for n in NSET:
            if L < n:
                continue
            g = self._gram(L, n)
            e = self.top[n].get(g)
            if e is None:
                e = self.base.ends[n].get(g)
            if e is not None and e < L and t[e - n : e] == t[L - n : L]:
                return n, e
        return 0, -1

    def draft(self, e: int, k: int) -> list:
        """k tokens copied from H[e:], continued periodically past the end of H."""
        L = len(self.t)
        out = []
        for j in range(k):
            src = e + j
            out.append(self.t[src] if src < L else out[src - L])
        return out


def certain_no_draft(n: int, tab: tuple) -> int:
    """How many upcoming passes are certain to draft nothing, whatever tokens they produce: one more token
    lengthens the longest earlier match by at most one. n is the match found now (the true match is below the
    next NSET length up)."""
    upper = {64: 64}
    asc = sorted(NSET)
    for a, b in pairwise(asc):
        upper[a] = b - 1
    true_max = 0 if n == 0 else upper[n]
    c = 0
    while c < 64:
        m = min(true_max + c, 64)
        found = max([x for x in NSET if x <= m], default=0)
        if any(tab[b] != 0 for b in range(bucket(found) + 1)):
            break
        c += 1
    return c
