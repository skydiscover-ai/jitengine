"""`group_prefix_points` (kvcache/prefix_cache.py) runs on the model thread over every pending head,
whenever the queue changes. It must give exactly the points of the all-pairs definition, in work that
grows with P log P, not P^2 (the all-pairs form took seconds at a few hundred pending long prompts,
and no decode step runs meanwhile). The work is counted, not timed: a wall-clock bound fails on a
loaded host."""

import random

import pytest

pytest.importorskip("mlx.core")  # the module imports MLX for the store's arrays

from jitengine_qwen3_8_flash_next_m4pro.kvcache import prefix_cache as pc


def all_pairs_points(heads, chunk, min_tokens):
    """The definition, as the earlier code computed it: every ordered pair of heads compared."""
    heads = [tuple(h) for h in heads]
    out = []
    for i, h in enumerate(heads):
        best = 0
        for j, other in enumerate(heads):
            if i != j:
                best = max(best, pc.common_prefix_len(h, other))
        a = (best // chunk) * chunk
        out.append(a if a >= min_tokens and a < len(h) else 0)
    return out


def _case(rng: random.Random):
    """P heads of up to 64 tokens over a small alphabet, many built from shared stems cut at, just
    below and just above grid points, with exact duplicates and prefixes of one another."""
    p = rng.randint(0, 12)
    alphabet = rng.choice([2, 3, 50])
    stems = [[rng.randrange(alphabet) for _ in range(rng.randint(0, 64))] for _ in range(3)]
    heads = []
    for _ in range(p):
        kind = rng.random()
        if heads and kind < 0.15:
            heads.append(list(rng.choice(heads)))  # a duplicate
        elif heads and kind < 0.25:
            h = rng.choice(heads)
            heads.append(h[: rng.randint(0, len(h))])  # a prefix of another head
        elif kind < 0.75:
            stem = rng.choice(stems)
            grid = rng.choice([1, 4, 8, 16])
            cut = min(len(stem), max(0, grid * rng.randint(0, 4) + rng.choice([-1, 0, 0, 1])))
            tail = [rng.randrange(alphabet) for _ in range(rng.randint(0, 64 - cut))]
            heads.append(stem[:cut] + tail)
        else:
            heads.append([rng.randrange(alphabet) for _ in range(rng.randint(0, 64))])
    if rng.random() < 0.5:
        heads = [tuple(h) for h in heads]
    return heads, rng.choice([1, 4, 16]), rng.choice([1, 8])


def test_points_equal_the_all_pairs_definition_on_random_cases():
    rng = random.Random(852)
    for k in range(500):
        heads, chunk, min_tokens = _case(rng)
        want = all_pairs_points(heads, chunk, min_tokens)
        assert pc.group_prefix_points(heads, chunk, min_tokens) == want, (
            k,
            heads,
            chunk,
            min_tokens,
        )


def test_the_scan_finds_the_same_common_prefix_as_the_binary_search():
    rng = random.Random(35)
    for _ in range(2000):
        n = rng.randint(0, 3000)
        a = [rng.randrange(3) for _ in range(n)]
        cut = rng.randint(0, n)
        b = a[:cut] + [rng.randrange(3) for _ in range(rng.randint(0, 200))]
        assert pc._scan_prefix_len(a, b) == pc.common_prefix_len(a, b)
        assert pc._scan_prefix_len(tuple(b), tuple(a)) == pc.common_prefix_len(b, a)


def _heads(p: int, length: int, shared: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    stem = [rng.randrange(248_000) for _ in range(shared)]
    return [stem + [rng.randrange(248_000) for _ in range(length - shared)] for _ in range(p)]


class _Reads:
    """A token sequence that counts the tokens read through it, by slice or one at a time."""

    def __init__(self, toks):
        self.toks, self.read = list(toks), 0

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, i):
        got = self.toks[i]
        self.read += len(got) if isinstance(i, slice) else 1
        return got


def test_the_scan_reads_the_common_prefix_and_at_most_one_block_more():
    """Work proportional to the common prefix, not to the heads' length: past it, the scan reads the
    rest of the block that differs (at most 1,024 tokens) and the differing token."""
    a = list(range(27_000))
    for n in (0, 1, 31, 32, 33, 1000, 1024, 4096, 26_999, 27_000):
        ra, rb = _Reads(a), _Reads(a[:n] + [-1] + a[n + 1 :])
        assert pc._scan_prefix_len(ra, rb) == n
        assert max(ra.read, rb.read) <= n + 1 + 1024, (n, ra.read, rb.read)


@pytest.mark.parametrize("shared", [0, 4096])
def test_hundreds_of_long_pending_heads_take_one_scan_per_neighbour_pair(shared, monkeypatch):
    """P = 256 heads of 27,000 tokens (the context bound at 40 GB), with random first tokens or a
    4,096-token common stem: one sort and P - 1 = 255 neighbour scans, where the all-pairs form
    compares P (P - 1) / 2 = 32,640 pairs."""
    big = _heads(256, 27_000, shared, seed=1)
    scans = []
    scan = pc._scan_prefix_len

    def counted(a, b):
        scans.append(scan(a, b))
        return scans[-1]

    monkeypatch.setattr(pc, "_scan_prefix_len", counted)
    pts = pc.group_prefix_points(big, 2048, 256)
    assert pts == [(shared // 2048) * 2048 if shared >= 2048 else 0] * 256
    assert len(scans) == 255
