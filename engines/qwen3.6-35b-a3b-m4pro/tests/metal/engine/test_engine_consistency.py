"""Qwen3.6 M4 Pro token for token, two ways, greedy on a tiny model: batched, prefix hit.

The engine promises that a batched row decodes its solo tokens (batched decode is row exact) and
that a prefix-store hit decodes the cold run's tokens ("a prefix-store hit bit-identical to a cold
run"). One test per property, so a divergence names the path that broke, at the first differing
token. Each runs on the 128-wide tiny model and on a 512-wide one, where the custom kernels' shape
conditions hold (K a multiple of 512) as on the measured model; both are 4-bit, the width this
engine serves. There is no offload-against-resident test: 4-bit experts have no segmented prefill
kernel, so a long prefill chunk takes stock's tiled GEMM resident and its per-pair kernel from the
pool, which round differently.
"""

import pytest

GEOMETRIES = [dict(bits=4, experts=16, hidden=None), dict(bits=4, experts=16, hidden=512)]
N = 24  # tokens per request
GEOMETRY_IDS = ["128", "512"]


def _prompts():
    """Prompts of different lengths, one past the prefix store's 256-token minimum."""
    return [[5, 6, 7], list(range(100, 137)), list(range(2000, 2090)), list(range(600, 900))]


def _first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _same(got, want, what):
    i = _first_difference(got, want)
    assert got == want, (
        f"{what}: first difference at token {i}: {got[i : i + 4]} vs {want[i : i + 4]}"
    )


def _gen(e, prompt):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    return e.generate(GenRequest(prompt_ids=prompt, max_tokens=N, stop_at_eos=False))


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["128", "512"])
def test_batched_rows_get_their_solo_tokens(tiny_engine, requires_mlx, geometry):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest

    e = tiny_engine(**geometry, prefix_cache_bytes=0, prefill_chunk=64)
    prompts = _prompts()
    solo = [_gen(e, p).token_ids for p in prompts]
    reqs = [GenRequest(prompt_ids=p, max_tokens=N, stop_at_eos=False) for p in prompts]
    batch = e.generate_batch(reqs)
    assert e.stats()["batcher"]["peak_rows"] > 1
    for i, (r, want) in enumerate(zip(batch, solo)):
        _same(r.token_ids, want, f"prompt {i} batched vs solo")


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["128", "512"])
def test_prefix_hits_get_the_cold_tokens(tiny_engine, requires_mlx, geometry):
    """A shared 300-token prefix (stored at the divergence of the first two requests, then hit),
    and a prompt sent three times (its end-of-prompt snapshot is admitted on the repeat)."""
    warm = tiny_engine(**geometry, prefix_cache_bytes=64 << 20)
    cold = tiny_engine(**geometry, prefix_cache_bytes=0)
    shared = list(range(600, 900))
    prompts = [shared + [i, i + 1, i + 2] for i in range(4)] + [shared[:290] + [7]] * 3
    hits = 0
    for i, p in enumerate(prompts):
        w = _gen(warm, p)
        hits += w.cached_prompt_tokens > 0
        _same(w.token_ids, _gen(cold, p).token_ids, f"prompt {i} (cached {w.cached_prompt_tokens})")
    assert hits >= 4, hits


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=GEOMETRY_IDS)
def test_appended_conversation_hits_and_gets_the_cold_tokens(tiny_engine, requires_mlx, geometry):
    """Agent turns: each prompt is the previous prompt, its reply and `new` fresh tokens, so every
    turn crosses a new grid boundary. Turn t must restore exactly the last grid point inside turn
    t-1's head (the prompt but its last token), and decode what a cache-off engine decodes."""
    warm = tiny_engine(**geometry, prefix_cache_bytes=64 << 20)
    cold = tiny_engine(**geometry, prefix_cache_bytes=0)
    chunk, new = 256, 300
    prompt = list(range(600, 1000))  # 400 tokens: off the 256 grid
    gen = _gen
    last_grid = 0
    for turn in range(4):
        w = gen(warm, prompt)
        _same(
            w.token_ids,
            gen(cold, prompt).token_ids,
            f"turn {turn} (cached {w.cached_prompt_tokens})",
        )
        assert w.cached_prompt_tokens == last_grid, (turn, w.cached_prompt_tokens, last_grid)
        last_grid = ((len(prompt) - 1) // chunk) * chunk
        start = 3000 + 1000 * turn
        prompt = prompt + w.token_ids + list(range(start, start + new))
