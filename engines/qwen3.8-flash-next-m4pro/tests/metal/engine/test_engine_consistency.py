"""Flash-Next token for token on the tiny model, greedy: prefix hits, pool size, batched rows.

What the engine promises: warm runs decode what cold runs decode ("prefix-store snapshots and warm
runs match cold ones bit for bit", engine.py); the result does not depend on the expert pool's
size or slot assignment (`Qwen4ExpTextSparseMoeBlock._gather_slots`), so a pool smaller than the
model, with misses, evictions and waves, decodes what a pool holding every record decodes; and the
batched decode step is row exact (`QLinear.rows`, the fused hyper-connection rows, per-row attention
and recurrence), so a row's tokens do not depend on which rows share its steps. It does not promise
that a batched row decodes what the row decodes alone: a lone row runs the fused decode kernels and
a batch the stock arithmetic, so against solo only the first token, which comes from the row's own
prefill, is compared. One test per property, so a divergence names the path that broke.
"""

N = 24  # tokens per request
# snapshots, and the admission window that makes an arrival group one prefill wave, on short prompts
OPTS = dict(prefix_grid_tokens=64, prefix_min_divergence_tokens=16)


def _req(prompt):
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

    return GenRequest(prompt_ids=prompt, max_tokens=N, stop_at_eos=False)


def _same(got, want, what):
    i = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
    assert got == want, (
        f"{what}: first difference at token {i}: {got[i : i + 4]} vs {want[i : i + 4]}"
    )


def test_prefix_hits_get_the_cold_tokens(flashnext_engine, requires_mlx):
    """A shared 100-token prefix (QSA prunes past 20 tokens, so the restored state includes the
    indexer's keys) stored at a divergence and hit, and a prompt sent three times."""
    warm = flashnext_engine(**OPTS, prefix_store_entries=8)
    cold = flashnext_engine(**OPTS, prefix_store_entries=0)
    shared = list(range(1000, 1100))
    prompts = [shared + [i, i + 1, i + 2] for i in range(4)] + [shared[:90] + [7]] * 3
    hits = 0
    for i, p in enumerate(prompts):
        w = warm.generate(_req(p))
        hits += w.cached_prompt_tokens > 0
        _same(
            w.token_ids,
            cold.generate(_req(p)).token_ids,
            f"prompt {i} (cached {w.cached_prompt_tokens})",
        )
    assert hits >= 3, hits


def test_appended_conversation_hits_and_gets_the_cold_tokens(flashnext_engine, requires_mlx):
    """Agent turns: each prompt is the previous prompt, its reply and `new` fresh tokens, so every
    turn crosses a new grid boundary. Turn t must restore exactly the last grid point inside turn
    t-1's head (the prompt but its last token), and decode what a cache-off engine decodes."""
    warm = flashnext_engine(**OPTS, prefix_store_entries=8)
    cold = flashnext_engine(**OPTS, prefix_store_entries=0)
    chunk, new = 64, 100
    prompt = list(range(1000, 1200))  # 200 tokens: off the 64 grid
    gen = lambda e, p: e.generate(_req(p))  # noqa: E731
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


def test_a_small_expert_pool_gets_the_full_pool_tokens(flashnext_engine, requires_mlx, monkeypatch):
    """A pool of 12 of the 32 expert records (one decode step routes 8), so most steps miss and
    evict. The plan's pool is capped rather than the budget cut, since a budget that small would
    also lower MLX's process-wide memory limit under the session's other engines; the extension
    pool, which would refill it from the idle budget, is off."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as fne

    full = flashnext_engine(prefix_store_entries=0)
    plan_budget = fne.plan_budget

    def capped(*args, **kwargs):
        plan = plan_budget(*args, **kwargs)
        plan.pool_slots = 12
        return plan

    monkeypatch.setattr(fne, "plan_budget", capped)
    small = flashnext_engine(prefix_store_entries=0, pool_ext_slots=0)
    assert small.expert_pool.slots == 12 and full.expert_pool.slots == 32
    for i, p in enumerate([[5, 6, 7], list(range(100, 130)), list(range(2000, 2070))]):
        _same(small.generate(_req(p)).token_ids, full.generate(_req(p)).token_ids, f"prompt {i}")
    assert small.stats()["expert_misses"] > 0


def test_a_batched_row_does_not_depend_on_its_companions(flashnext_engine, requires_mlx):
    """The same 40-token prompt decoded beside two different sets of equally long prompts (so all
    rows prefill in one wave and decode every step together); its first token is also solo's."""
    e = flashnext_engine(**OPTS, prefix_store_entries=0)
    p = list(range(300, 340))
    a = e.generate_batch([_req(q) for q in (p, list(range(400, 440)), list(range(500, 540)))])
    b = e.generate_batch(
        [_req(q) for q in (p, list(range(600, 640)), list(range(700, 740)), list(range(800, 840)))]
    )
    assert e.stats()["batcher"]["peak_rows"] > 1
    _same(b[0].token_ids, a[0].token_ids, "the row beside other companions")
    solo = e.generate(_req(p)).token_ids
    assert a[0].token_ids[0] == solo[0], (a[0].token_ids[:4], solo[:4])
