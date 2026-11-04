"""Exact fast sampler: the same token as mlx_lm's filters + categorical for the same key."""

import itertools


def _stock(sampler_mod, lp, key, t, p, k):
    import mlx.core as mx

    if 0 < p < 1:
        lp = sampler_mod.apply_top_p(lp, p)
    if k > 0:
        lp = sampler_mod.apply_top_k(lp, k)
    return mx.random.categorical(lp * (1.0 / t), axis=-1, key=key)


def test_exact_fast_sampler_matches_stock_draws(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import sampler as S

    V, n, bad = 248320, 0, []
    for scale, ties, t, p, k in itertools.product(
        [1.0, 4.0, 20.0], [False, True], [0.3, 1.0], [1.0, 0.95, 0.8], [0, 20, 40]
    ):
        for r in range(2):
            n += 1
            k1, _k2 = mx.random.split(mx.random.key(1000 + n))
            lg = mx.random.normal((1, V), key=k1) * scale
            if ties:
                lg = mx.round(lg * 2) / 2
            lp = S.logprobs_from_logits(lg.astype(mx.bfloat16))
            smp = S.Sampler(t, p, k, seed=n, max_steps=4, vocab=V)
            got = smp._sample_exact(lp, 0)
            want = _stock(S, lp, smp.key(0), t, p, k)
            assert got.shape == want.shape and got.dtype == want.dtype
            if got.item() != want.item():
                bad.append((scale, ties, t, p, k, r))
    assert not bad, f"{len(bad)}/{n} draws differ: {bad[:5]}"


def test_radix_sampler_and_batched_rows_match_stock_draws(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import sampler as S

    V = 248320
    bad = []
    for n, (scale, ties, t, p, k) in enumerate(
        itertools.product([1.0, 12.0], [False, True], [0.7, 1.0], [0.95, 1.0], [20, 40])
    ):
        B = 1 + n % 4
        lg = mx.random.normal((B, V), key=mx.random.key(2000 + n)) * scale
        if ties:
            lg = mx.round(lg * 2) / 2
        lg = lg.astype(mx.bfloat16)
        lps = [S.logprobs_from_logits(lg[b : b + 1]) for b in range(B)]
        want = [
            _stock(
                S,
                lps[b],
                S.Sampler(t, p, k, seed=n * 8 + b, max_steps=4, vocab=V).key(1),
                t,
                p,
                k,
            ).item()
            for b in range(B)
        ]
        solo = [
            S.Sampler(t, p, k, seed=n * 8 + b, max_steps=4, vocab=V, radix=True)
            ._sample_exact(lps[b], 1)
            .item()
            for b in range(B)
        ]
        smps = [
            S.Sampler(t, p, k, seed=n * 8 + b, max_steps=4, vocab=V, radix=True) for b in range(B)
        ]
        rows = S.sample_rows(mx.concatenate(lps, axis=0), smps, [1] * B).tolist()
        if not (solo == want and rows == want):
            bad.append((n, want, solo, rows))
    assert not bad, bad[:3]


def test_sorted_bounds_equal_full_counts(requires_mlx):
    """sampler.sorted_bounds (binary search) == mx.sum(a < x), mx.sum(a <= x) on sorted rows with
    heavy ties, -inf prefixes and x below / at / above every entry."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.sampler import sorted_bounds

    for t in range(40):
        B, n = 1 + t % 8, 1000 + 37 * t
        key = mx.random.key(500 + t)
        a = mx.random.randint(-6, 6, (B, n), key=key).astype(mx.float32)
        if t % 3 == 0:
            a = mx.where(mx.arange(n)[None, :] < 50 * (t % 5), -float("inf"), a)
        a = mx.sort(a, axis=-1).astype(mx.bfloat16 if t % 2 else mx.float32)
        picks = [a[:, :1], a[:, n // 2 : n // 2 + 1], a[:, -1:], a[:, -1:] + 1, a[:, :1] - 1]
        for x in picks:
            x = x.astype(a.dtype)
            lt, le = sorted_bounds(a, x)
            assert mx.array_equal(lt, mx.sum(a < x, axis=-1, keepdims=True)).item()
            assert mx.array_equal(le, mx.sum(a <= x, axis=-1, keepdims=True)).item()


def test_fused_tail_matches_op_chain_and_stock(requires_mlx):
    """sampler.draw_fused (the windowed draw after its sort as two kernels) == the op chain on
    tokens, the replayed row sum and the in-kernel uniform (selfcheck.check_sampler_tail, with and
    without the radix sort), and == mlx_lm's filters + categorical through Sampler.__call__."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro import sampler as S
    from jitengine_qwen3_6_35b_a3b_m4pro.sampler import check_sampler_tail

    assert check_sampler_tail(True)
    assert check_sampler_tail(False)
    V = 248320
    for t, (temp, p, k) in enumerate(((0.7, 0.95, 40), (1.0, 0.9, 20), (0.6, 0.8, 64))):
        lg = (mx.random.normal((1, V), key=mx.random.key(900 + t)) * (1.5 + 2 * t)).astype(
            mx.bfloat16
        )
        smp = S.Sampler(temp, p, k, 77 + t, 512, V, radix=True, fused_tail=True)
        lp = S.logprobs_from_logits(lg)
        for step in range(0, 600, 37):
            got = smp(lg, step)
            want = _stock(S, lp, smp.key(step), temp, p, k)
            assert mx.array_equal(got.reshape(-1), want.reshape(-1).astype(got.dtype)).item(), (
                t,
                step,
            )


def test_batched_logprobs_equal_per_row(requires_mlx):
    """logprobs_from_logits over [B, V] == the per-row [1, V] calls, bitwise (engine._draw relies on it)."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.sampler import logprobs_from_logits

    for t in range(12):
        x = (mx.random.normal((8, 248320), key=mx.random.key(40 + t)) * (0.3 + 2 * t)).astype(
            mx.bfloat16
        )
        per_row = mx.concatenate([logprobs_from_logits(x[i : i + 1]) for i in range(8)], axis=0)
        assert mx.array_equal(logprobs_from_logits(x), per_row).item()
        sel = mx.array([1, 3, 6])
        assert mx.array_equal(logprobs_from_logits(x[sel]), per_row[sel]).item()
