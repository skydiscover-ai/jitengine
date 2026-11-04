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

    from jitengine_qwen3_6_35b_a3b_m2 import sampler as S

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
            smp = S.Sampler(t, p, k, seed=n, vocab=V)
            got = smp._sample_exact(lp, 0)
            want = _stock(S, lp, smp.key(0), t, p, k)
            assert got.shape == want.shape and got.dtype == want.dtype
            if got.item() != want.item():
                bad.append((scale, ties, t, p, k, r))
    assert not bad, f"{len(bad)}/{n} draws differ: {bad[:5]}"


def test_radix_sampler_and_batched_rows_match_stock_draws(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2 import sampler as S

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
                S.Sampler(t, p, k, seed=n * 8 + b, vocab=V).key(1),
                t,
                p,
                k,
            ).item()
            for b in range(B)
        ]
        solo = [
            S.Sampler(t, p, k, seed=n * 8 + b, vocab=V, radix=True)._sample_exact(lps[b], 1).item()
            for b in range(B)
        ]
        smps = [S.Sampler(t, p, k, seed=n * 8 + b, vocab=V, radix=True) for b in range(B)]
        rows = S.sample_rows(mx.concatenate(lps, axis=0), smps, [1] * B).tolist()
        if not (solo == want and rows == want):
            bad.append((n, want, solo, rows))
    assert not bad, bad[:3]
