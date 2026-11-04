"""router_combine, Flash-Next's decode router top-k and MoE combine.

`topk_softmax` claims the top-k of a precise float32 softmax with ties going to the lower expert
id, in descending probability; the stock path it replaces is `Qwen4ExpTextSparseMoeBlock.route`
(precise softmax,
argpartition, renormalised). Ties and the order are checked exactly, the probabilities against an
exact float64 softmax as accurately as stock's. `moe_combine` claims the gathered pairs are summed in
their original order, with `sel` reading them in place; reading split parts through `sel` must give
the bits of reading the assembled pairs, and against the stock tail of
`Qwen4ExpTextSparseMoeBlock.fused_decode`
(score-weighted sum, shared expert, `hyper_inject`) it must be as accurate as stock. Geometry is
Flash-Next's: 512 experts, top-10, 4 streams of 2560.
"""

import numpy as np

E, K, HC, HID = 512, 10, 4, 2560


def _softmax64(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def test_topk_softmax_ties_go_to_the_lower_expert_id(requires_mlx):
    """Logits on a 0.5 grid tie exactly (equal logits give equal probabilities, unequal ones differ
    by a factor e^0.5), so the expected ids are a stable sort of the logits, descending."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.router_combine import topk_softmax

    for t in range(24):
        dt = mx.float32 if t % 3 else mx.bfloat16
        levels = 6 + 50 * (t % 4)  # from ~85 experts per level down to ~3
        grid = mx.random.randint(0, levels, (E,), key=mx.random.key(t)).astype(mx.float32) * 0.5
        logits = grid.astype(dt)
        inds, scores = topk_softmax(logits, K, norm_topk=bool(t % 2))
        want = np.argsort(-np.array(grid), kind="stable")[:K]
        assert np.array_equal(np.array(inds), want), t
        sc = np.array(scores)
        assert np.all(sc[:-1] >= sc[1:]), t


def test_topk_softmax_as_accurate_as_stock_route(requires_mlx, accuracy):
    """Continuous float32 logits at several scales: the same experts as argpartition's top-k, and
    the renormalised (or raw) top-k probabilities no less accurate than stock's."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.router_combine import topk_softmax

    for t, scale in enumerate((0.5, 2.0, 8.0)):
        for norm in (True, False):
            logits = mx.random.normal((E,), key=mx.random.key(100 + t)) * scale
            inds, scores = topk_softmax(logits, K, norm_topk=norm)
            gates = mx.softmax(logits, axis=-1, precise=True)
            sinds = mx.argpartition(gates, kth=-K, axis=-1)[-K:]
            assert set(np.array(inds).tolist()) == set(np.array(sinds).tolist()), (scale, norm)
            stock = gates[inds]  # stock's values for the kernel's ids, in the kernel's order
            if norm:
                stock = stock / stock.sum()
            p = _softmax64(accuracy.np64(logits))[np.array(inds)]
            exact = p / p.sum() if norm else p
            accuracy.no_worse_than_stock(scores, stock, exact, what=f"scale {scale} norm {norm}")


def _combine_inputs(mx, seed):
    key = lambda i: mx.random.key(seed + i)  # noqa: E731
    y = mx.random.normal((K, HID), key=key(0))
    scores = mx.softmax(mx.random.normal((K,), key=key(1)))
    sh = mx.random.normal((HID,), key=key(2))
    h2 = mx.random.normal((HC * HID,), key=key(3))
    inject = 2.0 * mx.sigmoid(mx.random.normal((HC,), key=key(4)))
    shg = mx.random.normal((1,), key=key(5))
    return y, scores, sh, h2, inject, shg


def test_moe_combine_reads_split_pairs_in_place(requires_mlx):
    """Pairs split over y1 and y2 and picked by `sel` (the pre-gathered and fetched parts, or a
    miss correction) give the bits of the assembled pairs read in order."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.router_combine import moe_combine

    y, scores, sh, h2, inject, shg = _combine_inputs(mx, 0)
    order = np.random.default_rng(1).permutation(K)  # pair j sits at row sel[j] of [y1; y2]
    stacked = y[mx.array(np.argsort(order))]  # row r of [y1; y2] holds pair order^-1[r]
    sel = order
    for split in (3, 7):
        y1, y2 = stacked[:split], stacked[split:]
        for gate in (None, shg):
            parts = moe_combine(y1, y2, sel, scores, sh, h2, inject, HC, HID, sh_gate=gate)
            whole = moe_combine(
                y, None, np.arange(K), scores, sh, h2, inject, HC, HID, sh_gate=gate
            )
            assert mx.array_equal(parts, whole).item(), (split, gate is None)


def test_moe_combine_as_accurate_as_the_stock_tail(requires_mlx, accuracy):
    """Against `(y * scores).sum + shared` then `hyper_inject`,
    Qwen4ExpTextSparseMoeBlock.fused_decode's stock tail."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m2.kernels.router_combine import moe_combine
    from jitengine_qwen3_8_flash_next_m2.model.hyper_connections import hyper_inject

    for seed in (0, 10, 20):
        y, scores, sh, h2, inject, shg = _combine_inputs(mx, seed)
        for gate in (None, shg):
            got = moe_combine(y, None, np.arange(K), scores, sh, h2, inject, HC, HID, sh_gate=gate)
            shared = sh if gate is None else mx.sigmoid(gate) * sh
            moe = (y[None, None] * scores.reshape(1, 1, K, 1)).sum(axis=-2) + shared
            stock = hyper_inject(moe, h2.reshape(1, 1, -1), inject.reshape(1, 1, -1)).reshape(-1)
            Y, S, H = accuracy.np64(y), accuracy.np64(scores), accuracy.np64(sh)
            if gate is not None:
                H = H / (1.0 + np.exp(-accuracy.np64(gate)))
            m = (Y * S[:, None]).sum(axis=0) + H
            exact = accuracy.np64(h2) + (accuracy.np64(inject)[:, None] * m[None, :]).reshape(-1)
            accuracy.no_worse_than_stock(got, stock, exact, what=f"seed {seed}")
