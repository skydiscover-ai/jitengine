"""router_combine, Flash-Next's decode router top-k and MoE combine.

`topk_softmax` claims the top-k of a precise float32 softmax with ties going to the lower expert
id, in descending probability; the stock path it replaces is `Qwen4ExpTextSparseMoeBlock.route`
(precise softmax, argpartition, renormalised). Ties and the order are checked exactly, the
probabilities against an exact float64 softmax as accurately as stock's. `moe_combine` claims the
gathered pairs are summed in their original order, with `sel` reading them in place; reading split
parts through `sel` must give the bits of reading the assembled pairs, and against the stock tail of
`Qwen4ExpTextSparseMoeBlock.fused_decode` (score-weighted sum, shared expert, `hyper_inject`) it
must be as accurate as stock. Geometry is Flash-Next's: 512 experts, top-10, 4 streams of 2560.
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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import topk_softmax

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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import topk_softmax

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


# The selection `topk_softmax` replaced: KTOP rounds of a threadgroup-wide arg-max, each round two simd
# reductions, a barrier, an NSG-way merge and a barrier. Its ids, their order and its scores are the bar.
_ROUNDS_REFERENCE = """
    constexpr int NSG = NT / 32;
    const uint t = thread_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    threadgroup float prob[E];
    threadgroup float redv[NSG];
    threadgroup int redi[NSG];
    threadgroup float sel_p[KTOP];
    threadgroup int sel_i[KTOP];

    // precise softmax: max, exp(x - max), sum
    float x = (t < E) ? float(logits[t]) : -INFINITY;
    float m = simd_max(x);
    if (simd_lid == 0) redv[simd_gid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = redv[0];
    #pragma unroll
    for (int i = 1; i < NSG; i++) m = metal::max(m, redv[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = (t < E) ? metal::exp(x - m) : 0.0f;
    float ssum = simd_sum(e);
    if (simd_lid == 0) redv[simd_gid] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ssum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NSG; i++) ssum += redv[i];
    float p = e / ssum;
    if (t < E) prob[t] = p;
    bool taken = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k rounds of arg-max over the untaken; a tie goes to the SMALLEST expert id
    for (int r = 0; r < KTOP; r++) {
      float cv = (t < E && !taken) ? p : -INFINITY;
      float mv = simd_max(cv);
      int ci = (cv == mv) ? int(t) : 0x7fffffff;
      int mi = simd_min(ci);
      if (simd_lid == 0) { redv[simd_gid] = mv; redi[simd_gid] = mi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float bv = redv[0]; int bi = redi[0];
      #pragma unroll
      for (int i = 1; i < NSG; i++) {
        if (redv[i] > bv || (redv[i] == bv && redi[i] < bi)) { bv = redv[i]; bi = redi[i]; }
      }
      if (int(t) == bi) taken = true;
      if (t == 0) { sel_p[r] = bv; sel_i[r] = bi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t < KTOP) {
      float denom = 0.0f;
      if (NORM) {
        #pragma unroll
        for (int r = 0; r < KTOP; r++) denom += sel_p[r];
      }
      inds[t] = sel_i[t];
      scores[t] = NORM ? sel_p[t] / denom : sel_p[t];
    }
"""


def _rounds_reference(mx, logits, k, norm):
    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import HEADER

    E = int(logits.shape[-1])
    nt = max(32, -(-E // 32) * 32)
    kern = mx.fast.metal_kernel(
        name="test_topk_rounds_reference",
        input_names=["logits"],
        output_names=["inds", "scores"],
        header=HEADER,
        source=_ROUNDS_REFERENCE,
        ensure_row_contiguous=True,
    )
    return kern(
        inputs=[logits.reshape(-1)],
        template=[("T", logits.dtype), ("E", E), ("KTOP", k), ("NT", nt), ("NORM", norm)],
        grid=(nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.int32, mx.float32],
    )


def _tie_cases(rng, z):
    """From one continuous logit vector: exact ties across the k-th place, a block of equal maxima
    wider than k, the 0.5 grid, and wide logits whose tail underflows to +0."""
    o = np.argsort(-z, kind="stable")
    a = z.copy()
    for src in (o[K - 1], o[K]):
        a[rng.choice(z.size, 3, replace=False)] = z[src]
    b = z.copy()
    b[rng.choice(z.size, K + 5, replace=False)] = z.max()
    c = (rng.integers(0, int(rng.integers(4, 200)), z.size) * 0.5).astype(np.float32)
    return [z, a, b, c, z * np.float32(400.0)]


def test_topk_softmax_selection_is_the_round_by_round_argmax(requires_mlx):
    """The per-simdgroup lists and their merge pick the round-by-round arg-max's ids in its order, and
    its scores, bit for bit: continuous logits, planted ties, equal maxima, underflow, in float32 and
    bfloat16, with and without the renorm, and at router widths that leave padding lanes."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import topk_softmax

    rng = np.random.default_rng(3)
    for t in range(40):
        # widths whose last simdgroup holds padding lanes, one with fewer live lanes than k
        e, k = (E, K) if t % 5 else ((200, 10), (40, 12))[t // 5 % 2]
        z = (rng.standard_normal(e) * (0.5, 2.0, 8.0)[t % 3]).astype(np.float32)
        for case in _tie_cases(rng, z) if e == E else [z, z * np.float32(400.0)]:
            for dt in (mx.float32, mx.bfloat16):
                logits = mx.array(case).astype(dt)
                for norm in (True, False):
                    got = topk_softmax(logits, k, norm)
                    want = _rounds_reference(mx, logits, k, norm)
                    assert np.array_equal(np.array(got[0]), np.array(want[0])), (t, dt, norm)
                    assert np.array(got[1]).tobytes() == np.array(want[1]).tobytes(), (t, dt, norm)


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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import moe_combine

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

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import moe_combine
    from jitengine_qwen3_8_flash_next_m4pro.model.hyper_connections import hyper_inject

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


def test_topk_softmax2_rows_are_the_one_row_kernel(requires_mlx):
    """`topk_softmax2` (both MTP verify rows in one launch) gives each row the bits of
    `topk_softmax` on that row alone: ids, order and probabilities, float32 and bfloat16, tied and
    continuous logits; the rows swapped differ."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import (
        topk_softmax,
        topk_softmax2,
    )

    for t in range(12):
        dt = mx.float32 if t % 2 else mx.bfloat16
        if t % 3 == 0:  # ties across the k-th place
            l0, l1 = (
                mx.random.randint(0, 20, (1, E), key=mx.random.key(300 + 2 * t + j)).astype(
                    mx.float32
                )
                * 0.5
                for j in range(2)
            )
        else:
            l0, l1 = (
                mx.random.normal((1, E), key=mx.random.key(300 + 2 * t + j)) * (1 + t)
                for j in range(2)
            )
        l0, l1 = l0.astype(dt), l1.astype(dt)
        norm = bool(t % 4 < 2)
        inds, scores = topk_softmax2(l0, l1, K, norm)
        rows = [topk_softmax(x.reshape(-1), K, norm) for x in (l0, l1)]
        want_i = np.concatenate([np.array(r[0]) for r in rows])
        want_s = np.concatenate([np.array(r[1]) for r in rows])
        assert np.array_equal(np.array(inds), want_i), t
        assert np.array(scores).tobytes() == want_s.tobytes(), t
        assert not np.array_equal(np.array(topk_softmax2(l1, l0, K, norm)[0]), want_i), t


def test_moe_combine2_rows_are_the_masked_one_row_combine(requires_mlx):
    """`moe_combine2` (both verify rows' speculative combine in one launch, the slot lookup inside)
    gives each row the bits of `spec_build_pair`'s per-row chain: take the slots at the layer's
    offset, zero the weight of a non-resident pair, `moe_combine` with the shared gate; float32 and
    bfloat16. Dropping the mask (every slot resident) differs."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels.router_combine import (
        moe_combine,
        moe_combine2,
        topk_softmax2,
    )

    rng = np.random.default_rng(31)
    for t in range(8):
        dt = mx.float32 if t % 2 else mx.bfloat16
        layer = int(rng.integers(0, 48))
        tab_np = np.where(rng.random(48 * E) < 0.5, -1, rng.integers(0, 9000, 48 * E)).astype(
            np.int32
        )
        tab, off = mx.array(tab_np), mx.array([layer * E], dtype=mx.int32)
        l0, l1 = (
            mx.array(rng.standard_normal((1, E)).astype(np.float32)).astype(dt) for _ in range(2)
        )
        inds, scores = topk_softmax2(l0, l1, K, True)
        y = mx.array(rng.standard_normal((2 * K, HID)).astype(np.float32)).astype(dt)
        sh = mx.array(rng.standard_normal((2, HID)).astype(np.float32)).astype(dt)
        h2 = mx.array(rng.standard_normal((2, HC * HID)).astype(np.float32)).astype(dt)
        inj = (2.0 * mx.sigmoid(mx.array(rng.standard_normal((2, HC)).astype(np.float32)))).astype(
            dt
        )
        g0, g1 = (
            mx.array(rng.standard_normal((1, 1)).astype(np.float32)).astype(dt) for _ in range(2)
        )
        got = moe_combine2(y, inds, scores, tab, off, sh, g0, g1, h2, inj, HC, HID)
        want = []
        for r, g in ((0, g0), (1, g1)):
            ir, sr = inds[r * K : (r + 1) * K], scores[r * K : (r + 1) * K]
            sc = mx.where(mx.take(tab, ir + layer * E) >= 0, sr, mx.zeros_like(sr))
            want.append(
                moe_combine(
                    y[r * K : (r + 1) * K],
                    None,
                    np.arange(K),
                    sc,
                    sh[r : r + 1],
                    h2[r : r + 1],
                    inj[r : r + 1],
                    HC,
                    HID,
                    sh_gate=g.reshape(-1),
                )
            )
        want = mx.stack(want)
        assert mx.array_equal(got, want).item(), t
        resident = moe_combine2(
            y, inds, scores, mx.array(np.abs(tab_np)), off, sh, g0, g1, h2, inj, HC, HID
        )
        assert not mx.array_equal(resident, want).item(), t
