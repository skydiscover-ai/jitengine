"""grouped_expert_qmv, Flash-Next's routed-expert kernel: a pair's row never depends on the other pairs.

grouped_expert_qmv.py claims every output row depends only on its activation row and its expert's bytes,
in a compile-time k-order, so batched and solo rows are bit-identical; Qwen4ExpTextSparseMoeBlock
relies on it to split pairs across calls (pre-gathered hits, extension slots, misses) and on the
result not depending on slot assignment. Against stock `mx.gather_qmm` (what `moe_kernel="stock"`
runs) only the summation order differs and no tolerance is stated, so the kernel must be as accurate
as stock against an exact float64 product. Shapes are Flash-Next's: gate/up 640 x 2560, down 2560 x
640.
"""

import numpy as np

E, K_TOP, ROWS = 12, 10, 4  # experts in the test pool, top-k, decode rows


def _routing(seed):
    """[ROWS, K_TOP] distinct expert ids per row, flattened row-major like
    Qwen4ExpTextSparseMoeBlock's `flat`."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(E)[:K_TOP] for _ in range(ROWS)]).astype(np.int64)


def _segments(mx, flat, slot_of, k, rows_are_pairs=False):
    """Segment arrays as `Qwen4ExpTextSparseMoeBlock._expert_mlp_sorted` builds them: pairs sorted
    by expert id.

    Returns (pos, [seg_slot, seg_start, seg_cnt, pair_row], max count, rows): `pos` maps sorted
    pairs to flat pairs and `rows` is each sorted pair's activation row.
    """
    pos = np.argsort(flat, kind="stable")
    keys = flat[pos]
    _, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    rows = np.arange(pos.size) if rows_are_pairs else pos // k
    arrs = [slot_of[keys][start], start, cnt, rows]
    return pos, [mx.array(a.astype(np.uint32)) for a in arrs], int(cnt.max()), rows


def _pool(mx, q4, n, k, seed, perm_seed=0):
    """[E, n, k] stacks; slot j holds expert perm[j], so slot ids are not expert ids.

    Returns ((w, s, b) stacks, slot_of[expert], per-expert (w, s, b)).
    """
    experts = [q4((n, k), seed=seed + e) for e in range(E)]
    perm = np.random.default_rng(perm_seed).permutation(E)
    stacks = tuple(mx.stack([experts[e][i] for e in perm]) for i in range(3))
    slot_of = np.empty(E, dtype=np.int64)
    slot_of[perm] = np.arange(E)
    return stacks, slot_of, experts


def test_each_pair_equals_the_pair_alone_in_any_slot(requires_mlx, q4):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk

    cases = (((640, 2560), mx.float32), ((2560, 640), mx.float32), ((640, 2560), mx.bfloat16))
    for (n, k), dt in cases:
        (w, s, b), slot_of, _ = _pool(mx, q4, n, k, seed=n)
        flat = _routing(seed=k)
        down = n > k  # the down projection reads one input row per pair, in sorted pair order
        x = mx.random.normal((flat.size if down else ROWS, k), key=mx.random.key(n + k))
        x = x.astype(dt)
        cfg = mk.config_for(n, k, dt)
        pos, seg, mc, rows = _segments(mx, flat, slot_of, K_TOP, rows_are_pairs=down)
        y = mk.grouped_qmv(x, w, s, b, *seg, flat.size, mc, *cfg, group_size=32)
        for p in range(flat.size):
            one = [np.array([v]) for v in (slot_of[flat[pos[p]]], 0, 1, rows[p])]
            one = [mx.array(a.astype(np.uint32)) for a in one]
            alone = mk.grouped_qmv(x, w, s, b, *one, 1, 1, *cfg, group_size=32)
            assert mx.array_equal(alone[0], y[p]).item(), (n, k, dt, p)
        # the same experts in other slots: every stack reversed
        _, seg_rev, _, _ = _segments(mx, flat, E - 1 - slot_of, K_TOP, rows_are_pairs=down)
        rev = (w[::-1], s[::-1], b[::-1])
        y_rev = mk.grouped_qmv(x, *rev, *seg_rev, flat.size, mc, *cfg, group_size=32)
        assert mx.array_equal(y_rev, y).item(), (n, k, dt)


def test_device_slot_variant_equals_the_canonical_call(requires_mlx, q4):
    """`Qwen4ExpTextSparseMoeBlock._gather_dev`'s calls (a segment per pair, int32 slots, some in
    the extension set, at `solo_config_for`): gate/up reading the one row, down reading a row per
    pair, give the canonical sorted call's rows (at `config_for`) in pair order; a pair whose slot is
    -1 (not resident) leaves the other rows unchanged."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk

    main = 8  # slots [0, 8) in the main arrays, [8, 12) in the extension arrays
    inds = np.random.default_rng(7).permutation(E)[:K_TOP]
    ar = mx.arange(K_TOP, dtype=mx.uint32)
    for (n, k), rows in (((640, 2560), mx.zeros((K_TOP,), dtype=mx.uint32)), ((2560, 640), ar)):
        (w, s, b), slot_of, _ = _pool(mx, q4, n, k, seed=5 + n, perm_seed=5)
        ext = (w[main:], s[main:], b[main:])
        x = mx.random.normal((1 if n < k else K_TOP, k), key=mx.random.key(6))
        cfg = mk.config_for(n, k, x.dtype)
        solo = mk.solo_config_for(n, k, x.dtype)
        assert solo[1:] == cfg[1:] and solo[0] != cfg[0]
        slots = slot_of[inds]
        assert (slots >= main).any() and (slots < main).any()

        def device(slots, x=x, w=w, s=s, b=b, ext=ext, rows=rows, cfg=solo):
            slots = mx.array(slots.astype(np.int32))
            ones = mx.ones((K_TOP,), dtype=mx.uint32)
            return mk.grouped_qmv_x(
                x,
                w[:main],
                s[:main],
                b[:main],
                ext,
                main,
                slots,
                ar,
                ones,
                rows,
                K_TOP,
                1,
                *cfg,
                group_size=32,
            )

        got = device(slots)
        # canonical: pairs sorted by expert, each reading its own row (k=1 maps pair p to row p)
        pos, seg, mc, _ = _segments(mx, inds.astype(np.int64), slot_of, 1)
        if n < k:
            seg[3] = mx.zeros_like(seg[3])  # gate/up: every pair reads the one row
        canon = mk.grouped_qmv(x, w, s, b, *seg, K_TOP, mc, *cfg, group_size=32)
        inv = np.empty_like(pos)
        inv[pos] = np.arange(pos.size)
        assert mx.array_equal(got, canon[mx.array(inv)]).item(), (n, k)
        missing = slots.copy()
        missing[3] = -1
        keep = mx.array(np.delete(np.arange(K_TOP), 3))
        assert mx.array_equal(device(missing)[keep], got[keep]).item(), (n, k)


def _stock_gather(mx, x, rows, stacks, slots):
    """Qwen4ExpTextSparseMoeBlock's stock branch: `gather_qmm(sorted_indices=False)` on each sorted
    pair's row, the pool's metadata in the activations' dtype (stock's float32 upcast at
    float32)."""
    w, s, b = stacks
    return mx.gather_qmm(
        x[mx.array(rows)][:, None, :],
        w,
        s.astype(x.dtype),
        b.astype(x.dtype),
        rhs_indices=mx.array(slots.astype(np.uint32)),
        transpose=True,
        group_size=32,
        bits=4,
        sorted_indices=False,
    )[:, 0]


def test_as_accurate_as_stock_gather_qmm(requires_mlx, q4, accuracy):
    """gate/up and down, at float32 (served) and bfloat16, each with its config."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk

    for (n, k), dt in (
        ((640, 2560), mx.float32),
        ((2560, 640), mx.float32),
        ((640, 2560), mx.bfloat16),
        ((2560, 640), mx.bfloat16),
    ):
        stacks, slot_of, experts = _pool(mx, q4, n, k, seed=3 * n, perm_seed=1)
        flat = _routing(seed=n)
        down = n > k
        x = mx.random.normal((flat.size if down else ROWS, k), key=mx.random.key(n)).astype(dt)
        pos, seg, mc, rows = _segments(mx, flat, slot_of, K_TOP, rows_are_pairs=down)
        cfg = mk.config_for(n, k, dt)
        got = mk.grouped_qmv(x, *stacks, *seg, flat.size, mc, *cfg, group_size=32)
        stock = _stock_gather(mx, x, rows, stacks, slot_of[flat[pos]])
        exact_w = [accuracy.weights(*e) for e in experts]
        xs = accuracy.np64(x)
        exact = np.stack([xs[rows[p]] @ exact_w[flat[pos[p]]].T for p in range(flat.size)])
        accuracy.no_worse_than_stock(got, stock, exact, what=f"{n}x{k} {dt}")


def test_fused_swiglu_variant_as_accurate_as_the_split_stock_path(requires_mlx, q4, accuracy):
    """Variant "fused" (gate and up read in one pass, SwiGLU in float32) against stock
    `swiglu(gather(gate), gather(up))`."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk
    from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu

    gate, slot_of, gate_e = _pool(mx, q4, 640, 2560, seed=11, perm_seed=2)
    up, _, up_e = _pool(mx, q4, 640, 2560, seed=11 + E, perm_seed=2)  # the same slot layout
    flat = _routing(seed=12)
    x = mx.random.normal((ROWS, 2560), key=mx.random.key(13))
    pos, seg, mc, rows = _segments(mx, flat, slot_of, K_TOP)
    cfg = mk.config_for(640, 2560, x.dtype)
    got = mk.grouped_swiglu(x, gate, up, *seg, flat.size, mc, *cfg, group_size=32)
    slots = slot_of[flat[pos]]
    stock = swiglu(_stock_gather(mx, x, rows, gate, slots), _stock_gather(mx, x, rows, up, slots))
    exact_g = [accuracy.weights(*e) for e in gate_e]
    exact_u = [accuracy.weights(*e) for e in up_e]
    xs = accuracy.np64(x)
    exact = []
    for p in range(flat.size):
        e = flat[pos[p]]
        g, u = xs[rows[p]] @ exact_g[e].T, xs[rows[p]] @ exact_u[e].T
        exact.append(g / (1.0 + np.exp(-g)) * u)
    accuracy.no_worse_than_stock(got, stock, np.stack(exact), what="fused swiglu")
