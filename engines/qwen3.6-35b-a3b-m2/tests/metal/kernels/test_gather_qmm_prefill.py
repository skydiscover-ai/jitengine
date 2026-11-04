"""The segmented prefill gather_qmm: tall and default tiles give stock gather_qmm's bits."""


def _qz(mx, shape, bits, seed):
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def test_seg_gather_tall_tiles_bitwise(requires_mlx):
    """The tall prefill tile matches the default tile and stock gather_qmm bit for bit."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.kernels import gather_qmm_prefill as ep
    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    E, D, H, k, T = 8, 2048, 512, 2, 700
    w, s, b = _qz(mx, (E, 2 * H, D), 3, 31)
    stack = QLinear(w, s, b, 64)
    x = mx.random.normal((T, D), key=mx.random.key(32)).astype(mx.bfloat16)
    inds = mx.argpartition(-mx.random.normal((T, E), key=mx.random.key(33)), kth=k - 1, axis=-1)[
        :, :k
    ]
    flat = inds.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat)
    xs = mx.repeat(x, k, axis=0)[order]
    idx = flat[order]
    assert xs.shape[0] >= ep.TALL_MIN_ROWS_PER_EXPERT * E  # the rule would pick the tall tile
    short = ep.seg_gather_qmm(xs, stack, idx, cfg=ep.CFG)
    tall = ep.seg_gather_qmm(xs, stack, idx, cfg=ep.CFG_TALL)
    auto = ep.seg_gather_qmm(xs, stack, idx)
    stock = mx.gather_qmm(
        xs[:, None],
        w,
        s,
        b,
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=3,
        sorted_indices=True,
    )[:, 0]
    assert mx.array_equal(short, tall).item() and mx.array_equal(auto, tall).item()
    assert mx.array_equal(stock, tall).item()
