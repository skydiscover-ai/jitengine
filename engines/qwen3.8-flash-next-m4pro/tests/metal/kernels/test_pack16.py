"""pack16, Flash-Next's compact expert slot: the encode kernel and the p16 grouped kernels.

kernels/pack16.py claims the GPU encoder stores each group as the host reference's word (or an
escape into the global table) and copies the weights verbatim, that the Metal decode returns the
original bf16 bits, and that it counts every group it cannot store exactly. grouped_expert_qmv.py
claims its p16 forms differ from the bf16 kernels only in the per-group load, so their outputs are
the bf16 kernels' outputs bit for bit, whole vectors, in every call shape the engine makes. Records
are Flash-Next-shaped (gate/up 640 x 2560, down 2560 x 640) and quantized as the checkpoint is.
"""

import numpy as np
import pytest

# every test here runs a custom Metal kernel
pytestmark = pytest.mark.usefixtures("requires_mlx")

PROJ_SHAPES = ((640, 2560), (640, 2560), (2560, 640))  # gate, up, down: (N, K)
S = 12  # records in the test pool


def _bits(mx, a):
    return np.asarray(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.asarray(a)


def _same(mx, a, b) -> bool:
    a, b = _bits(mx, a), _bits(mx, b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _fields():
    """The pack header's nine-field table for a Flash-Next record."""
    out, off = [], 0
    for p, (n, k) in zip(("gate_proj", "up_proj", "down_proj"), PROJ_SHAPES, strict=True):
        for t, dt, shape, nb in (
            ("weight", "U32", [n, k // 8], n * k // 2),
            ("scales", "BF16", [n, k // 32], n * k // 16),
            ("biases", "BF16", [n, k // 32], n * k // 16),
        ):
            out.append(
                {"name": f"{p}.{t}", "dtype": dt, "shape": shape, "bytes": nb, "offset": off}
            )
            off += nb
    return out


@pytest.fixture(scope="module")
def records():
    """S records quantized as the checkpoint is: per projection (w u32, s u16, b u16) stacks, and
    the records flattened in the pack's layout, u32 [S, record_bytes / 4]."""
    mx = pytest.importorskip("mlx.core")

    projs = []
    for i, (n, k) in enumerate(PROJ_SHAPES):
        w = mx.random.normal((S, n, k), key=mx.random.key(40 + i)) * k**-0.5
        q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=32, bits=4)
        s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        mx.eval(q, s, b)
        projs.append((np.asarray(q), np.asarray(s.view(mx.uint16)), np.asarray(b.view(mx.uint16))))
    flat = np.concatenate(
        [
            np.concatenate(
                [
                    w.reshape(S, -1),
                    s.reshape(S, -1).view(np.uint32),
                    b.reshape(S, -1).view(np.uint32),
                ],
                1,
            )
            for w, s, b in projs
        ],
        axis=1,
    )
    return projs, np.ascontiguousarray(flat)


def _encode_gpu(mx, geom, flat, table, slots=None, escape_out_of_table=False):
    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    n = flat.shape[0]
    slots = np.arange(n) if slots is None else slots
    stg = mx.array(flat)
    pool = [
        mx.zeros(
            (n, *(geom.wshape[p] if t == "w" else geom.gshape[p])),
            mx.uint32 if t == "w" else mx.uint16,
        )
        for p in range(3)
        for t in ("w", "s")
    ]
    dep = mx.zeros((1,), mx.uint32)
    mx.eval(stg, dep, *pool)
    err = p16.encode_records(
        geom,
        stg,
        pool,
        None,
        n,
        p16.exc_array(table),
        len(table),
        np.stack([np.arange(n), slots], 1),
        dep,
    )
    mx.eval(err, *pool)
    return int(err.item()), [np.asarray(a) for a in pool]


def test_the_encoder_stores_the_reference_words_and_the_weights_verbatim(records):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    projs, flat = records
    geom = p16.Geometry(_fields(), 32, 4)
    assert geom.slot_bytes == 2_764_800 and geom.record_bytes == 3_072_000
    table: list = []
    ref = [p16.encode_np(s, b, table)[0] for _w, s, b in projs]
    slots = np.random.default_rng(3).permutation(
        S
    )  # records land in slots that are not their index
    err, pool = _encode_gpu(mx, geom, flat, table, slots)
    assert err == 0
    for p, (w, s, b) in enumerate(projs):
        assert np.array_equal(pool[2 * p][slots], w), p
        assert np.array_equal(pool[2 * p + 1][slots], ref[p]), p
        ds, db = p16.decode_np(pool[2 * p + 1][slots], table)
        assert np.array_equal(ds, s) and np.array_equal(db, b), p


def test_an_unrepresentable_pair_escapes_through_the_table_or_is_counted(records):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    _projs, flat = records
    geom = p16.Geometry(_fields(), 32, 4)
    flat = flat.copy()
    # two groups no code can hold: a scale exponent below the code's range, and a bias off by 5 ulps
    u16 = flat.view(np.uint16)
    _ow, os_, ob = geom.offsets[1]
    u16[0, 2 * os_ + 7] = np.uint16((112 << 7) | 0x15)
    u16[3, 2 * ob + 11] = np.uint16(int(u16[3, 2 * ob + 11]) + 5)
    err, _ = _encode_gpu(mx, geom, flat, [])
    assert err == 2
    table: list = []
    for r in range(S):
        for p in range(3):
            _ow, so, bo = geom.offsets[p]
            n = geom.groups[p]
            p16.encode_np(u16[r, 2 * so : 2 * so + n], u16[r, 2 * bo : 2 * bo + n], table)
    assert len(table) == 2
    err, pool = _encode_gpu(mx, geom, flat, table)
    assert err == 0
    for r in range(S):
        for p in range(3):
            _ow, so, bo = geom.offsets[p]
            n = geom.groups[p]
            ds, db = p16.decode_np(pool[2 * p + 1][r].reshape(-1), table)
            assert np.array_equal(ds, u16[r, 2 * so : 2 * so + n]), (r, p)
            assert np.array_equal(db, u16[r, 2 * bo : 2 * bo + n]), (r, p)


def test_the_metal_decode_returns_the_original_bits(records):
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    projs, _ = records
    table: list = []
    for _w, s, b in projs:
        words, _ = p16.encode_np(s, b, table)
        words = words.ravel().copy()
        # forced escapes: every 997th group read through the table instead
        idx = np.arange(0, words.size, 997)
        for i in idx.tolist():
            key = int(s.ravel()[i]) | (int(b.ravel()[i]) << 16)
            if key not in table:
                table.append(key)
            words[i] = (table.index(key) << 5) | p16.ESC
        got = np.asarray(p16.decode_words(mx.array(words), p16.exc_array(table)))
        assert np.array_equal(got[:, 0], s.ravel()) and np.array_equal(got[:, 1], b.ravel())


def _p16_pool(mx, records, forced: bool):
    """(bf16 stacks, p16 stacks, exc) per projection over the S records."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    projs, _ = records
    table: list = []
    rng = np.random.default_rng(11)
    out = []
    for w, s, b in projs:
        words, _ = p16.encode_np(s, b, table)
        if forced:
            words = words.ravel().copy()
            for i in np.flatnonzero(rng.random(words.size) < 2e-4).tolist():
                key = int(s.ravel()[i]) | (int(b.ravel()[i]) << 16)
                if key not in table:
                    table.append(key)
                words[i] = (table.index(key) << 5) | p16.ESC
            words = words.reshape(s.shape)
        bf = (mx.array(w), mx.array(s).view(mx.bfloat16), mx.array(b).view(mx.bfloat16))
        out.append((bf, (mx.array(w), mx.array(words))))
    if forced:
        assert len(table) > 10
    exc = p16.exc_array(table)
    return out, exc


@pytest.mark.parametrize("forced", [False, True], ids=["clean", "forced_escapes"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_p16_grouped_calls_equal_the_bf16_calls_bit_for_bit(records, forced, dtype):
    """`_expert_mlp_sorted`'s calls: a miss correction (1 row, 3 pairs), batched decode (8 x 10)
    and a wave (64 x 10, segments of several tiles), gate/up then down, whole outputs."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk
    from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu

    dt = getattr(mx, dtype)
    pool, exc = _p16_pool(mx, records, forced)
    dep = mx.zeros((1,), mx.uint32)
    rng = np.random.default_rng(5 if forced else 6)
    for nrows, kk in ((1, 3), (8, 10), (64, 10)):
        pe = np.stack([rng.choice(S, kk, replace=False) for _ in range(nrows)]).reshape(-1)
        pos = np.argsort(pe, kind="stable")
        uniq, start, cnt = np.unique(pe[pos], return_index=True, return_counts=True)
        seg = [mx.array(a.astype(np.uint32)) for a in (uniq, start, cnt, pos // kk)]
        P, mc = pos.size, int(cnt.max())
        x = mx.random.normal((nrows, 2560), key=mx.random.key(int(rng.integers(1 << 30)))).astype(
            dt
        )
        ys = []
        for p in (0, 1):
            (w, s, b), (w2, words) = pool[p]
            cfg = mk.config_for(w.shape[1], s.shape[2] * 32, dt)
            y0 = mk.grouped_qmv(x, w, s, b, *seg, P, mc, *cfg, group_size=32)
            y1 = mk.grouped_qmv_p16(x, w2, words, exc, dep, *seg, P, mc, *cfg, group_size=32)
            assert _same(mx, y0, y1), (nrows, kk, p)
            ys.append((y0, y1))
        (w, s, b), (w2, words) = pool[2]
        cfg = mk.config_for(w.shape[1], s.shape[2] * 32, dt)
        ident = mx.arange(P, dtype=mx.uint32)
        d0 = mk.grouped_qmv(
            swiglu(*[y[0] for y in ys]), w, s, b, *seg[:3], ident, P, mc, *cfg, group_size=32
        )
        d1 = mk.grouped_qmv_p16(
            swiglu(*[y[1] for y in ys]),
            w2,
            words,
            exc,
            dep,
            *seg[:3],
            ident,
            P,
            mc,
            *cfg,
            group_size=32,
        )
        assert _same(mx, d0, d1), (nrows, kk, "down")


@pytest.mark.parametrize("forced", [False, True], ids=["clean", "forced_escapes"])
def test_p16_device_slot_calls_equal_the_bf16_calls_bit_for_bit(records, forced):
    """`_gather_dev`'s calls (10 pairs of one row, int32 slots, the upper half of the pool read as
    the extension set, one slot not resident, at `solo_config_for`): every resident pair's row,
    whole."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk
    from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu

    dt = mx.float32
    pool, exc = _p16_pool(mx, records, forced)
    dep = mx.zeros((1,), mx.uint32)
    main = S // 2
    k = 10
    slots = np.random.default_rng(8).permutation(S)[:k].astype(np.int32)
    slots[4] = -1
    keep = np.flatnonzero(slots >= 0)
    sl = mx.array(slots)
    ar, ones, zeros = (
        mx.arange(k, dtype=mx.uint32),
        mx.ones((k,), mx.uint32),
        mx.zeros((k,), mx.uint32),
    )
    x = mx.random.normal((1, 2560), key=mx.random.key(21)).astype(dt)
    outs = []
    for p in (0, 1):
        (w, s, b), (w2, words) = pool[p]
        cfg = mk.solo_config_for(w.shape[1], s.shape[2] * 32, dt)
        y0 = mk.grouped_qmv_x(
            x,
            w[:main],
            s[:main],
            b[:main],
            (w[main:], s[main:], b[main:]),
            main,
            sl,
            ar,
            ones,
            zeros,
            k,
            1,
            *cfg,
            group_size=32,
        )
        y1 = mk.grouped_qmv_x_p16(
            x,
            w2[:main],
            words[:main],
            exc,
            dep,
            (w2[main:], words[main:]),
            main,
            sl,
            ar,
            ones,
            zeros,
            k,
            1,
            *cfg,
            group_size=32,
        )
        assert _same(mx, y0[mx.array(keep)], y1[mx.array(keep)]), p
        outs.append((y0, y1))
    (w, s, b), (w2, words) = pool[2]
    cfg = mk.solo_config_for(w.shape[1], s.shape[2] * 32, dt)
    d0 = mk.grouped_qmv_x(
        swiglu(outs[0][0], outs[1][0]),
        w[:main],
        s[:main],
        b[:main],
        (w[main:], s[main:], b[main:]),
        main,
        sl,
        ar,
        ones,
        ar,
        k,
        1,
        *cfg,
        group_size=32,
    )
    d1 = mk.grouped_qmv_x_p16(
        swiglu(outs[0][1], outs[1][1]),
        w2[:main],
        words[:main],
        exc,
        dep,
        (w2[main:], words[main:]),
        main,
        sl,
        ar,
        ones,
        ar,
        k,
        1,
        *cfg,
        group_size=32,
    )
    assert _same(mx, d0[mx.array(keep)], d1[mx.array(keep)])


@pytest.mark.parametrize("forced", [False, True], ids=["clean", "forced_escapes"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_solo_gather_at_solo_vecs_equals_the_configs_launch(records, forced, dtype):
    """`_gather_dev`'s layer (gate, up, SwiGLU, down; 10 pairs of one row, one slot not resident) at
    `solo_config_for` (VECS 1) against the same calls at `config_for`'s VECS and at VECS 4, on the
    bf16 pool and on compact slots, with the upper half of the pool as the extension set and with
    none: every resident pair's row, whole. VECS never enters the k-order. The negative control, a
    different KL (another k-order), must be seen by the same compare."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk
    from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu

    dt = getattr(mx, dtype)
    pool, exc = _p16_pool(mx, records, forced)
    dep = mx.zeros((1,), mx.uint32)
    k = 10
    slots = np.random.default_rng(9).permutation(S)[:k].astype(np.int32)
    slots[6] = -1
    keep = mx.array(np.flatnonzero(slots >= 0))
    sl = mx.array(slots)
    ar, ones, zeros = (
        mx.arange(k, dtype=mx.uint32),
        mx.ones((k,), mx.uint32),
        mx.zeros((k,), mx.uint32),
    )
    x = mx.random.normal((1, 2560), key=mx.random.key(23)).astype(dt)

    def layer(codec, main, vecs=None, kl=None):
        def call(p, xin, rows):
            (w, s, b), (w2, words) = pool[p]
            cfg = list(mk.solo_config_for(w.shape[1], s.shape[2] * 32, dt))
            if vecs is not None:
                cfg[0] = vecs
            if kl is not None:
                cfg[1] = kl
            arrs = (w, s, b) if codec == "bf16" else (w2, words)
            head = tuple(a[:main] for a in arrs)
            ext = (
                tuple(a[main:] for a in arrs) if main < S else None
            )  # None: every slot in the main set
            if codec == "bf16":
                return mk.grouped_qmv_x(
                    xin, *head, ext, main, sl, ar, ones, rows, k, 1, *cfg, group_size=32
                )
            return mk.grouped_qmv_x_p16(
                xin, *head, exc, dep, ext, main, sl, ar, ones, rows, k, 1, *cfg, group_size=32
            )

        h = swiglu(call(0, x, zeros), call(1, x, zeros))
        return h, call(2, h, ar)

    shipped = mk.config_for(640, 2560, dt)[0]
    assert mk.solo_config_for(640, 2560, dt) == (mk.SOLO_VECS, *mk.config_for(640, 2560, dt)[1:])
    assert mk.SOLO_VECS == 1 and shipped != 1
    for codec in ("bf16", "p16"):
        for main in (S // 2, S):
            h1, y1 = layer(codec, main)
            for v in sorted({shipped, 4}):
                hv, yv = layer(codec, main, vecs=v)
                assert _same(mx, h1[keep], hv[keep]), (codec, main, v, "gate/up")
                assert _same(mx, y1[keep], yv[keep]), (codec, main, v, "down")
            _, ykl = layer(codec, main, kl=16)
            assert not _same(mx, y1[keep], ykl[keep]), (codec, main, "KL 16 not caught")


def test_a_flipped_code_changes_the_output(records):
    """The negative control: the comparison above would see one wrong word."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk

    pool, exc = _p16_pool(mx, records, False)
    (w, s, b), (w2, words) = pool[0]
    bad = np.asarray(words).copy()
    c = int(bad[2, 0, 0]) & 31
    bad[2, 0, 0] = (int(bad[2, 0, 0]) & ~31) | (c + 1 if c < 26 else c - 1)
    seg = [mx.array(np.array([v], np.uint32)) for v in (2, 0, 1, 0)]
    x = mx.random.normal((1, 2560), key=mx.random.key(3))
    cfg = mk.config_for(640, 2560, mx.float32)
    y0 = mk.grouped_qmv(x, w, s, b, *seg, 1, 1, *cfg, group_size=32)
    y1 = mk.grouped_qmv_p16(
        x, w2, mx.array(bad), exc, mx.zeros((1,), mx.uint32), *seg, 1, 1, *cfg, group_size=32
    )
    assert not _same(mx, y0, y1)


def test_a_read_after_an_encode_in_one_graph_sees_the_encoded_slot(records):
    """The encode writes the pool behind MLX's back; a gather that takes its error count as `dep`
    is ordered after it, so a slot encoded and read in one evaluation gives the bf16 result."""
    import mlx.core as mx

    from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as mk
    from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as p16

    projs, flat = records
    geom = p16.Geometry(_fields(), 32, 4)
    table: list = []
    for _w, s, b in projs:
        p16.encode_np(s, b, table)
    exc = p16.exc_array(table)
    stg = mx.array(flat)
    pool = [
        mx.zeros(
            (S, *(geom.wshape[p] if t == "w" else geom.gshape[p])),
            mx.uint32 if t == "w" else mx.uint16,
        )
        for p in range(3)
        for t in ("w", "s")
    ]
    dep0 = mx.zeros((1,), mx.uint32)
    mx.eval(stg, dep0, *pool)
    w, s, b = (mx.array(a) for a in projs[0])
    s, b = s.view(mx.bfloat16), b.view(mx.bfloat16)
    cfg = mk.config_for(640, 2560, mx.float32)
    for rnd in range(4):
        slot = rnd % S
        err = p16.encode_records(
            geom, stg, pool, None, S, exc, len(table), np.array([[slot, slot]]), dep0
        )
        seg = [mx.array(np.array([v], np.uint32)) for v in (slot, 0, 1, 0)]
        x = mx.random.normal((1, 2560), key=mx.random.key(70 + rnd))
        got = mk.grouped_qmv_p16(
            x, pool[0], pool[1], exc, mx.maximum(dep0, err), *seg, 1, 1, *cfg, group_size=32
        )
        want = mk.grouped_qmv(x, w, s, b, *seg, 1, 1, *cfg, group_size=32)
        mx.eval(got, want)
        assert _same(mx, got, want), rnd
