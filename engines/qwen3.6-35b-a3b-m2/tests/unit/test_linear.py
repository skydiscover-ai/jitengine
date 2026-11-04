"""The quantized wrappers must equal the MLX primitives they wrap, at 3, 4 and 8 bits."""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def test_qlinear_three_bit_matches_mlx(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    w = mx.random.normal((32, 256), key=mx.random.key(11)).astype(mx.bfloat16)
    x = mx.random.normal((1, 256), key=mx.random.key(12)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=3)
    layer = QLinear(packed, scales, biases, group_size=64)
    actual = layer(x)
    expected = mx.quantized_matmul(x, packed, scales, biases, transpose=True, group_size=64, bits=3)
    mx.eval(actual, expected)
    assert layer.bits == 3
    assert bool(mx.array_equal(actual, expected).item())


def test_qlinear_bits_inferred_for_each_width(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    for bits in (3, 4, 8):
        w = mx.random.normal((64, 512), key=mx.random.key(bits)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=bits)
        assert QLinear(q, s, b, 64).bits == bits


def test_expert_gather_three_bit_duplicated_and_reordered_ids(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    E, N, K = 6, 64, 256
    w = mx.random.normal((E, N, K), key=mx.random.key(3)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=3)
    lin = QLinear(q, s, b, 64)
    assert lin.bits == 3
    x = mx.random.normal((5, 1, 1, K), key=mx.random.key(4)).astype(mx.bfloat16)
    ids = mx.array([[4], [1], [4], [0], [5]], dtype=mx.uint32)  # duplicated, unsorted
    got = lin.gather(x, ids)
    for r in range(5):
        e = int(ids[r, 0].item())
        ref = mx.quantized_matmul(x[r, 0], q[e], s[e], b[e], transpose=True, group_size=64, bits=3)
        mx.eval(got, ref)
        assert bool(mx.array_equal(got[r, 0], ref).item()), f"row {r} expert {e}"


def test_eight_bit_router_on_three_bit_body(requires_mlx_any):
    """The checkpoint's routers are 8-bit next to a 3-bit body; each wrapper infers its own width."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    x = mx.random.normal((2, 256), key=mx.random.key(7)).astype(mx.bfloat16)
    wr = mx.random.normal((16, 256), key=mx.random.key(8)).astype(mx.bfloat16)
    wb = mx.random.normal((64, 256), key=mx.random.key(9)).astype(mx.bfloat16)
    router = QLinear(*mx.quantize(wr, group_size=64, bits=8), 64)
    body = QLinear(*mx.quantize(wb, group_size=64, bits=3), 64)
    assert (router.bits, body.bits) == (8, 3)
    mx.eval(router(x), body(x))


def test_expert_views_live_as_long_as_their_stack(requires_mlx_any):
    """`QLinear.expert_views` builds the one-expert views once and keeps them on the stack, so a
    dropped stack frees its buffers, as a second engine in the process or a resized expert pool
    needs."""
    import gc

    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

    def stack():
        w = mx.random.normal((64, 512, 1024), key=mx.random.key(0)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=3)
        mx.eval(q, s, b)
        return QLinear(q, s, b, 64)

    gc.collect()
    base = mx.get_active_memory()
    lin = stack()
    views = lin.expert_views()
    assert [v.shape[0] for v in views] == [1, 1, 1] and lin.expert_views() is views
    held = mx.get_active_memory() - base
    del lin, views
    gc.collect()
    assert held > 10 << 20 and mx.get_active_memory() - base < held // 10


def test_embedding_three_bit_matches_dequantize(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m2.linear import QEmbedding

    w = mx.random.normal((100, 128), key=mx.random.key(5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=3)
    emb = QEmbedding(q, s, b, 64)
    ids = mx.array([[3, 99, 3, 0]])
    got = emb.lookup(ids)
    full = mx.dequantize(q, s, b, group_size=64, bits=3)
    mx.eval(got, full)
    assert bool(mx.array_equal(got[0], full[mx.array([3, 99, 3, 0])]).item())
