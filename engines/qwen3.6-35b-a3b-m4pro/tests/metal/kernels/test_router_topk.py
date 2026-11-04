"""The router top-k and the fused router: the stock chain's results."""

from __future__ import annotations


def test_topk_stable_equals_argpartition_tail(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import topk_stable

    for t in range(200):
        key = mx.random.key(t)
        if t % 3 == 0:
            logits = (
                mx.random.randint(0, 5, (2, 1, 256), key=key).astype(mx.float32) * 0.5
            ).astype(mx.bfloat16)
        else:
            logits = (mx.random.normal((2, 1, 256), key=key) * (1 + t % 4)).astype(mx.bfloat16)
        gates = mx.softmax(logits, axis=-1, precise=True)
        want = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:].astype(mx.uint32)
        assert mx.array_equal(topk_stable(gates, 8), want).item(), t


def test_fused_router_bitwise_vs_stock_chain(requires_mlx):
    """routing.router_topk == softmax(precise) -> topk_stable -> gather -> (/ sum), including ties."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import check_router

    assert check_router(256, 8)
    assert check_router(128, 4)
