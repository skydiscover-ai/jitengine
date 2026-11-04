"""The router top-k's entry point: its contract, its coverage, and stock where the kernel does not
run."""

import pytest


def test_top_k_outside_the_contract_raises(requires_mlx_any):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import topk

    gates = mx.zeros((1, 1, 8))
    for k in (0, 9):
        with pytest.raises(ValueError):
            topk(gates, k)


def test_stock_runs_where_the_kernel_does_not(requires_mlx_any):
    """Past 1024 gates, or with the kernel off, top-k is stock argpartition (no Metal needed)."""
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import supported, topk

    assert supported(1024) and not supported(1025)
    for n, kernel in ((2048, True), (256, False)):
        gates = mx.softmax(mx.random.normal((2, 1, n), key=mx.random.key(n)), axis=-1)
        want = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
        assert mx.array_equal(topk(gates, 8, kernel=kernel), want).item()
