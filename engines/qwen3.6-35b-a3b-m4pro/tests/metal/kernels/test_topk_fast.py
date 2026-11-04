"""topk_fast gives a full sort's top-k values on vocabulary-sized rows, over the range of k it takes."""

from __future__ import annotations


def test_top_values_equal_a_full_sort(requires_mlx):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.topk_fast import topk

    for dtype in (mx.bfloat16, mx.float32):
        for k in (1, 9, 20, 64, 256):
            x = mx.random.normal((3, 248320), key=mx.random.key(k)).astype(dtype)
            values, _ = topk(x, k)
            want = mx.sort(x.astype(mx.float32), axis=-1)[:, ::-1][:, :k]
            assert mx.array_equal(values, want).item(), (dtype, k)
