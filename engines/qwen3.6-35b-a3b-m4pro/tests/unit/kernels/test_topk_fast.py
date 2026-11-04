"""topk_fast refuses a k outside the range its callers pass, before any launch."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("k", [0, 257, 1024])
def test_k_outside_the_supported_range_raises(requires_mlx_any, k):
    import mlx.core as mx

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.topk_fast import topk

    with pytest.raises(ValueError, match="takes k from 1 to 256"):
        topk(mx.zeros((1, 4096)), k)
