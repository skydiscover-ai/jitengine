"""The state of a linear-attention (Gated DeltaNet) layer: per request, the conv window and the
SSM state, and the fresh copy a snapshot takes of each."""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig


def _materialize(x: mx.array) -> mx.array:
    """Copy x into a fresh buffer; a DeltaNet conv array may itself be a buffer-sharing slice."""
    return x + mx.array(0, dtype=x.dtype)


class LinearStateSlot:
    """conv_state [1, K-1, conv_dim] (last K-1 inputs of the depthwise conv), ssm [1, Hv, Dv, Dk] fp32."""

    __slots__ = ("conv", "ssm")

    def __init__(self, cfg: Qwen3_5MoeConfig, dtype=mx.bfloat16):
        self.conv = mx.zeros((1, cfg.linear_conv_kernel_dim - 1, cfg.conv_dim), dtype=dtype)
        self.ssm = mx.zeros(
            (1, cfg.linear_num_value_heads, cfg.linear_value_head_dim, cfg.linear_key_head_dim),
            dtype=mx.float32,
        )
