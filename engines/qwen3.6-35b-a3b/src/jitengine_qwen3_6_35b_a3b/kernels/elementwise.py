"""Compiled elementwise ops (`mx.compile`) the model uses; the kernel checks use them as the
stock path the custom kernels must match."""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


# ----------------------------------------------------------------------------- fused elementwise
@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


@partial(mx.compile, shapeless=True)
def gated_silu_f32(h: mx.array, gate: mx.array, x: mx.array) -> mx.array:
    g = nn.silu(gate.astype(mx.float32))
    return (g * x.astype(mx.float32)).astype(h.dtype)
