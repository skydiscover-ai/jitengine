"""Small device arrays the kernels take as inputs, built and evaluated once per value: sizes fixed
per weight or per buffer and per-model scalars, so a launch builds only what changes per call."""

from __future__ import annotations

import functools

import mlx.core as mx


@functools.cache
def f32(*values: float) -> mx.array:
    """`values` as an fp32 array: metal_kernel templates take ints, bools and dtypes only."""
    a = mx.array([float(v) for v in values], dtype=mx.float32)
    mx.eval(a)
    return a


@functools.cache
def i32(*values: int) -> mx.array:
    """`values` as an int32 array."""
    a = mx.array([int(v) for v in values], dtype=mx.int32)
    mx.eval(a)
    return a
