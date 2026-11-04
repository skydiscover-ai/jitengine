"""Small device arrays the kernels take as inputs, built and evaluated once per value."""

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
def placeholder(dtype) -> mx.array:
    """An input the launch does not read. Eight elements: a one-element input lands in constant
    address space."""
    a = mx.zeros((8,), dtype=dtype)
    mx.eval(a)
    return a
