"""Exact small-M 4-bit matmul for decode batches of 13 to 16 rows.

Stock MLX runs a tiled kernel at 13 to 16 rows: it dequantizes to bf16, accumulates in fp32 with
8x8x8 matrix products in natural k order, and for small N splits K into P partitions that it sums in
order in fp32. This kernel repeats that arithmetic with a layout that reads each weight once, so its
output equals `mx.quantized_matmul` bit for bit. Below 13 rows stock runs a different kernel; its
exact replica was not faster, so those shapes stay on stock. Any shape outside the table below goes
to stock too.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_1_7b_m4pro.kernels._sources import _HEADER, _SRC_EXACT, _SRC_REDUCE

MIN_ROWS = 13
MAX_ROWS = 16
GROUP_ROWS = 16  # output rows one simdgroup owns
VALID_K = (2048, 6144, 8192)


def split_k(rows: int, k: int, n: int) -> int | None:
    """Partitions stock's qmm uses for a (rows, k, n) product, or None where this kernel is not exact.

    Found by probing `mx.quantized_matmul` on a grid of k and n with the pinned MLX; n alone sets it.
    For n of 3072 or 4096 with k below 6144, stock runs its qmv kernel below 15 rows."""
    if k not in VALID_K or not MIN_ROWS <= rows <= MAX_ROWS or n % GROUP_ROWS or n < 1024:
        return None
    if (rows, k, n) in DISABLED:
        return None
    if n == 1024:
        return 16
    if n >= 12288:
        return 1
    if n >= 6144:
        return 2 if n <= 8192 else None
    if n >= 3072:
        if n > 4096 or (k < 6144 and rows < 15):
            return None
        return 4
    return 8 if n == 2048 else None


_plans: dict = {}
_kernels: dict = {}
# (rows, k, n) shapes the startup self-check found different from stock; stock runs them.
DISABLED: set[tuple[int, int, int]] = set()


def disable(rows: int, k: int, n: int) -> None:
    DISABLED.add((rows, k, n))
    _plans.clear()


def plan(rows: int, k: int, segments: tuple[int, ...]):
    """(per-16-row-group partition counts, largest count) for a linear whose output rows are the
    concatenation of `segments` unfused projections, or None where any segment is not covered."""
    key = (rows, k, segments)
    if key not in _plans:
        counts = [split_k(rows, k, n) for n in segments]
        if any(c is None for c in counts):
            _plans[key] = None
        else:
            groups = [c for c, n in zip(counts, segments) for _ in range(n // GROUP_ROWS)]
            _plans[key] = (mx.array(groups, dtype=mx.int32), max(counts))
    return _plans[key]


def _build():
    if not _kernels:
        _kernels["main"] = mx.fast.metal_kernel(
            name="jit_qmm_exact_4b64",
            input_names=["x", "w", "scales", "biases", "gp"],
            output_names=["out"],
            source=_SRC_EXACT,
            header=_HEADER,
        )
        _kernels["reduce"] = mx.fast.metal_kernel(
            name="jit_qmm_exact_reduce",
            input_names=["parts", "gp"],
            output_names=["out"],
            source=_SRC_REDUCE,
        )
    return _kernels["main"], _kernels["reduce"]


def eligible(x, scales, biases, group_size: int, bits: int, mode: str, out_rows: int) -> bool:
    """True when the decode call has the layout this kernel was validated on."""
    return (
        biases is not None
        and bits == 4
        and group_size == 64
        and mode == "affine"
        and x.ndim == 3
        and x.shape[1] == 1
        and MIN_ROWS <= x.shape[0] <= MAX_ROWS
        and x.dtype == mx.bfloat16
        and scales.dtype == x.dtype
        and out_rows % GROUP_ROWS == 0
        and x.shape[2] % 64 == 0
    )


def qmm_exact(x, w, scales, biases, segments: tuple[int, ...] | None = None):
    """x (rows, 1, k) bf16 times the packed weight w (n, k/8). Returns (rows, 1, n), or None where
    the shape is outside the table and stock must run it."""
    rows = x.shape[0]
    k = x.shape[2]
    n = w.shape[0]
    p = plan(rows, k, segments or (n,))
    if p is None:
        return None
    groups, parts_n = p
    main, reduce = _build()
    x2 = x.reshape(rows, k)
    dtype = x.dtype if parts_n == 1 else mx.float32
    parts = main(
        inputs=[x2, w, scales, biases, groups],
        template=[("T", x.dtype), ("OT", dtype), ("M", rows), ("K", k), ("N", n)],
        grid=(32 * (n // GROUP_ROWS), parts_n, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(parts_n, rows, n)],
        output_dtypes=[dtype],
    )[0]
    if parts_n == 1:
        return parts.reshape(rows, 1, n)
    out = reduce(
        inputs=[parts, groups],
        template=[("T", x.dtype), ("M", rows), ("N", n)],
        grid=(rows * n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(rows, 1, n)
