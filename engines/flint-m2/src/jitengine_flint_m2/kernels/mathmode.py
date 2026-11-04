"""The Metal math mode every Flint kernel is compiled with, pinned.

MLX compiles each mx.fast.metal_kernel with compile_options={"math_mode": "safe" | "relaxed" | "fast"};
its default is "safe" (MLX 0.32.2), and the engine's bit-identity record (verify rows equal to one-token
passes, repeatable runs, renders equal to the measured build's, stock-MLX agreement) was measured under
that default. Every metal_kernel call in this package passes METAL_COMPILE_OPTIONS explicitly, so another
MLX's default (another Mac, a later release) cannot move the arithmetic silently.
tests/unit/kernels/test_mathmode.py scans the package's sources for any call that does not.

An MLX whose metal_kernel has no compile_options, or rejects "safe", cannot honour the pin: require()
refuses it when the engine is constructed, before any weight is allocated, whatever the decode backend
(the load check's compile stage would otherwise read the error as a megakernel failure and fall back to
stock MLX ops). The pin is never dropped silently.
"""

from __future__ import annotations

MATH_MODE = "safe"
# The one constant every mx.fast.metal_kernel call passes as compile_options= (MLX accepts a dict only).
METAL_COMPILE_OPTIONS = {"math_mode": MATH_MODE}


def require() -> None:
    """Raise RuntimeError unless this MLX's metal_kernel accepts the pinned compile options. Constructing
    the kernel object validates the options and compiles nothing (the Metal build happens at the first
    dispatch)."""
    import mlx.core as mx

    if METAL_COMPILE_OPTIONS != {"math_mode": MATH_MODE}:
        raise RuntimeError(f"METAL_COMPILE_OPTIONS was modified: {METAL_COMPILE_OPTIONS!r}")
    try:
        mx.fast.metal_kernel(
            name="flint_mathmode_check",
            input_names=["x"],
            output_names=["y"],
            source="y[0] = x[0];",
            compile_options=METAL_COMPILE_OPTIONS,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"MLX {getattr(mx, '__version__', '?')} cannot compile Metal kernels with "
            f"compile_options={METAL_COMPILE_OPTIONS!r} ({type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:200]}); Flint pins the Metal math mode it was verified under "
            f"and needs an MLX whose mx.fast.metal_kernel takes compile_options (0.32.2 does)"
        ) from exc
