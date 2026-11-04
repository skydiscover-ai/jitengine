"""Exact top-k routing: the indices `mx.argpartition(g, kth=-k)[..., -k:]` returns, in its order.

MLX's GPU argpartition is a stable merge sort, so the top k come in ascending (value, index) order
and tied values at the boundary resolve to the higher index. The order matters: the down projection
sums experts in slot order.
"""

from __future__ import annotations

import functools

import mlx.core as mx

_SRC = """
  const uint row = threadgroup_position_in_grid.x;
  const uint i = thread_position_in_threadgroup.x;
  const device T* g = gates + row * N;
  threadgroup float vals[N];
  vals[i] = float(g[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float v = vals[i];
  uint rank = 0;
  for (uint j = 0; j < N; j++) {
    const float u = vals[j];
    rank += (u < v) || (u == v && j < i);
  }
  if (rank >= N - K) out[row * K + (rank - (N - K))] = i;
"""


def supported(n: int) -> bool:
    """Whether the kernel covers n gates: one threadgroup sorts them all, so up to 1024."""
    return n <= 1024


def topk(gates: mx.array, k: int, kernel: bool = True) -> mx.array:
    """The k largest gates' expert ids, in `mx.argpartition(gates, kth=-k)[..., -k:]`'s order:
    `topk_stable` where it covers the gates and `kernel` says its load-time check passed, stock
    otherwise. 0 < k <= N is the op's contract: ValueError outside it."""
    n = gates.shape[-1]
    if not 0 < k <= n:
        raise ValueError(f"top-k takes 0 < k <= {n} gates, not k = {k}")
    if kernel and supported(n):
        return topk_stable(gates, k)
    return mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]


@functools.cache
def _topk_stable_kernel():
    return mx.fast.metal_kernel(
        name="moe_topk_stable", input_names=["gates"], output_names=["out"], source=_SRC
    )


def topk_stable(gates: mx.array, k: int) -> mx.array:
    """gates [..., N] (N <= 1024) -> uint32 [..., k] == mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]."""
    n = gates.shape[-1]
    rows = gates.size // n
    return _topk_stable_kernel()(
        inputs=[gates],
        template=[("T", gates.dtype), ("N", n), ("K", k)],
        grid=(n * rows, 1, 1),
        threadgroup=(n, 1, 1),
        output_shapes=[(rows, k)],
        output_dtypes=[mx.uint32],
    )[0].reshape(*gates.shape[:-1], k)


def check_topk(n_experts: int = 256, k: int = 8) -> bool:
    """Whether `topk_stable` returns stock `mx.argpartition`'s expert ids on 16 draws of bf16
    softmax gates over `n_experts`, every other draw full of ties."""
    for t in range(16):
        key = mx.random.key(1000 + t)
        logits = (
            (mx.random.randint(0, 5, (2, 1, n_experts), key=key).astype(mx.float32) * 0.5)
            if t % 2
            else mx.random.normal((2, 1, n_experts), key=key)
        ).astype(mx.bfloat16)
        gates = mx.softmax(logits, axis=-1, precise=True)
        want = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:].astype(mx.uint32)
        if not mx.array_equal(topk_stable(gates, k), want).item():
            return False
    return True
