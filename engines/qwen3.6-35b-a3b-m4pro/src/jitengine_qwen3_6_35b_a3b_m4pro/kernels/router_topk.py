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


# The whole router in one launch: softmax(precise), top-k, gather, sum, divide (the MoE block's five
# ops). The softmax is MLX's softmax_single_row on the first N/4 threads, the rank top-k above runs
# on all N, and the k-sum accumulates sequentially in bf16 as scores.sum(-1) does.
_FSRC = """
  // threadgroup: N threads (one per expert); softmax runs on the first N / 4 threads exactly as MLX
  constexpr int N_READS = 4;
  constexpr int SIMD_SIZE = 32;
  typedef float AccT;
  const uint row = threadgroup_position_in_grid.x;
  const int lid = thread_position_in_threadgroup.x;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  constexpr int SM_THREADS = ((N + N_READS - 1) / N_READS + SIMD_SIZE - 1) / SIMD_SIZE * SIMD_SIZE;
  const bool sm = lid < SM_THREADS;
  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  threadgroup T gates[N];
  AccT ld[N_READS];
  const device T* in = logits + row * N + lid * N_READS;
  if (sm) {
    if (lid * N_READS + N_READS <= N) {
      for (int i = 0; i < N_READS; i++) ld[i] = AccT(in[i]);
    } else {
      for (int i = 0; i < N_READS; i++) ld[i] = ((lid * N_READS + i) < N) ? AccT(in[i]) : Limits<AccT>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<AccT>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  AccT maxval = Limits<AccT>::finite_min;
  if (sm) {
    for (int i = 0; i < N_READS; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (simd_lane_id == 0) local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) local_max[0] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  AccT normalizer = 0;
  if (sm) {
    for (int i = 0; i < N_READS; i++) {
      AccT exp_x = fast::exp(ld[i] - maxval);
      ld[i] = exp_x;
      normalizer += exp_x;
    }
    normalizer = simd_sum(normalizer);
    if (simd_lane_id == 0) local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) local_normalizer[0] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  if (sm) {
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < N) gates[lid * N_READS + i] = T(ld[i] * normalizer);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // exact top-k (routing.py): rank in ascending (value, index) order
  threadgroup uint slot_idx[K];
  {
    const float v = float(gates[lid]);
    uint rank = 0;
    for (int j = 0; j < N; j++) {
      const float u = float(gates[j]);
      rank += (u < v) || (u == v && j < lid);
    }
    if (rank >= N - K) slot_idx[rank - (N - K)] = lid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    T sc[K];
    for (int j = 0; j < K; j++) { const uint e = slot_idx[j]; inds[row * K + j] = e; sc[j] = gates[e]; }
    if (NORM) {
      T tot = T(0);                // sequential in T: bitwise what scores.sum(-1) gives for k values
      for (int j = 0; j < K; j++) tot = tot + sc[j];
      for (int j = 0; j < K; j++) scores[row * K + j] = sc[j] / tot;
    } else {
      for (int j = 0; j < K; j++) scores[row * K + j] = sc[j];
    }
  }
"""


@functools.cache
def _router_kernel(key: bool):
    return mx.fast.metal_kernel(
        name=f"moe_router_fused_{int(key)}",
        input_names=["logits"],
        output_names=["inds", "scores"],
        source=_FSRC,
    )


def router_supported(logits: mx.array, k: int) -> bool:
    """Whether the fused router covers these logits: bf16, N a multiple of 32 up to 1024."""
    n = logits.shape[-1]
    return logits.dtype == mx.bfloat16 and 0 < k <= n <= 1024 and n % 32 == 0


def router_topk(logits: mx.array, k: int, norm: bool):
    """logits [..., N] bf16 -> (inds uint32 [..., k], scores [..., k]): bitwise the stock chain
    softmax(precise) / topk_stable / take_along_axis / (/ sum) of the MoE block."""
    key = bool(norm)
    n = logits.shape[-1]
    rows = logits.size // n
    inds, scores = _router_kernel(key)(
        inputs=[logits],
        template=[("T", logits.dtype), ("N", n), ("K", k), ("NORM", key)],
        grid=(n * rows, 1, 1),
        threadgroup=(n, 1, 1),
        output_shapes=[(rows, k), (rows, k)],
        output_dtypes=[mx.uint32, logits.dtype],
    )
    shp = (*logits.shape[:-1], k)
    return inds.reshape(shp), scores.reshape(shp)


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


def check_router(n_experts: int = 256, k: int = 8) -> bool:
    """router_topk vs the stock five-op router (softmax precise, topk_stable, gather, sum,
    divide) on tie-heavy, wide and near-flat logits, normalised and not."""
    for t in range(24):
        key = mx.random.key(2000 + t)
        B = 1 + t % 8
        kind = t % 4
        lg = mx.random.normal((B, 1, n_experts), key=key) * (2.0, 8.0, 1.0, 0.05)[kind]
        if kind == 2:
            lg = mx.random.randint(-4, 4, (B, 1, n_experts), key=key).astype(mx.float32)
        lg = lg.astype(mx.bfloat16)
        gates = mx.softmax(lg, axis=-1, precise=True)
        inds = topk_stable(gates, k)
        sc = mx.take_along_axis(gates, inds, axis=-1)
        for norm in (True, False):
            want = sc / sc.sum(axis=-1, keepdims=True) if norm else sc
            gi, gs = router_topk(lg, k, norm)
            if not (mx.array_equal(gi, inds).item() and mx.array_equal(gs, want).item()):
                return False
    return True
