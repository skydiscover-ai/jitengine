"""Fast per-row top-k over a vocabulary-sized row, for the relaxed variant's sampling.

mx.argpartition is a full stable merge sort on this MLX build. One threadgroup per row instead: (1) each of
the NT threads takes the max over its strided slice; the K-th largest of those NT maxima, tau, is a lower
bound on the row's K-th largest value (the K largest slice maxima are K distinct elements >= tau); (2) every
value >= tau is appended to threadgroup memory, up to CAP; (3) a bitonic sort of the candidates (descending
value, ascending id) yields the top K. The result is the exact top K whenever at most CAP values reach tau.
Values above tau lie only in the K - 1 slices whose maximum exceeds tau, so for K <= CAP // ceil(V / NT) + 1
(9 at this vocabulary) only values tied with tau can be dropped; for larger K a row with more than CAP values
at or above tau keeps an arbitrary CAP of them. Used only where exact tie order does not matter (the relaxed
path and the drafts), with K from 1 to MAX_K, the range its callers pass.
"""

from __future__ import annotations

import functools

import mlx.core as mx

NT = 1024
CAP = 2048
MAX_K = 256  # the largest K the callers pass (spec_sample._kk, MTPHead.draft_sampled)

_SRC = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const device T* xr = x + (size_t)row * V;
    threadgroup float smax[NT];
    threadgroup float cv[CAP];
    threadgroup int ci[CAP];
    threadgroup atomic_int cnt;
    // (1) per-thread max over a strided slice
    float m = -INFINITY;
    for (int i = tid; i < V; i += NT) m = max(m, float(xr[i]));
    smax[tid] = m;
    if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // bitonic sort of the NT maxima, descending
    for (uint k = 2; k <= NT; k <<= 1) {
      for (uint j = k >> 1; j > 0; j >>= 1) {
        uint ixj = tid ^ j;
        if (ixj > tid) {
          float a = smax[tid], b = smax[ixj];
          bool desc = (tid & k) == 0;
          if (desc ? (a < b) : (a > b)) { smax[tid] = b; smax[ixj] = a; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    const float tau = smax[K - 1];
    // (2) candidates >= tau
    for (int i = tid; i < V; i += NT) {
      float v = float(xr[i]);
      if (v >= tau) {
        int p = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
        if (p < CAP) { cv[p] = v; ci[p] = i; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n = min(atomic_load_explicit(&cnt, memory_order_relaxed), CAP);
    for (int i = tid; i < CAP; i += NT) if (i >= n) { cv[i] = -INFINITY; ci[i] = 0x7fffffff; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (3) bitonic sort of CAP candidates: descending value, ascending id on ties
    for (uint k = 2; k <= CAP; k <<= 1) {
      for (uint j = k >> 1; j > 0; j >>= 1) {
        for (uint t = tid; t < CAP; t += NT) {
          uint ixj = t ^ j;
          if (ixj > t) {
            float a = cv[t], b = cv[ixj];
            int ia = ci[t], ib = ci[ixj];
            bool a_first = (a > b) || (a == b && ia < ib);
            bool desc = (t & k) == 0;
            if (desc ? !a_first : a_first) { cv[t] = b; cv[ixj] = a; ci[t] = ib; ci[ixj] = ia; }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    for (int i = tid; i < K; i += NT) { vals[row * K + i] = cv[i]; ids[row * K + i] = ci[i]; }
"""


@functools.cache
def _topk_kernel():
    return mx.fast.metal_kernel(
        name="topk_fast_rows", input_names=["x"], output_names=["vals", "ids"], source=_SRC
    )


def topk(x: mx.array, k: int):
    """x [R, V] -> (vals [R, k] float32 descending, ids [R, k] int32)."""
    R, V = x.shape
    if not 1 <= k <= MAX_K:
        raise ValueError(f"topk_fast takes k from 1 to {MAX_K}, got {k}")
    vals, ids = _topk_kernel()(
        inputs=[x],
        template=[("T", x.dtype), ("V", V), ("K", k), ("NT", NT), ("CAP", CAP)],
        grid=(NT, R, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(R, k), (R, k)],
        output_dtypes=[mx.float32, mx.int32],
    )
    return vals, ids
