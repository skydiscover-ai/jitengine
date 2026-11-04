"""RMS norm fused with the GEMV activation pre-scale, bit-identical to mx.fast.rms_norm.

One launch computes the norm with MLX's arithmetic and, from its output, the pre-scaled activations
and group sums that `qmv_prescaled` and the expert GEMVs read, so the solo decode path skips a launch.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels import constants
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import PRESCALE_STORES, PRESCALE_SUM

# ---- rms_norm fused with the prologue: MLX's rms_single_row (rms_norm.metal v0.32.2, non-looped
# path) verbatim, then prescale() of the normalised row; both outputs match the stock ops bitwise
RMS_SRC = (
    """
  constexpr int SIMD_SIZE = 32;
  constexpr int N_READS = 4;
  const uint gid = threadgroup_position_in_grid.x;
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint axis_size = AXIS;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  threadgroup T ybuf[AXIS];
  float acc = 0;
  float thread_x[N_READS];
  const device T* xr = x + gid * size_t(axis_size) + lid * N_READS;
  const device T* wr = w + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] = xr[i];
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* yo = y + gid * size_t(axis_size) + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    T v = wr[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    yo[i] = v;
    ybuf[lid * N_READS + i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // dense_v4 prologue over this row: thread lid < axis/16 handles 16-slice lid
  if (lid < axis_size / 16) {
    const threadgroup T* xv = ybuf + lid * 16;
"""
    + PRESCALE_SUM
    + """      device float* o = xs + gid * size_t(axis_size) + lid * 16 + i;
"""
    + PRESCALE_STORES
    + """    xsum[gid * (axis_size / 16) + lid] = s_;
  }
"""
)


@functools.cache
def _rms_norm_pre_kernel():
    return mx.fast.metal_kernel(
        name="rms_norm_pre",
        input_names=["x", "w", "eps"],
        output_names=["y", "xs", "xsum"],
        source=RMS_SRC,
    )


def rms_norm_pre(x, w, eps: float):
    """x [..., D] (D % 16 == 0, D <= 4096), w [D] -> (rms_norm(x, w, eps), (xs, xsum))."""
    D = x.shape[-1]
    rows = x.size // D
    y, xs, xsum = _rms_norm_pre_kernel()(
        inputs=[x, w, constants.f32(eps)],
        template=[("T", x.dtype), ("AXIS", D)],
        grid=(rows * D // 4, 1, 1),
        threadgroup=(D // 4, 1, 1),
        output_shapes=[x.shape, (rows, D), (rows, D // 16)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )
    return y, (xs, xsum)


def check_rms_pre(hidden: int = 2048) -> bool:
    """rms_norm_pre: bitwise mx.fast.rms_norm and bitwise prescale() of its output."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import prescale

    for t in range(12):
        x = (mx.random.normal((1, hidden), key=mx.random.key(500 + t)) * (0.5 + t)).astype(
            mx.bfloat16
        )
        w = (mx.random.normal((hidden,), key=mx.random.key(600 + t)) * 0.2 + 1).astype(mx.bfloat16)
        ref = mx.fast.rms_norm(x, w, 1e-6)
        rxs, rsum = prescale(ref)
        y, (xs, xsum) = rms_norm_pre(x, w, 1e-6)
        if not (
            mx.array_equal(y, ref).item()
            and mx.array_equal(xs, rxs).item()
            and mx.array_equal(xsum, rsum).item()
        ):
            return False
    return True
