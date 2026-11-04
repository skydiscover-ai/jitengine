"""Gated DeltaNet decode glue as three Metal launches around the recurrence.

`qmv_multi` runs up to four 4-bit input projections in one launch, `gdn_pre` does the conv state
update, conv, silu, l2 norms, beta and g, and `gdn_post` computes sigmoid(z) * rms_norm(y, w).

Each `qmv_multi` row uses the dense kernel's fixed k-order, so its value does not depend on which
weight sets share the launch. Reduction orders are fixed at compile time; results differ from stock
by about 1e-7 relative. The load-time check (`check_glue`) holds a set's rows to that independence
bit for bit, and the outputs to `selfcheck.no_worse_than_stock` against the stock ops, the property
behind that figure.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m2.kernels.constants import f32, placeholder
from jitengine_qwen3_8_flash_next_m2.kernels.qmv_row_blocked import DEQUANTIZE4

HEADER = (
    """
#include <metal_simdgroup>
using namespace metal;

"""
    + DEQUANTIZE4
    + """inline float sigmoid_f(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
inline float softplus_f(float x) { return log1p(metal::exp(-metal::abs(x))) + metal::max(x, 0.0f); }
"""
)

# inputs: x, w1,s1,b1, w2,s2,b2, w3,s3,b3, w4,s4,b4  ->  y [M, N1+N2+N3+N4]      (M = 1 row per grid.z)
MULTI_SOURCE = """
    typedef float U;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int KG = K / GS;
    constexpr int slots_per_sg = 32 / KL;
    constexpr int rows_per_sg = slots_per_sg * RB;
    constexpr int rows_per_tg = rows_per_sg * NSG;
    constexpr int NROWS = N1 + N2 + N3 + N4;
    constexpr int wbytes = K * bits / 8;

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int m = tid.z;
    const short k_lane = simd_lid % KL;
    const short slot = simd_lid / KL;
    const int row0 = tid.y * rows_per_tg + rows_per_sg * simd_gid + slot * RB;
    const device T* xr_ = x + (ulong)m * (XSWI ? 2 * K : K);

    // per row: which weight set and the row inside it. The four sets may carry DIFFERENT
    // scale/bias dtypes (the dense policy stores unserved shapes' metadata as fp32), so scales
    // and biases are read through a switch on the set rather than through one typed pointer.
    const device uint8_t* wrow[RB];
    int setid[RB];
    ulong roff[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      int row = min(row0 + r, NROWS - 1);
      if (row < N1) { setid[r] = 0; roff[r] = row; wrow[r] = (const device uint8_t*)w1 + (ulong)row * wbytes; }
      else if (row < N1 + N2) { row -= N1; setid[r] = 1; roff[r] = row; wrow[r] = (const device uint8_t*)w2 + (ulong)row * wbytes; }
      else if (row < N1 + N2 + N3) { row -= N1 + N2; setid[r] = 2; roff[r] = row; wrow[r] = (const device uint8_t*)w3 + (ulong)row * wbytes; }
      else { row -= N1 + N2 + N3; setid[r] = 3; roff[r] = row; wrow[r] = (const device uint8_t*)w4 + (ulong)row * wbytes; }
    }
    U result[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) result[r] = 0;
    for (int g = k_lane; g < KG; g += KL) {
      U scale[RB], bias[RB];
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const ulong gi = roff[r] * KG + g;
        switch (setid[r]) {
          case 0: scale[r] = static_cast<U>(s1[gi]); bias[r] = static_cast<U>(b1[gi]); break;
          case 1: scale[r] = static_cast<U>(s2[gi]); bias[r] = static_cast<U>(b2[gi]); break;
          case 2: scale[r] = static_cast<U>(s3[gi]); bias[r] = static_cast<U>(b3[gi]); break;
          default: scale[r] = static_cast<U>(s4[gi]); bias[r] = static_cast<U>(b4[gi]); break;
        }
      }
      #pragma unroll
      for (int sc = 0; sc < GS / sub; sc++) {
        const int k0 = g * GS + sc * sub;
        U xr[sub];
        #pragma unroll
        for (int i = 0; i < sub; i++) {
          if (XSWI) {
            const float gt = float(xr_[k0 + i]);
            xr[i] = (gt * sigmoid_f(gt)) * float(xr_[K + k0 + i]);   // silu(gate) * up, fp32
          } else {
            xr[i] = static_cast<U>(xr_[k0 + i]);
          }
        }
        #pragma unroll
        for (int r = 0; r < RB; r++) {
          U w_dq[sub];
          dequantize4<U, sub>(wrow[r] + k0 * bits / 8, scale[r], bias[r], w_dq);
          U acc = 0;
          #pragma unroll
          for (int i = 0; i < sub; i++) acc += xr[i] * w_dq[i];
          result[r] += acc;
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      if (KL >= 32) result[r] += simd_shuffle_down(result[r], 16);
      if (KL >= 16) result[r] += simd_shuffle_down(result[r], 8);
      if (KL >= 8)  result[r] += simd_shuffle_down(result[r], 4);
      if (KL >= 4)  result[r] += simd_shuffle_down(result[r], 2);
      if (KL >= 2)  result[r] += simd_shuffle_down(result[r], 1);
    }
    if (k_lane == 0) {
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const int row = row0 + r;
        if (row < NROWS) y[(ulong)m * NROWS + row] = static_cast<T>(result[r]);
      }
    }
"""

# `qmv_multi` with VECS activation rows per threadgroup, so the weights stream once for all of them.
# VECS is a loop beside the single-row arithmetic, never inside it, so every row is bit-identical to
# `qmv_multi` on that row alone.
MULTIV_SOURCE = """
    typedef float U;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int KG = K / GS;
    constexpr int slots_per_sg = 32 / KL;
    constexpr int rows_per_sg = slots_per_sg * RB;
    constexpr int rows_per_tg = rows_per_sg * NSG;
    constexpr int NROWS = N1 + N2 + N3 + N4;
    constexpr int wbytes = K * bits / 8;

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int m0 = tid.z * VECS;
    const short k_lane = simd_lid % KL;
    const short slot = simd_lid / KL;
    const int row0 = tid.y * rows_per_tg + rows_per_sg * simd_gid + slot * RB;
    const device T* xv_[VECS];
    #pragma unroll
    for (int vv = 0; vv < VECS; vv++) xv_[vv] = x + (ulong)min(m0 + vv, M - 1) * (XSWI ? 2 * K : K);

    // per row: which weight set and the row inside it. The four sets may carry DIFFERENT
    // scale/bias dtypes (the dense policy stores unserved shapes' metadata as fp32), so scales
    // and biases are read through a switch on the set rather than through one typed pointer.
    const device uint8_t* wrow[RB];
    int setid[RB];
    ulong roff[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      int row = min(row0 + r, NROWS - 1);
      if (row < N1) { setid[r] = 0; roff[r] = row; wrow[r] = (const device uint8_t*)w1 + (ulong)row * wbytes; }
      else if (row < N1 + N2) { row -= N1; setid[r] = 1; roff[r] = row; wrow[r] = (const device uint8_t*)w2 + (ulong)row * wbytes; }
      else if (row < N1 + N2 + N3) { row -= N1 + N2; setid[r] = 2; roff[r] = row; wrow[r] = (const device uint8_t*)w3 + (ulong)row * wbytes; }
      else { row -= N1 + N2 + N3; setid[r] = 3; roff[r] = row; wrow[r] = (const device uint8_t*)w4 + (ulong)row * wbytes; }
    }
    U result[RB][VECS];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      #pragma unroll
      for (int vv = 0; vv < VECS; vv++) result[r][vv] = 0;
    }
    for (int g = k_lane; g < KG; g += KL) {
      U scale[RB], bias[RB];
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const ulong gi = roff[r] * KG + g;
        switch (setid[r]) {
          case 0: scale[r] = static_cast<U>(s1[gi]); bias[r] = static_cast<U>(b1[gi]); break;
          case 1: scale[r] = static_cast<U>(s2[gi]); bias[r] = static_cast<U>(b2[gi]); break;
          case 2: scale[r] = static_cast<U>(s3[gi]); bias[r] = static_cast<U>(b3[gi]); break;
          default: scale[r] = static_cast<U>(s4[gi]); bias[r] = static_cast<U>(b4[gi]); break;
        }
      }
      #pragma unroll
      for (int sc = 0; sc < GS / sub; sc++) {
        const int k0 = g * GS + sc * sub;
        U xr[VECS][sub];
        #pragma unroll
        for (int vv = 0; vv < VECS; vv++) {
          const device T* xr_ = xv_[vv];
          #pragma unroll
          for (int i = 0; i < sub; i++) {
            if (XSWI) {
              const float gt = float(xr_[k0 + i]);
              xr[vv][i] = (gt * sigmoid_f(gt)) * float(xr_[K + k0 + i]);   // silu(gate) * up, fp32
            } else {
              xr[vv][i] = static_cast<U>(xr_[k0 + i]);
            }
          }
        }
        #pragma unroll
        for (int r = 0; r < RB; r++) {
          U w_dq[sub];
          dequantize4<U, sub>(wrow[r] + k0 * bits / 8, scale[r], bias[r], w_dq);
          #pragma unroll
          for (int vv = 0; vv < VECS; vv++) {
            U acc = 0;
            #pragma unroll
            for (int i = 0; i < sub; i++) acc += xr[vv][i] * w_dq[i];
            result[r][vv] += acc;
          }
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      #pragma unroll
      for (int vv = 0; vv < VECS; vv++) {
        if (KL >= 32) result[r][vv] += simd_shuffle_down(result[r][vv], 16);
        if (KL >= 16) result[r][vv] += simd_shuffle_down(result[r][vv], 8);
        if (KL >= 8)  result[r][vv] += simd_shuffle_down(result[r][vv], 4);
        if (KL >= 4)  result[r][vv] += simd_shuffle_down(result[r][vv], 2);
        if (KL >= 2)  result[r][vv] += simd_shuffle_down(result[r][vv], 1);
      }
    }
    if (k_lane == 0) {
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const int row = row0 + r;
        #pragma unroll
        for (int vv = 0; vv < VECS; vv++)
          if (row < NROWS && m0 + vv < M) y[(ulong)(m0 + vv) * NROWS + row] = static_cast<T>(result[r][vv]);
      }
    }
"""


# inputs: conv (KC-1, C), ycat (NTOT), convw (C, KC), alog (HV), dtb (HV), epsv (1), scl (2: q_scale, k_scale)
# outputs: q (HK, DK), k (HK, DK), v (HV, DV), g (HV) f32, beta (HV), newconv (KC-1, C)
# one simdgroup per head block of D channels: heads [0, HK) are q, [HK, 2HK) are k, then HV of v
PRE_SOURCE = """
    constexpr int C = 2 * HK * DK + HV * DV;
    constexpr int NH = 2 * HK + HV;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint tid_tg = thread_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    if (tid.x == 0 && tid_tg < HV) {
      const int hv = tid_tg;
      beta[hv] = static_cast<T>(sigmoid_f(float(ycat[BOFF + hv])));
      g[hv] = metal::exp(-metal::exp(alog[hv]) * softplus_f(float(ycat[AOFF + hv]) + dtb[hv]));
    }
    const int head = tid.x * NSG + simd_gid;
    if (head >= NH) return;
    const bool is_v = head >= 2 * HK;
    const int D = is_v ? DV : DK;
    const int c0 = is_v ? (2 * HK * DK + (head - 2 * HK) * DV) : head * DK;
    const int per = D / 32;
    float o[DMAX / 32];
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < DMAX / 32; i++) {
      if (i < per) {
        const int c = c0 + simd_lid * per + i;
        // depthwise causal conv over [conv state rows | this position], then silu
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < KC - 1; j++) {
          const float xin = float(conv[(ulong)j * C + c]);
          acc += float(convw[(ulong)c * KC + j]) * xin;
          newconv[(ulong)j * C + c] = static_cast<T>(j + 1 < KC - 1 ? float(conv[(ulong)(j + 1) * C + c]) : float(ycat[QKVOFF + c]));
        }
        acc += float(convw[(ulong)c * KC + (KC - 1)]) * float(ycat[QKVOFF + c]);
        const float so = acc * sigmoid_f(acc);
        o[i] = so;
        ss += so * so;
      }
    }
    if (is_v) {
      const int hv = head - 2 * HK;
      #pragma unroll
      for (int i = 0; i < DMAX / 32; i++)
        if (i < per) v[(ulong)hv * DV + simd_lid * per + i] = static_cast<T>(o[i]);
      return;
    }
    ss = simd_sum(ss);
    const float rstd = metal::rsqrt(ss / float(D) + epsv[0]);
    const bool is_q = head < HK;
    const int hk = is_q ? head : head - HK;
    const float sc = is_q ? scl[0] : scl[1];
    #pragma unroll
    for (int i = 0; i < DMAX / 32; i++) {
      if (i < per) {
        const float val = (o[i] * rstd) * sc;
        if (is_q) q[(ulong)hk * DK + simd_lid * per + i] = static_cast<T>(val);
        else      k[(ulong)hk * DK + simd_lid * per + i] = static_cast<T>(val);
      }
    }
"""

# inputs: y (HV, DV), ycat (NTOT), normw (DV: one weight per head dim, shared by the heads), epsv  ->  act (HV*DV)
POST_SOURCE = """
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int hv = tid.x * NSG + simd_gid;
    if (hv >= HV) return;
    constexpr int per = DV / 32;
    float yv[per];
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < per; i++) {
      yv[i] = float(y[(ulong)hv * DV + simd_lid * per + i]);
      ss += yv[i] * yv[i];
    }
    ss = simd_sum(ss);
    const float rstd = metal::rsqrt(ss / float(DV) + epsv[0]);
    #pragma unroll
    for (int i = 0; i < per; i++) {
      const int c = hv * DV + simd_lid * per + i;
      const float normed = (yv[i] * rstd) * float(normw[simd_lid * per + i]);
      const float gate = sigmoid_f(float(ycat[ZOFF + c]));
      act[c] = static_cast<T>(gate * normed);
    }
"""


@functools.cache
def _kernel(which: str):
    if which == "multi":
        return mx.fast.metal_kernel(
            name="moe_engine_qmv_multi",
            input_names=[
                "x",
                "w1",
                "s1",
                "b1",
                "w2",
                "s2",
                "b2",
                "w3",
                "s3",
                "b3",
                "w4",
                "s4",
                "b4",
            ],
            output_names=["y"],
            header=HEADER,
            source=MULTI_SOURCE,
            ensure_row_contiguous=True,
        )
    if which == "multiv":
        return mx.fast.metal_kernel(
            name="moe_engine_qmv_multiv",
            input_names=[
                "x",
                "w1",
                "s1",
                "b1",
                "w2",
                "s2",
                "b2",
                "w3",
                "s3",
                "b3",
                "w4",
                "s4",
                "b4",
            ],
            output_names=["y"],
            header=HEADER,
            source=MULTIV_SOURCE,
            ensure_row_contiguous=True,
        )
    if which == "pre":
        return mx.fast.metal_kernel(
            name="moe_engine_gdn_pre",
            input_names=["conv", "ycat", "convw", "alog", "dtb", "epsv", "scl"],
            output_names=["q", "k", "v", "g", "beta", "newconv"],
            header=HEADER,
            source=PRE_SOURCE,
            ensure_row_contiguous=True,
        )
    return mx.fast.metal_kernel(
        name="moe_engine_gdn_post",
        input_names=["y", "ycat", "normw", "epsv"],
        output_names=["act"],
        header=HEADER,
        source=POST_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_multi(lins) -> bool:
    """1..4 QLinears sharing K, 4-bit, one group size, biases present, K % gs == 0."""
    if not 1 <= len(lins) <= 4:
        return False
    k, gs = lins[0].in_dims, lins[0].group_size
    return (
        all(
            lin.bits == 4 and lin.group_size == gs and lin.in_dims == k and lin.biases is not None
            for lin in lins
        )
        and k % gs == 0
        and gs % 8 == 0
    )


def qmv_multi(x: mx.array, lins, cfg=(16, 4, 2), swiglu: bool = False) -> mx.array:
    """x [M, K] -> [M, sum(out_dims)], rows end to end; `swiglu` takes x = [gate | up] [M, 2K]."""
    M, K = x.shape
    if swiglu:
        K //= 2
    KL, NSG, RB = cfg
    lins = list(lins)
    ns = [lin.out_dims for lin in lins] + [0] * (4 - len(lins))
    args = [x]
    sdt = lins[0].scales.dtype
    for lin in lins:
        args += [lin.weight, lin.scales, lin.biases]
    if len(lins) < 4:
        args += [placeholder(mx.uint32), placeholder(sdt), placeholder(sdt)] * (4 - len(lins))
    rows_per_tg = (32 // KL) * RB * NSG
    nrows = sum(ns)
    return _kernel("multi")(
        inputs=args,
        template=[
            ("T", x.dtype),
            ("KL", KL),
            ("NSG", NSG),
            ("RB", RB),
            ("GS", lins[0].group_size),
            ("K", K),
            ("N1", ns[0]),
            ("N2", ns[1]),
            ("N3", ns[2]),
            ("N4", ns[3]),
            ("XSWI", bool(swiglu)),
        ],
        grid=(32, NSG * ((nrows + rows_per_tg - 1) // rows_per_tg), M),
        threadgroup=(32, NSG, 1),
        output_shapes=[(M, nrows)],
        output_dtypes=[x.dtype],
    )[0]


def qmv_multi_v(x: mx.array, lins, cfg=(16, 4, 2), swiglu: bool = False, vecs: int = 2) -> mx.array:
    """`qmv_multi` with `vecs` rows per threadgroup (weights read once per threadgroup); bit-identical
    rows."""
    M, K = x.shape
    if swiglu:
        K //= 2
    KL, NSG, RB = cfg
    lins = list(lins)
    ns = [lin.out_dims for lin in lins] + [0] * (4 - len(lins))
    args = [x]
    sdt = lins[0].scales.dtype
    for lin in lins:
        args += [lin.weight, lin.scales, lin.biases]
    if len(lins) < 4:
        args += [placeholder(mx.uint32), placeholder(sdt), placeholder(sdt)] * (4 - len(lins))
    rows_per_tg = (32 // KL) * RB * NSG
    nrows = sum(ns)
    return _kernel("multiv")(
        inputs=args,
        template=[
            ("T", x.dtype),
            ("KL", KL),
            ("NSG", NSG),
            ("RB", RB),
            ("GS", lins[0].group_size),
            ("K", K),
            ("N1", ns[0]),
            ("N2", ns[1]),
            ("N3", ns[2]),
            ("N4", ns[3]),
            ("XSWI", bool(swiglu)),
            ("VECS", int(vecs)),
            ("M", M),
        ],
        grid=(32, NSG * ((nrows + rows_per_tg - 1) // rows_per_tg), (M + vecs - 1) // vecs),
        threadgroup=(32, NSG, 1),
        output_shapes=[(M, nrows)],
        output_dtypes=[x.dtype],
    )[0]


def supports_gdn(cfg) -> bool:
    """Whether `gdn_pre` and `gdn_post` take this config: key and value head dims that are
    multiples of 32 up to 256, and a value head count that is a multiple of the key head count."""
    return (
        cfg.linear_key_head_dim % 32 == 0
        and cfg.linear_value_head_dim % 32 == 0
        and cfg.linear_key_head_dim <= 256
        and cfg.linear_value_head_dim <= 256
        and cfg.linear_num_value_heads % cfg.linear_num_key_heads == 0
    )


def gdn_pre(
    conv_state: mx.array,
    ycat: mx.array,
    conv_w2d: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    cfg,
    l2_eps: float,
    q_scale: float,
    k_scale: float,
    offs: tuple,
    nsg: int = 4,
):
    """Conv state update, depthwise conv, silu, l2 norms, beta and g for one decode position.

    conv_state [KC-1, C], ycat [NTOT] (qkv | z | b | a at `offs`), conv_w2d [C, KC] -> (q [HK, DK],
    k [HK, DK], v [HV, DV], g [HV] f32, beta [HV], new_conv [KC-1, C]).
    """
    HK, DK = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    HV, DV = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    KC = cfg.linear_conv_kernel_dim
    C = 2 * HK * DK + HV * DV
    T = ycat.dtype
    nh = 2 * HK + HV
    qkv_off, _z_off, b_off, a_off = offs
    return _kernel("pre")(
        inputs=[
            conv_state,
            ycat,
            conv_w2d,
            a_log,
            dt_bias,
            f32(l2_eps),
            f32(q_scale, k_scale),
        ],
        template=[
            ("T", T),
            ("HK", HK),
            ("DK", DK),
            ("HV", HV),
            ("DV", DV),
            ("KC", KC),
            ("NSG", nsg),
            ("DMAX", max(DK, DV)),
            ("QKVOFF", qkv_off),
            ("BOFF", b_off),
            ("AOFF", a_off),
        ],
        grid=(32 * ((nh + nsg - 1) // nsg), nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(HK, DK), (HK, DK), (HV, DV), (HV,), (HV,), (KC - 1, C)],
        output_dtypes=[T, T, T, mx.float32, T, T],
    )


def gdn_post(
    y: mx.array, ycat: mx.array, norm_w: mx.array, cfg, eps: float, z_off: int, nsg: int = 4
) -> mx.array:
    """y [HV, DV], ycat (z at z_off), norm_w [DV] -> act [HV*DV] = sigmoid(z) * rms_norm(y, w)."""
    HV, DV = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    return _kernel("post")(
        inputs=[y, ycat, norm_w, f32(eps)],
        template=[("T", y.dtype), ("HV", HV), ("DV", DV), ("NSG", nsg), ("ZOFF", z_off)],
        grid=(32 * ((HV + nsg - 1) // nsg), nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(HV * DV,)],
        output_dtypes=[y.dtype],
    )[0]


# ----------------------------------------------------------------------------- load-time check
def check_glue(calls, act_dtype, gdn=None) -> bool:
    """The load-time check of the model's `qmv_multi` calls and, when `gdn` = (cfg, l2_eps,
    q_scale, k_scale, nsg) is given, of gdn_pre and gdn_post at that GDN geometry.

    `calls` lists each `qmv_multi` call as (weight sets, launch config, swiglu), a weight set being
    (out, in, bits, group_size, metadata dtype). Bit for bit (the claim above): each set's rows in
    the shared launch are the set launched alone, and gdn_pre's new conv state is an exact copy.
    Against the stock ops the model runs without the glue (`quantized_matmul` per set, and
    `Qwen4ExpTextGatedDeltaNet._decode_body` from the projections on, compiled), each output is no less
    accurate than stock, as `selfcheck.no_worse_than_stock` measures it. g's allowance adds the
    Metal fast log's absolute error bound on [0.5, 2], 2^-21 (MSL specification, table 8.2): the
    kernel's softplus takes log(1 + e^-|x|) there, so its error in g is at most 2^-21 g e^A_log."""
    if not all(_check_multi(*call, act_dtype, seed=40 * i) for i, call in enumerate(calls)):
        return False
    return gdn is None or (_check_pre(*gdn, act_dtype) and _check_post(gdn[0], gdn[4], act_dtype))


def _random_lins(sets, seed: int) -> list:
    """The weight sets (out, in, bits, group_size, metadata dtype) as random `sc.Linear`s; raises
    ValueError if `qmv_multi` does not take them."""
    lins = [
        sc.Linear(n, k, seed + i, gs, bits, meta) for i, (n, k, bits, gs, meta) in enumerate(sets)
    ]
    if not supports_multi(lins):
        raise ValueError(f"qmv_multi does not take the weight sets {sets}")
    return lins


def check_multi_v(calls, act_dtype) -> bool:
    """The load-time check of the verify pass's `qmv_multi_v` launches (two rows per threadgroup).

    `calls` lists them as `check_glue` lists the solo calls, (weight sets, launch config, swiglu):
    the verify pass launches the same weight sets as the solo path and no others. Bit for
    bit, the claim above: each of the two rows is what `qmv_multi` gives that row alone, which is
    what a solo step computes. The rows are read at an offset, as the model's are. The solo
    kernel has its own check against stock (`check_glue`), so equality to it is the whole
    criterion: a verify pass that matches it matches a solo step."""
    for i, (sets, launch, swiglu) in enumerate(calls):
        lins = _random_lins(sets, 40 * i + 17)
        k = lins[0].in_dims
        x = sc.normal((3, 2 * k if swiglu else k), 40 * i + 26, act_dtype)[1:]  # read at an offset
        both = qmv_multi_v(x, lins, launch, swiglu=swiglu, vecs=2)
        for r in range(2):
            alone = qmv_multi(x[r : r + 1], lins, launch, swiglu=swiglu)
            if not mx.array_equal(both[r : r + 1], alone).item():
                return False
    return True


def _check_multi(sets, launch, swiglu: bool, dt, seed: int) -> bool:
    lins = _random_lins(sets, seed)
    k = lins[0].in_dims
    x = sc.normal((2, 2 * k if swiglu else k), seed + 9, dt)[1:]  # read at an offset
    together = qmv_multi(x, lins, launch, swiglu=swiglu)
    at = 0
    for lin in lins:
        if len(lins) > 1:
            alone = qmv_multi(x, [lin], launch, swiglu=swiglu)
            if not mx.array_equal(together[:, at : at + lin.out_dims], alone).item():
                return False
        at += lin.out_dims
    xs = sc.f64(x)
    if swiglu:  # x = [gate | up]: the input is the shared expert's silu(gate) * up
        x = nn.silu(x[:, :k]) * x[:, k:]
        xs = xs[:, :k] / (1.0 + np.exp(-xs[:, :k])) * xs[:, k:]
    at = 0
    for lin in lins:
        rows = sc.sample(lin.out_dims)
        idx = mx.array(rows)
        got = together[:, at + idx]
        if not sc.no_worse_than_stock(got, lin.stock(x)[:, idx], lin.exact(xs, rows)):
            return False
        at += lin.out_dims
    return True


def _gdn_dims(cfg):
    hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    c = 2 * hk * dk + hv * dv
    return hk, dk, hv, dv, c, (0, c, c + hv * dv, c + hv * dv + hv)  # offsets of qkv | z | b | a


def _check_pre(cfg, l2_eps, q_scale, k_scale, nsg, dt) -> bool:
    hk, dk, hv, dv, c, offs = _gdn_dims(cfg)
    kc = cfg.linear_conv_kernel_dim
    ycat = sc.normal((offs[3] + hv,), 81, dt)
    conv = sc.normal((2, kc - 1, c), 82, dt)[1]  # a row slot read at an offset
    conv_w = (0.5 * sc.normal((c, kc, 1), 83)).astype(dt)
    a_log, dt_bias = 0.5 + 0.3 * sc.normal((hv,), 84), 0.5 * sc.normal((hv,), 85)
    mx.eval(conv_w, a_log, dt_bias)  # the compiled stock path captures them
    got = gdn_pre(
        conv, ycat, conv_w.reshape(c, kc), a_log, dt_bias, cfg, l2_eps, q_scale, k_scale, offs, nsg
    )

    @mx.compile
    def stock(conv, ycat):  # `_decode_body` from the projections on
        b, a = ycat[offs[2] : offs[2] + hv], ycat[offs[3] : offs[3] + hv]
        conv_input = mx.concatenate([conv[None], ycat[:c].reshape(1, 1, c)], axis=1)
        conv_out = nn.silu(mx.conv1d(conv_input, conv_w, groups=c))
        q, k, v = mx.split(conv_out, [hk * dk, 2 * hk * dk], axis=-1)
        q = q_scale * mx.fast.rms_norm(q.reshape(hk, dk), None, l2_eps)
        k = k_scale * mx.fast.rms_norm(k.reshape(hk, dk), None, l2_eps)
        g = mx.exp(-mx.exp(a_log) * nn.softplus(a.astype(mx.float32) + dt_bias))
        return q, k, v.reshape(hv, dv), g, mx.sigmoid(b), conv_input[0, 1:]

    *want, new_conv = stock(conv, ycat)
    if not mx.array_equal(got[5], new_conv).item():
        return False
    y, w = sc.f64(ycat), sc.f64(conv_w)[:, :, 0]
    co = (np.concatenate([sc.f64(conv), y[None, :c]], axis=0).T * w).sum(axis=1)
    co = co / (1.0 + np.exp(-co))

    def l2(z, s):
        return s * z / np.sqrt((z * z).mean(axis=-1, keepdims=True) + l2_eps)

    a = y[offs[3] : offs[3] + hv] + sc.f64(dt_bias)
    decay = np.exp(sc.f64(a_log))
    g = np.exp(-decay * (np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)))
    exact = (
        l2(co[: hk * dk].reshape(hk, dk), q_scale),
        l2(co[hk * dk : 2 * hk * dk].reshape(hk, dk), k_scale),
        co[2 * hk * dk :].reshape(hv, dv),
        g,
        1.0 / (1.0 + np.exp(-y[offs[2] : offs[2] + hv])),
    )
    slack = (0.0, 0.0, 0.0, 2.0**-21 * float((g * decay).max()), 0.0)
    return all(
        sc.no_worse_than_stock(gt, st, ex, slack=sl)
        for gt, st, ex, sl in zip(got[:5], want, exact, slack, strict=True)
    )


def _check_post(cfg, nsg, dt) -> bool:
    _, _, hv, dv, _, offs = _gdn_dims(cfg)
    eps = float(cfg.rms_norm_eps)
    ycat = sc.normal((offs[3] + hv,), 91, dt)
    y = sc.normal((hv, dv), 92, dt)
    norm_w = (1.0 + 0.1 * sc.normal((dv,), 93)).astype(dt)
    mx.eval(norm_w)  # the compiled stock path captures it
    got = gdn_post(y, ycat, norm_w, cfg, eps, offs[1], nsg)
    z = ycat[offs[1] : offs[2]].reshape(hv, dv)
    sigmoid = str(cfg.output_gate_type) == "sigmoid"

    @mx.compile
    def stock(y, z):  # `_decode_body`'s gated norm, with the model's gate
        zf = z.astype(mx.float32)
        gate = mx.sigmoid(zf) if sigmoid else nn.silu(zf)
        normed = mx.fast.rms_norm(y, norm_w, eps)
        return (gate * normed.astype(mx.float32)).astype(y.dtype).reshape(-1)

    ye, ze = sc.f64(y), sc.f64(z)
    normed = ye / np.sqrt((ye * ye).mean(axis=-1, keepdims=True) + eps) * sc.f64(norm_w)
    gate = 1.0 / (1.0 + np.exp(-ze))
    exact = (normed * (gate if sigmoid else ze * gate)).reshape(-1)
    return sc.no_worse_than_stock(got, stock(y, z), exact)
