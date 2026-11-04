"""3-bit routed-expert GEMV kernels on pre-scaled activations, bit-identical per row to stock qmv_fast.

The activations come from `qmv_prescaled.prescale`, so every product is exact in fp32 and only the
add order matters; each kernel keeps qmv_fast's order and `simd_sum`.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import (
    ACTIVATION_POINTERS,
    DIM_KG,
    DIMS_MK,
    HEADER,
    KW_BYTES,
    REGISTER_LOOP,
    RESULT_INIT,
    SCALE_SETUP,
    THREAD_IDS,
    prescale,
    weight_pointers,
)

# ---- routed gate+up for one row with the SwiGLU epilogue fused
GU_SIG = """
struct SigmoidX {   // mlx unary_ops.h Sigmoid, verbatim (as expert_solo.py)
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""


SOURCE_GU = (
    "\n"
    + SCALE_SETUP
    + "\n"
    + DIMS_MK
    + "    const int N = 2 * HID;\n"
    + THREAD_IDS
    + "\n"
    + KW_BYTES
    + DIM_KG
    + """    // threadgroup tid.y = (selected expert j, block of RPS hidden units); simdgroup 0 computes those
    // units' GATE rows, simdgroup 1 their UP rows (rows HID.. of the fused [2*HID, K] stack)
    // UB unit blocks per threadgroup: simdgroup sg = 2*blk + (0 gate | 1 up)
    const int nblk = HID / RPS;
    const int gblk = tid.y * UB + (sg >> 1);
    const int jsel = gblk / nblk;
    const int unit0 = (gblk % nblk) * RPS;
    const int out_row = (sg & 1) * HID + unit0;    // row within the expert's fused stack
    const int m0 = 0;
    const size_t erow = (size_t)inds[jsel] * (2 * HID) + out_row;

"""
    + weight_pointers("erow")
    + ACTIVATION_POINTERS
    + "\n"
    + RESULT_INIT
    + "\n"
    + REGISTER_LOOP
    + "\n"
    + """    threadgroup float gu_tg[2 * UB * RPS];
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
      U v = simd_sum(result[0][r]);
      if (lid == 0) gu_tg[sg * RPS + r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((sg & 1) == 0 && lid < RPS) {
      // expert_solo's epilogue verbatim: round gate and up to T, silu via MLX's Sigmoid, times up
      T gt = static_cast<T>(gu_tg[sg * RPS + lid]), ut = static_cast<T>(gu_tg[(sg + 1) * RPS + lid]);
      T sgm = SigmoidX{}(gt);
      T sl = gt * sgm;
      y[jsel * HID + unit0 + lid] = sl * ut;
    }
"""
)


GU_UB = 1


@functools.cache
def _gate_up_kernel():
    return mx.fast.metal_kernel(
        name="moe_gate_up_v4",
        input_names=["w", "scales", "biases", "xs", "xsum", "inds"],
        output_names=["y"],
        header=HEADER + GU_SIG,
        source=SOURCE_GU,
        ensure_row_contiguous=True,
    )


def expert_gate_up_prescaled(x, gate_up, inds, rps: int = 4, pre=None, ub: int | None = None):
    """One row x [1, K], inds [k] -> h [k, HID], bitwise expert_solo.expert_gate_up_exact."""
    hid = gate_up.out_dims // 2
    ne = inds.shape[0]
    ub = ub or GU_UB
    xs, xsum = pre if pre is not None else prescale(x)
    return _gate_up_kernel()(
        inputs=[*gate_up.expert_views(), xs, xsum, inds],
        template=[
            ("T", x.dtype),
            ("GS", 64),
            ("MB", 1),
            ("RPS", rps),
            ("NSG", 2),
            ("PF", 1),
            ("HID", hid),
            ("UB", ub),
        ],
        grid=(32, 2 * ne * (hid // rps), 1),
        threadgroup=(32, 2 * ub, 1),
        output_shapes=[(ne, hid)],
        output_dtypes=[x.dtype],
    )[0]


# ---- routed gate+up for M rows: dedup across rows, gate/up in sibling simdgroups, fused SwiGLU
GU_ROWS_SRC = """
  typedef float U;
  constexpr int group_size = GS;
  constexpr int block_size = 512;
  constexpr int scale_step_per_thread = group_size / 16;
  const int K = KK;
  const int Kw = K * 3 / 8, Kw16 = Kw / 2, Kg = K / group_size;
  const int P = NROWS * NE;
  const uint lid = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint3 tid = threadgroup_position_in_grid;
  const int nblk = HID / RPS;
  const int p = tid.y / nblk;
  const int unit0 = (tid.y % nblk) * RPS;
  const uint e = inds[p];
  for (int q = 0; DEDUP && q < p; q++) {
    if (inds[q] == e) return;              // an earlier pair owns this expert (whole threadgroup)
  }
  const size_t erow = (size_t)e * (2 * HID) + sg * HID + unit0;
  const device uint16_t* w0 = (const device uint16_t*)((const device uint8_t*)w + erow * Kw + lid * 6);
  const device T* s0 = scales + erow * Kg + lid / scale_step_per_thread;
  const device T* b0p = biases + erow * Kg + lid / scale_step_per_thread;
  threadgroup float gu_tg[2 * RPS];
  for (int q = p; q < (DEDUP ? P : p + 1); q++) {
    if (inds[q] != e) continue;            // uniform across the threadgroup
    const int row = q / NE;
    const device float* xr = xs + row * K + lid * 16;
    const device float* xsr = xsum + row * (K / 16) + lid;
    U result[RPS];
    #pragma unroll
    for (int r = 0; r < RPS; r++) result[r] = 0;
    for (int kb = 0; kb < K / block_size; kb++) {
      float4 xt4[4];
      const device float4* xv = (const device float4*)(xr + kb * block_size);
      #pragma unroll
      for (int j = 0; j < 4; j++) xt4[j] = xv[j];
      const U sum = xsr[kb * 32];
#define XT(idx) (xt4[(idx) >> 2][(idx) & 3])
      #pragma unroll
      for (int r = 0; r < RPS; r++) {
        const device uint16_t* wl = w0 + r * Kw16 + kb * (block_size * 3 / 16);
        const uint w01 = wl[0] | (uint(wl[1]) << 16);
        const uint w2 = wl[2];
        const U s = s0[r * Kg + kb * (block_size / group_size)];
        const U b = b0p[r * Kg + kb * (block_size / group_size)];
        U a = 0;
        #pragma unroll
        for (int ii = 0; ii < 2; ii++) {
          const uint c0 = ii == 0 ? (w01 & 0xff) : ((w01 >> 24) & 0xff);
          const uint c1 = ii == 0 ? ((w01 >> 8) & 0xff) : (w2 & 0xff);
          const uint c2 = ii == 0 ? ((w01 >> 16) & 0xff) : ((w2 >> 8) & 0xff);
          const int o = 8 * ii;
          a += U(c0 & 0x07) * XT(o);
          a += U(c0 & 0x38) * XT(o + 1);
          a += U(c0 & 0xc0) * XT(o + 2);
          a += U(c1 & 0x01) * (XT(o + 2) * 256.0f);
          a += U(c1 & 0x0e) * XT(o + 3);
          a += U(c1 & 0x70) * XT(o + 4);
          a += U(c1 & 0x80) * XT(o + 5);
          a += U(c2 & 0x03) * (XT(o + 5) * 256.0f);
          a += U(c2 & 0x1c) * XT(o + 6);
          a += U(c2 & 0xe0) * XT(o + 7);
        }
        result[r] += s * a + sum * b;
      }
    }
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
      U v = simd_sum(result[r]);
      if (lid == 0) gu_tg[sg * RPS + r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lid < RPS) {
      T gt = static_cast<T>(gu_tg[lid]), ut = static_cast<T>(gu_tg[RPS + lid]);
      T sgm = SigmoidX{}(gt);
      T sl = gt * sgm;
      y[q * HID + unit0 + lid] = sl * ut;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@functools.cache
def _gate_up_rows_kernel():
    return mx.fast.metal_kernel(
        name="moe_gate_up_v4_rows",
        input_names=["w", "scales", "biases", "xs", "xsum", "inds"],
        output_names=["y"],
        header=HEADER + GU_SIG,
        source=GU_ROWS_SRC,
        ensure_row_contiguous=True,
    )


def expert_gate_up_prescaled_rows(x, gate_up, inds, rps: int = 4, pre=None, dedup: bool = True):
    """x [M, K], inds [M, k] -> h [M*k, HID], each row bitwise expert_gate_up_exact of that row.

    `dedup` gives each distinct expert one threadgroup column for all rows that chose it; same bits.
    """
    M, K = x.shape
    ne = inds.shape[-1]
    hid = gate_up.out_dims // 2
    xs, xsum = pre if pre is not None else prescale(x)
    fi = inds.reshape(-1).astype(mx.uint32)
    return _gate_up_rows_kernel()(
        inputs=[*gate_up.expert_views(), xs, xsum, fi],
        template=[
            ("T", x.dtype),
            ("GS", 64),
            ("KK", K),
            ("HID", hid),
            ("NE", ne),
            ("NROWS", M),
            ("RPS", rps),
            ("DEDUP", bool(dedup)),
        ],
        grid=(32, 2 * M * ne * (hid // rps), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(M * ne, hid)],
        output_dtypes=[x.dtype],
        init_value=0,
    )[0]
