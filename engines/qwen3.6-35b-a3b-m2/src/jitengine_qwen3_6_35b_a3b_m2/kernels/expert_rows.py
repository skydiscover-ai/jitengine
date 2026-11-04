"""Bit-exact routed experts for the M rows of a batched decode step in one pass.

Each row equals `expert_solo` (the solo path) bit for bit; gate_up streams each distinct expert
once for all rows that selected it. 3-bit stacks only; callers keep the stock path otherwise.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kernels.expert_solo import DN_SRC, _header
from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_prescaled import prescale
from jitengine_qwen3_6_35b_a3b_m2.kernels.source import substitute

GU_ROWS_SRC = """
  // xs [M, K] fp32 pre-scaled, xsum [M, K/16]; inds [M*NE]; W [E, 2*HID, K*3/8]; h [M*NE, HID]
  constexpr int group_size = GS;
  constexpr int block_size = 512;
  constexpr int scale_step_per_thread = group_size / 16;
  typedef float U;
  const int K = KK;
  const int Kw = K * 3 / 8, Kw16 = Kw / 2;
  const int Kg = K / group_size;
  const int P = NROWS * NE;
  const uint p = threadgroup_position_in_grid.x;
  const uint sg = simdgroup_index_in_threadgroup, lid = thread_index_in_simdgroup;
  const uint e = inds[p];
  for (uint q = 0; q < p; q++) {
    if (inds[q] == e) return;                 // an earlier pair owns this expert
  }
  // every pair (row, slot) that selected e: at most one per row
  int prs[NROWS];
  int n = 0;
  for (int q = p; q < P; q++) {
    if (inds[q] == e) prs[n++] = q;
  }
  const int out_row = threadgroup_position_in_grid.y * (NSG * RPS) + sg * RPS;
  const device uint16_t* wg = (const device uint16_t*)((const device uint8_t*)w + (size_t)e * (2 * HID) * Kw
      + out_row * Kw + lid * 6);
  const device uint16_t* wu = wg + (size_t)HID * Kw16;
  const device T* sg_ = scales + (size_t)e * (2 * HID) * Kg + out_row * Kg + lid / scale_step_per_thread;
  const device T* bg_ = biases + (size_t)e * (2 * HID) * Kg + out_row * Kg + lid / scale_step_per_thread;
  const size_t up_g = (size_t)HID * Kg;

  // rows sharing this expert are processed CH at a time with register-resident arrays; a second
  // chunk re-reads the same weight bytes, which the first chunk just pulled into cache
  for (int c0 = 0; c0 < n; c0 += CH) {
  const int nc = min(CH, n - c0);
  U rg[CH][RPS], ru[CH][RPS];
  #pragma unroll
  for (int i = 0; i < CH; i++)
    #pragma unroll
    for (int r = 0; r < RPS; r++) { rg[i][r] = 0; ru[i][r] = 0; }
  int rowi[CH];
  #pragma unroll
  for (int i = 0; i < CH; i++) rowi[i] = prs[c0 + min(i, nc - 1)] / NE;

  for (int k = 0; k < K; k += block_size) {
    const int kb = k / block_size;
    float4 xt4[CH][4];
    U sum[CH];
    #pragma unroll
    for (int i = 0; i < CH; i++) {
      const device float4* xv = (const device float4*)(xs + rowi[i] * K + k + lid * 16);
      #pragma unroll
      for (int j = 0; j < 4; j++) xt4[i][j] = xv[j];
      sum[i] = xsum[rowi[i] * (K / 16) + kb * 32 + lid];
    }
#define XT(i, idx) (xt4[i][(idx) >> 2][(idx) & 3])
    #pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
      const device uint16_t* wbase = half_ == 0 ? wg : wu;
      const device T* sp = sg_ + (half_ == 0 ? 0 : up_g);
      const device T* bp = bg_ + (half_ == 0 ? 0 : up_g);
      #pragma unroll
      for (int r = 0; r < RPS; r++) {
        const device uint16_t* wl = wbase + r * Kw16 + kb * (block_size * 3 / 16);
        const uint w01 = wl[0] | (uint(wl[1]) << 16);
        const uint w2 = wl[2];
        const U s = sp[r * Kg + kb * (block_size / group_size)];
        const U b = bp[r * Kg + kb * (block_size / group_size)];
        U acc[CH];
        #pragma unroll
        for (int i = 0; i < CH; i++) acc[i] = 0;
        #pragma unroll
        for (int ii = 0; ii < 2; ii++) {
          const uint b0 = ii == 0 ? (w01 & 0xff) : ((w01 >> 24) & 0xff);
          const uint b1 = ii == 0 ? ((w01 >> 8) & 0xff) : (w2 & 0xff);
          const uint b2 = ii == 0 ? ((w01 >> 16) & 0xff) : ((w2 >> 8) & 0xff);
          const U q0 = b0 & 0x07, q1 = b0 & 0x38, q2 = b0 & 0xc0, q3 = b1 & 0x01, q4 = b1 & 0x0e;
          const U q5 = b1 & 0x70, q6 = b1 & 0x80, q7 = b2 & 0x03, q8 = b2 & 0x1c, q9 = b2 & 0xe0;
          const int o = 8 * ii;
          #pragma unroll
          for (int i = 0; i < CH; i++) {
            U a = acc[i];
            a += q0 * XT(i, o);
            a += q1 * XT(i, o + 1);
            a += q2 * XT(i, o + 2);
            a += q3 * (XT(i, o + 2) * 256.0f);
            a += q4 * XT(i, o + 3);
            a += q5 * XT(i, o + 4);
            a += q6 * XT(i, o + 5);
            a += q7 * (XT(i, o + 5) * 256.0f);
            a += q8 * XT(i, o + 6);
            a += q9 * XT(i, o + 7);
            acc[i] = a;
          }
        }
        #pragma unroll
        for (int i = 0; i < CH; i++) {
          if (half_ == 0) rg[i][r] += s * acc[i] + sum[i] * b;
          else ru[i][r] += s * acc[i] + sum[i] * b;
        }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < RPS; r++) {
    #pragma unroll
    for (int i = 0; i < CH; i++) {
      U g = simd_sum(rg[i][r]);
      U u = simd_sum(ru[i][r]);
      if (lid == 0 && i < nc) {
        T gt = static_cast<T>(g), ut = static_cast<T>(u);
        T sgm = SigmoidX{}(gt);
        T sl = gt * sgm;
        h[prs[c0 + i] * HID + out_row + r] = sl * ut;
      }
    }
  }
  }
"""

DN_ROWS_SRC = substitute(
    DN_SRC,
    (
        "  const int out_row = threadgroup_position_in_grid.y * (NSG * RPS) + simd_gid * RPS;",
        """  const int out_row = threadgroup_position_in_grid.y * (NSG * RPS) + simd_gid * RPS;
  const int rr = threadgroup_position_in_grid.x;       // the row: its own k experts, its own order
  h += rr * NE * HID;
  inds += rr * NE;
  scores += rr * NE;
  y += rr * D;""",
    ),
)


@functools.lru_cache(None)
def _kernels():
    gu = mx.fast.metal_kernel(
        name="moe_gate_up_rows_exact",
        input_names=["xs", "xsum", "w", "scales", "biases", "inds"],
        output_names=["h"],
        header=_header(),
        source=GU_ROWS_SRC,
    )
    dn = mx.fast.metal_kernel(
        name="moe_down_rows_exact",
        input_names=["h", "w", "scales", "biases", "inds", "scores"],
        output_names=["y"],
        header=_header(),
        source=DN_ROWS_SRC,
    )
    return gu, dn


GU_TILE = (4, 2)  # (rows per simdgroup, simdgroups)
CH = 1  # rows held in registers at once
DN_TILE = (4, 2)


def supported(gate_up, down) -> bool:
    """Whether the rows kernels apply to these gate/up and down stacks: 3-bit, group 64, with
    biases, K a multiple of 512, and a row tiling (`GU_TILE`, `DN_TILE`) that divides the output
    rows."""
    return (
        gate_up.bits == 3
        and down.bits == 3
        and gate_up.group_size == 64
        and down.group_size == 64
        and gate_up.biases is not None
        and down.biases is not None
        and gate_up.in_dims % 512 == 0
        and down.in_dims % 512 == 0
        and (gate_up.out_dims // 2) % (GU_TILE[0] * GU_TILE[1]) == 0
        and down.out_dims % (DN_TILE[0] * DN_TILE[1]) == 0
    )


def experts_rows_exact(x: mx.array, gate_up, down, inds: mx.array, scores: mx.array) -> mx.array:
    """x [M, K], inds/scores [M, k] -> y [M, D]; row r is bit-equal to `expert_solo` on x[r]."""
    gu, dn = _kernels()
    M, K = x.shape
    NE = inds.shape[-1]
    hid = gate_up.out_dims // 2
    D = down.out_dims
    xs, xsum = prescale(x)
    fi = inds.reshape(-1).astype(mx.uint32)
    rps, nsg = GU_TILE
    h = gu(
        inputs=[xs, xsum, gate_up.weight, gate_up.scales, gate_up.biases, fi],
        template=[
            ("T", x.dtype),
            ("GS", 64),
            ("KK", K),
            ("HID", hid),
            ("NE", NE),
            ("NROWS", M),
            ("CH", CH),
            ("RPS", rps),
            ("NSG", nsg),
        ],
        grid=(32 * M * NE, nsg * (hid // (rps * nsg)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M * NE, hid)],
        output_dtypes=[x.dtype],
        init_value=0,
    )[0]
    rps, nsg = DN_TILE
    y = dn(
        inputs=[h, down.weight, down.scales, down.biases, fi, scores.reshape(-1)],
        template=[
            ("T", x.dtype),
            ("BITS", 3),
            ("GS", 64),
            ("HID", hid),
            ("D", D),
            ("NE", NE),
            ("RPS", rps),
            ("NSG", nsg),
        ],
        grid=(32 * M, nsg * (D // (rps * nsg)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, D)],
        output_dtypes=[x.dtype],
    )[0]
    return y
