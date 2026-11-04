"""3-bit affine GEMV kernels over M input rows, bit-identical per row to stock qmv_fast at M = 1.

A prologue pre-scales activations exactly as qmv_fast's `load_vector` does, so every product is
exact in fp32 and only the add order matters; the kernels keep qmv_fast's order and `simd_sum`.
`qmv` is the entry point; `check` verifies it against the stock op for one weight, and the model
runs it once per dense weight shape before use.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels.selfcheck import random_quantized

HEADER = """
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
"""

# The pre-scale of one 16-wide slice, shared by `PRO_SOURCE` and the fused norm kernel
# (`rms_norm_prescale`): the running sum, then the eight stores scaled as `load_vector` scales them.
PRESCALE_SUM = """    typedef float U;
    U s_ = 0;
    for (int i = 0; i < 16; i += 8) {
      s_ += xv[i] + xv[i + 1] + xv[i + 2] + xv[i + 3] + xv[i + 4] + xv[i + 5] + xv[i + 6] + xv[i + 7];
"""

PRESCALE_STORES = """      o[0] = xv[i];
      o[1] = xv[i + 1] / 8.0f;
      o[2] = xv[i + 2] / 64.0f;
      o[3] = xv[i + 3] / 2.0f;
      o[4] = xv[i + 4] / 16.0f;
      o[5] = xv[i + 5] / 128.0f;
      o[6] = xv[i + 6] / 4.0f;
      o[7] = xv[i + 7] / 32.0f;
    }
"""

PRO_SOURCE = (
    """
    const uint gid = thread_position_in_grid.x;      // over M * K/16
    const int K = x_shape[1];
    const int per = K / 16;
    if ((int)gid >= x_shape[0] * per) return;
    const device T* xv = x + gid * 16;
"""
    + PRESCALE_SUM
    + """      device float* o = xs + gid * 16 + i;
"""
    + PRESCALE_STORES
    + """    xsum[gid] = s_;
"""
)

# ---- the pieces the three multi-row GEMV sources (SOURCE_REGISTERS, SOURCE_L1 and the routed
# gate+up in moe/expert_qmv_prescaled) have in common, each the exact text those sources had
# (the sources add blank lines between pieces where they had them)
SCALE_SETUP = """    typedef float U;
    constexpr int group_size = GS;
    constexpr int block_size = 16 * 32;
    constexpr int scale_step_per_thread = group_size / 16;
"""

DIMS_MK = """    const int M = xs_shape[0];
    const int K = xs_shape[1];
"""

DIM_N = """    const int N = w_shape[0];
"""

THREAD_IDS = """    const uint lid = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
"""

DIM_KG = """    const int Kg = K / group_size;
"""

KW_BYTES = """    const int Kw = K * 3 / 8;               // bytes per weight row
"""

ROW_BASE = """    const int out_row = tid.y * (NSG * RPS) + sg * RPS;
    const int m0 = tid.x * MB;
"""


_WEIGHT_POINTERS = """    const device uint16_t* wbase = (const device uint16_t*)((const device uint8_t*)w + {row} * Kw + lid * 6);
    const int Kw16 = Kw / 2;
    const device T* sp = scales + {row} * Kg + lid / scale_step_per_thread;
    const device T* bp = biases + {row} * Kg + lid / scale_step_per_thread;
"""


def weight_pointers(row: str) -> str:
    """Metal: this lane's weight, scale and bias pointers for output row `row` (a Metal
    expression: `out_row`, or the routed kernel's `erow`)."""
    return _WEIGHT_POINTERS.format(row=row)


ACTIVATION_POINTERS = """    const device float* xp[MB];
    const device float* xsp[MB];
    #pragma unroll
    for (int m = 0; m < MB; m++) {
      xp[m] = xs + min(m0 + m, M - 1) * K + lid * 16;
      xsp[m] = xsum + min(m0 + m, M - 1) * (K / 16) + lid;
    }
"""

RESULT_INIT = """    U result[MB][RPS];
    #pragma unroll
    for (int m = 0; m < MB; m++)
      #pragma unroll
      for (int r = 0; r < RPS; r++) result[m][r] = 0;
"""

# the register-activation main loop: prefetched weights, the block loop with qmv_fast's add order;
# ends at the loop's closing brace, before the epilogue
REGISTER_LOOP = """    uint16_t wv[RPS][3];
    U sv[RPS], bv[RPS];
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) wv[r][j] = wbase[r * Kw16 + j];
      sv[r] = sp[r * Kg];
      bv[r] = bp[r * Kg];
    }

    for (int k = 0; k < K; k += block_size) {
      // ---- activations of this block, pre-scaled once by the prologue (same arithmetic)
      float4 xt4[MB][4];
      U sum[MB];
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        const device float4* xv = (const device float4*)xp[m];
        #pragma unroll
        for (int j = 0; j < 4; j++) xt4[m][j] = xv[j];
        sum[m] = xsp[m][0];
        xp[m] += block_size;
        xsp[m] += 32;
      }
#define XT(m, idx) (xt4[m][(idx) >> 2][(idx) & 3])
#if PF
      // ---- prefetch the next block's weights while this one computes
      uint16_t nwv[RPS][3];
      U nsv[RPS], nbv[RPS];
      const bool more = (k + block_size) < K;
      if (more) {
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) nwv[r][j] = wbase[r * Kw16 + block_size * 3 / 16 + j];
          nsv[r] = sp[r * Kg + block_size / group_size];
          nbv[r] = bp[r * Kg + block_size / group_size];
        }
      }
#endif

      #pragma unroll
      for (int r = 0; r < RPS; r++) {
        U acc[MB];
        #pragma unroll
        for (int m = 0; m < MB; m++) acc[m] = 0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
          // bytes 3i, 3i+1, 3i+2 of the lane's 6 (little-endian words)
          const uint w01 = wv[r][0] | (uint(wv[r][1]) << 16);
          const uint w2 = wv[r][2];
          const uint b0 = i == 0 ? (w01 & 0xff) : ((w01 >> 24) & 0xff);
          const uint b1 = i == 0 ? ((w01 >> 8) & 0xff) : (w2 & 0xff);
          const uint b2 = i == 0 ? ((w01 >> 16) & 0xff) : ((w2 >> 8) & 0xff);
          const U q0 = b0 & 0x07, q1 = b0 & 0x38, q2 = b0 & 0xc0, q3 = b1 & 0x01, q4 = b1 & 0x0e;
          const U q5 = b1 & 0x70, q6 = b1 & 0x80, q7 = b2 & 0x03, q8 = b2 & 0x1c, q9 = b2 & 0xe0;
          #pragma unroll
          for (int m = 0; m < MB; m++) {
            const int o = 8 * i;
            U a = acc[m];
            a += q0 * XT(m, o);
            a += q1 * XT(m, o + 1);
            a += q2 * XT(m, o + 2);
            a += q3 * (XT(m, o + 2) * 256.0f);
            a += q4 * XT(m, o + 3);
            a += q5 * XT(m, o + 4);
            a += q6 * XT(m, o + 5);
            a += q7 * (XT(m, o + 5) * 256.0f);
            a += q8 * XT(m, o + 6);
            a += q9 * XT(m, o + 7);
            acc[m] = a;
          }
        }
        #pragma unroll
        for (int m = 0; m < MB; m++) result[m][r] += sv[r] * acc[m] + sum[m] * bv[r];
      }

      wbase += block_size * 3 / 16;
      sp += block_size / group_size;
      bp += block_size / group_size;
#if PF
      if (more) {
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) wv[r][j] = nwv[r][j];
          sv[r] = nsv[r];
          bv[r] = nbv[r];
        }
      }
#else
      if (k + block_size < K) {
        #pragma unroll
        for (int r = 0; r < RPS; r++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) wv[r][j] = wbase[r * Kw16 + j];
          sv[r] = sp[r * Kg];
          bv[r] = bp[r * Kg];
        }
      }
#endif
    }
"""

SOURCE_REGISTERS = (
    "\n"
    + SCALE_SETUP
    + "\n"
    + DIMS_MK
    + DIM_N
    + THREAD_IDS
    + "\n"
    + KW_BYTES
    + DIM_KG
    + ROW_BASE
    + "\n"
    + weight_pointers("out_row")
    + ACTIVATION_POINTERS
    + "\n"
    + RESULT_INIT
    + "\n"
    + REGISTER_LOOP
    + "\n"
    + """    #pragma unroll
    for (int r = 0; r < RPS; r++) {
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        U v = simd_sum(result[m][r]);
        if (lid == 0 && m0 + m < M && out_row + r < N) {
          y[(m0 + m) * N + out_row + r] = static_cast<T>(v);
        }
      }
    }
"""
)


@functools.cache
def kernel():
    """The Metal kernels, built once, as a pair: the multi-row GEMV `qmv_mrow_v4` and the
    activation pre-scale `qmv_mrow_v4_pro`."""
    return (
        mx.fast.metal_kernel(
            name="qmv_mrow_v4",
            input_names=["w", "scales", "biases", "xs", "xsum"],
            output_names=["y"],
            header=HEADER,
            source=SOURCE_REGISTERS,
            ensure_row_contiguous=True,
        ),
        mx.fast.metal_kernel(
            name="qmv_mrow_v4_pro",
            input_names=["x"],
            output_names=["xs", "xsum"],
            header=HEADER,
            source=PRO_SOURCE,
            ensure_row_contiguous=True,
        ),
    )


# Single-row prescaled GEMV with the prologue computed inline: each lane pre-scales its own 16 activations
# with PRO_SOURCE's expressions (bitwise the same xs / xsum), so a call with no shared prologue is one
# launch instead of two. Only worth it for short rows, hence INLINE_MAX_K.
INLINE_MAX_K = 1024
_A_BLOCK = """      float4 xt4[MB][4];
      U sum[MB];
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        const device float4* xv = (const device float4*)xp[m];
        #pragma unroll
        for (int j = 0; j < 4; j++) xt4[m][j] = xv[j];
        sum[m] = xsp[m][0];
        xp[m] += block_size;
        xsp[m] += 32;
      }"""
_A_INLINE = """      float4 xt4[MB][4];
      U sum[MB];
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        const device T* xv = xr[m] + k;
        U s_ = 0;
        float xo[16];
        for (int i = 0; i < 16; i += 8) {
          s_ += xv[i] + xv[i + 1] + xv[i + 2] + xv[i + 3] + xv[i + 4] + xv[i + 5] + xv[i + 6] + xv[i + 7];
          xo[i] = xv[i];
          xo[i + 1] = xv[i + 1] / 8.0f;
          xo[i + 2] = xv[i + 2] / 64.0f;
          xo[i + 3] = xv[i + 3] / 2.0f;
          xo[i + 4] = xv[i + 4] / 16.0f;
          xo[i + 5] = xv[i + 5] / 128.0f;
          xo[i + 6] = xv[i + 6] / 4.0f;
          xo[i + 7] = xv[i + 7] / 32.0f;
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) xt4[m][j] = float4(xo[4 * j], xo[4 * j + 1], xo[4 * j + 2], xo[4 * j + 3]);
        sum[m] = s_;
      }"""
_P_PTRS = """    const device float* xp[MB];
    const device float* xsp[MB];
    #pragma unroll
    for (int m = 0; m < MB; m++) {
      xp[m] = xs + min(m0 + m, M - 1) * K + lid * 16;
      xsp[m] = xsum + min(m0 + m, M - 1) * (K / 16) + lid;
    }"""
_P_INLINE = """    const device T* xr[MB];
    #pragma unroll
    for (int m = 0; m < MB; m++) xr[m] = x + min(m0 + m, M - 1) * K + lid * 16;"""
assert _A_BLOCK in SOURCE_REGISTERS and _P_PTRS in SOURCE_REGISTERS
SOURCE_INLINE = (
    SOURCE_REGISTERS.replace(_A_BLOCK, _A_INLINE)
    .replace(_P_PTRS, _P_INLINE)
    .replace("const int M = xs_shape[0];", "const int M = x_shape[0];")
    .replace("const int K = xs_shape[1];", "const int K = x_shape[1];")
)


@functools.cache
def _inline_prescale_kernel():
    return mx.fast.metal_kernel(
        name="qmv_v4_inline",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE_INLINE,
        ensure_row_contiguous=True,
    )


def qmv_inline_prescale(x, w, s, b, rps: int = 4, nsg: int = 2, group_size: int = 64):
    """x [1, K] bf16 -> [1, N]: bitwise qmv_rows_in_registers(x, ..., mb=1, pf=1) without the prescale launch."""
    M, _ = x.shape
    N = w.shape[0]
    return _inline_prescale_kernel()(
        inputs=[w, s, b, x],
        template=[
            ("T", x.dtype),
            ("GS", group_size),
            ("MB", 1),
            ("RPS", rps),
            ("NSG", nsg),
            ("PF", 1),
        ],
        grid=(32 * M, nsg * ((N + nsg * rps - 1) // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


def prescale(x):
    """x [M, K] bf16 -> (xs [M, K] fp32 pre-scaled exactly as load_vector does, xsum [M, K/16])."""
    _, P = kernel()
    M, K = x.shape
    n = M * (K // 16)
    return P(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, K), (M, K // 16)],
        output_dtypes=[mx.float32, mx.float32],
    )


def qmv_rows_in_registers(
    x, w, s, b, mb: int = 2, rps: int = 4, nsg: int = 2, pf: int = 1, group_size: int = 64, pre=None
):
    """x [M, K] against 3-bit weights `w` (scales `s`, biases `b`) -> [M, N] in x's dtype: `mb`
    input rows per threadgroup, their pre-scaled activations held in registers. `pre` is
    `prescale(x)` when the caller already has it."""
    M, _ = x.shape
    N = w.shape[0]
    nb = (M + mb - 1) // mb
    rows_per_tg = nsg * rps
    K_, _ = kernel()
    xs, xsum = pre if pre is not None else prescale(x)
    return K_(
        inputs=[w, s, b, xs, xsum],
        template=[
            ("T", x.dtype),
            ("GS", group_size),
            ("MB", mb),
            ("RPS", rps),
            ("NSG", nsg),
            ("PF", pf),
        ],
        grid=(32 * nb, nsg * ((N + rows_per_tg - 1) // rows_per_tg), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


# ---- multi-row blocks reading activations from L1 instead of registers, so 8 output rows per
# simdgroup do not spill; the same per-lane adds as SOURCE_REGISTERS (bit-identical, see calibrate_prescaled)
SOURCE_L1 = (
    "\n"
    + SCALE_SETUP
    + DIMS_MK
    + DIM_N
    + THREAD_IDS
    + "    const int Kw = K * 3 / 8;\n"
    + DIM_KG
    + ROW_BASE
    + weight_pointers("out_row")
    + ACTIVATION_POINTERS
    + RESULT_INIT
    + """
    for (int k = 0; k < K; k += block_size) {
      U sum[MB];
      #pragma unroll
      for (int m = 0; m < MB; m++) sum[m] = xsp[m][0];
      #pragma unroll(RU)
      for (int r = 0; r < RPS; r++) {
        const uint16_t w0 = wbase[r * Kw16], w1 = wbase[r * Kw16 + 1], w2_ = wbase[r * Kw16 + 2];
        const U sv = sp[r * Kg], bv = bp[r * Kg];
        U acc[MB];
        #pragma unroll
        for (int m = 0; m < MB; m++) acc[m] = 0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
          const uint w01 = w0 | (uint(w1) << 16);
          const uint w2 = w2_;
          const uint b0 = i == 0 ? (w01 & 0xff) : ((w01 >> 24) & 0xff);
          const uint b1 = i == 0 ? ((w01 >> 8) & 0xff) : (w2 & 0xff);
          const uint b2 = i == 0 ? ((w01 >> 16) & 0xff) : ((w2 >> 8) & 0xff);
          const U q0 = b0 & 0x07, q1 = b0 & 0x38, q2 = b0 & 0xc0, q3 = b1 & 0x01, q4 = b1 & 0x0e;
          const U q5 = b1 & 0x70, q6 = b1 & 0x80, q7 = b2 & 0x03, q8 = b2 & 0x1c, q9 = b2 & 0xe0;
          #pragma unroll
          for (int m = 0; m < MB; m++) {
            const device float4* xv = (const device float4*)(xp[m] + 8 * i);
            const float4 xa = xv[0], xb = xv[1];
            U a = acc[m];
            a += q0 * xa.x;
            a += q1 * xa.y;
            a += q2 * xa.z;
            a += q3 * (xa.z * 256.0f);
            a += q4 * xa.w;
            a += q5 * xb.x;
            a += q6 * xb.y;
            a += q7 * (xb.y * 256.0f);
            a += q8 * xb.z;
            a += q9 * xb.w;
            acc[m] = a;
          }
        }
        #pragma unroll
        for (int m = 0; m < MB; m++) result[m][r] += sv * acc[m] + sum[m] * bv;
      }
      #pragma unroll
      for (int m = 0; m < MB; m++) { xp[m] += block_size; xsp[m] += 32; }
      wbase += block_size * 3 / 16;
      sp += block_size / group_size;
      bp += block_size / group_size;
    }
    #pragma unroll
    for (int r = 0; r < RPS; r++) {
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        U v = simd_sum(result[m][r]);
        if (lid == 0 && m0 + m < M && out_row + r < N) y[(m0 + m) * N + out_row + r] = static_cast<T>(v);
      }
    }
"""
)


@functools.cache
def _rows_in_l1_kernel():
    return mx.fast.metal_kernel(
        name="qmv_rows_v10",
        input_names=["w", "scales", "biases", "xs", "xsum"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE_L1,
        ensure_row_contiguous=True,
    )


def qmv_rows_in_l1(
    x, w, s, b, mb: int = 2, rps: int = 8, nsg: int = 2, group_size: int = 64, pre=None
):
    """`qmv_rows_in_registers` with the pre-scaled activations read from L1, so that 8 output
    rows per simdgroup (`rps`) do not spill; the same per-lane adds."""
    M, _ = x.shape
    N = w.shape[0]
    nb = (M + mb - 1) // mb
    xs, xsum = pre if pre is not None else prescale(x)
    return _rows_in_l1_kernel()(
        inputs=[w, s, b, xs, xsum],
        template=[
            ("T", x.dtype),
            ("GS", group_size),
            ("MB", mb),
            ("RPS", rps),
            ("NSG", nsg),
            ("RU", rps),
        ],
        grid=(32 * nb, nsg * ((N + nsg * rps - 1) // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


def supports(w: mx.array, s: mx.array, b: mx.array | None, group_size: int) -> bool:
    """3-bit affine weights with biases, K a multiple of 512, a group size of 32, 64 or 128."""
    k = s.shape[-1] * group_size
    return (
        w.shape[-1] * 32 // k == 3
        and b is not None
        and k % 512 == 0
        and group_size in (32, 64, 128)
    )


def qmv(x: mx.array, w: mx.array, s: mx.array, b: mx.array, group_size: int, pre=None) -> mx.array:
    """x [M, K] -> [M, N], each row bitwise stock qmv_fast on that row alone: 2 or at least 4 rows
    through `qmv_rows_in_l1` when N % 16 == 0, one row without a shared prologue and with K up to
    INLINE_MAX_K through `qmv_inline_prescale`, the others through `qmv_rows_in_registers`."""
    m = x.shape[0]
    if (m == 2 or m >= 4) and w.shape[-2] % 16 == 0:
        return qmv_rows_in_l1(x, w, s, b, mb=2, rps=8, nsg=2, group_size=group_size, pre=pre)
    if m == 1 and pre is None and s.shape[-1] * group_size <= INLINE_MAX_K:
        return qmv_inline_prescale(x, w, s, b, rps=4, nsg=2, group_size=group_size)
    mb = m if m <= 3 else 2
    return qmv_rows_in_registers(
        x, w, s, b, mb=mb, rps=4, nsg=2, pf=1, group_size=group_size, pre=pre
    )


def check(w: mx.array, s: mx.array, b: mx.array, group_size: int, probe_rows: int = 8) -> bool:
    """`qmv` on 1 to probe_rows rows bitwise per-row `mx.quantized_matmul` (batched decode runs up
    to 8 rows)."""
    k = s.shape[-1] * group_size
    x = mx.random.normal((probe_rows, k), key=mx.random.key(0xD4)).astype(mx.bfloat16)
    solo = mx.concatenate(
        [
            mx.quantized_matmul(
                x[i : i + 1],
                w,
                scales=s,
                biases=b,
                transpose=True,
                group_size=group_size,
                bits=w.shape[-1] * 32 // k,
                mode="affine",
            )
            for i in range(probe_rows)
        ],
        axis=0,
    )
    mx.eval(solo)
    for m in range(1, probe_rows + 1):
        y = qmv(x[:m], w, s, b, group_size)
        mx.eval(y)
        if not bool(mx.array_equal(y, solo[:m]).item()):
            return False
    return True


def covers(bits: int, group_size: int) -> bool:
    """Whether the prescaled GEMV takes weights of this width: 3-bit affine, groups of 32, 64 or 128."""
    return bits == 3 and group_size in (32, 64, 128)


def check_dense(bits: int, group_size: int) -> bool:
    """The prescaled GEMV on 1 to 8 rows bitwise per-row mx.quantized_matmul, for three weight shapes.

    Returns False where `covers` is False.
    """
    for n, k in ((1024, 2048), (512, 4096), (2048, 512)):
        w = random_quantized((n, k), bits, n + k, group_size)
        if not (supports(*w, group_size) and check(*w, group_size)):
            return False
    return True
