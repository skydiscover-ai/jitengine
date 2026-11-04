"""Prefill's routed experts: MLX's gather_qmm_rhs arithmetic with segment-aligned tiles.

Each threadgroup owns up to `bm` sorted rows of a single expert and runs MLX's own tile GEMM once
(device templates inlined from the installed headers). MMA rows are independent, so every element
is bitwise what mx.gather_qmm computes on the sorted rows.
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels import constants
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.source import substitute

KDIR = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
ORDER = [
    "steel/defines.h",
    "steel/utils/type_traits.h",
    "steel/utils/integral_constant.h",
    "steel/utils.h",
    "steel/gemm/transforms.h",
    "steel/gemm/loader.h",
    "steel/gemm/mma.h",
    "quantized_utils.h",
    "quantized.h",
]


def header():
    """MLX's steel GEMM and quantized headers (`ORDER`) from the installed `mlx`, joined into one
    Metal header with their `#include` lines dropped and `align_M`, `align_N` and `align_K` made
    constants."""
    out = []
    for f in ORDER:
        src = open(os.path.join(KDIR, f)).read()
        src = "\n".join(
            line
            for line in src.splitlines()
            if not line.strip().startswith("#include") and "#pragma once" not in line
        )
        if f == "quantized.h":
            src = substitute(
                src,
                (
                    "constant bool align_M [[function_constant(200)]];",
                    "constant constexpr bool align_M = false;",
                ),
                (
                    "constant bool align_N [[function_constant(201)]];",
                    "constant constexpr bool align_N = true;",
                ),
                (
                    "constant bool align_K [[function_constant(202)]];",
                    "constant constexpr bool align_K = true;",
                ),
            )
        out.append(f"// ---- {f}\n" + src)
    return (
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\nusing namespace metal;\n"
        + "\n".join(out)
    )


BODY = """
  // template: T, GS, BITS; fixed bm=16 bn=32 bk=32 wm=1 wn=2 transpose=true (the stock dispatch)
  constexpr int BM = TBM, BN = TBN, BK = TBK, WM = TWM, WN = TWN;
  constexpr int bits = BITS, group_size = GS;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true, WM * WN * SIMD_SIZE, group_size, bits>;
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const int M = Mtot[0];
  const int N = Nd[0];
  const int K = Kd[0];
  const int tile = tid.y;
  if (tile >= ntiles[0]) return;
  const int y_row = tiles[2 * tile];
  const int nrows = tiles[2 * tile + 1];
  const int y_col = tid.x * BN;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = (size_t)N * K_w;
  const size_t stride_s = (size_t)N * K_g;
  const short tgp_bm = short(min(BM, M - y_row));
  const short tgp_bn = BN;
  auto wl = (const device uint8_t*)w;
  const device T* xp = x + (size_t)y_row * K;
  device T* yp = y + (size_t)y_row * N + y_col;
  wl += (size_t)y_col * K_w;
  const device T* sc = scales + (size_t)y_col * K_g;
  const device T* bi = biases + (size_t)y_col * K_g;
  const uint32_t index = indices[y_row];
  // MIXED: a piece of <= BM/4 or <= BM/2 rows runs a BM/4- or BM/2-row MMA and x loader (same
  // threadgroup shape, weight loader and K loop): the padding rows a full-height MMA would compute
  // are skipped. MMA rows are independent, so each stored element is the same arithmetic.
#define SEG_PART(BMH)                                                                                  \
  {                                                                                                    \
    using mma_h = mlx::steel::BlockMMA<T, T, BMH, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;  \
    using loader_xh = mlx::steel::BlockLoader<T, BMH, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;          \
    thread mma_h mma_op(simd_group_id, simd_lane_id);                                                  \
    thread loader_xh loader_x(xp, K, Xs, simd_group_id, simd_lane_id);                                 \
    thread loader_w_t loader_w(wl + index * stride_w, sc + index * stride_s, bi + index * stride_s, K, Ws, \
                               simd_group_id, simd_lane_id);                                           \
    const short hb = short(min(BMH, M - y_row));                                                       \
    if (hb == BMH) {                                                                                   \
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);                                     \
    } else {                                                                                           \
      gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, hb, tgp_bn, BK); \
    }                                                                                                  \
    mma_op.store_result_slice(yp, N, short2(0, 0), short2(BN, nrows));                                 \
    return;                                                                                            \
  }
  // a tier exists when its height is a whole number of 8-row fragments per simdgroup row (WM)
#define SEG_OK(D) (BM % (D * 8 * WM) == 0)
  if (MIXED && SEG_OK(8) && nrows <= BM / 8) SEG_PART((SEG_OK(8) ? BM / 8 : BM))
  if (MIXED && SEG_OK(4) && nrows <= BM / 4) SEG_PART((SEG_OK(4) ? BM / 4 : BM))
  if (MIXED && SEG_OK(2) && nrows <= BM / 2) SEG_PART((SEG_OK(2) ? BM / 2 : BM))
  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xp, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl + index * stride_w, sc + index * stride_s, bi + index * stride_s, K, Ws,
                             simd_group_id, simd_lane_id);
  if (tgp_bm == BM) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
  } else {
    gemm_loop_unaligned<false, true, true>(Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
  }
  mma_op.store_result_slice(yp, N, short2(0, 0), short2(BN, nrows));
"""

TILES_SRC = """
  // one thread: split the sorted rows into pieces of <= 16 rows of one expert
  int n = 0;
  int start = 0;
  const int P = Ptot[0];
  for (int r = 1; r <= P; r++) {
    if (r == P || indices[r] != indices[start] || r - start == TBM) {
      tiles[2 * n] = start;
      tiles[2 * n + 1] = r - start;
      n++;
      start = r;
    }
  }
  ntiles[0] = n;
"""

# one simdgroup per 16x32 tile; the tile shape does not change the arithmetic
CFG = dict(bm=16, bn=32, bk=32, wm=1, wn=1)


# taller tiles when the gather averages many rows per expert (a merged multi-prompt prefill or a
# long prompt), amortising each dequantised weight over more MMA rows; same arithmetic per element
CFG_TALL = dict(bm=32, bn=64, bk=32, wm=1, wn=2)
TALL_MIN_ROWS_PER_EXPERT = 64
# MIXED: one CFG_TALL kernel whose pieces of <= 8 / <= 16 rows run 8- / 16-row MMAs (see BODY);
# bitwise equal to the fixed tiles.
MIXED = True
# With >= 64 rows per expert on average, a 64-row top over the same 1x2 simdgroups with 32/16/8-row
# tiers for the remainders.
CFG_TALL64 = dict(bm=64, bn=64, bk=32, wm=1, wn=2)


@functools.cache
def _kernels():
    """(the segmented GEMM, the tile builder), compiled once."""
    h = header()
    gemm = mx.fast.metal_kernel(
        name="seg_gather_qmm",
        input_names=[
            "x",
            "w",
            "scales",
            "biases",
            "indices",
            "tiles",
            "ntiles",
            "Mtot",
            "Nd",
            "Kd",
        ],
        output_names=["y"],
        header=h,
        source=BODY,
        ensure_row_contiguous=True,
    )
    tiles = mx.fast.metal_kernel(
        name="seg_tiles",
        input_names=["indices", "Ptot"],
        output_names=["tiles", "ntiles"],
        source=TILES_SRC,
    )
    return gemm, tiles


def seg_gather_qmm(x, stack, idx, cfg=None):
    """x [P, K] rows sorted by expert idx [P]; `stack` a quantized expert stack [E, N, K]
    (`weight`, `scales`, `biases`, `group_size`, `bits`, `expert_views`) -> y [P, N]
    (transpose=True)."""
    E, N = stack.weight.shape[:2]
    mixed = MIXED and (cfg is None or bool(cfg.get("mixed")))
    if cfg is None:
        dense = x.shape[0] >= TALL_MIN_ROWS_PER_EXPERT * E
        cfg = (CFG_TALL64 if dense else CFG_TALL) if MIXED else (CFG_TALL if dense else CFG)
    P, K = x.shape
    bm, bn, bk, wm, wn = cfg["bm"], cfg["bn"], cfg["bk"], cfg["wm"], cfg["wn"]
    max_tiles = (P + bm - 1) // bm + E
    gemm, tiles_kernel = _kernels()
    pairs = mx.array([P], dtype=mx.int32)  # the one size that changes per call
    tiles, ntiles = tiles_kernel(
        inputs=[idx, pairs],
        template=[("TBM", bm)],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(2 * max_tiles,), (1,)],
        output_dtypes=[mx.int32, mx.int32],
    )
    wa, sa, ba = stack.expert_views()
    return gemm(
        inputs=[
            x,
            wa,
            sa,
            ba,
            idx,
            tiles,
            ntiles,
            pairs,
            constants.i32(N),
            constants.i32(K),
        ],
        template=[
            ("T", x.dtype),
            ("GS", stack.group_size),
            ("BITS", stack.bits),
            ("TBM", bm),
            ("TBN", bn),
            ("TBK", bk),
            ("TWM", wm),
            ("TWN", wn),
            ("MIXED", mixed),
        ],
        grid=(32 * wn * (N // bn), wm * max_tiles, 1),
        threadgroup=(32 * wn, wm, 1),
        output_shapes=[(P, N)],
        output_dtypes=[x.dtype],
    )[0]
