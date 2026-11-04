"""Prefill dense projections: MLX's `affine_qmm_t` arithmetic with larger tiles than stock.

MMA rows and columns are independent and the K order is unchanged, so on `eligible()` shapes the
output is bit-identical to `mx.quantized_matmul`; `linear.QLinear.calibrate_pf` re-verifies each
weight shape at first use.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kernels import constants
from jitengine_qwen3_6_35b_a3b_m2.kernels.gather_qmm_prefill import header

BODY = """
  constexpr int BM = TBM, BN = TBN, BK = 32, WM = TWM, WN = TWN;
  constexpr int bits = BITS, group_size = GS;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int M = Md[0], N = Nd[0], K = Kd[0];
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  const device T* xp = x + y_row * (int64_t)K;
  wl += y_col * K_w;
  const device T* sc = scales + y_col * K_g;
  const device T* bi = biases + y_col * K_g;
  device T* yp = y + y_row * (int64_t)N + y_col;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(xp, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, sc, bi, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(yp, N, short2(BN, num_els));
  } else {
    mma_op.store_result(yp, N);
  }
"""

MIN_ROWS = 192  # below this stock's 32-row tiles waste less than a 128-row tile does
SPLITK_BOUND = 1 << 18  # stock uses split-K (a different reduction order) when M * N <= this


def tile_for(m: int) -> tuple:
    """The (BM, BN, WM, WN) tile for `m` rows: 128 x 64 when `m % 128` is 0 or above 64,
    otherwise 64 x 32."""
    r = m % 128
    return (128, 64, 4, 1) if (r == 0 or r > 64) else (64, 32, 2, 1)


def eligible(m: int, n: int, k: int, bits: int) -> bool:
    """Whether `qmm_t` applies to `m` rows of an `n` x `k` weight of `bits` bits: at least
    `MIN_ROWS` rows, `m * n` above `SPLITK_BOUND` (below it stock splits K), `n` a multiple of
    64, `k` of 32, and 3-bit."""
    return m >= MIN_ROWS and m * n > SPLITK_BOUND and n % 64 == 0 and k % 32 == 0 and bits == 3


@functools.cache
def _qmm_t_kernel():
    return mx.fast.metal_kernel(
        name="dense_prefill_qmm_t",
        input_names=["x", "w", "scales", "biases", "Md", "Nd", "Kd"],
        output_names=["y"],
        header=header(),
        source=BODY,
        ensure_row_contiguous=True,
    )


def qmm_t(x, w, scales, biases, bits: int, group_size: int):
    """x [M, K] bf16 -> [M, N]; same bits as mx.quantized_matmul(transpose=True) where eligible()."""
    M, K = x.shape
    N = w.shape[0]
    bm, bn, wm, wn = tile_for(M)
    return _qmm_t_kernel()(
        inputs=[
            x,
            w,
            scales,
            biases,
            mx.array([M], mx.int32),
            constants.i32(N),
            constants.i32(K),
        ],
        template=[
            ("T", x.dtype),
            ("GS", group_size),
            ("BITS", bits),
            ("TBM", bm),
            ("TBN", bn),
            ("TWM", wm),
            ("TWN", wn),
        ],
        grid=(32 * wn * (N // bn), wm * ((M + bm - 1) // bm), 1),
        threadgroup=(32 * wn, wm, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
