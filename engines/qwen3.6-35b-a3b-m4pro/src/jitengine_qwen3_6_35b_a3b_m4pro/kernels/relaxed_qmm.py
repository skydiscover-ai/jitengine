"""Skinny-M quantized matmul on simdgroup matrices, for the relaxed variant's verify windows only.

Not row-exact: the per-row reduction order differs from qmv_fast's, so results differ from one-row decode at the
bf16 tail-ulp level. Never used by the exact engine.

Y[M, N] = X[M, K] @ W[N, K]^T for 4-bit affine weights (group 64), M <= 8. A threadgroup owns 8 output rows and its
NSG simdgroups split K. Per 8-wide k chunk a simdgroup multiplies an 8x8 tile of raw nibbles (as floats) by an 8x8
tile of X^T (the M rows, zero-padded to 8); after each 64-wide group the tile is folded with the group's scale and
bias: y += s * sum(q * x) + b * sum(x), with sum(x) per (row, group) from a small prologue kernel.
"""

from __future__ import annotations

import functools

import mlx.core as mx

_XS_SRC = """
    // xs[m, g] = sum of x[m, g*64 .. g*64+63] in fp32
    const uint i = thread_position_in_grid.x;       // m * G + g
    const int G = K / 64;
    if ((int)i >= M * G) return;
    const int m = i / G, g = i % G;
    const device T* p = x + (size_t)m * K + g * 64;
    float acc = 0.0f;
    for (int j = 0; j < 64; j++) acc += float(p[j]);
    xs[i] = acc;
"""

_QMM_SRC = """
    constexpr int G = K / 64;
    constexpr int GPS = G / NSG;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    // padded rows (>= M) read row M-1 and are never stored: no divergent branches
    const int m0 = min((int)fn, M - 1), m1 = min((int)fn + 1, M - 1);
    threadgroup float part[NSG * 64];
    for (int t = 0; t < CT; t++) {
      const int o = (threadgroup_position_in_grid.y * CT + t) * 8 + fm;
      const device uint32_t* wrow = w + (size_t)o * (K / 8);
      float acc0 = 0.0f, acc1 = 0.0f;
      for (int gi = 0; gi < GPS; gi++) {
        const int g = sg * GPS + gi;
        const device uint4* wv = (const device uint4*)(wrow + g * 8);
        const uint4 wa = wv[0], wb = wv[1];
        const uint words[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        simdgroup_float8x8 C = simdgroup_float8x8(0);
        #pragma unroll
        for (int c = 0; c < 8; c++) {
          const int kk = g * 64 + c * 8 + fm;
          simdgroup_float8x8 A, B;
          A.thread_elements()[0] = float((words[c] >> (4 * fn)) & 0xfu);
          A.thread_elements()[1] = float((words[c] >> (4 * (fn + 1))) & 0xfu);
          B.thread_elements()[0] = float(x[(size_t)m0 * K + kk]);
          B.thread_elements()[1] = float(x[(size_t)m1 * K + kk]);
          simdgroup_multiply_accumulate(C, A, B, C);
        }
        const float s_ = float(scales[(size_t)o * G + g]);
        const float b_ = float(biases[(size_t)o * G + g]);
        acc0 += s_ * C.thread_elements()[0] + b_ * xs[m0 * G + g];
        acc1 += s_ * C.thread_elements()[1] + b_ * xs[m1 * G + g];
      }
      if (NSG > 1) {
        part[sg * 64 + fm * 8 + fn] = acc0;
        part[sg * 64 + fm * 8 + fn + 1] = acc1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (sg == 0) {
        float r0 = acc0, r1 = acc1;
        for (int s2 = 1; s2 < NSG; s2++) { r0 += part[s2 * 64 + fm * 8 + fn]; r1 += part[s2 * 64 + fm * 8 + fn + 1]; }
        if (fn < M) y[(size_t)fn * N + o] = static_cast<T>(r0);
        if (fn + 1 < M) y[(size_t)(fn + 1) * N + o] = static_cast<T>(r1);
      }
      if (NSG > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@functools.cache
def _kern(name, src, inputs, outputs):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src)


def group_sums(x: mx.array) -> mx.array:
    """The sums of each 64-wide group of the rows of x [M, K], float32 [M, K / 64], that `qmm` reads."""
    M, K = x.shape
    return _kern("rq_xs", _XS_SRC, ("x",), ("xs",))(
        inputs=[x],
        template=[("T", x.dtype), ("K", K), ("M", M)],
        grid=(max(32, M * (K // 64)), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, K // 64)],
        output_dtypes=[mx.float32],
    )[0]


def geometry(N: int, K: int):
    """(simdgroups per threadgroup, `ct`: the blocks of 8 output columns each threadgroup covers) that
    `qmm` launches with for an [N, K] weight."""
    G = K // 64
    nsg = 2 if G % 2 == 0 else 1
    ct = 4 if N >= 65536 else 1
    return nsg, ct


MIN_ROWS = 6  # below this MLX's own small-M kernels are faster
MIN_N = 2048


def rows(x2: mx.array, w, scales, biases, bits: int, group_size: int):
    """Project x2 with this kernel in blocks of <= 8 rows where it applies, else return None."""
    M, K = x2.shape
    N = w.shape[0]
    if (
        M < MIN_ROWS
        or N < MIN_N
        or biases is None
        or not supported(min(M, 8), N, K, bits, group_size)
    ):
        return None
    if M <= 8:
        return qmm(x2, w, scales, biases)
    return mx.concatenate([qmm(x2[c : c + 8], w, scales, biases) for c in range(0, M, 8)], axis=0)


def supported(M: int, N: int, K: int, bits: int, group_size: int) -> bool:
    """Whether `qmm` takes these shapes: 4-bit group-64 weights, 1 to 8 rows, K a multiple of 64 and N
    a multiple of 8 x `ct` (`geometry`)."""
    _nsg, ct = geometry(N, K)
    return bits == 4 and group_size == 64 and 1 <= M <= 8 and K % 64 == 0 and N % (8 * ct) == 0


def qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    xs: mx.array | None = None,
    nsg: int | None = None,
    ct: int | None = None,
) -> mx.array:
    """x [M, K] (M <= 8), 4-bit affine g64 weights [N, K/8] -> [M, N] in x's dtype (relaxed numerics)."""
    M, K = x.shape
    N = w.shape[0]
    g_nsg, g_ct = geometry(N, K)
    nsg, ct = nsg or g_nsg, ct or g_ct
    xs = group_sums(x) if xs is None else xs
    return _kern("rq_qmm", _QMM_SRC, ("x", "w", "scales", "biases", "xs"), ("y",))(
        inputs=[x, w, scales, biases, xs],
        template=[("T", x.dtype), ("K", K), ("N", N), ("M", M), ("NSG", nsg), ("CT", ct)],
        grid=(32 * nsg, N // (8 * ct), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
