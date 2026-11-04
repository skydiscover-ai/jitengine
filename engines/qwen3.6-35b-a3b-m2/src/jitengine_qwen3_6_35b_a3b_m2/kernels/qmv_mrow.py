"""Input-row-blocked `qmv_fast`: MB input rows per threadgroup share each weight read.

Bit-identical to stock `qmv_fast` at any M: each output keeps MLX's sequential k-block accumulation
and 32-lane `simd_sum`, and each output row depends only on its own input row. `load_vector` and
`qdot` are verbatim from MLX's `quantized.h`, 3-bit pointer quirk included; keep them that way.
"""

from __future__ import annotations

import functools

import mlx.core as mx

HEADER = """
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constant constexpr int SIMD_SIZE = 32;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// Verbatim from kernel_lab/mlx_src/quantized.h::load_vector (the 3, 4 and 8 bit branches).
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;

  if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

// kernel_lab/mlx_src/quantized.h::qdot, with the weight bytes already staged in thread registers. Same
// values, same accumulation order, same final `scale * accum + sum * bias`.
template <typename U, int values_per_thread, int bits>
inline U qdot_regs(
    const thread uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  U accum = 0;

  if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    // Stock reads this as uint16_t; the staged bytes are little-endian the same way, so the
    // masked values -- and therefore every product and every partial sum -- are identical.
    for (int i = 0; i < (values_per_thread / 4); i++) {
      const uint32_t wsi = (uint32_t)w[2 * i] | ((uint32_t)w[2 * i + 1] << 8);
      accum +=
          (x_thread[4 * i] * (wsi & 0x000f) +
           x_thread[4 * i + 1] * (wsi & 0x00f0) +
           x_thread[4 * i + 2] * (wsi & 0x0f00) +
           x_thread[4 * i + 3] * (wsi & 0xf000));
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}
"""

# x [M, K], w [N, K*bytes_per_pack/pack_factor] uint32, s/b [N, K/GS]; *_shape is auto-injected.
SOURCE = """
    typedef float U;
    constexpr int group_size = GS;
    constexpr int bits = BITS;
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = NSG;
    constexpr int results_per_simdgroup = RPS;
    constexpr int pack_factor = get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    constexpr int bytes_per_thread = values_per_thread * bytes_per_pack / pack_factor;

    const int M = x_shape[0];
    const int in_vec_size = x_shape[1];
    const int out_vec_size = w_shape[0];

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;

    const device uint8_t* ws = (const device uint8_t*)w;

    thread U x_thread[MB][values_per_thread];
    thread U sum[MB];
    thread U result[MB][results_per_simdgroup];
    #pragma unroll
    for (int m = 0; m < MB; m++) {
      #pragma unroll
      for (int row = 0; row < results_per_simdgroup; row++) result[m][row] = 0;
    }

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    // Identical pointer arithmetic to stock, except that tid.x now selects a BLOCK of MB input
    // rows instead of one. Out-of-range rows are clamped to M-1 (read, never written), so a
    // ragged tail costs a duplicate read and nothing else.
    const int m0 = tid.x * MB;
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    const device T* xp[MB];
    #pragma unroll
    for (int m = 0; m < MB; m++) {
      xp[m] = x + min(m0 + m, M - 1) * in_vec_size + simd_lid * values_per_thread;
    }

    for (int k = 0; k < in_vec_size; k += block_size) {
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        sum[m] = load_vector<T, U, values_per_thread, bits>(xp[m], x_thread[m]);
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        // THE CHANGE: the weight, scale and bias of this output row are read ONCE per k-block and
        // applied to all MB input rows, instead of once per input row per k-block.
        const device uint8_t* wl = ws + row * in_vec_size_w;
        thread uint8_t wb[bytes_per_thread];
        #pragma unroll
        for (int i = 0; i < bytes_per_thread; i++) wb[i] = wl[i];

        U s = scales[row * in_vec_size_g];
        U b = biases[row * in_vec_size_g];

        #pragma unroll
        for (int m = 0; m < MB; m++) {
          result[m][row] += qdot_regs<U, values_per_thread, bits>(wb, x_thread[m], s, b, sum[m]);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      #pragma unroll
      for (int m = 0; m < MB; m++) xp[m] += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      #pragma unroll
      for (int m = 0; m < MB; m++) {
        result[m][row] = simd_sum(result[m][row]);
        if (simd_lid == 0 && m0 + m < M && out_row + row < out_vec_size) {
          y[(m0 + m) * out_vec_size + out_row + row] = static_cast<T>(result[m][row]);
        }
      }
    }
"""


@functools.cache
def kernel():
    """The compiled `affine_qmv_mrow` Metal kernel, built once."""
    return mx.fast.metal_kernel(
        name="affine_qmv_mrow",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def qmv_fast_mrow(
    x, w, s, b, mb: int = 2, group_size: int = 64, bits: int = 3, rps: int = 4, nsg: int = 2
):
    """Multi-input-row affine GEMV; returns [M, N] in x's dtype.

    Requires K % (values_per_thread * 32) == 0 and K % group_size == 0, like stock `qmv_fast`;
    callers fall back to `mx.quantized_matmul` otherwise.
    """
    M, _K = x.shape
    N = w.shape[0]
    n_blocks = (M + mb - 1) // mb
    rows_per_tg = nsg * rps
    grid = (32 * n_blocks, nsg * ((N + rows_per_tg - 1) // rows_per_tg), 1)
    return kernel()(
        inputs=[w, s, b, x],
        template=[
            ("T", x.dtype),
            ("GS", group_size),
            ("BITS", bits),
            ("MB", mb),
            ("RPS", rps),
            ("NSG", nsg),
        ],
        grid=grid,
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
