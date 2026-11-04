"""Row-blocked 4-bit dense GEMV for the decode step's dense linears, and its per-shape configs.

The kernel is MLX's `affine_qmv_wide` with RB output rows per lane; under fp32 activations it casts
the bf16 scales and biases per group in-kernel, where stock upcasts them on every call.

Each output row depends only on its own activation and weight rows, in a k-order set by KL alone
(not VECS, NSG, RB or M), so a batched row is bit-identical to the same row computed alone. KL is
therefore fixed per (N, K, dtype) in `CONFIGS`; only VECS may depend on M. The load-time check
(`check_dense`) holds rows to that bit for bit, and the kernel to `selfcheck.no_worse_than_stock`
against stock `quantized_matmul`.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc

# The dequantize helper that the dense, grouped-expert and GDN glue kernels all include in their
# headers.
DEQUANTIZE4 = """template <typename U, int N, typename W>
inline void dequantize4(const device uint8_t* w, U scale, U bias, W w_local) {
  const float s = float(scale);
  const float b = float(bias);
  float sc[2] = {s, s / 16.0f};
  for (int i = 0; i < (N / 2); i++) {
    w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
    w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
  }
}
"""

HEADER = (
    """
#include <metal_simdgroup>
using namespace metal;

// bits == 4 branch of mlx/backend/metal/kernels/quantized.h::dequantize
"""
    + DEQUANTIZE4
)

SOURCE = """
    typedef float U;
    constexpr int group_size = GS;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int slots_per_simdgroup = 32 / KL;              // row slots per simdgroup
    constexpr int rows_per_simdgroup = slots_per_simdgroup * RB;

    const int M = x_shape[0];
    const int in_vec_size = x_shape[1];
    const int out_vec_size = w_shape[0];

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;

    const short k_lane = simd_lid % KL;
    const short slot = simd_lid / KL;

    const int row0 = tid.y * (rows_per_simdgroup * NSG) + rows_per_simdgroup * simd_gid + slot * RB;
    const int vec0 = tid.x * VECS;

    const int in_vec_size_w = in_vec_size * bits / 8;
    const int in_vec_size_g = in_vec_size / group_size;

    const device uint8_t* wrow[RB];
    const device SB* srow[RB];
    const device SB* brow[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      const int row = min(row0 + r, out_vec_size - 1);
      wrow[r] = (const device uint8_t*)w + row * in_vec_size_w;
      srow[r] = scales + row * in_vec_size_g;
      brow[r] = biases + row * in_vec_size_g;
    }
    const device T* xv[VECS];
    #pragma unroll
    for (int v = 0; v < VECS; v++) {
      xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
    }

    U result[RB][VECS];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      #pragma unroll
      for (int v = 0; v < VECS; v++) result[r][v] = 0;
    }

    for (int g = k_lane; g < in_vec_size_g; g += KL) {
      U scale[RB], bias[RB];
      #pragma unroll
      for (int r = 0; r < RB; r++) { scale[r] = static_cast<U>(srow[r][g]); bias[r] = static_cast<U>(brow[r][g]); }
      #pragma unroll
      for (int sc = 0; sc < group_size / sub; sc++) {
        const int k0 = g * group_size + sc * sub;
        // activation sub-chunk, loaded once for all RB rows
        U xr[VECS][sub];
        #pragma unroll
        for (int v = 0; v < VECS; v++) {
          const device T* xc = xv[v] + k0;
          #pragma unroll
          for (int i = 0; i < sub; i++) xr[v][i] = static_cast<U>(xc[i]);
        }
        #pragma unroll
        for (int r = 0; r < RB; r++) {
          U w_dq[sub];
          dequantize4<U, sub>(wrow[r] + k0 * bits / 8, scale[r], bias[r], w_dq);
          #pragma unroll
          for (int v = 0; v < VECS; v++) {
            U acc = 0;
            #pragma unroll
            for (int i = 0; i < sub; i++) acc += xr[v][i] * w_dq[i];
            result[r][v] += acc;
          }
        }
      }
    }

    #pragma unroll
    for (int r = 0; r < RB; r++) {
      #pragma unroll
      for (int v = 0; v < VECS; v++) {
        if (KL >= 32) result[r][v] += simd_shuffle_down(result[r][v], 16);
        if (KL >= 16) result[r][v] += simd_shuffle_down(result[r][v], 8);
        if (KL >= 8)  result[r][v] += simd_shuffle_down(result[r][v], 4);
        if (KL >= 4)  result[r][v] += simd_shuffle_down(result[r][v], 2);
        if (KL >= 2)  result[r][v] += simd_shuffle_down(result[r][v], 1);
      }
    }

    if (k_lane == 0) {
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const int out_row = row0 + r;
        if (out_row < out_vec_size) {
          #pragma unroll
          for (int v = 0; v < VECS; v++) {
            if (vec0 + v < M) y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[r][v]);
          }
        }
      }
    }
"""


@functools.cache
def kernel():
    """The compiled `moe_engine_affine_qmv_wide_rb` kernel, built once."""
    return mx.fast.metal_kernel(
        name="moe_engine_affine_qmv_wide_rb",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def row_blocked_template(
    dtype, sb_dtype, vecs: int, k_lanes: int, nsg: int, rb: int, group_size: int
):
    """The `template` argument of a launch of the row-blocked kernel family (dense and grouped):
    activation dtype `dtype`, scale and bias dtype `sb_dtype`, and the shape parameters."""
    return [
        ("T", dtype),
        ("SB", sb_dtype),
        ("VECS", vecs),
        ("KL", k_lanes),
        ("NSG", nsg),
        ("RB", rb),
        ("GS", group_size),
    ]


def qmv_wide_rb(x, w, s, b, vecs: int, k_lanes: int, nsg: int, rb: int, group_size: int):
    """y[M, N] = x[M, K] @ dequant(w)[N, K]^T. `x` must be 2-D and row-contiguous."""
    M, _K = x.shape
    N = w.shape[0]
    rows_per_tg = (32 // k_lanes) * rb * nsg
    n_tiles = (M + vecs - 1) // vecs
    grid = (32 * n_tiles, nsg * ((N + rows_per_tg - 1) // rows_per_tg), 1)
    return kernel()(
        inputs=[w, s, b, x],
        template=row_blocked_template(x.dtype, s.dtype, vecs, k_lanes, nsg, rb, group_size),
        grid=grid,
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]


# --------------------------------------------------------------------------- engine-side policy

DENSE_KERNELS = ("rb", "stock")
DENSE_KERNEL_DEFAULT = "rb"

# Above this many rows (prompt processing) the shape goes to stock's tiled GEMM. `QLinear.rows()`
# splits a taller independent-row stack into groups of MAX_M, so the cap never changes row values.
MAX_M = 8

# (out_dims N, in_dims K, activation dtype) -> (KL, NSG, RB). One entry per shape and dtype: KL sets
# the k-order, so it is fixed here and never chosen from M.
CONFIGS: dict = {
    # --- float32: the default compute dtype.
    (10240, 2560, "float32"): (16, 4, 2),  # linear_attn.in_proj_qkv        x36/step
    (16384, 2560, "float32"): (16, 4, 2),  # in_proj_qkv + in_proj_z stacked (same KL -> same rows)
    (248320, 2560, "float32"): (16, 4, 2),  # lm_head                        x1
    (6144, 2560, "float32"): (16, 4, 2),  # linear_attn.in_proj_z          x36
    (2560, 6144, "float32"): (16, 4, 2),  # linear_attn.out_proj + self_attn.o_proj  x48
    (12288, 2560, "float32"): (16, 4, 2),  # self_attn.q_proj               x12
    (320, 10240, "float32"): (16, 2, 2),  # hc input_mix_weight_down (+ the mixer's)  x49
    (10240, 320, "float32"): (4, 4, 2),  # hc input_mix_weight_up   (+ the mixer's)  x49
    (4, 10240, "float32"): (32, 4, 1),  # hc block_inject_weight   x96/step
    # --- bfloat16: served at every M even where stock is faster at M = 1, because a shape served
    # at M = 8 but not at M = 1 would lose row exactness.
    (10240, 2560, "bfloat16"): (8, 4, 2),
    (248320, 2560, "bfloat16"): (8, 2, 2),
    (6144, 2560, "bfloat16"): (8, 4, 2),
    (2560, 6144, "bfloat16"): (8, 1, 2),
    (12288, 2560, "bfloat16"): (8, 4, 2),
    (320, 10240, "bfloat16"): (16, 4, 2),
    (10240, 320, "bfloat16"): (4, 2, 2),
}

NOISE_FLOOR_BASIS = (
    "same-shape spread in the sweep is 6.3-12.6% at M=1 (8.5% for "
    "2560x6144), so per-class differences below ~10% at M=1 are not resolvable "
    "there; entries were picked pooled-by-shape and count-weighted"
)

CONFIG_BASIS = (
    "Picked from a full (VECS, KL, NSG, RB) sweep measured on the M4 Pro, with one "
    "fixed KL per (shape, dtype). Against the best config per row count it costs +0.0 "
    "to +12.0% at M=1 (mostly inside the noise floor below) and +0.0% at M=8 on every "
    "shape but one (+3.8%). " + NOISE_FLOOR_BASIS
)

_ACT_DTYPES = {mx.bfloat16: "bfloat16", mx.float16: "float16", mx.float32: "float32"}


def dtype_name(dt) -> str:
    """The name of activation dtype `dt` as `CONFIGS` keys it: "bfloat16", "float16" or
    "float32"."""
    return _ACT_DTYPES.get(dt, str(dt).split(".")[-1])


def vecs_for(m: int, dt) -> int:
    """Activation vectors per threadgroup; outside the k-order, so it is free to depend on M."""
    if dtype_name(dt) == "float32":
        return 1 if m <= 1 else 2
    return max(1, min(m, 4))


def config_for(out_dims: int, in_dims: int, dt):
    """(KL, NSG, RB) for this shape and activation dtype, or None if the table does not serve it."""
    return CONFIGS.get((int(out_dims), int(in_dims), dtype_name(dt)))


def supports(
    bits: int, group_size: int, in_dims: int, has_biases: bool, scales_dtype, act_dtype
) -> bool:
    """Whether the vendored kernel can compute this linear at all (else: stock)."""
    return (
        int(bits) == 4
        and int(group_size) >= 8
        and int(group_size) % 8 == 0
        and int(in_dims) % int(group_size) == 0
        and bool(has_biases)
        and scales_dtype in _ACT_DTYPES
        and act_dtype in _ACT_DTYPES
    )


def served(shapes, act_dtype) -> list:
    """The (out, in, bits, group_size) of `shapes` the table serves at `act_dtype`, each once."""
    return [
        (n, k, bits, gs)
        for n, k, bits, gs in sorted(set(shapes))
        if config_for(n, k, act_dtype) is not None
        and supports(bits, gs, k, True, mx.bfloat16, act_dtype)
    ]


def check_dense(shapes, act_dtype) -> bool:
    """The load-time check, at each (out, in, bits, group_size) of `shapes` the table serves at
    `act_dtype`: a row computed with up to MAX_M - 1 others, read at an offset into its batch, is
    the row alone bit for bit (the claim above), at row counts on each side of every change of VECS,
    and the kernel is no less accurate than the stock `quantized_matmul` it replaces, as
    `selfcheck.no_worse_than_stock` measures it. Rows are independent and N is a runtime value, not
    a template constant, so each shape runs on its first two and a half row tiles."""
    for n, k, bits, gs in served(shapes, act_dtype):
        cfg = config_for(n, k, act_dtype)
        rows_per_tg = (32 // cfg[0]) * cfg[2] * cfg[1]
        lin = sc.Linear(min(n, 2 * rows_per_tg + rows_per_tg // 2 + 1), k, n + k, gs, bits)
        if not _check_rows(lin, cfg, act_dtype):
            return False
    return True


def _check_rows(lin, cfg, act_dtype) -> bool:
    def run(xs):
        vecs = vecs_for(xs.shape[0], act_dtype)
        return qmv_wide_rb(xs, lin.weight, lin.scales, lin.biases, vecs, *cfg, lin.group_size)

    x = sc.normal((MAX_M + 1, lin.in_dims), lin.out_dims, act_dtype)[1:]
    solo = mx.concatenate([run(x[i : i + 1]) for i in range(MAX_M)])
    for m in (2, 3, 4, 5, MAX_M):
        if not mx.array_equal(run(x[:m]), solo[:m]).item():
            return False
    return all(
        sc.no_worse_than_stock(run(x[:m]), lin.stock(x[:m]), lin.exact(sc.f64(x[:m])))
        for m in (1, MAX_M)
    )
