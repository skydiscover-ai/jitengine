"""Bit-exact fused single-token MoE expert kernels: gate+up with SwiGLU, and down with combine.

Bit-identity rests on MLX's own `load_vector`/`qdot` (read from the installed `quantized.h`), on
`qmv_fast`'s per-row arithmetic, and on combining experts as `mx.sum` does: each product rounded
to bf16, then a sequential bf16 sum in ascending expert order.
"""

from __future__ import annotations

import functools
import os
from typing import NamedTuple

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels.selfcheck import random_quantized

BITS = (3, 4, 8)
_TILE = {
    3: ((4, 2), (4, 2)),
    4: ((2, 2), (4, 2)),
    8: ((2, 2), (4, 2)),
}  # (gate_up, down): (rows/simdgroup, simdgroups)


def _mlx_quantized_h() -> str:
    base = os.path.join(
        os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels", "quantized.h"
    )
    with open(base) as f:
        return f.read()


@functools.lru_cache(None)
def _header() -> str:
    src = _mlx_quantized_h()

    def grab(pat):
        i = src.index(pat)
        j = src.index("\n}\n", i) + 3
        return src[src.rfind("template", 0, i) : j]

    head = src[: src.index("template <typename T, typename U, int values_per_thread, int bits>")]
    head = "\n".join(
        line
        for line in head.splitlines()
        if not line.startswith("#include") and "function_constant" not in line
    )
    return (
        head
        + "\n"
        + grab("inline U load_vector(")
        + "\n"
        + grab("inline U qdot(")
        + """
struct SigmoidX {   // mlx unary_ops.h Sigmoid, verbatim
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""
    )


GU_SRC = """
  // one row x [K]; experts inds[NE]; W [E, 2*HID, K*bits/32]; out h [NE, HID] = swiglu(gate, up)
  constexpr int bits = BITS; constexpr int group_size = GS;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;
  const int K = in_vec_size;
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const uint j = threadgroup_position_in_grid.x;
  const uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
  const int e = inds[j];
  const int out_row = threadgroup_position_in_grid.y * (NSG * RPS) + simd_gid * RPS;
  const device uint8_t* ws = (const device uint8_t*)w + (size_t)e * (2 * HID) * in_vec_size_w
      + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + (size_t)e * (2 * HID) * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bi = biases + (size_t)e * (2 * HID) * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xp = x + simd_lid * values_per_thread;
  thread U x_thread[values_per_thread];
  thread U rg[RPS] = {0};
  thread U ru[RPS] = {0};
  const size_t up_w = (size_t)HID * in_vec_size_w, up_g = (size_t)HID * in_vec_size_g;
  for (int k = 0; k < K; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(xp, x_thread);
    for (int row = 0; row < RPS; row++) {
      rg[row] += qdot<U, values_per_thread, bits>(ws + row * in_vec_size_w, x_thread,
                    (U)sc[row * in_vec_size_g], (U)bi[row * in_vec_size_g], sum);
      ru[row] += qdot<U, values_per_thread, bits>(ws + up_w + row * in_vec_size_w, x_thread,
                    (U)sc[up_g + row * in_vec_size_g], (U)bi[up_g + row * in_vec_size_g], sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xp += block_size;
  }
  for (int row = 0; row < RPS; row++) {
    U g = simd_sum(rg[row]);
    U u = simd_sum(ru[row]);
    if (simd_lid == 0) {
      T gt = static_cast<T>(g), ut = static_cast<T>(u);
      T sg = SigmoidX{}(gt);
      T sl = gt * sg;
      h[j * HID + out_row + row] = sl * ut;
    }
  }
"""

DN_SRC = """
  // h [NE, HID]; W [E, D, HID*bits/32]; out y [D] = sum_j bf16(down_j(h_j) * score_j) (bf16, j ascending)
  constexpr int bits = BITS; constexpr int group_size = GS;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;
  const int K = HID;
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
  const int out_row = threadgroup_position_in_grid.y * (NSG * RPS) + simd_gid * RPS;
  thread U x_thread[values_per_thread];
  thread U res[NE][RPS];
  for (int j = 0; j < NE; j++) {
    const int e = inds[j];
    const device uint8_t* ws = (const device uint8_t*)w + (size_t)e * D * in_vec_size_w
        + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    const device T* sc = scales + (size_t)e * D * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* bi = biases + (size_t)e * D * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xp = h + j * HID + simd_lid * values_per_thread;
    for (int row = 0; row < RPS; row++) res[j][row] = 0;
    for (int k = 0; k < K; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        res[j][row] += qdot<U, values_per_thread, bits>(ws + row * in_vec_size_w, x_thread,
                         (U)sc[row * in_vec_size_g], (U)bi[row * in_vec_size_g], sum);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / group_size;
      bi += block_size / group_size;
      xp += block_size;
    }
  }
  for (int row = 0; row < RPS; row++) {
    T acc = T(0);
    for (int j = 0; j < NE; j++) {
      U r = simd_sum(res[j][row]);
      T p = static_cast<T>(static_cast<T>(r) * scores[j]);   // stock rounds the product to bf16
      acc = static_cast<T>(acc + p);
    }
    if (simd_lid == 0) y[out_row + row] = acc;
  }
"""


@functools.lru_cache(None)
def _kernels():
    gu = mx.fast.metal_kernel(
        name="moe_gate_up_exact",
        input_names=["x", "w", "scales", "biases", "inds", "in_vec_size"],
        output_names=["h"],
        header=_header(),
        source=GU_SRC,
    )
    dn = mx.fast.metal_kernel(
        name="moe_down_exact",
        input_names=["h", "w", "scales", "biases", "inds", "scores"],
        output_names=["y"],
        header=_header(),
        source=DN_SRC,
    )
    return gu, dn


def supported(gate_up, down, x_dtype) -> bool:
    """True when the exact kernels apply to stacks gate_up [E, 2H, K] and down [E, D, H].

    Requires the stock qmv_fast dispatch (K % 512 == 0, group 64) and a row tiling that divides N.
    """
    if (
        gate_up.bits not in BITS
        or down.bits != gate_up.bits
        or gate_up.group_size != 64
        or down.group_size != 64
    ):
        return False
    if gate_up.biases is None or down.biases is None or x_dtype not in (mx.bfloat16, mx.float16):
        return False
    (gr, gn), (dr, dn) = _TILE[gate_up.bits]
    hid, dmodel = gate_up.out_dims // 2, down.out_dims
    return (
        gate_up.in_dims % 512 == 0
        and down.in_dims % 512 == 0
        and down.in_dims == hid
        and hid % (gr * gn) == 0
        and dmodel % (dr * dn) == 0
        and gate_up.scales.dtype == x_dtype
        and down.scales.dtype == x_dtype
    )


def expert_gate_up_exact(x: mx.array, gate_up, inds: mx.array) -> mx.array:
    """x [K] (one row), inds [k] uint32 -> h [k, H] = swiglu(gate, up), bitwise the stock value."""
    kgu, _ = _kernels()
    (rps, nsg), _ = _TILE[gate_up.bits]
    hid, K, ne = gate_up.out_dims // 2, x.shape[-1], inds.shape[-1]
    return kgu(
        inputs=[x, *gate_up.expert_views(), inds, K],
        template=[
            ("T", x.dtype),
            ("BITS", gate_up.bits),
            ("GS", 64),
            ("HID", hid),
            ("RPS", rps),
            ("NSG", nsg),
        ],
        grid=(32 * ne, nsg * (hid // (rps * nsg)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(ne, hid)],
        output_dtypes=[x.dtype],
    )[0]


def expert_down_exact(h: mx.array, down, inds: mx.array, scores: mx.array) -> mx.array:
    """h [k, H], inds [k], scores [k] -> y [D] = sum_j bf16(down_j(h_j) * score_j), bitwise stock."""
    _, kdn = _kernels()
    _, (rps, nsg) = _TILE[down.bits]
    ne, hid = h.shape
    dmodel = down.out_dims
    return kdn(
        inputs=[h, *down.expert_views(), inds, scores],
        template=[
            ("T", h.dtype),
            ("BITS", down.bits),
            ("GS", 64),
            ("HID", hid),
            ("D", dmodel),
            ("NE", ne),
            ("RPS", rps),
            ("NSG", nsg),
        ],
        grid=(32, nsg * (dmodel // (rps * nsg)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(dmodel,)],
        output_dtypes=[h.dtype],
    )[0]


# ---- down projection with one simdgroup per expert; the k partial rows meet in threadgroup memory
# and one thread per row combines them in stock order (bf16 products, bf16 sum by ascending slot)
DN_PAR_SRC = """
  constexpr int bits = BITS; constexpr int group_size = GS;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;
  const int K = HID;
  const int in_vec_size_w = K * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const uint j = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
  const int rr = threadgroup_position_in_grid.x;           // row (0 for a single decode row)
  const int out_row = threadgroup_position_in_grid.y * RPS;
  threadgroup float part[NE * RPS];
  h += rr * NE * HID;
  inds += rr * NE;
  scores += rr * NE;
  y += rr * D;
  const int e = inds[j];
  const device uint8_t* ws = (const device uint8_t*)w + (size_t)e * D * in_vec_size_w
      + out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + (size_t)e * D * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bi = biases + (size_t)e * D * in_vec_size_g + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xp = h + j * HID + simd_lid * values_per_thread;
  thread U x_thread[values_per_thread];
  thread U res[RPS];
  for (int row = 0; row < RPS; row++) res[row] = 0;
  for (int k = 0; k < K; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(xp, x_thread);
    for (int row = 0; row < RPS; row++) {
      res[row] += qdot<U, values_per_thread, bits>(ws + row * in_vec_size_w, x_thread,
                    (U)sc[row * in_vec_size_g], (U)bi[row * in_vec_size_g], sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
    xp += block_size;
  }
  for (int row = 0; row < RPS; row++) {
    U r = simd_sum(res[row]);
    if (simd_lid == 0) part[j * RPS + row] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (j == 0 && simd_lid < RPS) {
    const int row = simd_lid;
    T acc = T(0);
    for (int jj = 0; jj < NE; jj++) {
      T p = static_cast<T>(static_cast<T>(part[jj * RPS + row]) * scores[jj]);
      acc = static_cast<T>(acc + p);
    }
    y[out_row + row] = acc;
  }
"""

DN_PAR_RPS = 4


@functools.lru_cache(None)
def _kernel_dn_par():
    return mx.fast.metal_kernel(
        name="moe_down_exact_par",
        input_names=["h", "w", "scales", "biases", "inds", "scores"],
        output_names=["y"],
        header=_header(),
        source=DN_PAR_SRC,
    )


def expert_down_exact_par(
    h: mx.array, down, inds: mx.array, scores: mx.array, rows: int = 1, rps: int | None = None
) -> mx.array:
    """Bitwise expert_down_exact per row: h [rows*k, H] -> y [rows, D] ([D] if rows == 1)."""
    rps = rps or DN_PAR_RPS
    ne = inds.shape[-1] // rows
    hid = h.shape[-1]
    dmodel = down.out_dims
    y = _kernel_dn_par()(
        inputs=[h, *down.expert_views(), inds, scores],
        template=[
            ("T", h.dtype),
            ("BITS", down.bits),
            ("GS", 64),
            ("HID", hid),
            ("D", dmodel),
            ("NE", ne),
            ("RPS", rps),
        ],
        grid=(32 * rows, ne * (dmodel // rps), 1),
        threadgroup=(32, ne, 1),
        output_shapes=[(rows, dmodel)],
        output_dtypes=[h.dtype],
    )[0]
    return y.reshape(dmodel) if rows == 1 else y


# Gate+up+SwiGLU for R rows in one launch. Threadgroup row j of the grid is (row j / NE, slot j % NE):
# the single-row kernel's code with the input pointer moved to that row, so every (row, slot) output
# is the single-row kernel's bits. Pairs with expert_down_exact_par(rows=R).
GU_ROWS_SRC = GU_SRC.replace(
    "  const device T* xp = x + simd_lid * values_per_thread;",
    "  const device T* xp = x + (j / NE) * K + simd_lid * values_per_thread;",
)
assert GU_ROWS_SRC != GU_SRC


@functools.lru_cache(None)
def _kernel_gu_rows():
    return mx.fast.metal_kernel(
        name="moe_gate_up_exact_rows",
        input_names=["x", "w", "scales", "biases", "inds", "in_vec_size"],
        output_names=["h"],
        header=_header(),
        source=GU_ROWS_SRC,
    )


def expert_gate_up_exact_rows(x: mx.array, gate_up, inds: mx.array) -> mx.array:
    """x [R, K], inds [R, k] uint32 -> h [R*k, H]; row r's block is bitwise expert_gate_up_exact(x[r], inds[r])."""
    (rps, nsg), _ = _TILE[gate_up.bits]
    R, K = x.shape
    ne = inds.shape[-1]
    hid = gate_up.out_dims // 2
    return _kernel_gu_rows()(
        inputs=[x, *gate_up.expert_views(), inds.reshape(-1), K],
        template=[
            ("T", x.dtype),
            ("BITS", gate_up.bits),
            ("GS", 64),
            ("HID", hid),
            ("RPS", rps),
            ("NSG", nsg),
            ("NE", ne),
        ],
        grid=(32 * R * ne, nsg * (hid // (rps * nsg)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(R * ne, hid)],
        output_dtypes=[x.dtype],
    )[0]


class _Stack(NamedTuple):
    """A quantized expert stack [E, N, K] with the attributes the expert kernels read."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int
    in_dims: int
    out_dims: int

    @classmethod
    def random(cls, shape, bits, seed, group_size=64):
        """A stack of random bf16 weights of `shape` [E, N, K], quantized to `bits` in groups of
        `group_size`; `seed` fixes the weights."""
        w, s, b = random_quantized(shape, bits, seed, group_size)
        k = s.shape[-1] * group_size
        return cls(w, s, b, group_size, w.shape[-1] * 32 // k, k, w.shape[-2])

    def expert_views(self) -> tuple[mx.array, mx.array, mx.array]:
        """One-expert views of the stack's buffers, as `QLinear.expert_views` gives the model's."""
        views = (self.weight[:1], self.scales[:1], self.biases[:1])
        mx.eval(*views)
        return views

    def gather(self, x: mx.array, rhs_indices: mx.array) -> mx.array:
        """The stock expert-gathered matmul (`mx.gather_qmm`, transposed, unsorted indices)."""
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=False,
        )


def expert_checks(bits: int) -> tuple:
    """The kernels `check_experts` checks for expert stacks `bits` wide: the exact kernels take 3, 4
    and 8 bits, but the prescaled gate/up, the rows kernels and the segmented prefill GEMM are 3-bit
    only (their `supported` tests say so), so a wider checkpoint neither runs nor reports them."""
    if bits == 3:
        return ("exact_experts", "moe_v4", "rows_v4", "seg_prefill")
    return ("exact_experts",)


def check_experts(hid: int, d: int, bits: int, group_size: int) -> dict:
    """expert_solo vs the stock gather path, prescaled solo and multi-row experts vs expert_solo, and
    seg_gather_qmm vs sorted mx.gather_qmm, on stacks `bits` wide in groups of `group_size`; only the
    kernels `expert_checks(bits)` names."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.elementwise import swiglu
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_qmv_prescaled import (
        expert_gate_up_prescaled,
        expert_gate_up_prescaled_rows,
    )
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gather_qmm_prefill import seg_gather_qmm

    out = dict.fromkeys(expert_checks(bits), False)
    E = 24
    gu = _Stack.random((E, 2 * hid, d), bits, 1, group_size)
    dn = _Stack.random((E, d, hid), bits, 2, group_size)
    x = mx.random.normal((3, d), key=mx.random.key(5)).astype(mx.bfloat16)
    logits = mx.random.normal((3, E), key=mx.random.key(6))
    inds = mx.argpartition(-logits, kth=7, axis=-1)[:, :8].astype(mx.uint32)
    sc = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1).astype(mx.bfloat16)
    want_h = mx.stack([expert_gate_up_exact(x[r], gu, inds[r]) for r in range(3)])
    want_y = mx.stack([expert_down_exact(want_h[r], dn, inds[r], sc[r]) for r in range(3)])
    # the stock path a single token takes through the MoE block: gather, swiglu, gather, weighted sum
    stock = True
    for r in range(3):
        ii = inds[r].reshape(1, 1, 8)
        g, u = mx.split(gu.gather(mx.expand_dims(x[r].reshape(1, 1, d), (-2, -3)), ii), 2, axis=-1)
        hh = swiglu(g, u)
        yy = (dn.gather(hh, ii).squeeze(-2) * sc[r].reshape(1, 1, 8)[..., None]).sum(axis=-2)
        stock = stock and bool(
            mx.array_equal(want_h[r], hh.reshape(8, hid)).item()
            and mx.array_equal(want_y[r].reshape(-1), yy.reshape(-1)).item()
        )
    out["exact_experts"] = stock
    if bits != 3:
        return out
    out["moe_v4"] = stock and bool(
        mx.array_equal(expert_gate_up_prescaled(x[:1], gu, inds[0]), want_h[0]).item()
        and mx.array_equal(
            expert_down_exact_par(want_h[0], dn, inds[0], sc[0], rps=8), want_y[0]
        ).item()
    )
    h = expert_gate_up_prescaled_rows(x, gu, inds).reshape(3, 8, hid)
    y = expert_down_exact_par(
        h.reshape(24, hid), dn, inds.reshape(-1), sc.reshape(-1), rows=3, rps=8
    )
    out["rows_v4"] = stock and bool(
        mx.array_equal(h, want_h).item() and mx.array_equal(y.reshape(3, d), want_y).item()
    )
    # prefill: enough pairs per expert for stock to take gather_qmm_rhs
    S = 64
    xp = mx.random.normal((S, d), key=mx.random.key(7)).astype(mx.bfloat16)
    ip = mx.argpartition(-mx.random.normal((S, E), key=mx.random.key(8)), kth=7, axis=-1)[
        :, :8
    ].astype(mx.uint32)
    flat = ip.reshape(-1)
    order = mx.argsort(flat)
    idx = flat[order]
    xs = mx.expand_dims(xp, -2)[order // 8]
    ref = mx.gather_qmm(
        xs,
        gu.weight,
        gu.scales,
        gu.biases,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        sorted_indices=True,
    )
    got = seg_gather_qmm(xs.reshape(-1, d), gu, idx)
    ok = bool(mx.array_equal(ref.reshape(got.shape), got).item())
    # a skewed routing so pieces of every height (<= 8, <= 16, full 32 rows) occur
    S = 512
    xp = mx.random.normal((S, d), key=mx.random.key(9)).astype(mx.bfloat16)
    lg = mx.random.normal((S, E), key=mx.random.key(10)) + 3 * mx.random.normal(
        (1, E), key=mx.random.key(11)
    )
    flat = mx.argpartition(-lg, kth=7, axis=-1)[:, :8].astype(mx.uint32).reshape(-1)
    order = mx.argsort(flat)
    idx = flat[order]
    xs = mx.expand_dims(xp, -2)[order // 8]
    ref = mx.gather_qmm(
        xs,
        gu.weight,
        gu.scales,
        gu.biases,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        sorted_indices=True,
    )
    got = seg_gather_qmm(xs.reshape(-1, d), gu, idx)
    out["seg_prefill"] = ok and bool(mx.array_equal(ref.reshape(got.shape), got).item())
    return out
