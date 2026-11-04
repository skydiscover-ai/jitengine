"""Row-grouped 4-bit expert matvec Metal kernels for the pooled MoE path.

Pairs (row, expert) arrive sorted by expert id; threadgroup z serves one distinct expert, read from
slot `seg_slot[z]` of the `[S, ...]` pool, so an expert shared by c rows is read ceil(c / VECS)
times rather than c. Output is [P, N] in sorted pair order, in x's dtype. The kernel adapts MLX's
`affine_qmv_wide` to a gathered operand.

Every output row depends only on its activation row and its expert's bytes, in a compile-time
k-order, so batched and solo rows are bit-identical. That holds only while `CONFIGS` is keyed on
(N, K, dtype) and never on the number of rows; VECS alone may vary by call (it never enters the
k-order), and the solo gather runs at `SOLO_VECS` (`solo_config_for`). The load-time check
(`check_grouped`) holds rows to that bit for bit, and the kernel to `selfcheck.no_worse_than_stock`
against stock `gather_qmm`.
`grouped_qmv_u2`, the two-row kernel of the MTP verify pass, has its own check (`check_grouped_u2`)
that its pairs equal `grouped_qmv_x`'s, one row at a time, bit for bit.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as _p16
from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m4pro.kernels.constants import placeholder
from jitengine_qwen3_8_flash_next_m4pro.kernels.qmv_row_blocked import HEADER, row_blocked_template
from jitengine_qwen3_8_flash_next_m4pro.kernels.source import substitute


def _source(dual: bool) -> str:
    wsets = ["g", "u"] if dual else [""]
    decl = ""
    for t in wsets:
        decl += f"""
    const device uint8_t* wrow{t}[RB];
    const device SB* srow{t}[RB];
    const device SB* brow{t}[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {{
      const int row = min(row0 + r, out_vec_size - 1);
      wrow{t}[r] = (const device uint8_t*)w{t} + slot_w + (ulong)row * in_vec_size_w;
      srow{t}[r] = scales{t} + slot_g + (ulong)row * in_vec_size_g;
      brow{t}[r] = biases{t} + slot_g + (ulong)row * in_vec_size_g;
    }}
    U result{t}[RB][VECS];
    #pragma unroll
    for (int r = 0; r < RB; r++) {{
      #pragma unroll
      for (int v = 0; v < VECS; v++) result{t}[r][v] = 0;
    }}"""
    inner = ""
    for t in wsets:
        inner += f"""
        #pragma unroll
        for (int r = 0; r < RB; r++) {{
          U w_dq[sub];
          dequantize4<U, sub>(wrow{t}[r] + k0 * bits / 8, scale{t}[r], bias{t}[r], w_dq);
          #pragma unroll
          for (int v = 0; v < VECS; v++) {{
            U acc = 0;
            #pragma unroll
            for (int i = 0; i < sub; i++) acc += xr[v][i] * w_dq[i];
            result{t}[r][v] += acc;
          }}
        }}"""
    sb = "".join(
        f" scale{t}[r] = static_cast<U>(srow{t}[r][g]); bias{t}[r] = static_cast<U>(brow{t}[r][g]);"
        for t in wsets
    )
    sbdecl = " ".join(f"U scale{t}[RB], bias{t}[RB];" for t in wsets)
    reduce = ""
    for t in wsets:
        reduce += f"""
    #pragma unroll
    for (int r = 0; r < RB; r++) {{
      #pragma unroll
      for (int v = 0; v < VECS; v++) {{
        if (KL >= 32) result{t}[r][v] += simd_shuffle_down(result{t}[r][v], 16);
        if (KL >= 16) result{t}[r][v] += simd_shuffle_down(result{t}[r][v], 8);
        if (KL >= 8)  result{t}[r][v] += simd_shuffle_down(result{t}[r][v], 4);
        if (KL >= 4)  result{t}[r][v] += simd_shuffle_down(result{t}[r][v], 2);
        if (KL >= 2)  result{t}[r][v] += simd_shuffle_down(result{t}[r][v], 1);
      }}
    }}"""
    if dual:
        val = "(resultg[r][v] / (1.0f + metal::exp(-resultg[r][v]))) * resultu[r][v]"
    else:
        val = "result[r][v]"
    return f"""
    typedef float U;
    constexpr int group_size = GS;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int slots_per_simdgroup = 32 / KL;
    constexpr int rows_per_simdgroup = slots_per_simdgroup * RB;

    const int in_vec_size = x_shape[1];
    const int out_vec_size = w{wsets[0]}_shape[1];

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;

    const int j = tid.z;
    const int cnt = seg_cnt[j];
    const int vec0 = tid.x * VECS;
    if (vec0 >= cnt) return;                       // uniform over the threadgroup; no barriers follow
    const int start = seg_start[j];
    const ulong slot = seg_slot[j];

    const short k_lane = simd_lid % KL;
    const short rslot = simd_lid / KL;
    const int row0 = tid.y * (rows_per_simdgroup * NSG) + rows_per_simdgroup * simd_gid + rslot * RB;

    const int in_vec_size_w = in_vec_size * bits / 8;   // bytes per weight row
    const int in_vec_size_g = in_vec_size / group_size;
    const ulong slot_w = slot * (ulong)out_vec_size * (ulong)in_vec_size_w;   // bytes
    const ulong slot_g = slot * (ulong)out_vec_size * (ulong)in_vec_size_g;   // elements
    {decl}
    const device T* xv[VECS];
    #pragma unroll
    for (int v = 0; v < VECS; v++) {{
      xv[v] = x + (ulong)pair_row[start + min(vec0 + v, cnt - 1)] * in_vec_size;
    }}

    for (int g = k_lane; g < in_vec_size_g; g += KL) {{
      {sbdecl}
      #pragma unroll
      for (int r = 0; r < RB; r++) {{{sb} }}
      #pragma unroll
      for (int sc = 0; sc < group_size / sub; sc++) {{
        const int k0 = g * group_size + sc * sub;
        U xr[VECS][sub];
        #pragma unroll
        for (int v = 0; v < VECS; v++) {{
          const device T* xc = xv[v] + k0;
          #pragma unroll
          for (int i = 0; i < sub; i++) xr[v][i] = static_cast<U>(xc[i]);
        }}
        {inner}
      }}
    }}
    {reduce}
    if (k_lane == 0) {{
      #pragma unroll
      for (int r = 0; r < RB; r++) {{
        const int out_row = row0 + r;
        if (out_row < out_vec_size) {{
          #pragma unroll
          for (int v = 0; v < VECS; v++) {{
            if (vec0 + v < cnt) y[(ulong)(start + vec0 + v) * out_vec_size + out_row] = static_cast<T>({val});
          }}
        }}
      }}
    }}
"""


def _source_x() -> str:
    """`_source(False)` with int32 device-resolved slots, an extension and an annex array set.

    A negative slot (record not yet resident) is clamped to 0 so the read stays in bounds; its rows
    are garbage and never read. Slots >= MAIN index the extension set (`wx`, `scalesx`, `biasesx`),
    slots >= MAIN2 the annex set (`wa`, `scalesa`, `biasesa`): only the base pointer and the slot's
    offset in its set change, never the arithmetic, so a record gives the same rows in any set.
    """
    return substitute(
        _source(False),
        (
            "    const ulong slot = seg_slot[j];",
            "    const int slot_raw = max(int(seg_slot[j]), 0);\n"
            "    const bool in_ann = slot_raw >= MAIN2;\n"
            "    const bool in_ext = !in_ann && slot_raw >= MAIN;\n"
            "    const ulong slot = in_ann ? (ulong)(slot_raw - MAIN2)\n"
            "                     : in_ext ? (ulong)(slot_raw - MAIN) : (ulong)slot_raw;\n"
            "    const device uint8_t* wbase = in_ann ? (const device uint8_t*)wa\n"
            "        : in_ext ? (const device uint8_t*)wx : (const device uint8_t*)w;\n"
            "    const device SB* sbase = in_ann ? scalesa : in_ext ? scalesx : scales;\n"
            "    const device SB* bbase = in_ann ? biasesa : in_ext ? biasesx : biases;",
        ),
        (
            "      wrow[r] = (const device uint8_t*)w + slot_w + (ulong)row * in_vec_size_w;",
            "      wrow[r] = wbase + slot_w + (ulong)row * in_vec_size_w;",
        ),
        (
            "      srow[r] = scales + slot_g + (ulong)row * in_vec_size_g;",
            "      srow[r] = sbase + slot_g + (ulong)row * in_vec_size_g;",
        ),
        (
            "      brow[r] = biases + slot_g + (ulong)row * in_vec_size_g;",
            "      brow[r] = bbase + slot_g + (ulong)row * in_vec_size_g;",
        ),
    )


def _source_xl(base: str | None = None) -> str:
    """`_source_x` (or `base`, a form of it) resolving each slot in-kernel as `tab[seg_slot[j] +
    off]` from expert ids."""
    return substitute(
        _source_x() if base is None else base,
        (
            "    const int slot_raw = max(int(seg_slot[j]), 0);",
            "    const int slot_raw = max(int(tab[(ulong)seg_slot[j] + (ulong)off[0]]), 0);",
        ),
    )


def _annex(ann, fallback: tuple, ann_slots: int) -> tuple:
    """(annex arrays, MAIN2): the annex set's arrays and first slot, or, without one, `fallback`
    bound in its place (never read: no slot reaches 1 << 30)."""
    if ann is None:
        return tuple(fallback), 1 << 30
    return tuple(ann), int(ann_slots)


@functools.cache
def kernel_x():
    """The compiled `moe_engine_grouped_qmv_x` kernel, the device-slot form of `grouped_qmv`,
    built once."""
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_x",
        input_names=[
            "w",
            "scales",
            "biases",
            "wx",
            "scalesx",
            "biasesx",
            "wa",
            "scalesa",
            "biasesa",
            "x",
            "seg_slot",
            "seg_start",
            "seg_cnt",
            "pair_row",
        ],
        output_names=["y"],
        header=HEADER,
        source=_source_x(),
        ensure_row_contiguous=True,
    )


def grouped_qmv_x(
    x,
    w,
    s,
    b,
    ext,
    main_slots: int,
    seg_slot,
    seg_start,
    seg_cnt,
    pair_row,
    n_pairs: int,
    max_cnt: int,
    vecs: int = 4,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
    ann=None,
    ann_slots: int = 1 << 30,
):
    """`grouped_qmv` with signed int32 device slots; `ext` holds the slots >= `main_slots` and
    `ann` (w, s, b), when given, the slots >= `ann_slots`."""
    N = w.shape[1]
    if ext is None:
        wx, sx, bx = w, s, b
        main_slots = 1 << 30
    else:
        wx, sx, bx = ext
    (wa, sa, ba), ann_slots = _annex(ann, (wx, sx, bx), ann_slots)
    return kernel_x()(
        inputs=[w, s, b, wx, sx, bx, wa, sa, ba, x, seg_slot, seg_start, seg_cnt, pair_row],
        template=[
            *row_blocked_template(x.dtype, s.dtype, vecs, k_lanes, nsg, rb, group_size),
            ("MAIN", int(main_slots)),
            ("MAIN2", int(ann_slots)),
        ],
        grid=_grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, seg_slot.shape[0]),
        threadgroup=(32, nsg, 1),
        output_shapes=[(n_pairs, N)],
        output_dtypes=[x.dtype],
    )[0]


@functools.cache
def kernel(dual: bool):
    """The compiled grouped kernel, built once for each `dual`: `moe_engine_grouped_qmv_dual`
    reads the gate and up weights in one pass, `moe_engine_grouped_qmv` a single weight set."""
    names = (
        ["wg", "scalesg", "biasesg", "wu", "scalesu", "biasesu"]
        if dual
        else ["w", "scales", "biases"]
    )
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_dual" if dual else "moe_engine_grouped_qmv",
        input_names=names + ["x", "seg_slot", "seg_start", "seg_cnt", "pair_row"],
        output_names=["y"],
        header=HEADER,
        source=_source(dual),
        ensure_row_contiguous=True,
    )


def _grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, n_seg):
    rows_per_tg = (32 // k_lanes) * rb * nsg
    tiles = (max_cnt + vecs - 1) // vecs
    return (32 * tiles, nsg * ((N + rows_per_tg - 1) // rows_per_tg), n_seg)


def grouped_qmv(
    x,
    w,
    s,
    b,
    seg_slot,
    seg_start,
    seg_cnt,
    pair_row,
    n_pairs: int,
    max_cnt: int,
    vecs: int = 4,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
):
    """y[P, N] = for each pair p (sorted by expert): x[pair_row[p]] @ dequant(w[slot_of(p)])^T."""
    N = w.shape[1]
    return kernel(False)(
        inputs=[w, s, b, x, seg_slot, seg_start, seg_cnt, pair_row],
        template=row_blocked_template(x.dtype, s.dtype, vecs, k_lanes, nsg, rb, group_size),
        grid=_grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, seg_slot.shape[0]),
        threadgroup=(32, nsg, 1),
        output_shapes=[(n_pairs, N)],
        output_dtypes=[x.dtype],
    )[0]


def grouped_swiglu(
    x,
    gate,
    up,
    seg_slot,
    seg_start,
    seg_cnt,
    pair_row,
    n_pairs: int,
    max_cnt: int,
    vecs: int = 4,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
):
    """h[P, N] = silu(x @ gate_e^T) * (x @ up_e^T) per pair, gate and up read in one pass."""
    (wg, sg, bg), (wu, su, bu) = gate, up
    N = wg.shape[1]
    return kernel(True)(
        inputs=[wg, sg, bg, wu, su, bu, x, seg_slot, seg_start, seg_cnt, pair_row],
        template=row_blocked_template(x.dtype, sg.dtype, vecs, k_lanes, nsg, rb, group_size),
        grid=_grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, seg_slot.shape[0]),
        threadgroup=(32, nsg, 1),
        output_shapes=[(n_pairs, N)],
        output_dtypes=[x.dtype],
    )[0]


# ----------------------------------------------------------------------------- pack16 forms
# The same kernels over compact slots (`kernels/pack16.py`): one u16 word per (scale, bias) group in
# place of the two bf16 arrays, decoded to the original bf16 values at the per-group load. Each form
# is the bf16 kernel's own source with that load replaced, so weights, k-order, shuffle ladder and
# the VECS/KL/NSG/RB grid are its text. `dep` is unused by the arithmetic: it is the join of the
# encode kernels' outputs, the input through which MLX orders this read after their pool writes.
P16_HEADER = HEADER + _p16.P16_DECODE
_LOAD = " scale[r] = static_cast<U>(srow[r][g]); bias[r] = static_cast<U>(brow[r][g]);"
_LOAD_P16 = " p16_decode(srow[r][g], ebase, scale[r], bias[r]);"


def _source_p16() -> str:
    """`_source(False)` reading `words` and the global escape table `exc`."""
    return substitute(
        _source(False),
        (
            "      srow[r] = scales + slot_g + (ulong)row * in_vec_size_g;",
            "      srow[r] = words + slot_g + (ulong)row * in_vec_size_g;",
        ),
        (
            "      brow[r] = biases + slot_g + (ulong)row * in_vec_size_g;",
            "      brow[r] = srow[r];",
        ),
        (
            "    const device T* xv[VECS];",
            "    const device uint* ebase = exc;\n    const device T* xv[VECS];",
        ),
        (_LOAD, _LOAD_P16),
    )


def _source_x_p16() -> str:
    """`_source_x()` reading `words` / `wordsx` / `wordsa` and the global escape table `exc`."""
    return substitute(
        _source_x(),
        (
            "    const device SB* sbase = in_ann ? scalesa : in_ext ? scalesx : scales;",
            "    const device SB* sbase = in_ann ? wordsa : in_ext ? wordsx : words;",
        ),
        (
            "    const device SB* bbase = in_ann ? biasesa : in_ext ? biasesx : biases;",
            "    const device uint* ebase = exc;",
        ),
        (
            "      brow[r] = bbase + slot_g + (ulong)row * in_vec_size_g;",
            "      brow[r] = srow[r];",
        ),
        (_LOAD, _LOAD_P16),
    )


@functools.cache
def kernel_p16():
    """The compiled `moe_engine_grouped_qmv_p16` kernel, `kernel(False)` over compact slots."""
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_p16",
        input_names=[
            "w",
            "words",
            "exc",
            "x",
            "seg_slot",
            "seg_start",
            "seg_cnt",
            "pair_row",
            "dep",
        ],
        output_names=["y"],
        header=P16_HEADER,
        source=_source_p16(),
        ensure_row_contiguous=True,
    )


@functools.cache
def kernel_x_p16():
    """The compiled `moe_engine_grouped_qmv_x_p16` kernel, `kernel_x()` over compact slots."""
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_x_p16",
        input_names=[
            "w",
            "words",
            "wx",
            "wordsx",
            "wa",
            "wordsa",
            "exc",
            "x",
            "seg_slot",
            "seg_start",
            "seg_cnt",
            "pair_row",
            "dep",
        ],
        output_names=["y"],
        header=P16_HEADER,
        source=_source_x_p16(),
        ensure_row_contiguous=True,
    )


def grouped_qmv_p16(
    x,
    w,
    words,
    exc,
    dep,
    seg_slot,
    seg_start,
    seg_cnt,
    pair_row,
    n_pairs: int,
    max_cnt: int,
    vecs: int = 4,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
):
    """`grouped_qmv` over compact slots: `words` u16 [S, N, K/gs], `exc` the u32 escape table."""
    N = w.shape[1]
    return kernel_p16()(
        inputs=[w, words, exc, x, seg_slot, seg_start, seg_cnt, pair_row, dep],
        template=row_blocked_template(x.dtype, mx.uint16, vecs, k_lanes, nsg, rb, group_size),
        grid=_grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, seg_slot.shape[0]),
        threadgroup=(32, nsg, 1),
        output_shapes=[(n_pairs, N)],
        output_dtypes=[x.dtype],
    )[0]


def grouped_qmv_x_p16(
    x,
    w,
    words,
    exc,
    dep,
    ext,
    main_slots: int,
    seg_slot,
    seg_start,
    seg_cnt,
    pair_row,
    n_pairs: int,
    max_cnt: int,
    vecs: int = 4,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
    ann=None,
    ann_slots: int = 1 << 30,
):
    """`grouped_qmv_x` over compact slots; `ext` = (w, words) of the slots >= `main_slots`, `ann`
    the same of the slots >= `ann_slots`."""
    N = w.shape[1]
    if ext is None:
        wx, sx = w, words
        main_slots = 1 << 30
    else:
        wx, sx = ext
    (wa, sa), ann_slots = _annex(ann, (wx, sx), ann_slots)
    return kernel_x_p16()(
        inputs=[w, words, wx, sx, wa, sa, exc, x, seg_slot, seg_start, seg_cnt, pair_row, dep],
        template=[
            *row_blocked_template(x.dtype, mx.uint16, vecs, k_lanes, nsg, rb, group_size),
            ("MAIN", int(main_slots)),
            ("MAIN2", int(ann_slots)),
        ],
        grid=_grid(n_pairs, max_cnt, N, vecs, k_lanes, nsg, rb, seg_slot.shape[0]),
        threadgroup=(32, nsg, 1),
        output_shapes=[(n_pairs, N)],
        output_dtypes=[x.dtype],
    )[0]


# ----------------------------------------------------------------------------- engine-side policy
DEFAULT_CONFIG = (4, 8, 2, 2)  # (VECS, KL, NSG, RB)
CONFIG_BASIS = (
    "Measured on the M4 Pro at the 8-row decode union of 52 experts: (4,8,2,2) ran "
    "0.89x stock and (2,8,4,2) 1.07x; switching took a step from 42.86 to 35.55 ms. "
    "float32, the served dtype, uses (2,8,4,2); bfloat16 keeps the unmeasured default. "
    "VECS=8 runs at 0.22-0.27x on every shape."
)
SCORED_CONFIG = (2, 8, 4, 2)
# "split" beats "fused" for the engine's two grouped calls per layer (resident pairs, the rest).
DEFAULT_VARIANT = "split"
VARIANT_BASIS = (
    "split: 1.13-1.14x faster than fused in paired single-stream runs of an earlier "
    "build of this engine, which also made two grouped calls per layer"
)

# (out_dims N, in_dims K, activation dtype name) -> (VECS, KL, NSG, RB). One config per shape and
# dtype, never per row count, so batched and solo rows share a k-order.
CONFIGS: dict = {
    (640, 2560, "bfloat16"): DEFAULT_CONFIG,  # Qwen3.8-Flash-Next gate_proj / up_proj
    (2560, 640, "bfloat16"): DEFAULT_CONFIG,  # down_proj
    (640, 2560, "float32"): SCORED_CONFIG,
    (2560, 640, "float32"): SCORED_CONFIG,
}

_ACT_DTYPES = {mx.bfloat16: "bfloat16", mx.float16: "float16", mx.float32: "float32"}


def dtype_name(dt) -> str:
    """The name of activation dtype `dt` as `CONFIGS` keys it: "bfloat16", "float16" or
    "float32"."""
    return _ACT_DTYPES.get(dt, str(dt).split(".")[-1])


def config_for(out_dims: int, in_dims: int, dt) -> tuple:
    """(VECS, KL, NSG, RB) for this projection and activation dtype: its `CONFIGS` entry, else
    `DEFAULT_CONFIG`."""
    return CONFIGS.get((int(out_dims), int(in_dims), dtype_name(dt)), DEFAULT_CONFIG)


# The solo gather (`SpeculativeGatherMixin._gather_dev`: one row, a segment per pair) gives every
# segment one vector, so a VECS-2 tile computes that vector twice. Each vector's sum runs over the
# same k in the same order whatever VECS is, so its pairs are bit-identical to the `CONFIGS` launch;
# KL, NSG and RB stay the shape's.
SOLO_VECS = 1
SOLO_VECS_BASIS = (
    "VECS 1 for the one-row gather, measured on the M4 Pro over real records and routing: a layer "
    "(gate, up, SwiGLU, down) on pack16 compact slots ran x0.950 / x0.954 of VECS 2 in two runs, "
    "every output bit-identical."
)


def solo_config_for(out_dims: int, in_dims: int, dt) -> tuple:
    """`config_for` with VECS = `SOLO_VECS`: the launch of a call whose segments each hold one
    pair (the solo gather's)."""
    return (SOLO_VECS, *config_for(out_dims, in_dims, dt)[1:])


def supports(
    bits: int,
    group_size: int,
    in_dims: int,
    weight_dtype=mx.uint32,
    scales_dtype=mx.bfloat16,
    has_biases: bool = True,
    act_dtype=mx.bfloat16,
) -> bool:
    """Whether the vendored kernel computes this expert projection (else: stock gather_qmm)."""
    return (
        int(bits) == 4
        and int(group_size) >= 8
        and int(group_size) % 8 == 0
        and int(in_dims) % int(group_size) == 0
        and weight_dtype == mx.uint32
        and has_biases
        and scales_dtype in _ACT_DTYPES
        and act_dtype in _ACT_DTYPES
    )


def supports_geometry(
    bits: int, group_size: int, in_dims_list, scales_dtype_name: str = "BF16"
) -> bool:
    """`supports` from checkpoint-header facts alone, usable before any Metal allocation."""
    sd = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}.get(str(scales_dtype_name))
    return sd is not None and all(
        supports(bits, group_size, k, scales_dtype=sd) for k in in_dims_list
    )


def _source_u2(base: str | None = None) -> str:
    """`_source_xl` (or `base`, a form of it) over the 2*KP pairs of two activation rows, each
    expert's weights streamed once.

    Pairs 0..KP-1 are row 0's, KP..2KP-1 row 1's (`seg_slot` = their expert ids). Segment j < KP takes
    row 0's pair j and, when row 1 routed to the same expert, row 1's pair as its second vector;
    segment j >= KP takes row 1's pair unless row 0 has that expert. VECS never enters the k-order,
    so every pair is bit-identical to the solo gather's. Output row p = pair p."""
    return substitute(
        _source_xl() if base is None else base,
        (
            "    const int j = tid.z;\n"
            "    const int cnt = seg_cnt[j];\n"
            "    const int vec0 = tid.x * VECS;\n"
            "    if (vec0 >= cnt) return;                       // uniform over the threadgroup; no barriers follow\n"
            "    const int start = seg_start[j];\n",
            "    const int j = tid.z;\n"
            "    int pidx[2] = {j, j};\n"
            "    int cnt = 1;\n"
            "    if (j < KP) {\n"
            "      for (int m = 0; m < KP; m++) if (seg_slot[KP + m] == seg_slot[j]) { pidx[1] = KP + m; cnt = 2; }\n"
            "    } else {\n"
            "      for (int m = 0; m < KP; m++) if (seg_slot[m] == seg_slot[j]) return;\n"
            "    }\n"
            "    const int vec0 = tid.x * VECS;\n"
            "    if (vec0 >= cnt) return;                       // uniform over the threadgroup; no barriers follow\n",
        ),
        (
            "      xv[v] = x + (ulong)pair_row[start + min(vec0 + v, cnt - 1)] * in_vec_size;",
            "      xv[v] = x + (ulong)(XPAIR ? pidx[min(vec0 + v, cnt - 1)] : pidx[min(vec0 + v, cnt - 1)] / KP) * in_vec_size;",
        ),
        (
            "            if (vec0 + v < cnt) y[(ulong)(start + vec0 + v) * out_vec_size + out_row]",
            "            if (vec0 + v < cnt) y[(ulong)pidx[vec0 + v] * out_vec_size + out_row]",
        ),
    )


@functools.cache
def kernel_u2():
    """The compiled `moe_engine_grouped_qmv_u2` kernel, the two-row form of `kernel_x` for the MTP
    verify pass, built once."""
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_u2",
        input_names=[
            "w",
            "scales",
            "biases",
            "wx",
            "scalesx",
            "biasesx",
            "wa",
            "scalesa",
            "biasesa",
            "x",
            "seg_slot",
            "seg_start",
            "seg_cnt",
            "pair_row",
            "tab",
            "off",
        ],
        output_names=["y"],
        header=HEADER,
        source=_source_u2(),
        ensure_row_contiguous=True,
    )


def grouped_qmv_u2(
    x,
    w,
    s,
    b,
    ext,
    main_slots: int,
    inds2,
    tab,
    off,
    kp: int,
    xpair: bool,
    vecs: int = 2,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
    ann=None,
    ann_slots: int = 1 << 30,
):
    """The 2*kp pairs of two rows (`inds2` = row 0's kp expert ids then row 1's), each expert's weights
    read once. x: [2, K] (`xpair` False: pair p reads row p // kp) or [2*kp, K] (`xpair`: row p).
    `ann` (w, s, b), when given, holds the slots >= `ann_slots`."""
    N = w.shape[1]
    if ext is None:
        wx, sx, bx = w, s, b
        main_slots = 1 << 30
    else:
        wx, sx, bx = ext
    (wa, sa, ba), ann_slots = _annex(ann, (wx, sx, bx), ann_slots)
    d = placeholder(mx.int32)
    return kernel_u2()(
        inputs=[w, s, b, wx, sx, bx, wa, sa, ba, x, inds2, d, d, d, tab, off],
        template=[
            *row_blocked_template(x.dtype, s.dtype, vecs, k_lanes, nsg, rb, group_size),
            ("MAIN", int(main_slots)),
            ("MAIN2", int(ann_slots)),
            ("KP", int(kp)),
            ("XPAIR", bool(xpair)),
        ],
        grid=_grid(2 * kp, 2, N, vecs, k_lanes, nsg, rb, 2 * kp),
        threadgroup=(32, nsg, 1),
        output_shapes=[(2 * kp, N)],
        output_dtypes=[x.dtype],
    )[0]


def _source_u2_p16() -> str:
    """`_source_u2` over compact slots: the two-row text of `_source_x_p16()`, whose edits touch
    only the metadata pointers and the per-group load, never the segment set-up, the activation
    pointers or the output index that the two-row edits replace."""
    return _source_u2(_source_xl(_source_x_p16()))


@functools.cache
def kernel_u2_p16():
    """The compiled `moe_engine_grouped_qmv_u2_p16` kernel, `kernel_u2()` over compact slots."""
    return mx.fast.metal_kernel(
        name="moe_engine_grouped_qmv_u2_p16",
        input_names=[
            "w",
            "words",
            "wx",
            "wordsx",
            "wa",
            "wordsa",
            "exc",
            "x",
            "seg_slot",
            "seg_start",
            "seg_cnt",
            "pair_row",
            "dep",
            "tab",
            "off",
        ],
        output_names=["y"],
        header=P16_HEADER,
        source=_source_u2_p16(),
        ensure_row_contiguous=True,
    )


def grouped_qmv_u2_p16(
    x,
    w,
    words,
    exc,
    dep,
    ext,
    main_slots: int,
    inds2,
    tab,
    off,
    kp: int,
    xpair: bool,
    vecs: int = 2,
    k_lanes: int = 8,
    nsg: int = 2,
    rb: int = 2,
    group_size: int = 32,
    ann=None,
    ann_slots: int = 1 << 30,
):
    """`grouped_qmv_u2` over compact slots; `ext` = (w, words) of the slots >= `main_slots`, `ann`
    the same of the slots >= `ann_slots`."""
    N = w.shape[1]
    if ext is None:
        wx, sx = w, words
        main_slots = 1 << 30
    else:
        wx, sx = ext
    (wa, sa), ann_slots = _annex(ann, (wx, sx), ann_slots)
    d = placeholder(mx.int32)
    return kernel_u2_p16()(
        inputs=[w, words, wx, sx, wa, sa, exc, x, inds2, d, d, d, dep, tab, off],
        template=[
            *row_blocked_template(x.dtype, mx.uint16, vecs, k_lanes, nsg, rb, group_size),
            ("MAIN", int(main_slots)),
            ("MAIN2", int(ann_slots)),
            ("KP", int(kp)),
            ("XPAIR", bool(xpair)),
        ],
        grid=_grid(2 * kp, 2, N, vecs, k_lanes, nsg, rb, 2 * kp),
        threadgroup=(32, nsg, 1),
        output_shapes=[(2 * kp, N)],
        output_dtypes=[x.dtype],
    )[0]


# ----------------------------------------------------------------------------- load-time check
CHECK_SLOTS = 6  # pool slots the check fills; the last two are read as the extension set


def check_grouped(hidden, inter, bits, group_size, act_dtype, top_k, variant=DEFAULT_VARIANT):
    """The load-time check of the gate/up (`inter` x `hidden`) and down (`hidden` x `inter`) calls
    at `act_dtype`, on a pool whose slots are not expert ids, with one expert shared by more rows
    than two tiles hold. Bit for bit (the claim above): a pair launched alone is its row in the
    batch, and the same experts in other slots give the same rows; the device-slot form, with some
    slots in the extension set read at an offset, gives the canonical call's rows, and a slot that
    is not resident leaves the other rows alone. And no less accurate than stock `gather_qmm`, as
    `selfcheck.no_worse_than_stock` measures it; with variant "fused", `grouped_swiglu` too."""
    for k in (hidden, inter):
        if not supports(bits, group_size, k, act_dtype=act_dtype):
            raise ValueError(f"no grouped kernel for K={k}, {bits}-bit, group {group_size}")
    vecs = max(config_for(inter, hidden, act_dtype)[0], config_for(hidden, inter, act_dtype)[0])
    pairs = _Pairs(rows=2 * vecs + 1, per_row=min(int(top_k), CHECK_SLOTS - 1), seed=7)
    gate = _Pool(inter, hidden, bits, group_size, seed=1)
    down = _Pool(hidden, inter, bits, group_size, seed=2)
    x = sc.normal((pairs.rows, hidden), 3, act_dtype)  # a row per token
    h = sc.normal((pairs.size, inter), 4, act_dtype)  # a row per sorted pair
    kd = max(2, min(int(top_k), CHECK_SLOTS))
    ok = (
        gate.check(x, pairs, by_pair=False)
        and down.check(h, pairs, by_pair=True)
        and gate.check_device(x[1:2], kd, by_pair=False)
        and down.check_device(h[1 : kd + 1], kd, by_pair=True)
    )
    if ok and variant == "fused":
        up = _Pool(inter, hidden, bits, group_size, seed=5)  # gate's slot layout, other weights
        ok = _check_swiglu(gate, up, x, pairs)
    return ok


def check_grouped_u2(hidden, inter, bits, group_size, act_dtype, top_k):
    """The load-time check of the verify pass's `grouped_qmv_u2` calls: the gate and up projections
    (`inter` x `hidden`, one input row per token) and the down projection (`hidden` x `inter`, one
    input row per pair), as `spec_build_pair` launches them, at `act_dtype` and `top_k` pairs per row.

    Bit for bit, the claim above: each pair of the two rows is what `grouped_qmv_x` gives that pair
    in `_gather_dev`'s call for its row alone, which is what a solo step computes. The launch is the
    model's: expert ids resolved through the device slot table at a layer offset, some slots in the
    extension set and a run without one, the two rows sharing some experts (one weight stream, two
    vectors) and each routing to some the other does not, and two experts not resident (-1), whose
    pairs are not compared, as the model never reads them. The solo kernel has its own check against
    stock (`check_grouped`), so equality to it is the whole criterion."""
    for k in (hidden, inter):
        if not supports(bits, group_size, k, act_dtype=act_dtype):
            raise ValueError(f"no grouped kernel for K={k}, {bits}-bit, group {group_size}")
    route = _Routing2(max(1, int(top_k)), seed=21)
    x = sc.normal((3, hidden), 14, act_dtype)[1:]  # a row per token, read at an offset
    h = sc.normal((2 * route.kp, inter), 15, act_dtype)  # a row per pair, as the SwiGLU leaves them
    pools = (
        (_Pool(inter, hidden, bits, group_size, 11, route.slots), x, False),
        (_Pool(inter, hidden, bits, group_size, 12, route.slots), x, False),
        (_Pool(hidden, inter, bits, group_size, 13, route.slots), h, True),
    )
    return all(route.agrees(pool, rows, by_pair) for pool, rows, by_pair in pools)


class _Routing2:
    """Two rows' `kp` experts each for `check_grouped_u2`, and the device slot table that resolves them.

    Row 1 repeats the first half of row 0's experts and takes the rest from outside row 0. Two experts
    are not resident, one only row 0 routes to (when it has one) and one only row 1 routes to (else a
    third expert nobody routes to); the others fill `slots` slots in a random order. `tab` holds the
    slots at layer offset `n_experts`, below a first layer's entries that point at other slots."""

    def __init__(self, kp: int, seed: int):
        rng = np.random.default_rng(seed)
        self.kp, self.n_experts = kp, kp + (kp + 1) // 2 + 2
        row0 = rng.permutation(self.n_experts)[:kp]
        shared = row0[: max(1, kp // 2)]
        fresh = rng.permutation(np.setdiff1d(np.arange(self.n_experts), row0))
        row1 = rng.permutation(np.concatenate([shared, fresh[: kp - shared.size]]))
        free = [e for e in range(self.n_experts) if e not in row0 and e not in row1]
        only1 = [e for e in row1 if e not in shared]
        gone = row0[-1] if row0[-1] not in shared else free[0]
        out = {int(gone), int(next(e for e in [*only1[::-1], *free] if e != gone))}
        resident = [e for e in range(self.n_experts) if e not in out]
        self.slots = len(resident)
        self.main = max(1, self.slots - max(2, self.slots // 3))
        tab = rng.integers(0, self.slots, 2 * self.n_experts)  # layer 0's entries are other slots
        tab[self.n_experts :] = -1
        tab[self.n_experts + np.array(resident)] = rng.permutation(self.slots)
        self.inds = np.concatenate([row0, row1])
        self.tab = mx.array(tab.astype(np.int32))
        self.inds2 = mx.array(self.inds.astype(np.int32))
        self.off = mx.array([self.n_experts], dtype=mx.int32)
        self.resident = tab[self.n_experts + self.inds] >= 0

    def agrees(self, pool: _Pool, x, by_pair: bool) -> bool:
        """`grouped_qmv_u2` on the two rows against `grouped_qmv_x` on each row alone at the solo
        gather's launch (`solo_config_for`), over the pool's slots split into a main and an
        extension set, and then all in the main set."""
        kp, w, s, b = self.kp, *pool.stacks
        cfg = config_for(pool.n, pool.k, x.dtype)
        solo = solo_config_for(pool.n, pool.k, x.dtype)
        ar = mx.arange(kp, dtype=mx.uint32)
        ones, zeros = mx.ones((kp,), dtype=mx.uint32), mx.zeros((kp,), dtype=mx.uint32)
        for main in (self.main, None):
            if main is None:
                head, ext, main = (w, s, b), None, 1 << 30
            else:
                head, ext = (w[:main], s[:main], b[:main]), (w[main:], s[main:], b[main:])
            two = grouped_qmv_u2(
                x,
                *head,
                ext,
                main,
                self.inds2,
                self.tab,
                self.off,
                kp,
                by_pair,
                2,
                *cfg[1:],
                group_size=pool.gs,
            )
            for r in range(2):
                pairs = slice(r * kp, (r + 1) * kp)
                slots = mx.take(self.tab, self.inds2[pairs] + self.n_experts)
                alone = grouped_qmv_x(
                    x[pairs] if by_pair else x[r : r + 1],
                    *head,
                    ext,
                    main,
                    slots,
                    ar,
                    ones,
                    ar if by_pair else zeros,
                    kp,
                    1,
                    *solo,
                    group_size=pool.gs,
                )
                keep = mx.array(np.flatnonzero(self.resident[pairs]))
                if not mx.array_equal(two[pairs][keep], alone[keep]).item():
                    return False
        return True


class _Pairs:
    """(row, expert) pairs as Qwen4ExpTextSparseMoeBlock sorts them: `rows` rows of `per_row`
    distinct experts each, expert `SHARED` in every row, so its segment spans more than two
    tiles."""

    SHARED = 3

    def __init__(self, rows: int, per_row: int, seed: int):
        rng = np.random.default_rng(seed)
        others = np.array([e for e in range(CHECK_SLOTS) if e != self.SHARED])
        flat = []
        for _ in range(rows):
            row = rng.permutation(others)[: per_row - 1]
            flat.append(np.insert(row, rng.integers(per_row), self.SHARED))
        self.rows, self.per_row = rows, per_row
        self.flat = np.concatenate(flat)
        self.size = int(self.flat.size)
        self.pos = np.argsort(self.flat, kind="stable")  # sorted pair -> flat pair
        self.experts, self.start, self.count = np.unique(
            self.flat[self.pos], return_index=True, return_counts=True
        )

    def row_of(self, by_pair: bool) -> np.ndarray:
        """Each sorted pair's input row: its own (down) or its token's (gate/up)."""
        return np.arange(self.size) if by_pair else self.pos // self.per_row

    def segments(self, slot_of, by_pair: bool) -> list:
        """The kernel's `seg_slot`, `seg_start`, `seg_cnt` and `pair_row` for these pairs, as
        uint32 device arrays; `slot_of` maps an expert id to its pool slot."""
        arrs = (slot_of[self.experts], self.start, self.count, self.row_of(by_pair))
        return [mx.array(a.astype(np.uint32)) for a in arrs]

    def probes(self) -> list:
        """Sorted pairs to launch alone: the first, the last of the largest segment, the last."""
        big = int(np.argmax(self.count))
        return sorted({0, int(self.start[big] + self.count[big] - 1), self.size - 1})


class _Pool:
    """CHECK_SLOTS random n x k experts; slot j holds expert perm[j], so slots are not expert
    ids."""

    def __init__(self, n, k, bits, group_size, seed, slots=CHECK_SLOTS):
        self.lins = [sc.Linear(n, k, 100 * seed + e, group_size, bits) for e in range(slots)]
        perm = np.random.default_rng(0).permutation(slots)
        self.slot_of = np.empty(slots, dtype=np.int64)
        self.slot_of[perm] = np.arange(slots)
        self.stacks = [mx.stack([getattr(self.lins[e], f) for e in perm]) for f in _FIELDS]
        mx.eval(*self.stacks)
        self.n, self.k, self.gs, self.bits = n, k, group_size, bits

    def call(self, x, seg, n_pairs, max_cnt, stacks=None):
        """`grouped_qmv` on the segments `seg` over `stacks` (the pool's own by default), at this
        shape's `config_for` launch."""
        w, s, b = stacks or self.stacks
        cfg = config_for(self.n, self.k, x.dtype)
        return grouped_qmv(x, w, s, b, *seg, n_pairs, max_cnt, *cfg, group_size=self.gs)

    def stock(self, x, pairs, by_pair: bool):
        """Qwen4ExpTextSparseMoeBlock's stock branch on each sorted pair's row:
        `gather_qmm(sorted_indices=False)`, the metadata in the activations' dtype (stock's
        upcast)."""
        w, s, b = self.stacks
        slots = self.slot_of[pairs.flat[pairs.pos]]
        return mx.gather_qmm(
            x[mx.array(pairs.row_of(by_pair))][:, None, :],
            w,
            s.astype(x.dtype),
            b.astype(x.dtype),
            rhs_indices=mx.array(slots.astype(np.uint32)),
            transpose=True,
            group_size=self.gs,
            bits=self.bits,
            sorted_indices=False,
        )[:, 0]

    def exact(self, x, pairs, by_pair: bool, cols) -> np.ndarray:
        """Each sorted pair's row times its expert in float64, output columns `cols`."""
        rows, x64 = pairs.row_of(by_pair), sc.f64(x)
        out = np.empty((pairs.size, cols.size))
        for e, s0, c in zip(pairs.experts, pairs.start, pairs.count, strict=True):
            out[s0 : s0 + c] = self.lins[e].exact(x64[rows[s0 : s0 + c]], cols)
        return out

    def check(self, x, pairs, by_pair: bool) -> bool:
        """Whether the grouped call on `pairs` gives the same bits as each probe pair launched
        alone and as the pool with its slots reversed, and is no less accurate than stock
        `gather_qmm`."""
        y = self.call(x, pairs.segments(self.slot_of, by_pair), pairs.size, int(pairs.count.max()))
        rows = pairs.row_of(by_pair)
        for p in pairs.probes():
            slot = self.slot_of[pairs.flat[pairs.pos[p]]]
            one = [mx.array(np.array([v], dtype=np.uint32)) for v in (slot, 0, 1, rows[p])]
            if not mx.array_equal(self.call(x, one, 1, 1)[0], y[p]).item():
                return False
        rev = pairs.segments(CHECK_SLOTS - 1 - self.slot_of, by_pair)
        y_rev = self.call(
            x, rev, pairs.size, int(pairs.count.max()), [a[::-1] for a in self.stacks]
        )
        if not mx.array_equal(y_rev, y).item():
            return False
        cols = sc.sample(self.n)
        idx = mx.array(cols)
        stock = self.stock(x, pairs, by_pair)
        return sc.no_worse_than_stock(y[:, idx], stock[:, idx], self.exact(x, pairs, by_pair, cols))

    def check_device(self, x, kd: int, by_pair: bool) -> bool:
        """`Qwen4ExpTextSparseMoeBlock._gather_dev`'s call for one token's `kd` pairs, a segment
        each with int32 slots, those from `main` up in the extension set, at `solo_config_for`;
        gate/up read one row, down a row each. The canonical call runs at `config_for`."""
        main = CHECK_SLOTS - 2
        slots = np.array([main, 0, main + 1, 1, 2, 3][:kd], dtype=np.int64)
        w, s, b = self.stacks
        cfg = solo_config_for(self.n, self.k, x.dtype)
        ar, ones = mx.arange(kd, dtype=mx.uint32), mx.ones((kd,), dtype=mx.uint32)
        rows = ar if by_pair else mx.zeros((kd,), dtype=mx.uint32)

        def device(sl):
            return grouped_qmv_x(
                x,
                w[:main],
                s[:main],
                b[:main],
                (w[main:], s[main:], b[main:]),
                main,
                mx.array(sl.astype(np.int32)),
                ar,
                ones,
                rows,
                kd,
                1,
                *cfg,
                group_size=self.gs,
            )

        got = device(slots)
        pos = np.argsort(slots, kind="stable")  # canonical: a segment per slot, in slot order
        seg = (slots[pos], np.arange(kd), np.ones(kd), pos if by_pair else np.zeros(kd))
        canon = self.call(x, [mx.array(a.astype(np.uint32)) for a in seg], kd, 1)
        if not mx.array_equal(got, canon[mx.array(np.argsort(pos))]).item():
            return False
        missing = slots.copy()
        missing[1] = -1
        keep = mx.array(np.delete(np.arange(kd), 1))
        return mx.array_equal(device(missing)[keep], got[keep]).item()


_FIELDS = ("weight", "scales", "biases")


def _check_swiglu(gate: _Pool, up: _Pool, x, pairs: _Pairs) -> bool:
    """`grouped_swiglu` (gate and up in one pass, SwiGLU in float32) against stock
    `silu(gather(gate)) * gather(up)`."""
    seg = pairs.segments(gate.slot_of, False)
    cfg = config_for(gate.n, gate.k, x.dtype)
    got = grouped_swiglu(
        x,
        gate.stacks,
        up.stacks,
        *seg,
        pairs.size,
        int(pairs.count.max()),
        *cfg,
        group_size=gate.gs,
    )
    g, u = gate.stock(x, pairs, False), up.stock(x, pairs, False)
    stock = g * mx.sigmoid(g) * u
    cols = sc.sample(gate.n)
    eg, eu = gate.exact(x, pairs, False, cols), up.exact(x, pairs, False, cols)
    idx = mx.array(cols)
    return sc.no_worse_than_stock(got[:, idx], stock[:, idx], eg / (1.0 + np.exp(-eg)) * eu)
