"""On-device QSA mask for a prefill chunk of L queries (the device form of `mask_from_scores`).

Row i (position p = s0 + i) is plain causal when it has at most `topk` complete blocks; otherwise it
keeps the top `topk` blocks by (score desc, block asc), the numpy path's stable-argsort tie rule,
plus the ragged tail [nb * cr, p].

The kernel computes the scores itself, in another order than stock, so the load-time check states
its tolerance against stock's mask (`Qwen4ExpTextQSAIndexer.select`): causal rows and tails match
exactly, and a pruned row may differ only at blocks within `tol` of a near tie at the top-k boundary
of the exact scores, `tol` = 2 (9 e + 4 ulp): e is stock's largest score error against float64, the
kernel's taken to be at most 8 times it as the decode kernel's are checked, and a reordering takes
both scores of the pair to move.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m4pro.kernels.constants import f32
from jitengine_qwen3_8_flash_next_m4pro.kernels.qsa_indexer_decode import (
    exact_scores,
    rope_table,
    stock_scores,
)

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# inputs: q (L, NH, HD) raw, w1 (HD), bk (nbmax, HD), cs (L, RD) cos table, sn (L, RD) sin table (the stock
# `rope_tables` values, duplicated over the two halves), dims (5: L, P, s0, CR, TOPK), fl (3: eps, base, scale)
# output: mask (L, P) bool.   One threadgroup of NT threads per query row.
SOURCE = """
    constexpr int NSG = NT / 32;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int i = threadgroup_position_in_grid.x;
    const int L = dims[0], P = dims[1], s0 = dims[2], CR = dims[3], TOPK = dims[4];
    const float eps = fl[0], scale = fl[2];
    const int p = s0 + i;
    const int nbi = (p + 1) / CR;
    device bool* row = mask + (ulong)i * P;
    if (nbi <= TOPK) {                                   // the causal row: every position <= p
      for (int c = t; c < P; c += NT) row[c] = (c <= p);
      return;
    }
    threadgroup float qf[NH * HD];
    threadgroup float s[NBMAX];
    // phase 1: normed, roped query heads of this row
    if (sg < NH) {
      const device T* qh = q + ((ulong)i * NH + sg) * HD;
      float ss = 0.0f;
      for (int d = lane; d < HD; d += 32) { const float v = float(qh[d]); ss += v * v; }
      ss = simd_sum(ss);
      const float rstd = metal::rsqrt(ss / float(HD) + eps);
      for (int d = lane; d < HD; d += 32) qf[sg * HD + d] = float(qh[d]) * rstd * w1[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < NH) {
      constexpr int HALF = RD / 2;
      // rotate-half with the STOCK tables: xr*cos + rot*sin, rot = [-x1 | x0]
      for (int d = lane; d < HALF; d += 32) {
        const float c0 = cs[(ulong)i * RD + d], s0_ = sn[(ulong)i * RD + d];
        const float c1 = cs[(ulong)i * RD + d + HALF], s1_ = sn[(ulong)i * RD + d + HALF];
        const float x0 = qf[sg * HD + d], x1 = qf[sg * HD + d + HALF];
        qf[sg * HD + d] = x0 * c0 + (-x1) * s0_;
        qf[sg * HD + d + HALF] = x1 * c1 + x0 * s1_;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // phase 2: scores of the nbi valid blocks
    for (int j = t; j < nbi; j += NT) {
      float tot = 0.0f;
      #pragma unroll
      for (int h = 0; h < NH; h++) {
        float dd = 0.0f;
        for (int d = 0; d < HD; d++) dd += qf[h * HD + d] * bk[(ulong)j * HD + d];
        tot += metal::max(dd, 0.0f);
      }
      s[j] = tot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // phase 3: ranks among the valid blocks -> kept blocks' positions
    for (int j = sg; j < nbi; j += NSG) {
      const float sj = s[j];
      int rank = 0;
      for (int k = lane; k < nbi; k += 32) {
        const float sk = s[k];
        rank += (sk > sj || (sk == sj && k < j)) ? 1 : 0;
      }
      rank = simd_sum(rank);
      if (lane == 0) {
        const bool keep = rank < TOPK;
        for (int c = 0; c < CR; c++) row[j * CR + c] = keep;
      }
    }
    // the ragged tail [nbi*CR, p] is visible; past p nothing is
    for (int c = nbi * CR + t; c < P; c += NT) row[c] = (c <= p);
"""

NBMAX = 4096


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moe_engine_indexer_prefill_mask",
        input_names=["q", "w1", "bk", "cs", "sn", "dims", "fl"],
        output_names=["mask"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def supports(nh: int, hd: int, rope_dims: int, nb: int) -> bool:
    """Whether the kernel covers `nh` indexer heads (1 to 32) of width `hd` (a multiple of 32),
    `rope_dims` rotated dims (even, at most `hd`) and `nb` blocks (0 to `NBMAX`)."""
    return (
        1 <= nh <= 32
        and hd % 32 == 0
        and rope_dims % 2 == 0
        and rope_dims <= hd
        and 0 <= nb <= NBMAX
    )


def prefill_mask(
    q: mx.array,
    w1: mx.array,
    bk: mx.array,
    cos: mx.array,
    sin: mx.array,
    s0: int,
    L: int,
    P: int,
    cr: int,
    topk: int,
    eps: float,
    base: float,
    rope_dims: int,
    hd: int,
) -> mx.array:
    """Raw q [L, nh, hd] at s0..s0+L-1, block keys bk [P // cr, hd] -> bool mask [1, 1, L, P]."""
    nh = q.shape[1]
    nt = 1024
    nb = bk.shape[0]
    dims = mx.array([int(L), int(P), int(s0), int(cr), int(topk)], dtype=mx.int32)
    mask = _kernel()(
        inputs=[
            q.reshape(L, nh, hd),
            w1.reshape(-1),
            bk.reshape(max(nb, 1), hd),
            cos.reshape(L, -1),
            sin.reshape(L, -1),
            dims,
            f32(eps, base, hd**-0.5),
        ],
        template=[
            ("T", q.dtype),
            ("NH", nh),
            ("HD", hd),
            ("RD", int(rope_dims)),
            ("NT", nt),
            ("NBMAX", NBMAX),
        ],
        grid=(nt * L, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(L, P)],
        output_dtypes=[mx.bool_],
    )[0]
    return mask[None, None]


# ----------------------------------------------------------------------------- load-time check
def check_prefill_mask(nh, hd, rope_dims, cr, topk, eps, base, act_dtype) -> bool:
    """The load-time check at the model's indexer geometry, against stock's mask (stock scores and
    the numpy tie rule) within the tolerance in the module docstring: chunks of rows on each side of
    the first pruned position, of the first row past one pass of the score loop (1024 valid
    blocks), and of the last position NBMAX blocks allow. Rows are independent threadgroups, so a
    few rows on each side of a switch stand for a chunk of any length."""
    if not supports(nh, hd, rope_dims, topk + 1):
        raise ValueError(f"no prefill mask kernel for {nh} heads of {hd}, rope {rope_dims}")
    w1 = 1.0 + 0.1 * sc.normal((hd,), 71)
    rows = 8
    ends = {cr * (topk + 1) + 4, cr * 1025 + 4, cr * NBMAX + cr - 1}  # P of each chunk
    for t, P in enumerate(sorted(e for e in ends if e // cr <= NBMAX)):
        s0 = P - rows
        q = sc.normal((rows, nh, hd), 72 + t, act_dtype)
        bk = sc.normal((P // cr, hd), 75 + t)
        cos, sin = rope_table(np.arange(s0, P), rope_dims, base)
        args = (q, w1, bk, cos, sin, s0, rows, P, cr, topk, eps, base, rope_dims, hd)
        got = np.array(prefill_mask(*args))[0, 0]
        stock = stock_scores(q, w1, bk, cos, sin, eps, rope_dims)
        exact = exact_scores(q, w1, bk, cos, sin, eps, rope_dims)
        want = _stock_mask(np.array(stock), s0, P, cr, topk)
        err = float(np.abs(sc.f64(stock) - exact).max())
        tol = 2 * (9 * err + 4 * sc.U32 * float(np.abs(exact).max()))
        if not _differs_only_at_ties(got, want, exact, tol, s0, cr, topk):
            return False
    return True


def _stock_mask(scores: np.ndarray, s0: int, P: int, cr: int, topk: int) -> np.ndarray:
    """`Qwen4ExpTextQSAIndexer.mask_from_scores`: causal, minus the blocks a pruned row does not
    keep."""
    pos = np.arange(s0, s0 + scores.shape[0])
    mask = np.arange(P)[None, :] <= pos[:, None]
    for r, nbi in enumerate((pos + 1) // cr):
        if nbi > topk:
            for j in np.argsort(-scores[r, :nbi], kind="stable")[topk:]:
                mask[r, j * cr : (j + 1) * cr] = False
    return mask


def _differs_only_at_ties(got, want, exact, tol, s0, cr, topk) -> bool:
    for i in np.nonzero((got != want).any(axis=1))[0]:
        nbi = (s0 + i + 1) // cr
        cols = np.nonzero(got[i] != want[i])[0]
        if nbi <= topk or cols.max() >= nbi * cr:  # a causal row or a tail differs
            return False
        row = np.sort(exact[i, :nbi])[::-1]
        edge = 0.5 * (row[topk - 1] + row[topk])
        if row[topk - 1] - row[topk] > tol:
            return False
        if any(abs(exact[i, j] - edge) > tol for j in np.unique(cols // cr)):
            return False
    return True
