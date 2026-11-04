"""QSA indexer at decode in one launch: query norm + partial rope, block scores, top-k mask.

Same arithmetic as `Qwen4ExpTextQSAIndexer.scores` + `select_device` in a fixed order; ties rank as
in the packed-key argsort (score descending, then block index ascending). The load-time check
(`check_indexer`) holds the mask to that rule on the kernel's own scores bit for bit, and the scores
to `selfcheck.no_worse_than_stock` against stock.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m4pro.kernels.constants import f32
from jitengine_qwen3_8_flash_next_m4pro.kernels.qsa_decode_mask import RANK_MASK_BODY

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# inputs: q (NH, HD) raw, w1 (HD) fp32 (1+w), bk (nb, HD) fp32, dims (3: nb, P, pos), fl (3: eps, base, scale)
# outputs: mask (P) bool, sc (nb) fp32
SOURCE = (
    """
    constexpr int NSG = NT / 32;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int NB = dims[0], P = dims[1], pos = dims[2];
    const float eps = fl[0], scale = fl[2];
    threadgroup float qf[NH * HD];
    threadgroup float s[NBMAX];
    threadgroup float red[NH];

    // ---- phase 1: normed, roped queries. One simdgroup per head (NH <= NSG).
    if (sg < NH) {
      float ss = 0.0f;
      for (int i = lane; i < HD; i += 32) { const float v = float(q[sg * HD + i]); ss += v * v; }
      ss = simd_sum(ss);
      const float rstd = metal::rsqrt(ss / float(HD) + eps);
      for (int i = lane; i < HD; i += 32) qf[sg * HD + i] = float(q[sg * HD + i]) * rstd * w1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < NH) {
      // rotate-half over the first RD dims with the STOCK cos/sin tables (1 row): xr*cos + rot*sin
      constexpr int HALF = RD / 2;
      for (int i = lane; i < HALF; i += 32) {
        const float x0 = qf[sg * HD + i], x1 = qf[sg * HD + i + HALF];
        qf[sg * HD + i] = x0 * cs[i] + (-x1) * sn[i];
        qf[sg * HD + i + HALF] = x1 * cs[i + HALF] + x0 * sn[i + HALF];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 2: block scores = sum_h relu(qf[h] . bk[j]) * scale
    for (int j = t; j < NB; j += NT) {
      float tot = 0.0f;
      #pragma unroll
      for (int h = 0; h < NH; h++) {
        float d = 0.0f;
        for (int i = 0; i < HD; i++) d += qf[h * HD + i] * bk[(ulong)j * HD + i];
        tot += metal::max(d, 0.0f);
      }
      s[j] = tot * scale;
      sc[j] = s[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 3: ranks -> mask (one simdgroup per block, lanes split the comparisons)
    for (int j = sg; j < NB; j += NSG) {
"""
    + RANK_MASK_BODY
    + """    for (int p = NB * CR + t; p < P; p += NT) mask[p] = true;
"""
)

NBMAX = 4096


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moe_engine_indexer_fused",
        input_names=["q", "w1", "bk", "cs", "sn", "dims", "fl"],
        output_names=["mask", "sc"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def supports(nh: int, hd: int, rope_dims: int, nb: int) -> bool:
    """Whether the kernel covers `nh` indexer heads (1 to 32) of width `hd` (a multiple of 32),
    `rope_dims` rotated dims (even, at most `hd`) and `nb` blocks (1 to `NBMAX`)."""
    return (
        1 <= nh <= 32
        and hd % 32 == 0
        and rope_dims % 2 == 0
        and rope_dims <= hd
        and 1 <= nb <= NBMAX
    )


def indexer_mask(
    q: mx.array,
    w1: mx.array,
    bk: mx.array,
    cos: mx.array,
    sin: mx.array,
    pos: int,
    nb: int,
    cr: int,
    topk: int,
    P: int,
    eps: float,
    base: float,
    rope_dims: int,
    hd: int,
):
    """Return (mask [1, 1, 1, P] bool, scores [nb] fp32) for raw query heads q [nh, hd] at `pos`.

    w1 [hd] is the (1+w) norm weight; bk [nb, hd] holds the pooled, normed, roped block keys.
    """
    nh = q.shape[0]
    nt = 1024
    dims = mx.array([int(nb), int(P), int(pos)], dtype=mx.int32)
    mask, sc = _kernel()(
        inputs=[
            q.reshape(nh, hd),
            w1.reshape(-1),
            bk.reshape(nb, hd),
            cos.reshape(-1),
            sin.reshape(-1),
            dims,
            f32(eps, base, hd**-0.5),
        ],
        template=[
            ("NH", nh),
            ("HD", hd),
            ("RD", int(rope_dims)),
            ("CR", int(cr)),
            ("TOPK", int(topk)),
            ("NT", nt),
            ("NBMAX", NBMAX),
        ],
        grid=(nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(P,), (nb,)],
        output_dtypes=[mx.bool_, mx.float32],
    )
    return mask[None, None, None, :], sc


# ----------------------------------------------------------------------------- load-time check
def check_indexer(nh, hd, rope_dims, cr, topk, eps, base, act_dtype) -> bool:
    """The load-time check at the model's indexer geometry, the raw query read at an offset (the
    batched step's row slice): the mask is exactly the tie rule on the kernel's own scores (its
    second output), and those scores are no less accurate than stock `Qwen4ExpTextQSAIndexer.scores`
    on the same rope tables and block keys (`selfcheck.no_worse_than_stock`). At the fewest blocks
    that prune, on each side of one pass of the score loop (1024 threads), and at NBMAX; relu leaves
    about one block in 2^nh at an exact 0, so the boundary can fall among tied zeros."""
    if not supports(nh, hd, rope_dims, topk + 1):
        raise ValueError(f"no fused indexer for {nh} heads of {hd}, rope {rope_dims}")
    w1 = 1.0 + 0.1 * sc.normal((hd,), 61)
    sizes = sorted(nb for nb in {topk + 1, 1024, 1025, NBMAX} if topk < nb <= NBMAX)
    for t, nb in enumerate(sizes):
        P = nb * cr + t % cr
        q = sc.normal((2, nh, hd), 62 + t, act_dtype)[1]
        bk = sc.normal((nb, hd), 66 + t)
        cos, sin = rope_table(np.array([P - 1]), rope_dims, base)
        args = (q, w1, bk, cos, sin, P - 1, nb, cr, topk, P, eps, base, rope_dims, hd)
        mask, scores = indexer_mask(*args)
        tail = np.ones(P - nb * cr, dtype=bool)
        want = np.concatenate([np.repeat(keep_blocks(np.array(scores), topk), cr), tail])
        if not np.array_equal(np.array(mask).reshape(-1), want):
            return False
        stock = stock_scores(q[None], w1, bk, cos, sin, eps, rope_dims)
        exact = exact_scores(q[None], w1, bk, cos, sin, eps, rope_dims)
        if not sc.no_worse_than_stock(scores[None], stock, exact):
            return False
    return True


def rope_table(positions, dims: int, base: float):
    """cos/sin [P, dims] float32 tables at integer `positions`, as `qsa_indexer.rope_tables` makes
    them (the stock tables both the kernels and stock read)."""
    inv = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    f = mx.array(np.asarray(positions)).astype(mx.float32)[:, None] * inv[None, :]
    cos, sin = mx.cos(f), mx.sin(f)
    return mx.concatenate([cos, cos], axis=-1), mx.concatenate([sin, sin], axis=-1)


def _rope(x, cos, sin, rope_dims: int, cat):
    h = rope_dims // 2
    xr = x[..., :rope_dims]
    rot = cat([-xr[..., h:], xr[..., :h]], axis=-1)
    return cat([xr * cos + rot * sin, x[..., rope_dims:]], axis=-1)


def stock_scores(q, w1, bk, cos, sin, eps: float, rope_dims: int) -> mx.array:
    """`Qwen4ExpTextQSAIndexer.scores` from raw queries q [L, nh, hd] on the given rope tables [L,
    rope_dims] and block keys [nb, hd]: the `(1 + w)` norm, the rope, and relu(q . k) summed over
    heads, as [L, nb]."""
    x = q.astype(mx.float32)
    x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * w1
    x = _rope(x.transpose(1, 0, 2), cos, sin, rope_dims, mx.concatenate)
    return mx.maximum(x @ bk.T, 0.0).sum(axis=0) * q.shape[-1] ** -0.5


def exact_scores(q, w1, bk, cos, sin, eps: float, rope_dims: int) -> np.ndarray:
    """`stock_scores` in float64."""
    x = sc.f64(q)
    x = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * sc.f64(w1)
    x = _rope(x.transpose(1, 0, 2), sc.f64(cos), sc.f64(sin), rope_dims, np.concatenate)
    return np.maximum(x @ sc.f64(bk).T, 0.0).sum(axis=0) * q.shape[-1] ** -0.5


def keep_blocks(scores: np.ndarray, topk: int) -> np.ndarray:
    """The tie rule: the `topk` blocks first by (score descending, block ascending)."""
    keep = np.zeros(scores.size, dtype=bool)
    keep[np.argsort(-scores, kind="stable")[:topk]] = True
    return keep
