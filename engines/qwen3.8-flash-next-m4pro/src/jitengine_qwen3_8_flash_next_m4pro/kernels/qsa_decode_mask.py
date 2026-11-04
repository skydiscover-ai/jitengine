"""Single-launch Metal kernel for the QSA decode mask that `Qwen4ExpTextQSAIndexer.select_device`
builds.

Block j is kept iff fewer than `topk` blocks precede it in (score descending, block index ascending)
order, the packed-key argsort's tie rule; every position of the trailing partial block is visible.
The load-time check (`check_qsa_mask`) holds it to the stock packed-key mask bit for bit.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

# The body of the per-block loop that ranks each block by its score and keeps the top `TOPK`, through
# the loop's closing brace; `qsa_indexer_decode` splices it into its phase 3.
RANK_MASK_BODY = """      const float sj = s[j];
      int rank = 0;
      for (int i = lane; i < NB; i += 32) {
        const float si = s[i];
        rank += (si > sj || (si == sj && i < j)) ? 1 : 0;
      }
      rank = simd_sum(rank);
      if (lane == 0) {
        const bool keep = rank < TOPK;
        #pragma unroll
        for (int c = 0; c < CR; c++) mask[j * CR + c] = keep;
      }
    }
"""

# NB and P are runtime values: a template parameter that varies with the context length would
# JIT-compile a new kernel every token.
SOURCE = (
    """
    // one SIMDGROUP per block j: its 32 lanes split the comparison loop over i and simd_sum the
    // rank. NSG simdgroups per threadgroup, NTG threadgroups over the blocks; every threadgroup
    // stages the whole score vector (a few KB) once.
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.x;
    const uint ntg = threads_per_grid.x / NT;
    const int NB = sc_shape[0];
    const int P = dims[0];
    threadgroup float s[NBMAX];
    for (int j = t; j < NB; j += NT) s[j] = float(sc[j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int NSG = NT / 32;
    for (int j = g * NSG + sg; j < NB; j += ntg * NSG) {
"""
    + RANK_MASK_BODY
    + """    if (g == 0) for (int p = NB * CR + t; p < P; p += NT) mask[p] = true;
"""
)


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moe_engine_qsa_mask",
        input_names=["sc", "dims"],
        output_names=["mask"],
        source=SOURCE,
        ensure_row_contiguous=True,
    )


NBMAX = 4096  # 16 KB of threadgroup memory; a 16k-token context at compress ratio 4


def supports(nb: int) -> bool:
    """Whether the kernel covers `nb` scored blocks: 1 to `NBMAX`, the length of the score array
    it stages in threadgroup memory."""
    return 1 <= nb <= NBMAX


def qsa_mask(sc: mx.array, nb: int, cr: int, topk: int, P: int) -> mx.array:
    """sc [nb] fp32 block scores -> bool mask [1, 1, 1, P] of the positions to attend."""
    nt = 256  # 8 simdgroups = 8 blocks per pass
    ntg = max(1, min(24, -(-nb // 32)))  # ~32 blocks per threadgroup
    out = _kernel()(
        inputs=[sc.reshape(-1), mx.array([int(P)], dtype=mx.int32)],
        template=[("CR", int(cr)), ("TOPK", int(topk)), ("NT", nt), ("NBMAX", NBMAX)],
        grid=(nt * ntg, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(P,)],
        output_dtypes=[mx.bool_],
    )[0]
    return out[None, None, None, :]


# ----------------------------------------------------------------------------- load-time check
def check_qsa_mask(cr: int, topk: int) -> bool:
    """The load-time check at the model's compress ratio `cr` and `topk` blocks: on the same scores,
    bit for bit the mask of `Qwen4ExpTextQSAIndexer.select_device`'s stock path, the packed-key
    argsort, ties included: scores with hundreds of exact ties, all zero, and mostly zero (as relu
    leaves them), at the fewest blocks that prune, past one pass of the score staging (256 threads),
    past the last growth of the threadgroup count (24 of them), and at NBMAX, with ragged tails of
    each length."""
    sizes = sorted(nb for nb in {topk + 1, 257, 23 * 32 + 1, NBMAX} if topk < nb <= NBMAX)
    for t, nb in enumerate(sizes):
        for kind in range(3):
            key = mx.random.key(10 * t + kind)
            if kind == 0:  # four values: ties across the k-th place
                sc = mx.random.randint(0, 4, (nb,), key=key).astype(mx.float32)
            elif kind == 1:
                sc = mx.zeros((nb,), dtype=mx.float32)
            else:
                sc = mx.maximum(mx.random.normal((nb,), key=key) - 1.5, 0.0)
            P = nb * cr + (t + kind) % cr
            got = qsa_mask(sc, nb, cr, topk, P)
            if not mx.array_equal(got, _stock_mask(sc, nb, cr, topk, P)).item():
                return False
    return True


def _stock_mask(sc, nb: int, cr: int, topk: int, P: int) -> mx.array:
    """`Qwen4ExpTextQSAIndexer.select_device`'s stock mask: each score mapped to an order-preserving
    int and packed with its block index, so any correct sort gives the stable tie order."""
    u = (-sc).view(mx.int32)
    k = mx.where(u < 0, mx.array(np.int32(-2147483648), dtype=mx.int32) - u, u).astype(mx.int64)
    keep = mx.argsort(mx.argsort((k << 21) | mx.arange(nb, dtype=mx.int64))) < topk
    full = mx.repeat(keep, cr)
    if P > nb * cr:
        full = mx.concatenate([full, mx.ones((P - nb * cr,), dtype=mx.bool_)])
    return full[None, None, None, :]
