"""Metal kernels around the MoE expert gathers on the decode path: router top-k and the combine.

`topk_softmax` takes the top-k of a precise fp32 softmax (ties to the lower expert id).
`moe_combine` sums the gathered pairs in their original order, sequentially over k as the stock
reduce does, adds the shared expert and applies the hyper-connection inject. The load-time check
(`check_router_combine`) holds the ids and the split reads to that bit for bit, and the
probabilities and the combine to `selfcheck.no_worse_than_stock` against the stock ops.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m2.kernels.constants import placeholder

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# inputs: logits (E) -> inds (KTOP) int32, scores (KTOP) fp32; one threadgroup of NT >= E threads
TOPK_SOURCE = """
    constexpr int NSG = NT / 32;
    const uint t = thread_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    threadgroup float prob[E];
    threadgroup float redv[NSG];
    threadgroup int redi[NSG];
    threadgroup float sel_p[KTOP];
    threadgroup int sel_i[KTOP];

    // precise softmax: max, exp(x - max), sum
    float x = (t < E) ? float(logits[t]) : -INFINITY;
    float m = simd_max(x);
    if (simd_lid == 0) redv[simd_gid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = redv[0];
    #pragma unroll
    for (int i = 1; i < NSG; i++) m = metal::max(m, redv[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = (t < E) ? metal::exp(x - m) : 0.0f;
    float ssum = simd_sum(e);
    if (simd_lid == 0) redv[simd_gid] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ssum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NSG; i++) ssum += redv[i];
    float p = e / ssum;
    if (t < E) prob[t] = p;
    bool taken = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // k rounds of arg-max over the untaken; a tie goes to the SMALLEST expert id
    for (int r = 0; r < KTOP; r++) {
      float cv = (t < E && !taken) ? p : -INFINITY;
      float mv = simd_max(cv);
      int ci = (cv == mv) ? int(t) : 0x7fffffff;
      int mi = simd_min(ci);
      if (simd_lid == 0) { redv[simd_gid] = mv; redi[simd_gid] = mi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float bv = redv[0]; int bi = redi[0];
      #pragma unroll
      for (int i = 1; i < NSG; i++) {
        if (redv[i] > bv || (redv[i] == bv && redi[i] < bi)) { bv = redv[i]; bi = redi[i]; }
      }
      if (int(t) == bi) taken = true;
      if (t == 0) { sel_p[r] = bv; sel_i[r] = bi; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t < KTOP) {
      float denom = 0.0f;
      if (NORM) {
        #pragma unroll
        for (int r = 0; r < KTOP; r++) denom += sel_p[r];
      }
      inds[t] = sel_i[t];
      scores[t] = NORM ? sel_p[t] / denom : sel_p[t];
    }
"""

# inputs: y1 (P1, HID), y2 (P2, HID), sel (KTOP) int32, scores (KTOP), sh (HID), h2 (S*HID), inj (S), shg (1: shared gate logit, SHGATE)
# -> h3 (S*HID)                                                   one thread per column of `mixed`
COMBINE_SOURCE = """
    const uint c = thread_position_in_grid.x;
    if (c >= HID) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < KTOP; k++) {
      const int r = sel[k];
      const float yv = (r < P1) ? float(y1[(ulong)r * HID + c]) : float(y2[(ulong)(r - P1) * HID + c]);
      acc += yv * float(scores[k]);
    }
    const float moe = acc + (SHGATE ? (1.0f / (1.0f + metal::exp(-float(shg[0])))) * float(sh[c]) : float(sh[c]));
    #pragma unroll
    for (int s = 0; s < S; s++) {
      const ulong i = (ulong)s * HID + c;
      h3[i] = static_cast<T>(float(h2[i]) + moe * float(inj[s]));
    }
"""


@functools.cache
def _kernel(which: str):
    if which == "topk":
        return mx.fast.metal_kernel(
            name="moe_engine_topk_softmax",
            input_names=["logits"],
            output_names=["inds", "scores"],
            header=HEADER,
            source=TOPK_SOURCE,
            ensure_row_contiguous=True,
        )
    return mx.fast.metal_kernel(
        name="moe_engine_moe_combine",
        input_names=["y1", "y2", "sel", "scores", "sh", "h2", "inj", "shg"],
        output_names=["h3"],
        header=HEADER,
        source=COMBINE_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_topk(num_experts: int, k: int) -> bool:
    """Whether `topk_softmax` takes this router: at most 1024 experts (one threadgroup holds
    them) and a top `k` of at most 64."""
    return 1 <= k <= num_experts <= 1024 and k <= 64


def topk_softmax(logits: mx.array, k: int, norm_topk: bool):
    """logits [E] -> (inds [k] int32 in descending probability, scores [k] fp32)."""
    E = int(logits.shape[-1])
    nt = max(32, -(-E // 32) * 32)
    inds, scores = _kernel("topk")(
        inputs=[logits.reshape(-1)],
        template=[
            ("T", logits.dtype),
            ("E", E),
            ("KTOP", int(k)),
            ("NT", nt),
            ("NORM", bool(norm_topk)),
        ],
        grid=(nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(k,), (k,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return inds, scores


def moe_combine(
    y1: mx.array,
    y2: mx.array | None,
    sel: np.ndarray,
    scores: mx.array,
    sh: mx.array,
    h2: mx.array,
    inject: mx.array,
    hc: int,
    hid: int,
    sh_gate: mx.array | None = None,
) -> mx.array:
    """h3 [hc*hid] = h2 + outer(sum_p scores[p] * y[sel[p]] + sh, inject).

    y1 [P1, hid] and optional y2 [P2, hid] hold the gathered pairs; `sel[p]` is the row of pair p in
    [y1; y2]. scores [k], sh [hid], h2 [hc*hid], inject [hc].
    """
    k = int(sel.size)
    P1 = int(y1.shape[0])
    y2 = placeholder(y1.dtype) if y2 is None else y2
    seli = _sel_array(sel)
    nt = 256
    shg = placeholder(mx.float32) if sh_gate is None else sh_gate.reshape(-1).astype(mx.float32)
    return _kernel("combine")(
        inputs=[
            y1,
            y2,
            seli,
            scores.reshape(-1),
            sh.reshape(-1),
            h2.reshape(-1),
            inject.reshape(-1),
            shg,
        ],
        template=[
            ("T", h2.dtype),
            ("HID", hid),
            ("S", hc),
            ("KTOP", k),
            ("P1", P1),
            ("SHGATE", sh_gate is not None),
        ],
        grid=(-(-hid // nt) * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(hc * hid,)],
        output_dtypes=[h2.dtype],
    )[0]


@functools.cache
def _identity_selection(k: int) -> mx.array:
    a = mx.array(np.arange(k, dtype=np.int32))
    mx.eval(a)
    return a


def _sel_array(sel: np.ndarray) -> mx.array:
    """`sel` as an int32 device array; the identity selection is cached per length."""
    k = int(sel.size)
    if np.array_equal(sel, np.arange(k)):
        return _identity_selection(k)
    return mx.array(np.ascontiguousarray(sel, dtype=np.int32))


# ----------------------------------------------------------------------------- load-time check
def check_router_combine(num_experts, top_k, norm_topk, hc, hid, act_dtype) -> bool:
    """The load-time check at the model's router and residual geometry. `topk_softmax`: on logits
    that tie across the k-th place, the experts in descending order with ties to the lower id, bit
    for bit; on continuous logits, the experts stock `Qwen4ExpTextSparseMoeBlock.route` picks, with probabilities no
    less accurate than stock's (`selfcheck.no_worse_than_stock`). `moe_combine`: pairs split over
    y1 and y2 and read through `sel`, as the solo decode's two gathers and the speculative miss
    correction pass them, give the bits of the assembled pairs; against the stock tail of
    `Qwen4ExpTextSparseMoeBlock.fused_decode` (weighted sum, shared expert, inject) it is no less accurate than
    stock."""
    return _check_topk(int(num_experts), int(top_k), bool(norm_topk), act_dtype) and _check_combine(
        int(top_k), int(hc), int(hid), act_dtype
    )


def _check_topk(E: int, k: int, norm: bool, dt) -> bool:
    levels = max(1, E // (2 * (k + 1)))  # about 2(k + 1) experts a level: ties across the k-th
    for t in range(3):
        grid = mx.random.randint(0, levels, (E,), key=mx.random.key(t)).astype(mx.float32) * 0.5
        inds, scores = topk_softmax(grid.astype(dt), k, norm)
        got, p = np.array(inds), np.array(scores)
        if not np.array_equal(got, np.argsort(-np.array(grid), kind="stable")[:k]):
            return False
        if not np.all(p[:-1] >= p[1:]):
            return False
    # continuous logits with a gap after the k-th, so rounding cannot reorder the boundary
    z = np.array((2.0 * sc.normal((E,), 11)).astype(dt).astype(mx.float32))
    z[np.argsort(-z, kind="stable")[k:]] -= 0.25  # exact in bfloat16 below 32
    logits = mx.array(z).astype(dt)
    inds, scores = topk_softmax(logits, k, norm)
    gates = mx.softmax(logits, axis=-1, precise=True)
    picked = mx.argpartition(gates, kth=-k, axis=-1)[-k:]
    if set(np.array(inds).tolist()) != set(np.array(picked).tolist()):
        return False
    stock = gates[inds]
    stock = stock / stock.sum() if norm else stock
    e = np.exp(z - z.max())
    p = (e / e.sum())[np.array(inds)]
    return sc.no_worse_than_stock(scores, stock, p / p.sum() if norm else p)


def _check_combine(k: int, hc: int, hid: int, dt) -> bool:
    y = sc.normal((k, hid), 21, dt)
    scores = mx.softmax(sc.normal((k,), 22))
    sh, h2 = sc.normal((hid,), 23, dt), sc.normal((hc * hid,), 24, dt)
    inject = (2.0 * mx.sigmoid(sc.normal((hc,), 25))).astype(dt)
    for gate in (None, sc.normal((1,), 26)):
        args = (scores, sh, h2, inject, hc, hid)
        whole = moe_combine(y, None, np.arange(k), *args, sh_gate=gate)
        for y1, y2, sel in _split_pairs(y, k):
            if not mx.array_equal(moe_combine(y1, y2, sel, *args, sh_gate=gate), whole).item():
                return False
        shared = sh if gate is None else mx.sigmoid(gate) * sh
        moe = (y * scores[:, None]).sum(axis=0) + shared
        stock = (h2.reshape(hc, hid) + moe[None] * inject[:, None]).reshape(-1).astype(h2.dtype)
        hs = sc.f64(sh) if gate is None else sc.f64(sh) / (1.0 + np.exp(-sc.f64(gate)))
        m = (sc.f64(y) * sc.f64(scores)[:, None]).sum(axis=0) + hs
        exact = sc.f64(h2) + (sc.f64(inject)[:, None] * m[None]).reshape(-1)
        if not sc.no_worse_than_stock(whole, stock, exact):
            return False
    return True


def _split_pairs(y, k: int):
    """(y1, y2, sel) for the k pairs of `y` split two ways: into two gathers in another order (the
    solo decode's parts), and as the speculative rows with the first and last pairs recomputed."""
    if k < 2:
        return []
    order = np.random.default_rng(1).permutation(k)  # pair j sits at row order[j] of [y1; y2]
    stacked = y[mx.array(np.argsort(order))]
    miss = np.array([0, k - 1])
    stale = mx.concatenate([mx.zeros_like(y[:1]), y[1 : k - 1], mx.zeros_like(y[:1])])
    sel = np.arange(k)
    sel[miss] = k + np.arange(miss.size)
    return [(stacked[: k // 2], stacked[k // 2 :], order), (stale, y[mx.array(miss)], sel)]
