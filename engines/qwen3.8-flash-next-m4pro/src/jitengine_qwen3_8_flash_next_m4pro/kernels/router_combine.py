"""Metal kernels around the MoE expert gathers on the decode path: router top-k and the combine.

`topk_softmax` takes the top-k of a precise fp32 softmax (ties to the lower expert id), per-simdgroup
lists merged by one simdgroup.
`moe_combine` sums the gathered pairs in their original order, sequentially over k as the stock
reduce does, adds the shared expert and applies the hyper-connection inject.
`topk_softmax2` and `moe_combine2` are the two positions of the MTP verify pass in one launch each:
every row runs the one-row source on its own row (the combine also resolving the pairs' slots), so
each row is the one-row kernel's bits. The load-time check (`check_router_combine`) holds the ids,
the split reads and the two-row launches' rows to that bit for bit, and the probabilities and the
combine to `selfcheck.no_worse_than_stock` against the stock ops.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m4pro.kernels.constants import placeholder

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# inputs: logits (E) -> inds (KTOP) int32, scores (KTOP) fp32; one threadgroup of NT >= E threads.
# The selection runs in two stages with no threadgroup barrier inside either loop: each simdgroup takes
# its own top-KTOP in (probability descending, id ascending) order, then simdgroup 0 merges the NSG
# sorted lists (one head per lane). Selection is exact under that total order, so the ids, their order
# and the probabilities are those of KTOP rounds of a threadgroup-wide arg-max (the kernel this replaced,
# kept as the reference in tests/metal/kernels/test_router_combine.py); the renorm sums them in
# selection order as before.
TOPK_SOURCE = """
    constexpr int NSG = NT / 32;
    const uint t = thread_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    threadgroup float redv[NSG];
    threadgroup float cand_p[NSG * KTOP];
    threadgroup int cand_i[NSG * KTOP];
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

    // each simdgroup's own top-KTOP, (probability descending, id ascending), no threadgroup barrier
    bool taken = false;
    for (int r = 0; r < KTOP; r++) {
      float cv = (t < E && !taken) ? p : -INFINITY;
      float mv = simd_max(cv);
      int ci = (cv == mv) ? int(t) : 0x7fffffff;
      int mi = simd_min(ci);
      if (int(t) == mi) taken = true;
      if (simd_lid == 0) { cand_p[simd_gid * KTOP + r] = mv; cand_i[simd_gid * KTOP + r] = mi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // simdgroup 0 merges the NSG sorted lists, lane j holding list j's head; a tie goes to the
    // SMALLEST expert id (the lists hold disjoint id ranges, so the winning head is unique)
    if (simd_gid == 0) {
      const bool live = simd_lid < NSG;
      int pos = 0;
      float hv = live ? cand_p[simd_lid * KTOP] : -INFINITY;
      int hi = live ? cand_i[simd_lid * KTOP] : 0x7fffffff;
      for (int r = 0; r < KTOP; r++) {
        float mv = simd_max(hv);
        int ci = (hv == mv) ? hi : 0x7fffffff;
        int mi = simd_min(ci);
        if (simd_lid == 0) { sel_p[r] = mv; sel_i[r] = mi; }
        if (live && hi == mi) {
          pos++;
          hv = (pos < KTOP) ? cand_p[simd_lid * KTOP + pos] : -INFINITY;
          hi = (pos < KTOP) ? cand_i[simd_lid * KTOP + pos] : 0x7fffffff;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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


# inputs: logits0, logits1 (E each) -> inds (2*KTOP) int32, scores (2*KTOP) fp32, row 0's then row 1's.
# Threadgroup r runs TOPK_SOURCE on row r, reading and writing through the names it uses.
TOPK2_SOURCE = (
    """
    const device T* logits = threadgroup_position_in_grid.x == 0 ? logits0 : logits1;
    device int* inds = inds2 + threadgroup_position_in_grid.x * KTOP;
    device float* scores = scores2 + threadgroup_position_in_grid.x * KTOP;
"""
    + TOPK_SOURCE
)

# inputs: y (2*KTOP, HID) the gathered pairs, inds (2*KTOP) int32, scores (2*KTOP) fp32, tab (ids) int32
# slot table, off (1) int32 the layer's first id, sh (2, HID), shg0, shg1 (1 each: shared gate logits,
# SHGATE), h2 (2, S*HID), inj (2, S) -> h3 (2, S*HID); grid row r is position r. COMBINE_SOURCE per row,
# its pairs read in place (sel = identity) and each weighed as `spec_build` does: zero when the
# expert's slot is not resident.
COMBINE2_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (c >= HID) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < KTOP; k++) {
      const int p = int(r) * KTOP + k;
      const float sc = (tab[inds[p] + off[0]] >= 0) ? float(scores[p]) : 0.0f;
      const float yv = float(y[(ulong)p * HID + c]);
      acc += yv * sc;
    }
    const float moe = acc + (SHGATE ? (1.0f / (1.0f + metal::exp(-float(r == 0 ? shg0[0] : shg1[0])))) * float(sh[(ulong)r * HID + c]) : float(sh[(ulong)r * HID + c]));
    #pragma unroll
    for (int s = 0; s < S; s++) {
      const ulong i = (ulong)r * S * HID + (ulong)s * HID + c;
      h3[i] = static_cast<T>(float(h2[i]) + moe * float(inj[r * S + s]));
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
    if which == "topk2":
        return mx.fast.metal_kernel(
            name="moe_engine_topk_softmax2",
            input_names=["logits0", "logits1"],
            output_names=["inds2", "scores2"],
            header=HEADER,
            source=TOPK2_SOURCE,
            ensure_row_contiguous=True,
        )
    if which == "combine2":
        return mx.fast.metal_kernel(
            name="moe_engine_moe_combine2",
            input_names=["y", "inds", "scores", "tab", "off", "sh", "shg0", "shg1", "h2", "inj"],
            output_names=["h3"],
            header=HEADER,
            source=COMBINE2_SOURCE,
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


def supports_pair(num_experts: int, k: int) -> bool:
    """Whether `topk_softmax2` takes this router: `supports_topk`'s, with at least eight experts so
    each row's logits are a device buffer (a smaller input lands in constant address space)."""
    return supports_topk(num_experts, k) and num_experts >= 8


def topk_softmax2(logits0: mx.array, logits1: mx.array, k: int, norm_topk: bool):
    """`topk_softmax` of two rows in one launch: logits0, logits1 (E elements each) -> (inds [2k]
    int32, scores [2k] fp32), row 0's k then row 1's, each row `topk_softmax`'s bits."""
    E = int(logits0.size)
    nt = max(32, -(-E // 32) * 32)
    inds, scores = _kernel("topk2")(
        inputs=[logits0, logits1],
        template=[
            ("T", logits0.dtype),
            ("E", E),
            ("KTOP", int(k)),
            ("NT", nt),
            ("NORM", bool(norm_topk)),
        ],
        grid=(2 * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(2 * k,), (2 * k,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return inds, scores


def moe_combine2(
    y: mx.array,
    inds: mx.array,
    scores: mx.array,
    tab: mx.array,
    off: mx.array,
    sh: mx.array,
    shg0: mx.array | None,
    shg1: mx.array | None,
    h2: mx.array,
    inject: mx.array,
    hc: int,
    hid: int,
) -> mx.array:
    """Two positions' speculative combine in one launch: h3 [2, hc*hid], row r the bits of
    `moe_combine(y[r*k:(r+1)*k], None, arange(k), where(tab[inds_r + off] >= 0, scores_r, 0), sh[r],
    h2[r], inject[r], sh_gate=shg_r)`.

    y [2k, hid] holds row 0's k gathered pairs then row 1's (`_gather_pair`); inds, scores [2k] as
    `topk_softmax2` gives them; tab the device slot table and off [1] the layer's first id; sh [2,
    hid], h2 [2, hc*hid], inject [2, hc]; shg0, shg1 each row's shared gate logit (any float dtype:
    read as fp32, as `moe_combine`'s exact cast does), or both None.
    """
    k = int(inds.size) // 2
    gate = shg0 is not None
    g0 = shg0 if gate else placeholder(mx.float32)
    g1 = shg1 if gate else placeholder(mx.float32)
    nt = 256
    return _kernel("combine2")(
        inputs=[y, inds, scores, tab, off, sh, g0, g1, h2, inject],
        template=[
            ("T", h2.dtype),
            ("HID", hid),
            ("S", hc),
            ("KTOP", k),
            ("SHGATE", gate),
        ],
        grid=(-(-hid // nt) * nt, 2, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(2, hc * hid)],
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
    for bit; on continuous logits, the experts stock `Qwen4ExpTextSparseMoeBlock.route` picks, with
    probabilities no less accurate than stock's (`selfcheck.no_worse_than_stock`). `moe_combine`:
    pairs split over y1 and y2 and read through `sel`, as the solo decode's two gathers and the
    speculative miss correction pass them, give the bits of the assembled pairs; against the stock
    tail of `Qwen4ExpTextSparseMoeBlock.fused_decode` (weighted sum, shared expert, inject) it is no
    less accurate than stock. `topk_softmax2` and `moe_combine2`: each row the one-row kernel's bits
    (the combine's row after the masked weights `spec_build_pair` passes it)."""
    return (
        _check_topk(int(num_experts), int(top_k), bool(norm_topk), act_dtype)
        and _check_combine(int(top_k), int(hc), int(hid), act_dtype)
        and _check_pair(int(num_experts), int(top_k), bool(norm_topk), int(hc), int(hid), act_dtype)
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


def _check_pair(E: int, k: int, norm: bool, hc: int, hid: int, dt) -> bool:
    if not supports_pair(E, k):
        return True  # never launched: the pair build takes its eager body
    l0, l1 = (sc.normal((1, E), 31 + r, dt) for r in range(2))
    inds, scores = topk_softmax2(l0, l1, k, norm)
    rows = [topk_softmax(x.reshape(-1), k, norm) for x in (l0, l1)]
    if not (
        mx.array_equal(inds, mx.concatenate([r[0] for r in rows])).item()
        and mx.array_equal(scores, mx.concatenate([r[1] for r in rows])).item()
    ):
        return False
    tab = mx.array(np.where(np.arange(2 * E) % 3 == 0, -1, np.arange(2 * E)).astype(np.int32))
    off = mx.array([E], dtype=mx.int32)
    y, sh = sc.normal((2 * k, hid), 33, dt), sc.normal((2, hid), 34, dt)
    h2 = sc.normal((2, hc * hid), 35, dt)
    inject = (2.0 * mx.sigmoid(sc.normal((2, hc), 36))).astype(dt)
    g = [sc.normal((1, 1), 37 + r, dt) for r in range(2)]
    got = moe_combine2(y, inds, scores, tab, off, sh, g[0], g[1], h2, inject, hc, hid)
    for r in range(2):
        ir, sr = inds[r * k : (r + 1) * k], scores[r * k : (r + 1) * k]
        w = mx.where(mx.take(tab, ir + E) >= 0, sr, mx.zeros_like(sr))
        one = moe_combine(
            y[r * k : (r + 1) * k],
            None,
            np.arange(k),
            w,
            sh[r : r + 1],
            h2[r : r + 1],
            inject[r : r + 1],
            hc,
            hid,
            sh_gate=g[r].reshape(-1),
        )
        if not mx.array_equal(got[r], one).item():
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
