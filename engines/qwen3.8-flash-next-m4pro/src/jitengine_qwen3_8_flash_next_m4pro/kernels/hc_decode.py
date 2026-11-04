"""Decode-step hyper-connection (`Qwen4ExpTextGatedResidual._decode_body`) as fused 4-bit Metal
kernels.

`pre`/`down` apply the optional residual inject, per-stream rstd and the mix_down and block_inject
matvecs over the normed rows; `up` applies silu, mix_up, the sigmoid gate and the stream mean.
Stock arithmetic in a different, fixed reduction order, so it is used on the decode path only and
prefill and scoring stay bit-exact. The load-time check (`check_hc`) holds a batched row to the row
alone bit for bit, and the outputs to `selfcheck.no_worse_than_stock` against the stock ops.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m4pro.kernels.constants import f32, placeholder

HEADER = """
#include <metal_simdgroup>
using namespace metal;

template <typename U, int N, typename W>
inline void dequantize4(const device uint8_t* w, U scale, U bias, W w_local) {
  const float s = float(scale);
  const float b = float(bias);
  float sc[2] = {s, s / 16.0f};
  for (int i = 0; i < (N / 2); i++) {
    w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
    w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
  }
}
inline float sigmoid_f(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
"""


# inputs: h, bo, inj, epsv -> outputs: h2, rstd     (one threadgroup of NT threads per batch row)
PRE_SOURCE = """
    constexpr int K = S * HID;
    constexpr int NSG = NT / 32;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint tid_tg = thread_index_in_threadgroup;
    const int m = threadgroup_position_in_grid.x;
    const ulong bo0 = (ulong)m * HID, inj0 = (ulong)m * S;
    const device T* hrow = h + (ulong)m * K;
    device T* h2row = h2 + (ulong)m * K;
    threadgroup float red[NSG * S];
    #pragma unroll
    for (int s = 0; s < S; s++) {
      float acc = 0.0f;
      const float is = INJ ? float(inj[inj0 + s]) : 0.0f;
      for (int c = tid_tg; c < HID; c += NT) {
        float v = float(hrow[s * HID + c]);
        if (INJ) { v += float(bo[bo0 + c]) * is; h2row[s * HID + c] = static_cast<T>(v); }
        acc += v * v;
      }
      acc = simd_sum(acc);
      if (simd_lid == 0) red[simd_gid * S + s] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_tg < S) {
      float tot = 0.0f;
      #pragma unroll
      for (int gi = 0; gi < NSG; gi++) tot += red[gi * S + tid_tg];
      rstd[m * S + tid_tg] = metal::rsqrt(tot / float(HID) + epsv[0]);
    }
"""

# inputs: h, w1, wd, sd, bd, wi, si, bi, bo, inj, epsv, rstd_in  ->  outputs: y, injraw, rstd, h2
DOWN_SOURCE = """
    typedef float U;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int K = S * HID;
    constexpr int KG = K / GS;
    constexpr int slots_per_sg = 32 / KL;
    constexpr int rows_per_sg = slots_per_sg * RB;
    constexpr int rows_per_tg = rows_per_sg * NSG;
    constexpr int NT = 32 * NSG;
    constexpr int NROWS = ND + NI;
    constexpr int wbytes = K * bits / 8;

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint tid_tg = thread_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int m = tid.z;
    const bool writer = (tid.y == 0);

    const device T* hrow = h + (ulong)m * K;
    // bo / inj are indexed through the parameter itself: small arrays may land in constant space
    const ulong bo0 = (ulong)m * HID, inj0 = (ulong)m * S;
    device T* h2row = h2 + (ulong)m * K;

    // ---- phase 1: (residual inject) + per-stream sum of squares (skipped when PRE did it)
    threadgroup float red[NSG * S];
    float rs[S];
    if (PRE) {
      #pragma unroll
      for (int s = 0; s < S; s++) rs[s] = rstd_in[m * S + s];
    } else {
    #pragma unroll
    for (int s = 0; s < S; s++) {
      float acc = 0.0f;
      const float is = INJ ? float(inj[inj0 + s]) : 0.0f;
      for (int c = tid_tg; c < HID; c += NT) {
        float v = float(hrow[s * HID + c]);
        if (INJ) {
          v += float(bo[bo0 + c]) * is;
          if (writer) h2row[s * HID + c] = static_cast<T>(v);
        }
        acc += v * v;
      }
      acc = simd_sum(acc);
      if (simd_lid == 0) red[simd_gid * S + s] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int s = 0; s < S; s++) {
      float tot = 0.0f;
      #pragma unroll
      for (int gi = 0; gi < NSG; gi++) tot += red[gi * S + s];
      rs[s] = metal::rsqrt(tot / float(HID) + epsv[0]);
    }
    if (writer && tid_tg < S) rstd[m * S + tid_tg] = rs[tid_tg];
    }
    if (NROWS == 0) return;

    // ---- phase 2: the two matvecs over normed(h2), rows [0, ND) of wd then [ND, ND+NI) of wi
    const short k_lane = simd_lid % KL;
    const short slot = simd_lid / KL;
    const int row0 = tid.y * rows_per_tg + rows_per_sg * simd_gid + slot * RB;

    const device uint8_t* wrow[RB];
    const device SB* srow[RB];
    const device SB* brow[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      int row = min(row0 + r, NROWS - 1);
      if (row < ND) {
        wrow[r] = (const device uint8_t*)wd + (ulong)row * wbytes;
        srow[r] = sd + (ulong)row * KG;
        brow[r] = bd + (ulong)row * KG;
      } else {
        row -= ND;
        wrow[r] = (const device uint8_t*)wi + (ulong)row * wbytes;
        srow[r] = si + (ulong)row * KG;
        brow[r] = bi + (ulong)row * KG;
      }
    }
    U result[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) result[r] = 0;

    for (int g = k_lane; g < KG; g += KL) {
      U scale[RB], bias[RB];
      #pragma unroll
      for (int r = 0; r < RB; r++) { scale[r] = static_cast<U>(srow[r][g]); bias[r] = static_cast<U>(brow[r][g]); }
      #pragma unroll
      for (int sc = 0; sc < GS / sub; sc++) {
        const int k0 = g * GS + sc * sub;
        const int s = k0 / HID;
        const float rsd = rs[s];
        const float is = INJ ? float(inj[inj0 + s]) : 0.0f;
        U xr[sub];
        #pragma unroll
        for (int i = 0; i < sub; i++) {
          const int k = k0 + i;
          float v = float(hrow[k]);
          if (INJ) v += float(bo[bo0 + (k - s * HID)]) * is;
          xr[i] = v * rsd * float(w1[k]);
        }
        #pragma unroll
        for (int r = 0; r < RB; r++) {
          U w_dq[sub];
          dequantize4<U, sub>(wrow[r] + k0 * bits / 8, scale[r], bias[r], w_dq);
          U acc = 0;
          #pragma unroll
          for (int i = 0; i < sub; i++) acc += xr[i] * w_dq[i];
          result[r] += acc;
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < RB; r++) {
      if (KL >= 32) result[r] += simd_shuffle_down(result[r], 16);
      if (KL >= 16) result[r] += simd_shuffle_down(result[r], 8);
      if (KL >= 8)  result[r] += simd_shuffle_down(result[r], 4);
      if (KL >= 4)  result[r] += simd_shuffle_down(result[r], 2);
      if (KL >= 2)  result[r] += simd_shuffle_down(result[r], 1);
    }
    if (k_lane == 0) {
      #pragma unroll
      for (int r = 0; r < RB; r++) {
        const int row = row0 + r;
        if (row < ND) y[(ulong)m * ND + row] = static_cast<T>(result[r]);
        else if (row < NROWS) injraw[(ulong)m * NI + (row - ND)] = static_cast<T>(result[r]);
      }
    }
"""

# inputs: hn, w1, rstd, y, wu, su, bu, injraw  ->  outputs: mixed, inject
# lane layout inside a simdgroup: LPR lanes per weight row, S rows (one per stream) per column,
# CPS = 32 / (S * LPR) columns per simdgroup; NSG simdgroups per threadgroup.
UP_SOURCE = """
    typedef float U;
    constexpr int bits = 4;
    constexpr int sub = 8;
    constexpr int K = S * HID;
    constexpr int KG2 = ND / GS;
    constexpr int CPS = 32 / (S * LPR);
    constexpr int NT = 32 * NSG;
    constexpr int wbytes = ND * bits / 8;

    const uint simd_lid = thread_index_in_simdgroup;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint tid_tg = thread_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int m = tid.z;

    threadgroup float wv[ND];
    for (int j = tid_tg; j < ND; j += NT) {
      const float t = float(y[(ulong)m * ND + j]) / float(S);
      wv[j] = t * sigmoid_f(t);
    }
    if (NI > 0 && tid.x == 0 && tid_tg < NI) {
      inject[(ulong)m * NI + tid_tg] = static_cast<T>(2.0f * sigmoid_f(float(injraw[(ulong)m * NI + tid_tg]) / float(S)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int cl = simd_lid / (S * LPR);                 // column within the simdgroup
    const int s = (simd_lid / LPR) % S;                  // stream
    const short kl = simd_lid % LPR;
    const int c = (tid.x * NSG + simd_gid) * CPS + cl;
    const int row = s * HID + min(c, HID - 1);
    const device uint8_t* wrow = (const device uint8_t*)wu + (ulong)row * wbytes;
    const device SB* srow = su + (ulong)row * KG2;
    const device SB* brow = bu + (ulong)row * KG2;

    U acc = 0;
    for (int g = kl; g < KG2; g += LPR) {
      const U scale = static_cast<U>(srow[g]);
      const U bias = static_cast<U>(brow[g]);
      #pragma unroll
      for (int sc = 0; sc < GS / sub; sc++) {
        const int k0 = g * GS + sc * sub;
        U w_dq[sub];
        dequantize4<U, sub>(wrow + k0 * bits / 8, scale, bias, w_dq);
        U a = 0;
        #pragma unroll
        for (int i = 0; i < sub; i++) a += wv[k0 + i] * w_dq[i];
        acc += a;
      }
    }
    if (LPR >= 16) acc += simd_shuffle_down(acc, 8);
    if (LPR >= 8)  acc += simd_shuffle_down(acc, 4);
    if (LPR >= 4)  acc += simd_shuffle_down(acc, 2);
    if (LPR >= 2)  acc += simd_shuffle_down(acc, 1);
    float v = 0.0f;
    if (kl == 0) {
      const float normed = float(hn[(ulong)m * K + row]) * rstd[m * S + s] * float(w1[row]);
      v = sigmoid_f(acc) * normed;
    }
    if (S >= 8) v += simd_shuffle_down(v, 4 * LPR);
    if (S >= 4) v += simd_shuffle_down(v, 2 * LPR);
    if (S >= 2) v += simd_shuffle_down(v, LPR);
    if (kl == 0 && s == 0 && c < HID) mixed[(ulong)m * HID + c] = static_cast<T>(v / float(S));
"""


@functools.cache
def _kernel(which: str):
    if which == "down":
        return mx.fast.metal_kernel(
            name="moe_engine_hc_down",
            input_names=[
                "h",
                "w1",
                "wd",
                "sd",
                "bd",
                "wi",
                "si",
                "bi",
                "bo",
                "inj",
                "epsv",
                "rstd_in",
            ],
            output_names=["y", "injraw", "rstd", "h2"],
            header=HEADER,
            source=DOWN_SOURCE,
            ensure_row_contiguous=True,
        )
    if which == "pre":
        return mx.fast.metal_kernel(
            name="moe_engine_hc_pre",
            input_names=["h", "bo", "inj", "epsv"],
            output_names=["h2", "rstd"],
            header=HEADER,
            source=PRE_SOURCE,
            ensure_row_contiguous=True,
        )
    return mx.fast.metal_kernel(
        name="moe_engine_hc_up",
        input_names=["hn", "w1", "rstd", "y", "wu", "su", "bu", "injraw"],
        output_names=["mixed", "inject"],
        header=HEADER,
        source=UP_SOURCE,
        ensure_row_contiguous=True,
    )


def supports(
    hc: int, hid: int, lowrank: int, gs: int, bits: int, mix_down, mix_up, block_inject
) -> bool:
    """Whether the fused kernels handle this geometry (4-bit affine mixes, one group size)."""
    if bits != 4 or hc not in (2, 4, 8):
        return False
    if (hid * hc) % gs or lowrank % gs or gs % 8 or hid % 8:
        return False
    for lin in (mix_down, mix_up) + ((block_inject,) if block_inject is not None else ()):
        if lin.bits != 4 or lin.group_size != gs or lin.biases is None:
            return False
    if mix_down.out_dims != lowrank or mix_down.in_dims != hc * hid:
        return False
    if mix_up.out_dims != hc * hid or mix_up.in_dims != lowrank:
        return False
    if block_inject is not None and (
        block_inject.out_dims != hc or block_inject.in_dims != hc * hid
    ):
        return False
    return True


def hc_fused(
    h: mx.array,
    w1: mx.array,
    mix_down,
    mix_up,
    block_inject,
    hc: int,
    hid: int,
    eps: float,
    block_out: mx.array | None = None,
    inject_in: mx.array | None = None,
    down_cfg=(16, 4, 1),
    up_cfg=(8, 4),
    pre: bool = True,
    pre_nt: int = 256,
    _stop: str = "",
):
    """Return (mixed [M, hid], inject [M, hc] or None, h2 [M, hc*hid]) for h [M, hc*hid].

    With block_out [M, hid] and inject_in [M, hc], h2 is the residual after that inject, else h.
    `down_cfg` = (KL, NSG, RB) and `up_cfg` = (lanes per row, simdgroups) shape the launches;
    `pre` computes the inject and norm statistics in their own launch.
    """
    M, K = h.shape
    T = h.dtype
    ND = mix_down.out_dims
    NI = block_inject.out_dims if block_inject is not None else 0
    INJ = 1 if block_out is not None else 0
    gs = mix_down.group_size
    KL, NSG, RB = down_cfg
    LPR, UNSG = up_cfg
    rows_per_tg = (32 // KL) * RB * NSG
    nrows = ND + NI
    wi, si, bi = _inject_operands(block_inject, mix_down, NI)
    bo, inj = _block_operands(block_out, inject_in, INJ, T)
    if _stop == "none":  # diagnostics only: no launch at all
        return h[:, :hid], None, h
    if pre:
        h2, rstd, hn = _launch_pre(h, bo, inj, eps, T, hc, hid, INJ, pre_nt, M, K)
        if _stop == "pre":
            return hn[:, :hid] + rstd[:, :1], None, hn
        tmpl = [
            ("T", T),
            ("SB", mix_down.scales.dtype),
            ("KL", KL),
            ("NSG", NSG),
            ("RB", RB),
            ("GS", gs),
            ("S", hc),
            ("HID", hid),
            ("ND", ND),
            ("NI", NI),
            ("INJ", 0),
            ("PRE", 1),
        ]
        y, injraw, _, _ = _kernel("down")(
            inputs=[
                hn,
                w1,
                mix_down.weight,
                mix_down.scales,
                mix_down.biases,
                wi,
                si,
                bi,
                placeholder(T),
                placeholder(T),
                f32(eps),
                rstd,
            ],
            template=tmpl,
            grid=(32, NSG * ((nrows + rows_per_tg - 1) // rows_per_tg), M),
            threadgroup=(32, NSG, 1),
            output_shapes=[(M, ND), (M, NI) if NI else (8,), (8,), (8,)],
            output_dtypes=[T, T, mx.float32, T],
        )
    else:
        tmpl = [
            ("T", T),
            ("SB", mix_down.scales.dtype),
            ("KL", KL),
            ("NSG", NSG),
            ("RB", RB),
            ("GS", gs),
            ("S", hc),
            ("HID", hid),
            ("ND", ND),
            ("NI", NI),
            ("INJ", INJ),
            ("PRE", 0),
        ]
        y, injraw, rstd, h2 = _kernel("down")(
            inputs=[
                h,
                w1,
                mix_down.weight,
                mix_down.scales,
                mix_down.biases,
                wi,
                si,
                bi,
                bo,
                inj,
                f32(eps),
                placeholder(mx.float32),
            ],
            template=tmpl,
            grid=(32, NSG * ((nrows + rows_per_tg - 1) // rows_per_tg), M),
            threadgroup=(32, NSG, 1),
            output_shapes=[(M, ND), (M, NI) if NI else (8,), (M, hc), (M, K) if INJ else (8,)],
            output_dtypes=[T, T, mx.float32, T],
        )
        hn = h2 if INJ else h
    if _stop == "down":
        return hn[:, :hid] + y[:, :1], None, hn
    cps = 32 // (hc * LPR)
    cols_per_tg = cps * UNSG
    mixed, inject = _kernel("up")(
        inputs=[hn, w1, rstd, y, mix_up.weight, mix_up.scales, mix_up.biases, injraw],
        template=[
            ("T", T),
            ("SB", mix_up.scales.dtype),
            ("NSG", UNSG),
            ("LPR", LPR),
            ("GS", gs),
            ("S", hc),
            ("HID", hid),
            ("ND", ND),
            ("NI", NI),
        ],
        grid=(32 * ((hid + cols_per_tg - 1) // cols_per_tg), UNSG, M),
        threadgroup=(32, UNSG, 1),
        output_shapes=[(M, hid), (M, NI) if NI else (8,)],
        output_dtypes=[T, T],
    )
    return mixed, (inject if NI else None), hn


def _inject_operands(block_inject, mix_down, NI: int) -> tuple:
    """The inject projection's weight, scales and biases, or placeholders without an inject."""
    wi, si, bi = (
        (block_inject.weight, block_inject.scales, block_inject.biases)
        if NI
        else (
            placeholder(mx.uint32),
            placeholder(mix_down.scales.dtype),
            placeholder(mix_down.scales.dtype),
        )
    )
    return wi, si, bi


def _block_operands(block_out: mx.array | None, inject_in: mx.array | None, INJ: int, T) -> tuple:
    """The block output and inject to fold in, or placeholders when none is given."""
    bo = block_out if INJ else placeholder(T)
    inj = inject_in if INJ else placeholder(T)
    return bo, inj


def _launch_pre(
    h: mx.array,
    bo: mx.array,
    inj: mx.array,
    eps: float,
    T,
    hc: int,
    hid: int,
    INJ: int,
    pre_nt: int,
    M: int,
    K: int,
) -> tuple:
    """The pre launch: the residual after the inject and the norm statistics."""
    h2, rstd = _kernel("pre")(
        inputs=[h, bo, inj, f32(eps)],
        template=[("T", T), ("S", hc), ("HID", hid), ("INJ", INJ), ("NT", pre_nt)],
        grid=(pre_nt * M, 1, 1),
        threadgroup=(pre_nt, 1, 1),
        output_shapes=[(M, K) if INJ else (8,), (M, hc)],
        output_dtypes=[T, mx.float32],
    )
    hn = h2 if INJ else h
    return h2, rstd, hn


# ----------------------------------------------------------------------------- load-time check
def check_hc(hc, hid, lowrank, bits, group_size, eps, act_dtype, meta_dtypes, launch, rows) -> bool:
    """The load-time check at the model's hyper-connection geometry, with mix_down, mix_up and
    block_inject storing their metadata in `meta_dtypes` and `launch` = (down_cfg, up_cfg, pre_nt)
    as `Qwen4ExpTextGatedResidual.fused` passes them. Bit for bit (the claim
    `Qwen4ExpTextGatedResidual` relies on): each row of a batch of `rows`, the most it sends, is the
    row alone, read at an offset into the batch, with and without the residual inject folded in. And
    no less accurate than the stock ops of `Qwen4ExpTextGatedResidual._decode_body`, compiled as the
    model compiles them (after `hyper_inject` when an inject is folded in), as
    `selfcheck.no_worse_than_stock` measures it."""
    shapes = ((lowrank, hc * hid), (hc * hid, lowrank), (hc, hc * hid))
    lins = [
        sc.Linear(n, k, 71 + i, group_size, bits, meta)
        for i, ((n, k), meta) in enumerate(zip(shapes, meta_dtypes, strict=True))
    ]
    if not supports(hc, hid, lowrank, group_size, bits, *lins):
        raise ValueError(f"no fused hyper-connection for {hc} x {hid}, rank {lowrank}")
    w1 = 1.0 + 0.1 * sc.normal((hc * hid,), 74)
    mx.eval(w1)  # the compiled stock body captures it, as the model's captures its weights
    h = sc.normal((rows + 1, hc * hid), 75, act_dtype)[1:]
    block_out = sc.normal((rows, hid), 76, act_dtype)
    inject = (2.0 * mx.sigmoid(sc.normal((rows, hc), 77))).astype(act_dtype)
    down_cfg, up_cfg, pre_nt = launch

    def fused(x, *extra):
        bo, inj = extra or (None, None)
        return hc_fused(x, w1, *lins, hc, hid, eps, bo, inj, down_cfg, up_cfg, True, pre_nt)

    for extra in ((), (block_out, inject)):
        batch = fused(h, *extra)
        for i in sorted({0, rows // 2, rows - 1}):
            alone = fused(h[i : i + 1], *(a[i : i + 1] for a in extra))
            for got, want in zip(batch, alone, strict=True):
                if not mx.array_equal(got[i : i + 1], want).item():
                    return False
        for m in (1, rows):
            part = tuple(a[:m] for a in extra)
            got = fused(h[:m], *part)
            stock = _stock_hc(h[:m], w1, lins, hc, hid, eps, part)
            exact = _exact_hc(h[:m], w1, lins, hc, hid, eps, part)
            for g, s_, e in zip(got, stock, exact, strict=True):
                if not sc.no_worse_than_stock(g, s_, e):
                    return False
    return True


def _stock_hc(h, w1, lins, hc, hid, eps, folded):
    """`Qwen4ExpTextGatedResidual._decode_body` on each row, compiled, after `hyper_inject` when an
    inject is folded in: (mixed, inject, the residual it read)."""
    md, mu, mi = lins

    def body(h, *folded):
        if folded:
            bo, inj = folded
            h = h + (bo[:, None, :] * inj[:, :, None]).reshape(h.shape)
        xg = h.reshape(-1, hc, hid)
        normed = (mx.fast.rms_norm(xg, None, eps) * w1.reshape(hc, hid)).astype(xg.dtype)
        normed = normed.reshape(h.shape)
        w = mx.sigmoid(mu.stock(nn.silu(md.stock(normed) / hc))).reshape(-1, hc, hid)
        mixed = mx.mean(w * normed.reshape(-1, hc, hid), axis=-2)
        return mixed, 2.0 * mx.sigmoid(mi.stock(normed) / hc), h

    return mx.compile(body)(h, *folded)


def _exact_hc(h, w1, lins, hc, hid, eps, folded):
    """The same formulas in float64 on the exact quantized weights."""
    md, mu, mi = lins
    h = sc.f64(h)
    if folded:
        bo, inj = (sc.f64(a) for a in folded)
        h = h + (inj[:, :, None] * bo[:, None, :]).reshape(h.shape)
    xg = h.reshape(-1, hc, hid)
    rstd = 1.0 / np.sqrt((xg * xg).mean(axis=-1, keepdims=True) + eps)
    normed = (xg * rstd * sc.f64(w1).reshape(hc, hid)).reshape(h.shape)
    y = md.exact(normed) / hc
    w = 1.0 / (1.0 + np.exp(-mu.exact(y / (1.0 + np.exp(-y)))))
    mixed = (w.reshape(-1, hc, hid) * normed.reshape(-1, hc, hid)).mean(axis=-2)
    return mixed, 2.0 / (1.0 + np.exp(-mi.exact(normed) / hc)), h
