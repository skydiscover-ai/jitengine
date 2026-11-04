"""Decode attention (one query position) that reads each K/V row once per KV head.

One simdgroup owns HPS query heads of one KV head over a chunk of keys and skips masked keys; a
second launch merges the chunks' (m, l, acc) partials. Accumulation is fp32; results match MLX's
unfused SDPA path (its fallback at head dim 256) up to summation order, which the load-time check
states as a tolerance: against an exact float64 attention, the kernel's largest error is at most
8 times stock `scaled_dot_product_attention`'s on the same mask, plus four float32 ulps of the
output's scale (`selfcheck.no_worse_than_stock`).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck as sc
from jitengine_qwen3_8_flash_next_m2.kernels.constants import f32

HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

# q (H, D) scaled in-kernel by sc[0], k/v (Hkv, CAP, D), mask (P) bool, dims (P, nch)
# grid (32 * nch, GQ / HPS, Hkv), threadgroup (32, 1, 1)
PART_SOURCE = """
    constexpr int GQ = H / HKV;
    constexpr int PER = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint3 tid = threadgroup_position_in_grid;
    const int chunk = tid.x, hg = tid.y, kvh = tid.z;
    const int P = dims[0], NCH = dims[1];
    const int h0 = kvh * GQ + hg * HPS;
    const int k0 = chunk * CH, k1 = min(P, k0 + CH);
    const int CAP = k_shape[1];
    float qv[HPS][PER], acc[HPS][PER], m[HPS], l[HPS];
    #pragma unroll
    for (int j = 0; j < HPS; j++) {
      m[j] = -INFINITY; l[j] = 0.0f;
      #pragma unroll
      for (int i = 0; i < PER; i++) { qv[j][i] = float(q[(ulong)(h0 + j) * D + lane * PER + i]) * sc[0]; acc[j][i] = 0.0f; }
    }
    const device T* kb = k + (ulong)kvh * CAP * D + lane * PER;
    const device T* vb = v + (ulong)kvh * CAP * D + lane * PER;
    for (int kk = k0; kk < k1; kk++) {
      if (HAS_MASK && !mask[kk]) continue;
      float kr[PER], vr[PER];
      #pragma unroll
      for (int i = 0; i < PER; i++) kr[i] = float(kb[(ulong)kk * D + i]);
      #pragma unroll
      for (int i = 0; i < PER; i++) vr[i] = float(vb[(ulong)kk * D + i]);
      #pragma unroll
      for (int j = 0; j < HPS; j++) {
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < PER; i++) d += qv[j][i] * kr[i];
        d = simd_sum(d);
        const float mn = metal::max(m[j], d);
        const float c = metal::exp(m[j] - mn), p = metal::exp(d - mn);
        l[j] = l[j] * c + p;
        #pragma unroll
        for (int i = 0; i < PER; i++) acc[j][i] = acc[j][i] * c + p * vr[i];
        m[j] = mn;
      }
    }
    #pragma unroll
    for (int j = 0; j < HPS; j++) {
      const ulong pi = (ulong)(h0 + j) * NCH + chunk;
      if (lane == 0) { pm[pi] = m[j]; pl[pi] = l[j]; }
      #pragma unroll
      for (int i = 0; i < PER; i++) pacc[pi * D + lane * PER + i] = acc[j][i];
    }
"""

REDUCE_SOURCE = """
    constexpr int PER = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int h = threadgroup_position_in_grid.x * NSG + sg;
    if (h >= H) return;
    const int NCH = dims[1];
    float M = -INFINITY;
    for (int c = 0; c < NCH; c++) M = metal::max(M, pm[(ulong)h * NCH + c]);
    float L = 0.0f, acc[PER];
    #pragma unroll
    for (int i = 0; i < PER; i++) acc[i] = 0.0f;
    for (int c = 0; c < NCH; c++) {
      const ulong pi = (ulong)h * NCH + c;
      const float mc = pm[pi];
      if (mc == -INFINITY) continue;
      const float w = metal::exp(mc - M);
      L += pl[pi] * w;
      #pragma unroll
      for (int i = 0; i < PER; i++) acc[i] += pacc[pi * D + lane * PER + i] * w;
    }
    const float inv = 1.0f / L;
    #pragma unroll
    for (int i = 0; i < PER; i++) o[(ulong)h * D + lane * PER + i] = static_cast<T>(acc[i] * inv);
"""


@functools.cache
def _kernel(which: str):
    if which == "part":
        return mx.fast.metal_kernel(
            name="moe_engine_sdpa_decode2_part",
            input_names=["q", "k", "v", "mask", "dims", "sc"],
            output_names=["pm", "pl", "pacc"],
            header=HEADER,
            source=PART_SOURCE,
            ensure_row_contiguous=True,
        )
    return mx.fast.metal_kernel(
        name="moe_engine_sdpa_decode2_reduce",
        input_names=["pm", "pl", "pacc", "dims"],
        output_names=["o"],
        header=HEADER,
        source=REDUCE_SOURCE,
        ensure_row_contiguous=True,
    )


@functools.lru_cache(maxsize=16)
def _dims(P: int, nch: int) -> mx.array:
    """(P, nch) as a device array, cached so a step's attention layers share it."""
    return mx.array([int(P), int(nch)], dtype=mx.int32)


def supports(H: int, Hkv: int, D: int, hps: int) -> bool:
    """Whether the kernel covers `H` query heads over `Hkv` KV heads of head dim `D`, `hps` heads
    per simdgroup: `D` a multiple of 32 from 32 to 512, `H` a multiple of `Hkv`, and `H // Hkv` a
    multiple of `hps`."""
    return D % 32 == 0 and 32 <= D <= 512 and H % Hkv == 0 and (H // Hkv) % hps == 0


def decode_masked(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    P: int,
    scale: float,
    mask: mx.array,
    chunk: int = 64,
    hps: int = 6,
) -> mx.array:
    """Attend q [1, H, 1, D] over the first P rows of keys/values [1, Hkv, CAP, D].

    mask is [.., P] bool. Returns [1, H, 1, D]. `P` is a runtime input because a template constant
    would recompile the kernel every token.
    """
    _, H, _, D = q.shape
    Hkv, CAP = keys.shape[1], keys.shape[2]
    GQ = H // Hkv
    nch = -(-P // chunk)
    dims = _dims(P, nch)
    T = q.dtype
    pm, pl, pacc = _kernel("part")(
        inputs=[
            q.reshape(H, D),
            keys.reshape(Hkv, CAP, D),
            values.reshape(Hkv, CAP, D),
            mask.reshape(-1),
            dims,
            f32(scale),
        ],
        template=[
            ("T", T),
            ("H", H),
            ("HKV", Hkv),
            ("D", D),
            ("CH", int(chunk)),
            ("HPS", int(hps)),
            ("HAS_MASK", True),
        ],
        grid=(32 * nch, GQ // hps, Hkv),
        threadgroup=(32, 1, 1),
        output_shapes=[(H, nch), (H, nch), (H, nch, D)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    nsg = 4
    o = _kernel("reduce")(
        inputs=[pm, pl, pacc, dims],
        template=[("T", T), ("H", H), ("D", D), ("NSG", nsg)],
        grid=(32 * (-(-H // nsg)), nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(H, D)],
        output_dtypes=[T],
    )[0]
    return o.reshape(1, H, 1, D)


# ----------------------------------------------------------------------------- load-time check
def check_decode_masked(heads, kv_heads, head_dim, act_dtype, chunk, hps, cr, topk) -> bool:
    """The load-time check at the model's attention geometry and launch (`chunk`, `hps`), on masks
    of whole `cr`-position blocks, `topk` of them kept, plus the visible ragged tail, as the QSA
    indexer builds them: at the fewest positions that prune (a ragged last chunk) with the first
    block masked, at one more with a random block masked, and with whole chunks masked between kept
    ends. The query is read at an offset, and the K/V buffers run past P with NaN under a mask that
    stays true there, so a kernel that read past P would return NaN. Held to the tolerance in the
    module docstring."""
    if not supports(heads, kv_heads, head_dim, hps):
        raise ValueError(
            f"no masked decode for {heads}/{kv_heads} heads of {head_dim} at hps {hps}"
        )
    first = cr * (topk + 1)  # P - 1 reaches the indexer's `prune_from`
    gap = 2 * -(-chunk // cr)  # masked blocks enough to cover a whole chunk wherever it falls
    cases = ((first, "first"), (first + 1, "random"), (cr * (topk + gap) + cr - 1, "ends"))
    for t, (P, kind) in enumerate(cases):
        cap = -(-(P + 1) // 256) * 256 + 256
        q = sc.normal((2, heads, 1, head_dim), 30 + t, act_dtype)[1:]
        kv = [sc.normal((1, kv_heads, P, head_dim), 33 + 2 * t + i, act_dtype) for i in (0, 1)]
        pad = mx.full((1, kv_heads, cap - P, head_dim), float("nan"), dtype=act_dtype)
        kbuf, vbuf = (mx.concatenate([a, pad], axis=2) for a in kv)
        keep = _blocks(P // cr, topk, kind, seed=t)
        m = np.concatenate([np.repeat(keep, cr), np.ones(P - keep.size * cr, dtype=bool)])
        mask = mx.array(m)[None, None, None, :]
        runs_on = mx.concatenate([mask, mx.ones((1, 1, 1, cap - P), dtype=mx.bool_)], axis=-1)
        got = decode_masked(q, kbuf, vbuf, P, head_dim**-0.5, runs_on, chunk=chunk, hps=hps)
        stock = mx.fast.scaled_dot_product_attention(
            q, kbuf[:, :, :P], vbuf[:, :, :P], scale=head_dim**-0.5, mask=mask
        )
        exact = _exact_attention(q, *kv, m, head_dim**-0.5)
        if not sc.no_worse_than_stock(got.reshape(heads, -1), stock.reshape(heads, -1), exact):
            return False
    return True


def _blocks(nb: int, topk: int, kind: str, seed: int) -> np.ndarray:
    """`topk` kept blocks of `nb`: all but the first ones, a random choice, or both ends."""
    keep = np.zeros(nb, dtype=bool)
    if kind == "first":
        keep[nb - topk :] = True
    elif kind == "random":
        keep[np.random.default_rng(seed).permutation(nb)[:topk]] = True
    else:
        keep[: topk // 2] = keep[nb - (topk - topk // 2) :] = True
    return keep


def _exact_attention(q, k, v, mask, scale) -> np.ndarray:
    """[H, D] float64: softmax over the unmasked keys of each head's KV head."""
    q, k, v = sc.f64(q)[0, :, 0], sc.f64(k)[0], sc.f64(v)[0]
    group = q.shape[0] // k.shape[0]
    out = np.empty(q.shape)
    for h in range(q.shape[0]):
        s = (k[h // group][mask] @ q[h]) * scale
        p = np.exp(s - s.max())
        out[h] = (p / p.sum()) @ v[h // group][mask]
    return out
