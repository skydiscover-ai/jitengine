"""Bit-exact fused glue for the Gated DeltaNet block on decode and verify rows.

Each kernel reproduces the composed MLX op chain bit for bit: the same op order, the same dtype at every
intermediate (bf16 ops round after each op), MLX's own Sigmoid functor, and the rms_single_row reduction
(4 consecutive reads per lane, simd_sum, precise::rsqrt, `w * T(x * inv)`). `check_gdn_exact` verifies them
bitwise against the composed ops on this device.

  gdn_pre_rows:  conv window append, depthwise causal conv, silu, head split, q/k rms_norm and scales, and the
                 new conv windows, for M rows of one slot (or one row per slot when batched).
  gdn_post_rows: gated_silu_f32(y, z, rms_norm(y, w, eps)), the out_proj activation, in one launch.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels import constants

PRE_SRC = """
    // grid: (32 * NH, M, 1), threadgroup (32, 1, 1): one simdgroup per (row m, head h); lane covers 4 channels
    constexpr int C = 2 * HK * DK + HV * DV;
    constexpr int NH = 2 * HK + HV;
    const uint lane = thread_index_in_simdgroup;
    const int h = threadgroup_position_in_grid.x;
    const int m = threadgroup_position_in_grid.y;
    const bool is_v = h >= 2 * HK;
    const int c0 = is_v ? (2 * HK * DK + (h - 2 * HK) * DV) : h * DK;
    const int cl = c0 + lane * 4;
    // seq rows: [conv state rows 0..NK-1 | qkv rows 0..M-1]; row m's window is seq[m .. m+NK]
    T xo[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int c = cl + i;
      T win[NK + 1];
      #pragma unroll
      for (int j = 0; j <= NK; j++) {
        // verify rows (BATCHED=0): one slot, consecutive positions; batched rows (BATCHED=1): row m has its own window
        const int r = m + j;
        win[j] = BATCHED ? (j < NK ? conv[(m * NK + j) * C + c] : ycat[m * NTOT + QKVOFF + c])
                         : (r < NK ? conv[r * C + c] : ycat[(r - NK) * NTOT + QKVOFF + c]);
      }
      CONV_EXPR
      // nn.silu (compiled x * sigmoid(x)) in T
      T sg = Sigmoid()(cv);
      xo[i] = cv * sg;
      // the new conv window after row m: seq[m+1 .. m+NK]
      #pragma unroll
      for (int j = 0; j < NK; j++) newconv[(m * NK + j) * C + c] = win[j + 1];
    }
    if (is_v) {
      #pragma unroll
      for (int i = 0; i < 4; i++) vout[(m * HV + (h - 2 * HK)) * DV + lane * 4 + i] = xo[i];
      return;
    }
    // mx.fast.rms_norm(x, None, 1e-6) over DK (= rms_single_row: 4 reads per lane, simd_sum, precise::rsqrt)
    float acc = 0;
    float tx[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { tx[i] = xo[i]; acc += tx[i] * tx[i]; }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / DK + 1e-6f);
    const bool is_q = h < HK;
    const T sc = is_q ? scl[0] : scl[1];
    device T* o = is_q ? (qout + (m * HK + h) * DK) : (kout + (m * HK + (h - HK)) * DK);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      T nrm = static_cast<T>(1.0f) * static_cast<T>(tx[i] * inv);
      o[lane * 4 + i] = sc * nrm;
    }
"""

SIGMOID_HDR = """
struct Sigmoid {
  template <typename U>
  U operator()(U x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""

POST_SRC = """
    // grid (32 * HV, M, 1): one simdgroup per (row, value head); y [M, HV, DV], z at ycat[m, ZOFF + h*DV + .]
    const uint lane = thread_index_in_simdgroup;
    const int h = threadgroup_position_in_grid.x;
    const int m = threadgroup_position_in_grid.y;
    const device YT* yr = y + (m * HV + h) * DV + lane * 4;
    float acc = 0;
    float tx[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { tx[i] = yr[i]; acc += tx[i] * tx[i]; }
    acc = simd_sum(acc);
    const float inv = metal::precise::rsqrt(acc / DV + eps[0]);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = lane * 4 + i;
      YT nrm = w[d] * static_cast<YT>(tx[i] * inv);
      float zf = static_cast<float>(ycat[m * NTOT + ZOFF + h * DV + d]);
      float g = zf * Sigmoid()(zf);
      out[(m * HV + h) * DV + d] = static_cast<YT>(g * static_cast<float>(nrm));
    }
"""

CONV_VARIANTS = {
    # window accumulated in order in fp32, then cast (the form that matches MLX)
    "seq_f32": "float a = 0; for (int j = 0; j <= NK; j++) a += float(win[j]) * float(convw[c * (NK + 1) + j]); T cv = static_cast<T>(a);",
    "seq_f32_fma": "float a = 0; for (int j = 0; j <= NK; j++) a = metal::fma(float(win[j]), float(convw[c * (NK + 1) + j]), a); T cv = static_cast<T>(a);",
    "rev_f32": "float a = 0; for (int j = NK; j >= 0; j--) a += float(win[j]) * float(convw[c * (NK + 1) + j]); T cv = static_cast<T>(a);",
    "seq_T": "T a = 0; for (int j = 0; j <= NK; j++) a += win[j] * convw[c * (NK + 1) + j]; T cv = a;",
}


@functools.cache
def _kern(name, src, inputs, outputs, header=""):
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        header=SIGMOID_HDR + header,
        source=src,
    )


CONV_FORM = "seq_f32"


@functools.cache
def _scales(DK, T):
    inv = DK**-0.5
    a = mx.ones((1,), dtype=T)
    scl = mx.concatenate([(inv**2) * a, inv * a])  # how `scalar * bf16 array` casts the scalar
    mx.eval(scl)
    return scl


def gdn_pre_rows(
    conv, ycat, conv_w2d, cfg, qkv_off: int, form: str | None = None, batched: bool = False
):
    """Fused conv, silu, split and q/k rms_norm for M rows.

    conv [NK, C] (slot state), ycat [M, NTOT] (fused in_proj rows), conv_w2d [C, NK+1] ->
    q [M, HK, DK], k [M, HK, DK], v [M, HV, DV] (all T), newconv [M, NK, C] (the window after each row).
    batched=True: conv [M, NK, C], row m is one position of its own slot (the batched decode step).
    """
    HK, DK = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    HV, DV = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    M, NTOT = ycat.shape
    NK, C = conv.shape[-2:]
    form = form or CONV_FORM
    src = PRE_SRC.replace("CONV_EXPR", CONV_VARIANTS[form])
    k = _kern(
        "gdn_pre_rows_" + form,
        src,
        ("conv", "ycat", "convw", "scl"),
        ("qout", "kout", "vout", "newconv"),
    )
    T = ycat.dtype
    # the scalars of `(inv_scale ** 2) * rms_norm(q)` and `inv_scale * rms_norm(k)`, cast to T as MLX does
    scl = _scales(DK, T)
    return k(
        inputs=[conv, ycat, conv_w2d, scl],
        template=[
            ("T", T),
            ("HK", HK),
            ("DK", DK),
            ("HV", HV),
            ("DV", DV),
            ("NK", NK),
            ("NTOT", NTOT),
            ("QKVOFF", qkv_off),
            ("BATCHED", int(bool(batched))),
        ],
        grid=(32 * (2 * HK + HV), M, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, HK, DK), (M, HK, DK), (M, HV, DV), (M, NK, C)],
        output_dtypes=[T, T, T, T],
    )


def gdn_post_rows(y, ycat, norm_w, cfg, eps: float, z_off: int):
    """Fused rms_norm of y and silu gate with z; y [M, HV, DV], ycat [M, NTOT] (z at z_off), norm_w [DV] -> [M, HV*DV]."""
    HV, DV = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    M = y.shape[0]
    NTOT = ycat.shape[-1]
    k = _kern("gdn_post_rows", POST_SRC, ("y", "ycat", "w", "eps"), ("out",))
    return k(
        inputs=[y, ycat, norm_w, constants.f32(eps)],
        template=[("YT", y.dtype), ("HV", HV), ("DV", DV), ("NTOT", NTOT), ("ZOFF", z_off)],
        grid=(32 * HV, M, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, HV * DV)],
        output_dtypes=[y.dtype],
    )[0]


def check_gdn_exact(cfg) -> bool:
    """Check gdn_pre_rows and gdn_post_rows are bitwise equal to the composed MLX chain at M = 1..8."""
    import mlx.nn as nn

    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.elementwise import gated_silu_f32

    HK, DK = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    HV, DV = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    NK = cfg.linear_conv_kernel_dim - 1
    if DK != 128 or DV != 128:
        return False
    C = 2 * HK * DK + HV * DV
    NTOT = C + HV * DV + 2 * HV
    T = mx.bfloat16
    for t in range(4):
        key = lambda i: mx.random.key(1300 + 17 * t + i)  # noqa: E731, B023  used only in this iteration
        conv_w = (mx.random.normal((C, NK + 1, 1), key=key(0)) * (0.2 + 0.2 * t)).astype(T)
        conv = (mx.random.normal((NK, C), key=key(1)) * (0.5 + t)).astype(T)
        w = (1 + 0.1 * mx.random.normal((DV,), key=key(2))).astype(T)
        for M in (1, 2, 3, 5, 8):
            ycat = (mx.random.normal((M, NTOT), key=key(10 + M)) * (0.5 + t)).astype(T)
            qkv = ycat[:, :C].reshape(1, M, C)
            seq = mx.concatenate([conv[None], qkv], axis=1)
            windows = mx.concatenate([seq[:, j : j + NK + 1, :] for j in range(M)], axis=0)
            convs = mx.concatenate(
                [mx.contiguous(seq[:, j + 1 : j + 1 + NK, :]) for j in range(M)], axis=0
            )
            co = nn.silu(mx.conv1d(windows, conv_w, groups=C))[:, -1:, :]
            q, k, v = mx.split(co, [HK * DK, 2 * HK * DK], axis=-1)
            inv = DK**-0.5
            q = (inv**2) * mx.fast.rms_norm(q.reshape(M, 1, HK, DK), None, 1e-6)
            k = inv * mx.fast.rms_norm(k.reshape(M, 1, HK, DK), None, 1e-6)
            got = gdn_pre_rows(conv, ycat, conv_w.reshape(C, NK + 1), cfg, 0)
            for a_, b_ in zip(
                got, (q.reshape(M, HK, DK), k.reshape(M, HK, DK), v.reshape(M, HV, DV), convs)
            ):
                if not mx.array_equal(a_, b_).item():
                    return False
            # batched rows: each row is one position of its own slot (Qwen3_5MoeGatedDeltaNet._step_rows_glue)
            prev = (mx.random.normal((M, NK, C), key=key(50 + M)) * (0.5 + t)).astype(T)
            ci = mx.concatenate([prev, ycat[:, :C].reshape(M, 1, C)], axis=1)
            co = nn.silu(mx.conv1d(ci, conv_w, groups=C))
            q, k, v = mx.split(co, [HK * DK, 2 * HK * DK], axis=-1)
            q = (inv**2) * mx.fast.rms_norm(q.reshape(M, 1, HK, DK), None, 1e-6)
            k = inv * mx.fast.rms_norm(k.reshape(M, 1, HK, DK), None, 1e-6)
            got = gdn_pre_rows(prev, ycat, conv_w.reshape(C, NK + 1), cfg, 0, batched=True)
            for a_, b_ in zip(
                got,
                (
                    q.reshape(M, HK, DK),
                    k.reshape(M, HK, DK),
                    v.reshape(M, HV, DV),
                    ci[:, -NK:, :],
                ),
            ):
                if not mx.array_equal(a_, b_).item():
                    return False
            y = (mx.random.normal((M, HV, DV), key=key(30 + M)) * (0.1 + 0.2 * t)).astype(T)
            z = ycat[:, C : C + HV * DV].reshape(M, 1, HV, DV)
            yy = y.reshape(M, 1, HV, DV)
            ref = gated_silu_f32(yy, z, mx.fast.rms_norm(yy, w, cfg.rms_norm_eps)).reshape(
                M, HV * DV
            )
            if not mx.array_equal(gdn_post_rows(y, ycat, w, cfg, cfg.rms_norm_eps, C), ref).item():
                return False
    return True
