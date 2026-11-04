"""Gated DeltaNet recurrence as Metal kernels (mx.fast.metal_kernel).

Per (batch, value head hv, step t), with fp32 state S [Dv, Dk] and q, k from key head
hv // (Hv/Hk): S = S * g; d = (v_t - S k_t) * beta; S = S + d k_t^T; y_t = S q_t.
Inputs keep their own dtypes (g and state fp32) so numerics match the reference. The load-time
check (`check_recurrence`) holds the staged variant to the single launch bit for bit, and both to
`selfcheck.no_worse_than_stock` against `gdn_recurrence_ops`; `check_pair` holds the MTP verify
pass's two-row launch (`gdn_recurrence_pair`) to two single launches bit for bit.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck as sc

_SOURCE = """
    auto n = thread_position_in_grid.z;          // b * Hv + hv
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;     // q, k: [B, T, Hk, Dk]
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;     // v, y: [B, T, Hv, Dv]
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;    // state: [B, Hv, Dv, Dk]
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }

    auto g_ = g + b_idx * T * Hv;                        // g, beta: [B, T, Hv]
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float gt = static_cast<float>(g_[hv_idx]);
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gt;
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);
      float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
        out += state[i] * static_cast<float>(q_[s_idx]);
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
"""


# `_SOURCE` with the inputs of TB steps staged into threadgroup memory (coalesced), so the step loop
# makes no dependent device loads. Operations, operands and order (both simd_sum trees included)
# match `_SOURCE`, so results are bit-identical. Threadgroup (32, 4, 1): 4 dv rows of one head.
_SOURCE_STAGED = """
    auto n = thread_position_in_grid.z;          // b * Hv + hv
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    constexpr int NT = 32 * 4;

    const device InT* q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    const device InT* k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    const device InT* v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    device InT* y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
    const device float* g_ = g + b_idx * T * Hv;
    const device InT* beta_ = beta + b_idx * T * Hv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_local = thread_position_in_threadgroup.y;
    auto dv_idx = thread_position_in_grid.y;
    const uint tid = dv_local * 32 + dk_idx;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);

    threadgroup float ks[TB * Dk];
    threadgroup float qs[TB * Dk];
    threadgroup float vs[TB * 4];
    threadgroup float gs[TB];
    threadgroup float bs[TB];

    for (int t0 = 0; t0 < T; t0 += TB) {
      const int nb = min(TB, T - t0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = tid; e < nb * Dk; e += NT) {
        const int tt = e / Dk, d = e % Dk;
        ks[e] = static_cast<float>(k_[(t0 + tt) * Hk * Dk + d]);
        qs[e] = static_cast<float>(q_[(t0 + tt) * Hk * Dk + d]);
      }
      if (tid < nb) { gs[tid] = g_[(t0 + tid) * Hv + hv_idx]; bs[tid] = static_cast<float>(beta_[(t0 + tid) * Hv + hv_idx]); }
      if (tid < nb * 4) {
        const int tt = tid / 4, r = tid % 4;
        vs[tid] = static_cast<float>(v_[(t0 + tt) * Hv * Dv + (dv_idx - dv_local) + r]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int tt = 0; tt < nb; ++tt) {
        const float gt = gs[tt];
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          const int s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * gt;
          kv_mem += state[i] * ks[tt * Dk + s_idx];
        }
        kv_mem = simd_sum(kv_mem);
        const float delta = (vs[tt * 4 + dv_local] - kv_mem) * bs[tt];
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          const int s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + ks[tt * Dk + s_idx] * delta;
          out += state[i] * qs[tt * Dk + s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) y_[(t0 + tt) * Hv * Dv + dv_idx] = static_cast<InT>(out);
      }
    }
    for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
"""


@functools.cache
def _kernel_staged():
    return mx.fast.metal_kernel(
        name="moe_engine_gdn_recurrence_staged",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE_STAGED,
    )


def gdn_recurrence_staged(q, k, v, g, beta, state, tb: int = 16):
    """`gdn_recurrence` with staged inputs (bit-identical); for prefill-length T."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kernel_staged()(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("TB", int(tb)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moe_engine_gdn_recurrence",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE,
    )


def gdn_recurrence(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
):
    """Run the recurrence; returns (y [B, T, Hv, Dv] in q.dtype, new state [B, Hv, Dv, Dk]).

    q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; g, beta: [B, T, Hv]; state: [B, Hv, Dv, Dk] fp32.
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _kernel()(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gdn_recurrence_step(q, k, v, g, beta, state, hk: int, dk: int, hv: int, dv: int):
    """`gdn_recurrence` at B = T = 1 with its inputs and y unexpanded: q, k [Hk, Dk], v [Hv, Dv],
    g, beta [Hv] -> (y [Hv, Dv], new state). The same launch: the kernel reads them flat."""
    return _kernel()(
        inputs=[q, k, v, g, beta, state, 1],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
        ],
        grid=(32, dv, hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(hv, dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


# `_SOURCE` over two positions passed as separate inputs (rows 0 and 1 of the MTP verify pass), in
# one launch and with no store of the state between them. It is cut from `_SOURCE`'s own text: the
# state load, the loop body (every per-step statement) and the state store, each where it occurs
# exactly once, so the step cannot drift from the single launch; if an anchor is missing there is
# no pair launch (`pair_available`), its check fails and `GDN_LAZY` stays off. Two
# `gdn_recurrence_step` launches store the fp32 state after row 0 and reload it; here it stays in
# the same fp32 registers, so y0, y1 and the final state are theirs bit for bit (`check_pair`). The state after row 0 is not
# written: `model/gdn.py` `LazyRow0` stands in for it.
_LOOP_HEAD = "    for (int t = 0; t < T; ++t) {\n"
_LOOP_STEP = """      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
"""
_STATE_LOAD = """    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }
"""
_STATE_STORE = """    for (int i = 0; i < n_per_t; ++i) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
"""
_PAIR_HEAD = """
    auto n = thread_position_in_grid.z;          // hv (one sequence)
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;    // state: [Hv, Dv, Dk]
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

"""
_PAIR_ROWS = """
    for (int t = 0; t < 2; ++t) {
      auto q_ = (t == 0 ? q0 : q1) + hk_idx * Dk;        // q, k: [Hk, Dk] per row
      auto k_ = (t == 0 ? k0 : k1) + hk_idx * Dk;
      auto v_ = (t == 0 ? v0 : v1) + hv_idx * Dv;        // v, y: [Hv, Dv] per row
      auto g_ = (t == 0 ? g0 : g1);                      // g, beta: [Hv] per row
      auto beta_ = (t == 0 ? beta0 : beta1);
      auto y = (t == 0 ? y0 : y1) + hv_idx * Dv;
"""


def _once(src: str, anchor: str) -> str:
    if src.count(anchor) != 1:
        raise ValueError(f"anchor occurs {src.count(anchor)} times in _SOURCE: {anchor[:50]!r}")
    return anchor


def _pair_source(src: str) -> str | None:
    """`_SOURCE`'s load, loop body and store around the two rows' pointers; None if an anchor is
    not found exactly once."""
    try:
        start = src.index(_once(src, _LOOP_HEAD)) + len(_LOOP_HEAD)
        body = src[start : src.index(_once(src, _LOOP_STEP))]
        load, store = _once(src, _STATE_LOAD), _once(src, _STATE_STORE)
    except ValueError:
        return None
    return _PAIR_HEAD + load + _PAIR_ROWS + body + "    }\n" + store


_SOURCE_PAIR = _pair_source(_SOURCE)


def pair_available() -> bool:
    """Whether `_SOURCE_PAIR` was cut (every anchor found once), so `gdn_recurrence_pair` exists."""
    return _SOURCE_PAIR is not None


@functools.cache
def _kernel_pair():
    return mx.fast.metal_kernel(
        name="moe_engine_gdn_recurrence_pair",
        input_names=["q0", "k0", "v0", "g0", "beta0", "q1", "k1", "v1", "g1", "beta1", "state_in"],
        output_names=["y0", "y1", "state_out"],
        source=_SOURCE_PAIR,
    )


def gdn_recurrence_pair(row0, row1, state, hk: int, dk: int, hv: int, dv: int):
    """Two `gdn_recurrence_step` launches, row 0's and then row 1's on the state row 0 left, as one:
    row0, row1 = (q, k, v, g, beta) of each position, unexpanded as the step takes them; state
    [1, Hv, Dv, Dk] float32 -> (y0 [Hv, Dv], y1 [Hv, Dv], the state after row 1), each the two
    launches' bits. The state after row 0 is not written."""
    if state.dtype != mx.float32:
        # the two launches store and reload the state between the rows: only fp32 makes that exact
        raise ValueError(f"gdn_recurrence_pair needs a float32 state, not {state.dtype}")
    q = row0[0]
    return _kernel_pair()(
        inputs=[*row0, *row1, state],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
        ],
        grid=(32, dv, hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(hv, dv), (hv, dv), state.shape],
        output_dtypes=[q.dtype, q.dtype, state.dtype],
    )


def gdn_recurrence_ops(q, k, v, g, beta, state):
    """Plain-ops reference of the same recurrence: the kernel's check and test compare with it, and
    `model/gdn.py` runs it when `GDN_IMPL` is "ops" or the kernel is off."""
    _B, T, Hk, _Dk = k.shape
    Hv, _Dv = v.shape[2:]
    rep = Hv // Hk
    qf = mx.repeat(q.astype(mx.float32), rep, axis=2)
    kf = mx.repeat(k.astype(mx.float32), rep, axis=2)
    vf = v.astype(mx.float32)
    gf = g.astype(mx.float32)
    bf = beta.astype(mx.float32)
    S = state
    ys = []
    for t in range(T):
        S = S * gf[:, t][..., None, None]
        kt = kf[:, t]  # [B, Hv, Dk]
        kv = (S * kt[..., None, :]).sum(-1)  # [B, Hv, Dv]
        d = (vf[:, t] - kv) * bf[:, t][..., None]
        S = S + kt[..., None, :] * d[..., None]
        ys.append((S * qf[:, t][..., None, :]).sum(-1))
    return mx.stack(ys, axis=1).astype(q.dtype), S


# ----------------------------------------------------------------------------- load-time check
def check_recurrence(hk, hv, dk, dv, act_dtype, tb: int) -> bool:
    """The load-time check at the model's GDN geometry, inputs read at an offset into a batch of two
    (the batched step's row slices). Bit for bit (the claim above): at the lengths the model gives
    `gdn_recurrence_staged` with its `tb` (more than one position), on each side of one staging tile
    and across two, the staged variant gives `gdn_recurrence`'s bits. And at one step (decode) and
    at a tile and one, the kernel is no less accurate than `gdn_recurrence_ops`, the plain-ops
    definition, against an exact float64 run, as `selfcheck.no_worse_than_stock` measures it."""
    for t in sorted({1, 2, max(2, tb - 1), tb, tb + 1, 2 * tb + 1}):
        args = _recurrence_inputs(hk, hv, dk, dv, t, act_dtype)
        y, s = gdn_recurrence(*args)
        if t > 1:
            ys, ss = gdn_recurrence_staged(*args, tb=tb)
            if not (mx.array_equal(ys, y).item() and mx.array_equal(ss, s).item()):
                return False
        if t in (1, tb + 1):
            exact = _exact_recurrence(*args)
            stock = gdn_recurrence_ops(*args)
            if not all(map(sc.no_worse_than_stock, (y, s), stock, exact, strict=True)):
                return False
    return True


def check_pair(hk, hv, dk, dv, act_dtype) -> bool:
    """The load-time check of `gdn_recurrence_pair` at the model's GDN geometry: on two input sets
    (positions 0 and 1 of `_recurrence_inputs`' rows, read at an offset as the step's are), y0, y1
    and the final state are the two `gdn_recurrence_step` launches' bit for bit, signed zeros and
    NaN payloads included. False when there is no pair launch (`pair_available`)."""
    if not pair_available():
        return False
    for t in (2, 3):
        q, k, v, g, beta, state = _recurrence_inputs(hk, hv, dk, dv, t, act_dtype)
        rows = [tuple(a[0, r] for a in (q, k, v, g, beta)) for r in (0, 1)]
        y0, s0 = gdn_recurrence_step(*rows[0], state, hk, dk, hv, dv)
        y1, s1 = gdn_recurrence_step(*rows[1], s0, hk, dk, hv, dv)
        got = gdn_recurrence_pair(rows[0], rows[1], state, hk, dk, hv, dv)
        if not all(map(same_bits, got, (y0, y1, s1), strict=False)):
            return False
    return True


def same_bits(a: mx.array, b: mx.array) -> bool:
    """Whether a and b hold the same bit patterns (`array_equal` takes -0.0 for 0.0, fails NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    u = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(u), b.view(u)).item())


def _recurrence_inputs(hk, hv, dk, dv, t, dt):
    """q and k l2-normalised and scaled as the block does, g and beta in (0, 1), a float32 state,
    each the second row of a batch of two, as the batched step slices them."""
    key = 100 * t
    q = mx.fast.rms_norm(sc.normal((2, t, hk, dk), key), None, 1e-6 / dk) / dk
    k = mx.fast.rms_norm(sc.normal((2, t, hk, dk), key + 1), None, 1e-6 / dk) * dk**-0.5
    v = sc.normal((2, t, hv, dv), key + 2)
    g = mx.exp(-mx.exp(0.3 * sc.normal((2, t, hv), key + 3)) * 0.5)
    beta = mx.sigmoid(sc.normal((2, t, hv), key + 4))
    state = 0.1 * sc.normal((2, hv, dv, dk), key + 5)
    batch = (q.astype(dt), k.astype(dt), v.astype(dt), g, beta.astype(dt), state)
    mx.eval(*batch)
    return tuple(a[1:] for a in batch)


def _exact_recurrence(q, k, v, g, beta, state):
    q, k, v, g, beta, s = (sc.f64(a) for a in (q, k, v, g, beta, state))
    rep = v.shape[2] // k.shape[2]
    q, k = np.repeat(q, rep, axis=2), np.repeat(k, rep, axis=2)
    ys = []
    for t in range(q.shape[1]):
        s = s * g[:, t, :, None, None]
        d = (v[:, t] - (s * k[:, t, :, None, :]).sum(-1)) * beta[:, t, :, None]
        s = s + k[:, t, :, None, :] * d[..., None]
        ys.append((s * q[:, t, :, None, :]).sum(-1))
    return np.stack(ys, axis=1), s
