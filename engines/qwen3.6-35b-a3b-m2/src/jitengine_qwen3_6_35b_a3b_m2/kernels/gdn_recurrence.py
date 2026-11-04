"""Gated DeltaNet recurrence as Metal kernels.

One 32-lane simdgroup holds a row S[dv, :] of the fp32 state in registers (the layout of mlx_lm's
`_make_gated_delta_kernel`), and the time loop runs sequentially inside the kernel.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kernels.source import substitute

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
    """Run T positions of the recurrence; returns (y [B, T, Hv, Dv] in q.dtype, new state).

    Shapes: q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk] fp32.
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


# One position for B requests in one launch: row b reads its own state array (state_in<b>) and the
# new states are written stacked [B, Hv, Dv, Dk]; each thread does the single-row kernel's work.
ROWS_MAXB = 8
_ROWS_SELECT = (
    "#define SELECT_STATE (b_idx == 0 ? state_in0 : b_idx == 1 ? state_in1 : b_idx == 2 ? state_in2 : "
    "b_idx == 3 ? state_in3 : b_idx == 4 ? state_in4 : b_idx == 5 ? state_in5 : b_idx == 6 ? state_in6 : "
    "state_in7)\n"
)
_SOURCE_ROWS = substitute(
    _SOURCE,
    (
        "    auto i_state = state_in + (n * Dv + dv_idx) * Dk;    // state: [B, Hv, Dv, Dk]",
        "    auto i_state = SELECT_STATE + (hv_idx * Dv + dv_idx) * Dk;   // row b_idx's own [1, Hv, Dv, Dk]",
    ),
)


@functools.cache
def _rows_kernel():
    return mx.fast.metal_kernel(
        name="moe_engine_gdn_recurrence_rows",
        input_names=[
            "q",
            "k",
            "v",
            "g",
            "beta",
            *[f"state_in{i}" for i in range(ROWS_MAXB)],
            "T",
        ],
        output_names=["y", "state_out"],
        header=_ROWS_SELECT,
        source=_SOURCE_ROWS,
    )


def gdn_recurrence_rows(q, k, v, g, beta, states: list):
    """One position for B rows with separate states; row b is bitwise gdn_recurrence on row b."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if T != 1 or not 1 <= B <= ROWS_MAXB or len(states) != B:
        raise ValueError(
            f"gdn_recurrence_rows takes one position for 1 to {ROWS_MAXB} rows and one state per "
            f"row, got {T} positions, {B} rows and {len(states)} states"
        )
    padded = list(states) + [states[0]] * (ROWS_MAXB - B)
    return _rows_kernel()(
        inputs=[q, k, v, g, beta, *padded, T],
        template=[
            ("InT", q.dtype),
            ("StT", states[0].dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[q.dtype, states[0].dtype],
    )


def supported_rows(states: list) -> bool:
    """Whether the rows kernel covers these rows: 2 to ROWS_MAXB of them, every state float32."""
    return 2 <= len(states) <= ROWS_MAXB and all(s.dtype == mx.float32 for s in states)


def recurrence_rows(q, k, v, g, beta, states: list, kernel: bool = True) -> tuple:
    """One position for B rows, each with its own state: the rows kernel in one launch where it
    covers them and `kernel` says its load-time check passed, `gdn_recurrence` row by row
    otherwise. Returns y [B, 1, Hv, Dv] and each row's new state."""
    if kernel and supported_rows(states):
        y, stacked = gdn_recurrence_rows(q, k, v, g, beta, states)
        return y, [stacked[i : i + 1] for i in range(len(states))]
    ys, new_states = [], []
    for i, s in enumerate(states):
        rows = slice(i, i + 1)
        y_i, s_i = gdn_recurrence(q[rows], k[rows], v[rows], g[rows], beta[rows], s)
        ys.append(y_i)
        new_states.append(s_i)
    return mx.concatenate(ys, axis=0), new_states


def check_gdn_rows() -> bool:
    """gdn_recurrence_rows: row b bitwise the single-row gdn_recurrence on its own state."""
    B, Hk, Hv, D = 3, 16, 32, 128
    r = lambda s, k: mx.random.normal(s, key=mx.random.key(k))  # noqa: E731
    q, k = r((B, 1, Hk, D), 1).astype(mx.bfloat16), r((B, 1, Hk, D), 2).astype(mx.bfloat16)
    v = r((B, 1, Hv, D), 3).astype(mx.bfloat16)
    g, beta = mx.sigmoid(r((B, 1, Hv), 4)), mx.sigmoid(r((B, 1, Hv), 5)).astype(mx.bfloat16)
    st = [r((1, Hv, D, D), 6 + b) for b in range(B)]
    ys, ss = zip(
        *[
            gdn_recurrence(
                q[b : b + 1], k[b : b + 1], v[b : b + 1], g[b : b + 1], beta[b : b + 1], st[b]
            )
            for b in range(B)
        ]
    )
    y2, s2 = gdn_recurrence_rows(q, k, v, g, beta, st)
    return bool(
        mx.array_equal(mx.concatenate(ys), y2).item()
        and mx.array_equal(mx.concatenate(ss), s2).item()
    )
