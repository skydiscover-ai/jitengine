"""The Gated DeltaNet block (`Qwen4ExpTextGatedDeltaNet`): input projections, causal conv, the gated
delta-rule recurrence and the gated output norm, with its compiled elementwise helpers, the
recurrence choice (`GDN_IMPL`), the MTP verify pass's lazy rollback state (`GDN_LAZY`, `LazyRow0`)
and the fused decode launch config.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_decode as _gf
from jitengine_qwen3_8_flash_next_m4pro.kernels.gdn_recurrence import (
    gdn_recurrence,
    gdn_recurrence_ops,
    gdn_recurrence_pair,
    gdn_recurrence_staged,
    gdn_recurrence_step,
    pair_available,
)
from jitengine_qwen3_8_flash_next_m4pro.kvcache.linear_state import LinearStateSlot
from jitengine_qwen3_8_flash_next_m4pro.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m4pro.model.common import GDN_FUSED_MULTI
from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

# GDN recurrence: "kernel" (`kernels.gdn_recurrence`) or "ops" (the plain-ops definition).
GDN_IMPL = "kernel"


GDN_FUSED_NSG = 4  # measured simdgroups per threadgroup of the pre and post launches


GDN_STAGED_TB = 8  # steps per staging tile of the multi-position recurrence


# The MTP verify pass's compiled pair build runs both rows' recurrences as one launch
# (`gdn_recurrence_pair`) and keeps the state after row 0, the rollback point, as `LazyRow0`, made
# only by a rollback; off, it runs two `gdn_recurrence_step` launches and stores that state. The
# same bits either way.
GDN_LAZY = True


@partial(mx.compile, shapeless=True)
def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """Per-step state decay factor exp(g) = exp(-exp(A_log) * softplus(a + dt_bias)), in fp32."""
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
    )


@partial(mx.compile, shapeless=True)
def gated_silu_f32(h: mx.array, gate: mx.array, x: mx.array) -> mx.array:
    g = nn.silu(gate.astype(mx.float32))
    return (g * x.astype(mx.float32)).astype(h.dtype)


@partial(mx.compile, shapeless=True)
def gated_sigmoid_f32(h: mx.array, gate: mx.array, x: mx.array) -> mx.array:
    g = mx.sigmoid(gate.astype(mx.float32))
    return (g * x.astype(mx.float32)).astype(h.dtype)


def _detach(x: mx.array, materialize: bool) -> mx.array:
    """Copy `x` into a fresh buffer via x * 1, which, unlike `mx.contiguous`, never aliases."""
    out = x * mx.array(1, dtype=x.dtype)
    if materialize:
        mx.eval(out)
    return out


class LazyRow0:
    """The DeltaNet state after row 0 of a lazy pair build (`fused_decode2_rows(lazy=True)`), kept
    as what makes it: the state before the pass and row 0's recurrence inputs. It is the verify
    pass's rollback point (`snaps`), and a slot holds it only between a restore and the next build:

    - `advance(row1)`, a p + 1-only rebuild (`fused_decode` on a slot holding it): row 1's output
      and state from the state before the pass by the pair launch, row 0's inputs kept and row 1's
      new, which is the stored build's row-1 launch on its row-0 state bit for bit and reads and
      writes the state once, as that launch does;
    - `state()`, a rollback (`rollback1`): the row-0 launch the stored build runs
      (`gdn_recurrence_step`) on the same inputs, so that build's rollback state bit for bit. It
      runs once (a second call returns the same array), then lets the inputs go, and is counted on
      the block (`lazy_materialized`).
    """

    __slots__ = ("block", "pre", "row0", "ssm")

    def __init__(self, block, pre: mx.array, row0):
        self.block, self.pre, self.row0, self.ssm = block, pre, tuple(row0), None

    def _dims(self) -> tuple:
        cfg = self.block.cfg
        hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        return hk, dk, cfg.linear_num_value_heads, cfg.linear_value_head_dim

    @property
    def pending(self) -> bool:
        """Whether the state is still unmade (`state()` not called)."""
        return self.ssm is None

    def state(self) -> mx.array:
        if self.ssm is None:
            self.ssm = gdn_recurrence_step(*self.row0, self.pre, *self._dims())[1]
            self.pre = self.row0 = None
            self.block.lazy_materialized += 1
        return self.ssm

    def advance(self, row1) -> tuple:
        """(y [Hv, Dv], state after row 1) from this point, row1 = row 1's (q, k, v, g, beta)."""
        if self.ssm is not None:
            return gdn_recurrence_step(*row1, self.ssm, *self._dims())
        _, y1, ssm = gdn_recurrence_pair(self.row0, row1, self.pre, *self._dims())
        return y1, ssm


def _recurrence(q, k, v, g, beta, state, kernel: bool = True):
    if GDN_IMPL == "ops" or not kernel:
        return gdn_recurrence_ops(q, k, v, g, beta, state)
    if k.shape[1] > 1:
        # multi-position inputs use the staged-input variant (bit-identical)
        return gdn_recurrence_staged(q, k, v, g, beta, state, tb=GDN_STAGED_TB)
    return gdn_recurrence(q, k, v, g, beta, state)


# ----------------------------------------------------------------------------- blocks
class Qwen4ExpTextGatedDeltaNet:
    """Gated DeltaNet block (`Qwen4ExpTextGatedDeltaNet`).

    Takes this checkpoint's split projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`,
    `in_proj_a`) or the fused `in_proj_qkvz` / `in_proj_ba` layout, chosen from the weight names.
    """

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        p: str,
        compute_dtype=mx.float32,
        kernels_off: frozenset = frozenset(),
    ):
        self.cfg = cfg
        self.dtype = compute_dtype
        self.fused = (p + "in_proj_qkvz.weight") in w
        if self.fused:
            self.in_proj_qkvz = qlinear_from(w, p + "in_proj_qkvz", cfg)
            self.in_proj_ba = qlinear_from(w, p + "in_proj_ba", cfg)
        else:
            self.in_proj_qkv = qlinear_from(w, p + "in_proj_qkv", cfg)
            self.in_proj_z = qlinear_from(w, p + "in_proj_z", cfg)
            self.in_proj_b = qlinear_from(w, p + "in_proj_b", cfg)
            self.in_proj_a = qlinear_from(w, p + "in_proj_a", cfg)
        self.out_proj = qlinear_from(w, p + "out_proj", cfg)
        self.conv_w = w[p + "conv1d.weight"].astype(
            compute_dtype
        )  # [conv_dim, K, 1] depthwise, MLX layout
        self.A_log = w[p + "A_log"].astype(mx.float32)
        self.dt_bias = w[p + "dt_bias"].astype(mx.float32)
        self.norm_w = w[p + "norm.weight"].astype(compute_dtype)  # the gated norm: plain w
        self.eps = float(cfg.rms_norm_eps)
        self.n_keep = cfg.linear_conv_kernel_dim - 1
        self.sigmoid_gate = str(cfg.output_gate_type) == "sigmoid"
        # l2norm(x) = x * rsqrt(sum(x^2) + 1e-6); rms_norm puts its eps on the mean, so the sum's
        # 1e-6 is eps / Dk on the mean, and rms_norm = sqrt(Dk) * l2norm. Hence the two scales.
        dk = cfg.linear_key_head_dim
        self.l2_eps = 1e-6 / dk
        self.q_scale = 1.0 / dk  # sqrt(Dk) l2norm(q) / Dk = l2norm(q) / sqrt(Dk)
        self.k_scale = dk**-0.5  # sqrt(Dk) l2norm(k) / sqrt(Dk) = l2norm(k)
        mx.eval(self.conv_w, self.A_log, self.dt_bias, self.norm_w)
        self.recurrence_kernel_ok = "gdn_kernel" not in kernels_off
        # `gdn_recurrence_pair` for `GDN_LAZY`: cut from the recurrence source and its check passed
        self.pair_ok = "gdn_pair" not in kernels_off and pair_available()
        self.lazy_materialized = 0  # `LazyRow0` states made (rollbacks)
        self.lazy_consumed = 0  # `LazyRow0` states the pair launch advanced (p + 1-only rebuilds)
        self._compiled = mx.compile(self._decode_body)
        self.fused_ok = (
            "gdn_fused" not in kernels_off
            and not self.fused
            and _gf.supports_gdn(cfg)
            and _gf.supports_multi(
                [self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a]
            )
        )
        if self.fused_ok:
            self._proj_lins = [self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a]
            n1, n2, n3 = (lin.out_dims for lin in self._proj_lins[:3])
            self._proj_offs = (0, n1, n1 + n2, n1 + n2 + n3)
            self.conv_w2d = self.conv_w.reshape(self.conv_w.shape[0], -1)
            mx.eval(self.conv_w2d)

    def fused_decode(self, x: mx.array, slot: LinearStateSlot) -> mx.array:
        """One decode position, B = S = 1: x [1, 1, hidden] -> out [1, 1, hidden]; advances `slot`.
        A slot holding a `LazyRow0` (a p + 1-only rebuild on a lazy rollback point) is advanced by
        `LazyRow0.advance`, the stored launch's bits."""
        ycat = _gf.qmv_multi(x.reshape(1, -1), self._proj_lins, GDN_FUSED_MULTI)[0]
        q, k, v, g, beta, new_conv = _gf.gdn_pre(
            slot.conv[0],
            ycat,
            self.conv_w2d,
            self.A_log,
            self.dt_bias,
            self.cfg,
            self.l2_eps,
            self.q_scale,
            self.k_scale,
            self._proj_offs,
            GDN_FUSED_NSG,
        )
        slot.conv = new_conv[None]
        if isinstance(slot.ssm, LazyRow0):
            y, slot.ssm = slot.ssm.advance((q, k, v, g, beta))
        else:
            y, slot.ssm = _recurrence(
                q[None, None],
                k[None, None],
                v[None, None],
                g[None, None],
                beta[None, None],
                slot.ssm,
                self.recurrence_kernel_ok,
            )
            y = y[0, 0]
        act = _gf.gdn_post(
            y, ycat, self.norm_w, self.cfg, self.eps, self._proj_offs[1], GDN_FUSED_NSG
        )
        return self.out_proj(act.reshape(1, 1, -1))

    def fused_decode2(self, x: mx.array, slot: LinearStateSlot, on_row0) -> mx.array:
        """Two consecutive positions of one sequence, x [2, 1, hidden] -> out [2, 1, hidden].

        `fused_decode`'s arithmetic for each row: the input projections share one `qmv_multi_v`
        launch and out_proj one `QLinear.rows` call (both bit-identical per row to the one-row
        calls; `rows` splits where a shape's launch is not); the conv, recurrence and gated norm run
        per position in order, so row 1 starts from the state row 0 left. `on_row0()` is called
        once row 0's state is in the slot.
        """
        ycat = _gf.qmv_multi_v(x.reshape(2, -1), self._proj_lins, GDN_FUSED_MULTI, vecs=2)
        acts = []
        for r in range(2):
            q, k, v, g, beta, new_conv = _gf.gdn_pre(
                slot.conv[0],
                ycat[r],
                self.conv_w2d,
                self.A_log,
                self.dt_bias,
                self.cfg,
                self.l2_eps,
                self.q_scale,
                self.k_scale,
                self._proj_offs,
                GDN_FUSED_NSG,
            )
            slot.conv = new_conv[None]
            y, slot.ssm = _recurrence(
                q[None, None],
                k[None, None],
                v[None, None],
                g[None, None],
                beta[None, None],
                slot.ssm,
                self.recurrence_kernel_ok,
            )
            acts.append(
                _gf.gdn_post(
                    y[0, 0],
                    ycat[r],
                    self.norm_w,
                    self.cfg,
                    self.eps,
                    self._proj_offs[1],
                    GDN_FUSED_NSG,
                )
            )
            if r == 0:
                on_row0()
        return self.out_proj.rows(mx.stack(acts, axis=0).reshape(2, 1, -1))

    def rows_ok(self) -> bool:
        """Whether `fused_decode2_rows` takes this block: the fused glue and the recurrence kernel."""
        return bool(self.fused_ok) and self.recurrence_kernel_ok and GDN_IMPL == "kernel"

    def rows_lazy(self) -> bool:
        """Whether `fused_decode2_rows` runs lazy: `GDN_LAZY` and the pair launch's check passed."""
        return GDN_LAZY and self.pair_ok

    def fused_decode2_rows(
        self, x: mx.array, conv: mx.array, ssm: mx.array, lazy: bool = False
    ) -> tuple:
        """`fused_decode2` on flat rows, x [2, hidden] -> (out [2, hidden], (conv, ssm) after row 0,
        (conv, ssm) after row 1), with no view launches between its kernels: the glue reads its row
        of the projections in place (`row`), the conv state is written at its slot shape and the
        recurrence takes its inputs unexpanded. The same launches on the same values, in order.

        `lazy`: both rows' recurrences run as one `gdn_recurrence_pair` launch, after both rows'
        pre kernels and before both post kernels, and the state after row 0 is not made: in its
        place come row 0's recurrence inputs, (conv, q, k, v, g, beta) after row 0, which
        `LazyRow0` turns into it with the stored build's own launch. Every output is the same bits.
        """
        cfg = self.cfg
        hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        ycat = _gf.qmv_multi_v(x, self._proj_lins, GDN_FUSED_MULTI, vecs=2)
        if lazy:
            pres = []
            for r in range(2):
                *row, conv = self._pre_row(conv, ycat, r)
                pres.append((conv, row))
            (conv0, row0), (conv1, row1) = pres
            y0, y1, ssm = gdn_recurrence_pair(row0, row1, ssm, hk, dk, hv, dv)
            acts = [self._post_row(y, ycat, r) for r, y in enumerate((y0, y1))]
            out = self.out_proj.rows(mx.concatenate(acts, axis=0))
            return out, (conv0, *row0), (conv1, ssm)
        acts, states = [], []
        for r in range(2):
            q, k, v, g, beta, conv = self._pre_row(conv, ycat, r)
            y, ssm = gdn_recurrence_step(q, k, v, g, beta, ssm, hk, dk, hv, dv)
            acts.append(self._post_row(y, ycat, r))
            states.append((conv, ssm))
        return self.out_proj.rows(mx.concatenate(acts, axis=0)), states[0], states[1]

    def _pre_row(self, conv: mx.array, ycat: mx.array, r: int) -> tuple:
        """`gdn_pre` on row r of the two-row projections: (q, k, v, g, beta, conv after row r)."""
        cfg = self.cfg
        hk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        return _gf.gdn_pre(
            conv,
            ycat,
            self.conv_w2d,
            self.A_log,
            self.dt_bias,
            cfg,
            self.l2_eps,
            self.q_scale,
            self.k_scale,
            self._proj_offs,
            GDN_FUSED_NSG,
            row=r,
            conv_shape=(1, cfg.linear_conv_kernel_dim - 1, 2 * hk * dk + hv * dv),
        )

    def _post_row(self, y: mx.array, ycat: mx.array, r: int) -> mx.array:
        """`gdn_post` on row r: the gated norm of y with row r's z, [1, Hv * Dv]."""
        cfg = self.cfg
        hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        return _gf.gdn_post(
            y,
            ycat,
            self.norm_w,
            cfg,
            self.eps,
            self._proj_offs[1],
            GDN_FUSED_NSG,
            row=r,
            out_shape=(1, hv * dv),
        )

    def _decode_body(self, x: mx.array, conv: mx.array, ssm: mx.array):
        """One decode position as one compiled graph: (x, conv, ssm) -> (out, new conv, new ssm)."""
        cfg = self.cfg
        B, S = 1, 1
        qkv, z, b, a = self._project(x, B, S)
        conv_input = mx.concatenate([conv, qkv], axis=1)
        new_conv = conv_input[:, -self.n_keep :, :] * mx.array(1, dtype=conv_input.dtype)
        conv_out = nn.silu(mx.conv1d(conv_input, self.conv_w, groups=cfg.conv_dim))
        q, k, v = self._qkv_heads(conv_out, B, S)
        beta = mx.sigmoid(b)
        g = mx.exp(
            -mx.exp(self.A_log.astype(mx.float32))
            * nn.softplus(a.astype(mx.float32) + self.dt_bias.astype(mx.float32))
        )
        y, new_ssm = _recurrence(q, k, v, g, beta, ssm, self.recurrence_kernel_ok)
        normed = mx.fast.rms_norm(y, self.norm_w, self.eps)
        if self.sigmoid_gate:
            gate = mx.sigmoid(z.astype(mx.float32))
        else:
            gate = nn.silu(z.astype(mx.float32))
        out = (gate * normed.astype(mx.float32)).astype(y.dtype).reshape(B, S, -1)
        return self.out_proj(out), new_conv, new_ssm

    def linears(self) -> list:
        return (
            [self.in_proj_qkvz, self.in_proj_ba]
            if self.fused
            else [self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a]
        ) + [self.out_proj]

    def _split_qkvz_ba(self, qkvz: mx.array, ba: mx.array, B: int, S: int):
        cfg = self.cfg
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dn, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        r = nv // nk
        q, k, v, z = mx.split(qkvz.reshape(B, S, nk, -1), [dn, 2 * dn, 2 * dn + r * dv], axis=-1)
        b, a = mx.split(ba.reshape(B, S, nk, -1), [r], axis=-1)
        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        return mixed_qkv, z.reshape(B, S, nv, dv), b.reshape(B, S, nv), a.reshape(B, S, nv)

    def _project(self, x: mx.array, B: int, S: int, rows: bool = False):
        if self.fused:
            lin = (lambda m: m.rows) if rows else (lambda m: m)
            return self._split_qkvz_ba(lin(self.in_proj_qkvz)(x), lin(self.in_proj_ba)(x), B, S)
        dense = (lambda m: m.rows) if rows else (lambda m: m)
        return (
            dense(self.in_proj_qkv)(x),
            dense(self.in_proj_z)(x).reshape(
                B, S, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim
            ),
            dense(self.in_proj_b)(x),
            dense(self.in_proj_a)(x),
        )

    def _qkv_heads(self, conv_out: mx.array, B: int, S: int):
        cfg = self.cfg
        kd = cfg.key_dim
        q, k, v = mx.split(conv_out, [kd, 2 * kd], axis=-1)
        q = q.reshape(B, S, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.reshape(B, S, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.reshape(B, S, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        q = self.q_scale * mx.fast.rms_norm(q, None, self.l2_eps)
        k = self.k_scale * mx.fast.rms_norm(k, None, self.l2_eps)
        return q, k, v

    def _output(self, y: mx.array, z: mx.array, B: int, S: int, rows: bool = False) -> mx.array:
        normed = mx.fast.rms_norm(y, self.norm_w, self.eps)
        out = gated_sigmoid_f32(y, z, normed) if self.sigmoid_gate else gated_silu_f32(y, z, normed)
        out = out.reshape(B, S, -1)
        return self.out_proj.rows(out) if rows else self.out_proj(out)

    def __call__(
        self, x: mx.array, slot: LinearStateSlot, offset: int, trace: dict | None = None
    ) -> mx.array:
        cfg = self.cfg
        B, S, _ = x.shape
        if trace is None and B == 1 and S == 1 and self.fused_ok:
            return self.fused_decode(x, slot)
        if trace is None and B == 1 and S == 1:
            out, slot.conv, slot.ssm = self._compiled(x, slot.conv, slot.ssm)
            return out
        qkv, z, b, a = self._project(x, B, S)

        conv_input = mx.concatenate([slot.conv, qkv], axis=1)
        # a copy, not a view, so the slot does not pin the chunk buffer
        slot.conv = _detach(conv_input[:, -self.n_keep :, :], S > self.n_keep)
        conv_out = nn.silu(mx.conv1d(conv_input, self.conv_w, groups=cfg.conv_dim))

        q, k, v = self._qkv_heads(conv_out, B, S)
        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        y, slot.ssm = _recurrence(q, k, v, g, beta, slot.ssm, self.recurrence_kernel_ok)
        return self._output(y, z, B, S)

    def step(self, x: mx.array, slots: list, offsets: list) -> mx.array:
        """One decode position for B rows (x [B, 1, D]); only the recurrence runs per row."""
        cfg = self.cfg
        B = x.shape[0]
        qkv, z, b, a = self._project(x, B, 1, rows=True)

        prev = mx.concatenate([s.conv for s in slots], axis=0) if B > 1 else slots[0].conv
        conv_input = mx.concatenate([prev, qkv], axis=1)
        tail = conv_input[:, -self.n_keep :, :]
        for i, s in enumerate(slots):
            s.conv = _detach(tail[i : i + 1], False)
        conv_out = nn.silu(mx.conv1d(conv_input, self.conv_w, groups=cfg.conv_dim))

        q, k, v = self._qkv_heads(conv_out, B, 1)
        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        ys, new_states = [], []
        for i, s in enumerate(slots):
            y_i, s_i = _recurrence(
                q[i : i + 1],
                k[i : i + 1],
                v[i : i + 1],
                g[i : i + 1],
                beta[i : i + 1],
                s.ssm,
                self.recurrence_kernel_ok,
            )
            ys.append(y_i)
            new_states.append(s_i)
        for s, s_i in zip(slots, new_states):
            s.ssm = s_i
        y = mx.concatenate(ys, axis=0) if B > 1 else ys[0]
        return self._output(y, z, B, 1, rows=True)
