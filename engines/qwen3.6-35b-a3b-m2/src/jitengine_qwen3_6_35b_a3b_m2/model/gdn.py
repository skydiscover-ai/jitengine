"""The Gated DeltaNet (linear attention) block, for one row and for a step of batched decode rows."""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from jitengine_qwen3_6_35b_a3b_m2.kernels.elementwise import gated_silu_f32
from jitengine_qwen3_6_35b_a3b_m2.kernels.gdn_recurrence import gdn_recurrence, recurrence_rows
from jitengine_qwen3_6_35b_a3b_m2.kvcache.linear_state import LinearStateSlot
from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig
from jitengine_qwen3_6_35b_a3b_m2.model.kernel_choice import KernelChoice
from jitengine_qwen3_6_35b_a3b_m2.model.weight import FUSIONS, _fuse, _qlin


@partial(mx.compile, shapeless=True)
def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


class Qwen3_5MoeGatedDeltaNet:
    def __init__(self, cfg: Qwen3_5MoeConfig, w: dict, p: str, fuse: frozenset = FUSIONS):
        gs = cfg.q_group_size
        self.cfg = cfg
        self.paths = KernelChoice()
        names = [p + n for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
        self.in_proj = _fuse(w, names, gs) if "gdn" in fuse else None
        if self.in_proj is None:
            self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a = (
                _qlin(w, n, gs) for n in names
            )
        self.out_proj = _qlin(w, p + "out_proj", gs)
        self.conv_w = w[p + "conv1d.weight"]  # [conv_dim, K, 1] depthwise
        self.A_log = w[p + "A_log"]
        self.dt_bias = w[p + "dt_bias"]
        self.norm_w = w[p + "norm.weight"]
        self.eps = cfg.rms_norm_eps
        self.n_keep = cfg.linear_conv_kernel_dim - 1

    def __call__(self, x: mx.array, slot: LinearStateSlot, offset: int, pre=None) -> mx.array:
        out, slot.conv, slot.ssm = self.pure(x, slot.conv, slot.ssm, pre=pre)
        return out

    def pure(self, x: mx.array, conv: mx.array, ssm: mx.array, pre=None):
        """(out, new conv window, new ssm): the block as a pure function of its state."""
        cfg = self.cfg
        B, S, _ = x.shape
        if self.in_proj is not None:
            qkv, z, b, a = self.in_proj(x, pre=pre) if pre is not None else self.in_proj(x)
        else:
            qkv, z, b, a = (
                self.in_proj_qkv(x),
                self.in_proj_z(x),
                self.in_proj_b(x),
                self.in_proj_a(x),
            )
        z = z.reshape(B, S, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        conv_input = mx.concatenate([conv, qkv], axis=1)
        new_conv = mx.contiguous(conv_input[:, -self.n_keep :, :])
        conv_out = nn.silu(mx.conv1d(conv_input, self.conv_w, groups=cfg.conv_dim))

        kd = cfg.key_dim
        q, k, v = mx.split(conv_out, [kd, 2 * kd], axis=-1)
        q = q.reshape(B, S, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.reshape(B, S, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.reshape(B, S, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        inv_scale = cfg.linear_key_head_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        y, new_ssm = gdn_recurrence(q, k, v, g, beta, ssm)

        out = gated_silu_f32(y, z, mx.fast.rms_norm(y, self.norm_w, self.eps))
        return self.out_proj(out.reshape(B, S, -1)), new_conv, new_ssm

    def step(self, x: mx.array, slots: list, offsets: list) -> mx.array:
        """One decode position for B rows: x [B, 1, D], one LinearStateSlot per row."""
        cfg = self.cfg
        B = x.shape[0]
        if self.in_proj is not None:
            qkv, z, b, a = self.in_proj.rows(x)
        else:
            qkv, z = self.in_proj_qkv.rows(x), self.in_proj_z.rows(x)
            b, a = self.in_proj_b.rows(x), self.in_proj_a.rows(x)
        z = z.reshape(B, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        prev = mx.concatenate([s.conv for s in slots], axis=0) if B > 1 else slots[0].conv
        conv_input = mx.concatenate([prev, qkv], axis=1)
        tail = conv_input[:, -self.n_keep :, :]
        for i, s in enumerate(slots):
            s.conv = mx.contiguous(tail[i : i + 1])
        conv_out = nn.silu(mx.conv1d(conv_input, self.conv_w, groups=cfg.conv_dim))

        kd = cfg.key_dim
        q, k, v = mx.split(conv_out, [kd, 2 * kd], axis=-1)
        q = q.reshape(B, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.reshape(B, 1, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.reshape(B, 1, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

        inv_scale = cfg.linear_key_head_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        y, new_states = recurrence_rows(
            q, k, v, g, beta, [s.ssm for s in slots], kernel=self.paths.gdn_rows
        )
        for s, s_i in zip(slots, new_states):
            s.ssm = s_i

        out = gated_silu_f32(y, z, mx.fast.rms_norm(y, self.norm_w, self.eps))
        return self.out_proj.rows(out.reshape(B, 1, -1))
