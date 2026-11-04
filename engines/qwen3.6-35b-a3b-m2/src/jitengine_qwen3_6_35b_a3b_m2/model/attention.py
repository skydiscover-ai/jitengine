"""The gated attention block (full attention), for one row and for a step of batched decode rows."""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kvcache.kv_cache import KVSlot
from jitengine_qwen3_6_35b_a3b_m2.model.config import Qwen3_5MoeConfig
from jitengine_qwen3_6_35b_a3b_m2.model.kernel_choice import KernelChoice
from jitengine_qwen3_6_35b_a3b_m2.model.weight import FUSIONS, _fuse, _qlin


class Qwen3_5MoeAttention:
    def __init__(self, cfg: Qwen3_5MoeConfig, w: dict, p: str, fuse: frozenset = FUSIONS):
        gs = cfg.q_group_size
        self.cfg = cfg
        names = [p + n for n in ("q_proj", "k_proj", "v_proj")]
        self.qkv_proj = _fuse(w, names, gs) if "attn" in fuse else None
        if self.qkv_proj is None:
            self.q_proj, self.k_proj, self.v_proj = (_qlin(w, n, gs) for n in names)
        self.o_proj = _qlin(w, p + "o_proj", gs)
        self.q_norm_w = w[p + "q_norm.weight"]
        self.k_norm_w = w[p + "k_norm.weight"]
        self.eps = cfg.rms_norm_eps
        self.scale = cfg.head_dim**-0.5
        self.rope_dims = cfg.rope_dims
        self.rope_base = cfg.rope_theta
        self.paths = KernelChoice()

    def __call__(self, x: mx.array, slot: KVSlot, offset, pre=None) -> mx.array:
        q, k, v, gate = self.pre(x, offset, pre=pre)
        keys, values = slot.append(k, v)
        if self.paths.exact_sdpa and q.shape[2] == 1:
            from jitengine_qwen3_6_35b_a3b_m2.kernels.sdpa_decode import decode

            return self.post(decode(q, slot.keys, slot.values, slot.offset, self.scale), gate)
        return self.post(self.attend(q, keys, values), gate)

    def pre(self, x: mx.array, offset, pre=None):
        """Projections, q/k norms and rope: q [B, H, L, D], k/v [B, Hkv, L, D], flat gate.

        `offset` may be an int or a scalar array; rope is bit-identical for both.
        """
        cfg = self.cfg
        B, L, _ = x.shape
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        if self.qkv_proj is not None:
            qg, k, v = self.qkv_proj(x, pre=pre) if pre is not None else self.qkv_proj(x)
        else:
            qg, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        qg = qg.reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        k = k.reshape(B, L, Hkv, D)
        v = v.reshape(B, L, Hkv, D)
        q = mx.fast.rms_norm(q, self.q_norm_w, self.eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, self.k_norm_w, self.eps).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = mx.fast.rope(
            q, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )
        return q, k, v, gate

    def attend(self, q, keys, values):
        L = q.shape[2]
        return mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask="causal" if L > 1 else None
        )

    def post(self, out, gate):
        B, _H, L, _D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    def step(self, x: mx.array, slots: list, offsets: list) -> mx.array:
        """One decode position for B rows: x [B, 1, D], one KVSlot and one rope offset per row.

        Rope and the KV append stay per row because mx.fast.rope takes a single scalar offset.
        """
        cfg = self.cfg
        B = x.shape[0]
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        if self.qkv_proj is not None:
            qg, k, v = self.qkv_proj.rows(x)
        else:
            qg, k, v = self.q_proj.rows(x), self.k_proj.rows(x), self.v_proj.rows(x)
        qg = qg.reshape(B, 1, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, 1, -1)
        k = k.reshape(B, 1, Hkv, D)
        v = v.reshape(B, 1, Hkv, D)

        q = mx.fast.rms_norm(q, self.q_norm_w, self.eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, self.k_norm_w, self.eps).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        outs = []
        if self.paths.exact_sdpa and self.paths.sdpa_rows:
            from jitengine_qwen3_6_35b_a3b_m2.kernels.sdpa_decode import decode_rows

            qs = []
            for i, (slot, off) in enumerate(zip(slots, offsets)):
                qi = mx.fast.rope(
                    q[i : i + 1],
                    self.rope_dims,
                    traditional=False,
                    base=self.rope_base,
                    scale=1.0,
                    offset=off,
                )
                ki = mx.fast.rope(
                    k[i : i + 1],
                    self.rope_dims,
                    traditional=False,
                    base=self.rope_base,
                    scale=1.0,
                    offset=off,
                )
                slot.append(ki, v[i : i + 1])
                qs.append(qi)
            # every row's attention in one launch when all of them take the exact kernel
            out = decode_rows(
                mx.concatenate(qs, axis=0),
                [s.keys for s in slots],
                [s.values for s in slots],
                [s.offset for s in slots],
                self.scale,
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            return self.o_proj.rows(out * mx.sigmoid(gate))
        for i, (slot, off) in enumerate(zip(slots, offsets)):
            qi = mx.fast.rope(
                q[i : i + 1],
                self.rope_dims,
                traditional=False,
                base=self.rope_base,
                scale=1.0,
                offset=off,
            )
            ki = mx.fast.rope(
                k[i : i + 1],
                self.rope_dims,
                traditional=False,
                base=self.rope_base,
                scale=1.0,
                offset=off,
            )
            keys, values = slot.append(ki, v[i : i + 1])
            if self.paths.exact_sdpa:
                from jitengine_qwen3_6_35b_a3b_m2.kernels.sdpa_decode import decode

                outs.append(decode(qi, slot.keys, slot.values, slot.offset, self.scale))
                continue
            outs.append(
                mx.fast.scaled_dot_product_attention(qi, keys, values, scale=self.scale, mask=None)
            )
        out = mx.concatenate(outs, axis=0)
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return self.o_proj.rows(out * mx.sigmoid(gate))
