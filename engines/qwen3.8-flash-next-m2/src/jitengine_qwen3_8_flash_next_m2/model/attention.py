"""The sparse-attention block (`Qwen4ExpTextAttention`): gated output, `(1 + w)` q/k norms, partial
rope and the QSA mask from its indexer, with its query blocking and decode launch settings.
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m2.kernels import gdn_decode as _gf
from jitengine_qwen3_8_flash_next_m2.kvcache.kv_cache import KVSlot
from jitengine_qwen3_8_flash_next_m2.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m2.model.common import _IN_SPEC_BUILD
from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig
from jitengine_qwen3_8_flash_next_m2.model.qsa_indexer import Qwen4ExpTextQSAIndexer

# multi-weight kernel config for the four attention input projections: measured (KL, NSG, RB)
INPUT_PROJS_CFG = (8, 4, 2)


# Decode attention over QSA-masked keys reading each K/V row once per KV head (sdpa_decode_masked.py).
MASKED_SDPA_CHUNK = 128  # measured keys per chunk


MASKED_SDPA_HEADS_PER_SIMDGROUP = 2  # measured query heads per simdgroup


# Queries per sdpa call in single-row prefill, bounding the [1, H, L, P] score block. Splitting
# queries is exact, since each query's softmax spans its own keys (splitting keys would not be).
# 0 disables it.
ATTN_QUERY_BLOCK = 256


class Qwen4ExpTextAttention:
    """`Qwen4ExpTextAttention`: gated output, `(1 + w)` q/k norms, partial rope and the QSA mask.

    The mask is "causal" (L > 1) or None (L == 1) below the indexer's `prune_from`, else a bool
    [1, 1, L, P] mask from the indexer.
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
        self.q_proj = qlinear_from(w, p + "q_proj", cfg)
        self.k_proj = qlinear_from(w, p + "k_proj", cfg)
        self.v_proj = qlinear_from(w, p + "v_proj", cfg)
        self.o_proj = qlinear_from(w, p + "o_proj", cfg)
        self.q_norm_w1 = 1.0 + w[p + "q_norm.weight"].astype(mx.float32)
        self.k_norm_w1 = 1.0 + w[p + "k_norm.weight"].astype(mx.float32)
        mx.eval(self.q_norm_w1, self.k_norm_w1)
        self.indexer = Qwen4ExpTextQSAIndexer(cfg, w, p + "indexer.", compute_dtype, kernels_off)
        # q, k, v and the indexer qk share K and quantization, so decode runs them as one launch
        self._input_projs = [self.q_proj, self.k_proj, self.v_proj, self.indexer.qk_proj]
        self.input_projs_ok = "gdn_fused" not in kernels_off and _gf.supports_multi(
            self._input_projs
        )
        self.masked_sdpa_ok = "sdpa_decode_masked" not in kernels_off
        self.eps = float(cfg.rms_norm_eps)
        self.scale = cfg.head_dim**-0.5
        self.rope_dims = cfg.rope_dims
        self.rope_base = cfg.rope_theta

    def linears(self) -> list:
        return [self.q_proj, self.k_proj, self.v_proj, self.o_proj] + self.indexer.linears()

    def _norm(self, x: mx.array, w1: mx.array) -> mx.array:
        return (mx.fast.rms_norm(x, None, self.eps) * w1).astype(x.dtype)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=int(offset)
        )

    def _qkv(self, x: mx.array, B: int, L: int, rows: bool = False):
        cfg = self.cfg
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dense = (lambda m: m.rows) if rows else (lambda m: m)
        qg = dense(self.q_proj)(x).reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        k = dense(self.k_proj)(x).reshape(B, L, Hkv, D)
        v = dense(self.v_proj)(x).reshape(B, L, Hkv, D)
        q = self._norm(q, self.q_norm_w1).transpose(0, 2, 1, 3)
        k = self._norm(k, self.k_norm_w1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return q, k, v, gate

    def _fused_input_projections(self, x: mx.array):
        """`_qkv` and the indexer's `project` for one token of one row, from one GEMV over every
        input projection (`_input_projs`): q, k, v, the output gate, the indexer's q and raw k."""
        cfg = self.cfg
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        cat = _gf.qmv_multi(x.reshape(1, -1), self._input_projs, INPUT_PROJS_CFG)
        n1, n2, n3 = (lin.out_dims for lin in self._input_projs[:3])
        qg = cat[:, :n1].reshape(1, 1, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(1, 1, -1)
        k = cat[:, n1 : n1 + n2].reshape(1, 1, Hkv, D)
        v = cat[:, n1 + n2 : n1 + n2 + n3].reshape(1, 1, Hkv, D)
        q = self._norm(q, self.q_norm_w1).transpose(0, 2, 1, 3)
        k = self._norm(k, self.k_norm_w1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        qk = cat[:, n1 + n2 + n3 :]
        nq = self.indexer.nh * self.indexer.hd
        iq = qk[..., :nq].reshape(1, 1, self.indexer.nh, self.indexer.hd)
        raw_k = qk[..., nq:].reshape(1, 1, 1, self.indexer.hd)
        return q, k, v, gate, iq, raw_k

    def _sdpa(self, q, keys, values, sel, mask, offset: int, L: int):
        """sdpa over query blocks of `ATTN_QUERY_BLOCK`, arithmetically identical to one call.

        Unpruned (`sel is None`): MLX's "causal" mask aligns the last query with the last key, so
        each block's keys are truncated to `[:offset + b1]` to keep its alignment. Pruned: the block
        takes rows `sel[b0:b1]` with all columns, since column j is absolute position j.
        """
        blk = ATTN_QUERY_BLOCK
        if blk <= 0 or L <= blk:
            return mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=self.scale, mask=mask
            )
        outs = []
        for b0 in range(0, L, blk):
            b1 = min(b0 + blk, L)
            qb = q[:, :, b0:b1, :]
            if sel is None:
                end = offset + b1
                outs.append(
                    mx.fast.scaled_dot_product_attention(
                        qb,
                        keys[:, :, :end, :],
                        values[:, :, :end, :],
                        scale=self.scale,
                        mask="causal",
                    )
                )
            elif isinstance(sel, mx.array):
                outs.append(
                    mx.fast.scaled_dot_product_attention(
                        qb, keys, values, scale=self.scale, mask=sel[:, :, b0:b1, :]
                    )
                )
            else:
                outs.append(
                    mx.fast.scaled_dot_product_attention(
                        qb, keys, values, scale=self.scale, mask=mx.array(sel[b0:b1])[None, None]
                    )
                )
        return mx.concatenate(outs, axis=2)

    def __call__(
        self, x: mx.array, slot: KVSlot, offset: int, trace: dict | None = None
    ) -> mx.array:
        """One row: x [1, L, hidden] at absolute positions [offset, offset + L)."""
        B, L, _ = x.shape
        if B != 1:
            raise InvariantError("the single-row path carries one row (the batched step is `step`)")
        if L == 1 and trace is None and self.input_projs_ok:
            q, k, v, gate, iq, raw_k = self._fused_input_projections(x)
        else:
            q, k, v, gate = self._qkv(x, B, L)
            iq, raw_k = self.indexer.project(x)
        q = self._rope(q, offset)
        k = self._rope(k, offset)
        keys, values = slot.append(k, v, raw_k)
        if L == 1 and trace is None:
            sel, mask, out = self._attend_decode(q, keys, values, gate, iq, slot, offset, L)
        elif L > 1 and trace is None and self.indexer.device_prefill_ok(slot.idx_keys().shape[2]):
            sel, out = self._attend_device_prefill(q, keys, values, iq, slot, offset, L)
        else:
            sel, mask, out = self._attend_host_select(q, keys, values, gate, iq, slot, offset, L)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if trace is not None:
            trace.update(
                q=q,
                k=k,
                v=v,
                gate=gate,
                idx_q=iq,
                idx_raw_k=raw_k,
                attn_out=out,
                sel_mask=sel,
                mask_kind=("bool" if sel is not None else str(mask)),
            )
        return self.o_proj(out * mx.sigmoid(gate))

    def _attend_decode(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        gate: mx.array,
        iq: mx.array,
        slot: KVSlot,
        offset: int,
        L: int,
    ) -> tuple:
        """One decode position: the device-side selection mask, then masked or plain sdpa."""
        dm = self.indexer.select_device(iq, slot.idx_keys(), offset, cache=slot.idx_bk)
        if dm is not None and not _IN_SPEC_BUILD[0]:
            mx.async_eval(q, keys, values, gate, dm)
        sel, mask = (None, None) if dm is None else (True, dm)
        if dm is not None and self.masked_sdpa_ok:
            from jitengine_qwen3_8_flash_next_m2.kernels import (
                sdpa_decode_masked as _sdpa_masked,
            )

            out = _sdpa_masked.decode_masked(
                q,
                slot.keys,
                slot.values,
                slot.offset,
                self.scale,
                dm,
                chunk=MASKED_SDPA_CHUNK,
                hps=MASKED_SDPA_HEADS_PER_SIMDGROUP,
            )
        else:
            out = self._sdpa(q, keys, values, None if dm is None else True, mask, offset, L)
        return sel, mask, out

    def _attend_device_prefill(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        iq: mx.array,
        slot: KVSlot,
        offset: int,
        L: int,
    ) -> tuple:
        """A prefill chunk selected on the device: the device mask, causal when it selects nothing."""
        dm = self.indexer.select_prefill_device(iq, slot.idx_keys(), offset, cache=slot.idx_bk)
        if dm is None:
            out = self._sdpa(q, keys, values, None, "causal", offset, L)
        else:
            out = self._sdpa(q, keys, values, dm, dm, offset, L)
        sel = dm
        return sel, out

    def _attend_host_select(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        gate: mx.array,
        iq: mx.array,
        slot: KVSlot,
        offset: int,
        L: int,
    ) -> tuple:
        """The general path (traced or short prefill): the host-side selection, then sdpa."""
        sel = self.indexer.select(
            iq, slot.idx_keys(), offset, dispatch=(q, keys, values, gate), cache=slot.idx_bk
        )
        mask = ("causal" if L > 1 else None) if sel is None else mx.array(sel)[None, None]
        out = self._sdpa(q, keys, values, sel, mask, offset, L)
        return sel, mask, out

    def decode2(self, x: mx.array, slot: KVSlot, offset: int, on_row0) -> mx.array:
        """Decode positions offset and offset + 1 of one sequence, x [2, 1, hidden].

        The input projections and o_proj share one launch each (rows bit-identical to the solo
        kernels); each position then runs the solo decode path in order, so position 1 attends over
        position 0's keys. `on_row0()` is called once position 0 is in the slot.
        """
        cfg = self.cfg
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        cat2 = _gf.qmv_multi_v(x.reshape(2, -1), self._input_projs, INPUT_PROJS_CFG, vecs=2)
        n1, n2, n3 = (lin.out_dims for lin in self._input_projs[:3])
        nq = self.indexer.nh * self.indexer.hd
        outs = []
        for r in range(2):
            cat = cat2[r : r + 1]
            off = offset + r
            qg = cat[:, :n1].reshape(1, 1, H, 2 * D)
            q, gate = mx.split(qg, 2, axis=-1)
            gate = gate.reshape(1, 1, -1)
            k = cat[:, n1 : n1 + n2].reshape(1, 1, Hkv, D)
            v = cat[:, n1 + n2 : n1 + n2 + n3].reshape(1, 1, Hkv, D)
            q = self._norm(q, self.q_norm_w1).transpose(0, 2, 1, 3)
            k = self._norm(k, self.k_norm_w1).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            qk = cat[:, n1 + n2 + n3 :]
            iq = qk[..., :nq].reshape(1, 1, self.indexer.nh, self.indexer.hd)
            raw_k = qk[..., nq:].reshape(1, 1, 1, self.indexer.hd)
            q = self._rope(q, off)
            k = self._rope(k, off)
            keys, values = slot.append(k, v, raw_k)
            dm = self.indexer.select_device(iq, slot.idx_keys(), off, cache=slot.idx_bk)
            if dm is not None and self.masked_sdpa_ok:
                from jitengine_qwen3_8_flash_next_m2.kernels import (
                    sdpa_decode_masked as _sdpa_masked,
                )

                out = _sdpa_masked.decode_masked(
                    q,
                    slot.keys,
                    slot.values,
                    slot.offset,
                    self.scale,
                    dm,
                    chunk=MASKED_SDPA_CHUNK,
                    hps=MASKED_SDPA_HEADS_PER_SIMDGROUP,
                )
            else:
                out = self._sdpa(q, keys, values, None if dm is None else True, dm, off, 1)
            outs.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(gate))
            if r == 0:
                on_row0()
        return self.o_proj(mx.concatenate(outs, axis=0))

    def decode2_ok(self) -> bool:
        return bool(self.input_projs_ok)

    def step(self, x: mx.array, slots: list, offsets: list) -> mx.array:
        """One decode position for B rows (x [B, 1, hidden]); rope, KV and sdpa run per row."""
        B = x.shape[0]
        q, k, v, gate = self._qkv(x, B, 1, rows=True)
        iq, raw_k = self.indexer.project(x, rows=True)
        outs = []
        for i, (slot, off) in enumerate(zip(slots, offsets)):
            qi = self._rope(q[i : i + 1], off)
            ki = self._rope(k[i : i + 1], off)
            keys, values = slot.append(ki, v[i : i + 1], raw_k[i : i + 1])
            # the device-side mask: same selection as numpy, without a per-row host sync
            mask = self.indexer.select_device(
                iq[i : i + 1], slot.idx_keys(), off, cache=slot.idx_bk
            )
            outs.append(
                mx.fast.scaled_dot_product_attention(qi, keys, values, scale=self.scale, mask=mask)
            )
        out = mx.concatenate(outs, axis=0) if B > 1 else outs[0]
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return self.o_proj.rows(out * mx.sigmoid(gate))
