"""The solo-decode path of `Qwen4ExpTextSparseMoeBlock` (one row, one token), as a mixin.

`fused_decode` routes with the fused top-k kernel and runs the shared expert as two
multi-weight launches where it can (`router_and_shared`, `shared_fused`); it then takes the
speculative gather when the layer allows it, else the pooled gather, and adds the routed and
shared outputs into the residual with one `moe_combine` kernel (stock ops when the layer's
experts take several waves).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_decode as _gf
from jitengine_qwen3_8_flash_next_m4pro.kernels import router_combine as _mf
from jitengine_qwen3_8_flash_next_m4pro.model.common import GDN_FUSED_MULTI
from jitengine_qwen3_8_flash_next_m4pro.model.hyper_connections import hyper_inject


class SoloDecodeMixin:
    """One row, one token: `fused_decode`, with its router and fused shared expert."""

    def shared_fused(self, x: mx.array):
        """(ungated shared output [1, 1, hid], gate logit [1]); `moe_combine` applies the gate."""
        gu = _gf.qmv_multi(
            x.reshape(1, -1), [self.shared_gate_proj, self.shared_up_proj], GDN_FUSED_MULTI
        )
        sh = _gf.qmv_multi(gu, [self.shared_down_proj], GDN_FUSED_MULTI, swiglu=True)
        return sh.reshape(x.shape), self.shared_gate(x).reshape(-1)

    def fused_decode(
        self,
        x: mx.array,
        h2: mx.array,
        inject: mx.array,
        telem=None,
        build_only: bool = False,
    ):
        """Solo-decode MoE of x [1, 1, hid] injected into h2; returns h3 [1, 1, hc*hid]."""
        inds, scores, sh, shg = self.router_and_shared(x)
        if self.spec_ok and self.grouped_for(x.dtype):
            if build_only:
                return self.spec_build(x, inds, scores, sh, shg, h2, inject, telem)
            return self._spec_decode(x, inds, scores, sh, shg, h2, inject, telem)
        inds3 = inds.reshape(1, 1, self.top_k)
        parts = self._experts_pooled(
            x, inds3, telem, overlap=(sh,) if shg is None else (sh, shg), parts=True
        )
        if parts is None:
            y = self._experts_pooled(x, inds3, telem, overlap=(sh,))
            shared = sh if shg is None else mx.sigmoid(shg) * sh
            y = (y * scores.reshape(1, 1, self.top_k, 1)).sum(axis=-2) + shared
            return hyper_inject(y, h2, inject)
        order = np.concatenate([p for _, p in parts]) if len(parts) > 1 else parts[0][1]
        sel = np.empty_like(order)
        sel[order] = np.arange(order.size, dtype=order.dtype)
        y1 = parts[0][0].reshape(-1, self.hid)
        y2 = parts[1][0].reshape(-1, self.hid) if len(parts) > 1 else None
        h3 = _mf.moe_combine(y1, y2, sel, scores, sh, h2, inject, self.hc, self.hid, sh_gate=shg)
        return h3.reshape(h2.shape)

    def router_and_shared2(self, x: mx.array) -> list:
        """`router_and_shared` for two consecutive positions, x [2, 1, hid]: [(inds, scores, sh,
        shg)] per row. The shared expert's launches take both rows (`qmv_multi_v`, bit-identical per
        row to `shared_fused`'s `qmv_multi`); the router and the shared gate run per row."""
        gu = _gf.qmv_multi_v(
            x.reshape(2, -1), [self.shared_gate_proj, self.shared_up_proj], GDN_FUSED_MULTI, vecs=2
        )
        sh = _gf.qmv_multi_v(gu, [self.shared_down_proj], GDN_FUSED_MULTI, swiglu=True, vecs=2)
        out = []
        for r in range(2):
            xr = x[r : r + 1]
            xg = xr if self.router_dtype is None else xr.astype(self.router_dtype)
            inds, scores = _mf.topk_softmax(self.gate(xg).reshape(-1), self.top_k, self.norm_topk)
            out.append(
                (inds, scores, sh[r : r + 1].reshape(xr.shape), self.shared_gate(xr).reshape(-1))
            )
        return out

    def router_pair(self, x: mx.array) -> tuple:
        """`router_and_shared2` on flat rows, x [2, hid] -> (inds [2k], scores [2k], sh [2, hid],
        shg0, shg1): the same launches, except that one `topk_softmax2` takes both rows' router
        logits (each row `topk_softmax`'s bits). Each row's gate and shared gate stay their own
        one-row matvec, as in `router_and_shared`; shg_r is that row's [1, 1] logit."""
        gu = _gf.qmv_multi_v(
            x, [self.shared_gate_proj, self.shared_up_proj], GDN_FUSED_MULTI, vecs=2
        )
        sh = _gf.qmv_multi_v(gu, [self.shared_down_proj], GDN_FUSED_MULTI, swiglu=True, vecs=2)
        logits, shg = [], []
        for r in range(2):
            xr = x[r : r + 1]
            xg = xr if self.router_dtype is None else xr.astype(self.router_dtype)
            logits.append(self.gate(xg))
            shg.append(self.shared_gate(xr))
        inds, scores = _mf.topk_softmax2(logits[0], logits[1], self.top_k, self.norm_topk)
        return inds, scores, sh, shg[0], shg[1]

    def router_and_shared(self, x: mx.array):
        """Single-row decode (inds, scores, sh, shg)."""
        xr = x if self.router_dtype is None else x.astype(self.router_dtype)
        inds, scores = _mf.topk_softmax(self.gate(xr).reshape(-1), self.top_k, self.norm_topk)
        if self.shared_fused_ok:
            sh, shg = self.shared_fused(x)
        else:
            sh, shg = self.shared(x), None
        return inds, scores, sh, shg
