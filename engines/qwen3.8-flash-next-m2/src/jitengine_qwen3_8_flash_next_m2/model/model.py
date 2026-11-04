"""Qwen3.8-Flash-Next (`qwen4_exp`) forward pass.

Gated DeltaNet and sparse-attention blocks, each followed by a 512-expert MoE, wired through
hyper-connections over a [B, S, hc * hidden] residual, with a hashed n-gram embedding (PLE).

Every weight-derived array is evaluated in the constructor that creates it: MLX binds a lazy node
to the stream of the thread that built it, and the model is built on one thread and run on another.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.kernels import gdn_decode as _gf
from jitengine_qwen3_8_flash_next_m2.kernels import grouped_expert_qmv as _mk
from jitengine_qwen3_8_flash_next_m2.kernels import qmv_row_blocked as _dk
from jitengine_qwen3_8_flash_next_m2.kernels import router_combine as _mf
from jitengine_qwen3_8_flash_next_m2.kvcache.linear_state import LinearStateSlot
from jitengine_qwen3_8_flash_next_m2.kvcache.state import RequestState
from jitengine_qwen3_8_flash_next_m2.linear import (
    QEmbedding,
    apply_dense_policy,
    calibrate_row_limits,
    qlinear_from,
)
from jitengine_qwen3_8_flash_next_m2.model.attention import Qwen4ExpTextAttention
from jitengine_qwen3_8_flash_next_m2.model.common import (
    _DEFER,
    _IN_SPEC_BUILD,
    GDN_FUSED_MULTI,
    _dispatch,
)
from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig
from jitengine_qwen3_8_flash_next_m2.model.gdn import (
    GDN_FUSED_NSG,
    Qwen4ExpTextGatedDeltaNet,
    _recurrence,
)
from jitengine_qwen3_8_flash_next_m2.model.hyper_connections import (
    HC_BASIS,
    HC_EXCLUDED,
    HC_PATH_DEFAULT,
    HC_REGIONS,
    Qwen4ExpTextGatedResidual,
    hyper_inject,
)
from jitengine_qwen3_8_flash_next_m2.model.moe import MOE_KERNEL_DEFAULT, Qwen4ExpTextSparseMoeBlock
from jitengine_qwen3_8_flash_next_m2.model.ple import NGramIndex, Qwen4ExpTextPLELayer

logger = logging.getLogger(__name__)

# When a wave's activations are evaluated ("none", "layer" or "stage"); never what is computed.
WAVE_EVAL = "none"


VSTATS = {
    "n": 0,
    "prebuilt": 0,
    "build": 0.0,
    "settle0": 0.0,
    "settle1": 0.0,
    "rebuild2": 0,
    "rebuild1": 0,
    "rebuild_t": 0.0,
}


# ------------------------------------------------------------ fused hyper-connection elementwise
# Compiled hyper-connection regions. Under `shapeless=True` a traced `.shape` is baked in and
# silently reused at later shapes, so no region reads `.shape`; callers do every reshape.
# Two links stay stock because compiling them is not bit-identical: sigmoid fused into the
# `w * normed` multiply (it skips rounding the sigmoid to storage precision), and the inject gate
# `2 * sigmoid(y / hc)` (differs at rank 3, which can flip a router tie).


class Qwen4ExpTextDecoderLayer:
    """`Qwen4ExpTextDecoderLayer`: [PLE], hc -> block -> inject, hc -> MoE -> inject."""

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        i: int,
        pool=None,
        sync_each_layer: bool = False,
        wave_eval: str = WAVE_EVAL,
        compute_dtype=mx.float32,
        ple_store=None,
        ngram=None,
        router_dtype=None,
        moe_kernel: str = MOE_KERNEL_DEFAULT,
        moe_kernel_variant: str | None = None,
        hc_path: str = HC_PATH_DEFAULT,
        kernels_off: frozenset = frozenset(),
    ):
        p = f"model.layers.{i}."
        self.idx = int(i)
        self.hc_path = str(hc_path)
        self.dtype = compute_dtype
        self.wave_eval = str(wave_eval)
        self.is_attention = cfg.is_attention(i)
        self.block = (
            Qwen4ExpTextAttention(cfg, w, p + "self_attn.", compute_dtype, kernels_off)
            if self.is_attention
            else Qwen4ExpTextGatedDeltaNet(cfg, w, p + "linear_attn.", compute_dtype, kernels_off)
        )
        self.moe = Qwen4ExpTextSparseMoeBlock(
            cfg,
            w,
            p + "mlp.",
            pool=pool,
            layer_idx=i,
            sync_each_layer=sync_each_layer,
            router_dtype=router_dtype,
            moe_kernel=moe_kernel,
            moe_kernel_variant=moe_kernel_variant,
            kernels_off=kernels_off,
        )
        self.hc_attn = Qwen4ExpTextGatedResidual(
            cfg,
            w,
            p + "attn_hyper_connection.",
            compute_dtype=compute_dtype,
            hc_path=hc_path,
            kernels_off=kernels_off,
        )
        self.hc_mlp = Qwen4ExpTextGatedResidual(
            cfg,
            w,
            p + "mlp_hyper_connection.",
            compute_dtype=compute_dtype,
            hc_path=hc_path,
            kernels_off=kernels_off,
        )
        self.ple = None
        if cfg.has_ple(i):
            if ple_store is None or ngram is None:
                raise ValueError(
                    f"layer {i} carries the PLE but no PLE store / n-gram index was given"
                )
            self.ple = Qwen4ExpTextPLELayer(
                cfg, w, p + "ple.", ple_store, ngram, compute_dtype, hc_path=hc_path
            )

    def linears(self) -> list:
        out = (
            self.block.linears()
            + self.moe.linears()
            + self.hc_attn.linears()
            + self.hc_mlp.linears()
        )
        if self.ple is not None:
            out += [self.ple.key_proj, self.ple.value_proj]
        return out

    def spec_eligible(self, dtype) -> bool:
        """Whether this layer can be built speculatively (all fused paths available)."""
        return (
            self.hc_attn.fused_ok
            and self.hc_mlp.fused_ok
            and self.moe.fused_ok
            and self.moe.spec_ok
            and self.moe.grouped_for(dtype)
        )

    def _compiled_ok(self) -> bool:
        """Whether the speculative build runs as two compiled functions, bit-identical to the eager
        body; attention and PLE layers stay eager (per-token KV shapes, host-side n-gram reads)."""
        return (
            self.ple is None
            and not self.is_attention
            and getattr(self.block, "fused_ok", False)
            and self.moe.shared_fused_ok
        )

    def _router_half_body(self, h, conv, ssm, rstd=None):
        """Up to the router: (inds, scores, sh, shg, h2, inj2, x2, new_conv, new_ssm)."""
        mixed, inj, _ = self.hc_attn.fused(h, rstd=rstd)
        s = LinearStateSlot.__new__(LinearStateSlot)
        s.conv, s.ssm = conv, ssm
        out = self.block.fused_decode(mixed, s)
        mixed, inj2, h2 = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        moe = self.moe
        inds, scores, sh, shg = moe.router_and_shared(mixed)
        return inds, scores, sh, shg, h2, inj2, mixed.reshape(-1, mixed.shape[-1]), s.conv, s.ssm

    def _expert_half_body(self, x2, inds, scores, sh, shg, h2, inj2, tab):
        """The speculative gather + combine: (y_spec, h3)."""
        moe = self.moe
        k = moe.top_k
        _, _, _, ar_np, _ = moe._spec_consts(k)
        slots_dev = mx.take(tab, inds + int(moe.layer_idx * moe.pool.num_experts))
        y_spec = moe._gather_dev(x2, slots_dev, k)
        sc = mx.where(slots_dev >= 0, scores, mx.zeros_like(scores))
        h3 = _mf.moe_combine(y_spec, None, ar_np, sc, sh, h2, inj2, moe.hc, moe.hid, sh_gate=shg)
        return y_spec, h3.reshape(h2.shape)

    def drop_compiled(self) -> None:
        """Forget compiled builds; they hold the pool's arrays (extension included) as constants."""
        for a in ("_router_half", "_expert_half", "_comp_key"):
            if hasattr(self, a):
                delattr(self, a)

    def _compiled_fns(self):
        pool = self.moe.pool
        key = (bool(pool.ext_active), id(getattr(pool, "gate_proj_ext", None)))
        if getattr(self, "_comp_key", None) != key:
            self._router_half = mx.compile(self._router_half_body)
            self._expert_half = mx.compile(self._expert_half_body)
            self._comp_key = key
        return self._router_half, self._expert_half

    def build_spec2(
        self, h0: mx.array, h1: mx.array, slot, offset: int, ta, da, ple_state, on_row0, telem=None
    ):
        """Build positions offset (h0, token ta) and offset + 1 (h1, da) of one sequence.

        Launches whose rows are computed exactly as the single row is are shared: both
        hyper-connections, the input projections and shared expert (`qmv_multi_v`), out_proj /
        o_proj, and the routed experts (`spec_build_pair`). The DeltaNet recurrence and attention run
        per position in order (position 1 reads position 0's new state), and each position gets an
        ordinary one-row MoE record, so settles are the solo ones. `on_row0()` is called once
        position 0's state is in the slot (the rollback point). Returns (rec0, rec1).
        """
        attn2 = (
            self.is_attention
            and self.ple is None
            and self.block.decode2_ok()
            and self.moe.shared_fused_ok
            and self.hc_attn.fused_ok
            and self.hc_mlp.fused_ok
            and self.moe.fused_ok
            and self.moe.spec_ok
            and self.moe.grouped_for(h0.dtype)
        )
        # the PLE layer: its per-token PLE runs per position first (each advances the PLE state), then
        # the layer is the shared GDN path; the rollback point keeps the PLE state after position 0
        gdn_ple = (
            self.ple is not None
            and not self.is_attention
            and getattr(self.block, "fused_ok", False)
            and self.moe.shared_fused_ok
            and self.hc_attn.fused_ok
            and self.hc_mlp.fused_ok
            and self.moe.fused_ok
            and self.moe.spec_ok
            and self.moe.grouped_for(h0.dtype)
        )
        if gdn_ple:
            h0 = h0 + self.ple(h0, ta, ple_state, None)
            pl0 = (ple_state.context, ple_state.conv, ple_state.pending)
            h1 = h1 + self.ple(h1, da, ple_state, None)
            outer = on_row0

            def on_row0():
                outer(pl0)
        elif not (self._compiled_ok() or attn2):
            r0 = self.build_spec(h0, slot, offset, tokens=ta, ple_state=ple_state, telem=telem)
            on_row0()
            r1 = self.build_spec(h1, slot, offset + 1, tokens=da, ple_state=ple_state, telem=telem)
            return r0, r1
        qmv2 = lambda x, lins, sw=False: _gf.qmv_multi_v(  # noqa: E731
            x, lins, GDN_FUSED_MULTI, swiglu=sw, vecs=2
        )
        h = mx.concatenate([h0, h1], axis=0)  # [2, 1, hc*hid]
        mixed, inj, _ = self.hc_attn.fused(h)
        if self.is_attention:
            out = self.block.decode2(mixed, slot, offset, on_row0)
        else:
            out = self._gdn2(mixed, slot, qmv2, on_row0)
        mixed, inj2, h2 = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        recs = self._moe2(mixed, h2, inj2, qmv2)
        for r in recs:
            r["telem"] = telem
        return recs

    def _gdn2(self, mixed, slot, qmv2, on_row0):
        blk = self.block
        ycat = qmv2(mixed.reshape(2, -1), blk._proj_lins)
        acts = []
        for r in range(2):
            q, k, v, g, beta, new_conv = _gf.gdn_pre(
                slot.conv[0],
                ycat[r],
                blk.conv_w2d,
                blk.A_log,
                blk.dt_bias,
                blk.cfg,
                blk.l2_eps,
                blk.q_scale,
                blk.k_scale,
                blk._proj_offs,
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
            )
            acts.append(
                _gf.gdn_post(
                    y[0, 0], ycat[r], blk.norm_w, blk.cfg, blk.eps, blk._proj_offs[1], GDN_FUSED_NSG
                )
            )
            if r == 0:
                on_row0()
        return blk.out_proj(mx.stack(acts, axis=0).reshape(2, 1, -1))

    def _moe2(self, mixed, h2, inj2, qmv2):
        moe = self.moe
        x2 = mixed.reshape(2, -1)
        gu = qmv2(x2, [moe.shared_gate_proj, moe.shared_up_proj])
        sh = qmv2(gu, [moe.shared_down_proj], True)
        rt = []
        for r in range(2):
            xr = mixed[r : r + 1]
            xg = xr if moe.router_dtype is None else xr.astype(moe.router_dtype)
            inds, scores = _mf.topk_softmax(moe.gate(xg).reshape(-1), moe.top_k, moe.norm_topk)
            shg = moe.shared_gate(xr).reshape(-1)
            rt.append(
                (
                    xr,
                    inds,
                    scores,
                    sh[r : r + 1].reshape(xr.shape),
                    shg,
                    h2[r : r + 1],
                    inj2[r : r + 1],
                )
            )
        recs = moe.spec_build_pair(*[[rt[0][j], rt[1][j]] for j in range(7)])
        return recs[0], recs[1]

    def build_spec(
        self,
        h: mx.array,
        slot,
        offset: int,
        tokens=None,
        ple_state=None,
        telem=None,
        ple_done: bool = False,
    ) -> dict:
        """Build and dispatch this layer on a possibly speculative input, without a host sync.

        Returns the MoE record for `spec_settle`, with the speculative output at rec["h3"].
        """
        if self.ple is not None and not ple_done:
            h = h + self.ple(h, tokens, ple_state, None)
        if self._compiled_ok():
            router_half, expert_half = self._compiled_fns()
            inds, scores, sh, shg, h2, inj2, x2, slot.conv, slot.ssm = router_half(
                h, slot.conv, slot.ssm
            )
            _dispatch(inds, scores)  # the router's own command buffer
            pool = self.moe.pool
            if pool._inflight and _DEFER[0] is None:
                pool.harvest_ready()
            tab = pool.device_slot_table()
            snap = pool.layer_snapshot(self.idx)
            y_spec, h3 = expert_half(x2, inds, scores, sh, shg, h2, inj2, tab)
            _dispatch(h3)
            return {
                "x2": x2,
                "inds": inds,
                "scores": scores,
                "sh": sh,
                "shg": shg,
                "h2": h2,
                "inject": inj2,
                "y_spec": y_spec,
                "h3": h3,
                "telem": telem,
                "snap": snap,
            }
        mixed, inj, _ = self.hc_attn.fused(h)
        _IN_SPEC_BUILD[0] = True
        try:
            out = self.block(mixed, slot, offset)
        finally:
            _IN_SPEC_BUILD[0] = False
        mixed, inj2, h = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        rec = self.moe.fused_decode(mixed, h, inj2, telem, build_only=True)
        rec["h3"] = rec["h3"].reshape(h.shape)
        return rec

    def __call__(
        self,
        h: mx.array,
        slot,
        offset: int,
        tokens: mx.array | None = None,
        ple_state=None,
        trace: dict | None = None,
        telem=None,
    ) -> mx.array:
        """One row: h [1, S, hc*hid]; the PLE layer also needs `tokens` [1, S] and `ple_state`.

        `telem` only observes expert ids already on the host; the engine passes it on solo decode.
        """
        if self.ple is not None:
            ple_out = self.ple(h, tokens, ple_state, trace)
            if trace is not None:
                trace["ple_out"] = ple_out
            h = h + ple_out
        if trace is not None:
            trace["h_after_ple"] = h
        if (
            trace is None
            and h.shape[0] == 1
            and h.shape[1] == 1
            and self.hc_attn.fused_ok
            and self.hc_mlp.fused_ok
        ):
            # fused hyper-connections: hc_mlp folds in the attention inject and returns the
            # injected residual
            spec = self.moe.fused_ok and self.moe.spec_ok
            if spec and self.moe.grouped_for(h.dtype):
                # the PLE was applied above; `ple_done` keeps build_spec from applying it twice
                rec = self.build_spec(
                    h, slot, offset, tokens=tokens, ple_state=ple_state, telem=telem, ple_done=True
                )
                return self.moe.spec_settle(rec)[0]
            h = self._decode_fused(h, slot, offset, spec, telem)
            return h
        h = self._forward_unfused(h, slot, offset, trace, telem)
        return h

    def _decode_fused(self, h: mx.array, slot, offset: int, spec: bool, telem) -> mx.array:
        """The fused single-position path: block, MLP hyper-connection and MoE, committed asynchronously."""
        mixed, inj, _ = self.hc_attn.fused(h)
        out = self.block(mixed, slot, offset)
        if not spec:
            # the speculative path commits the block with the router's buffer itself
            mx.async_eval(out)
        mixed, inj2, h = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        if self.moe.fused_ok:
            h = self.moe.fused_decode(mixed, h, inj2, telem)
        else:
            out = self.moe(mixed, telem)
            h = hyper_inject(out, h, inj2, path=self.hc_path)
        if not spec:
            mx.async_eval(h)
        return h

    def _forward_unfused(
        self, h: mx.array, slot, offset: int, trace: dict | None, telem
    ) -> mx.array:
        """The general path: block, MLP and MoE, each followed by its inject, with the trace recorded."""
        mixed, inj = self.hc_attn(h)
        out = self.block(mixed, slot, offset, trace=trace)
        if trace is None:
            # submit the block now so the GPU runs it while the host builds the rest of the layer
            mx.async_eval(out)
        if trace is not None:
            trace["mixed_attn"], trace["inject_attn"], trace["block_out"] = mixed, inj, out
        h = hyper_inject(out, h, inj, path=self.hc_path)
        mixed, inj = self.hc_mlp(h)
        out = self.moe(mixed, telem, trace=trace)
        if trace is not None:
            (
                trace["stream_after_attn"],
                trace["mixed_mlp"],
                trace["inject_mlp"],
                trace["moe_out"],
            ) = h, mixed, inj, out
        h = hyper_inject(out, h, inj, path=self.hc_path)
        if trace is None:
            mx.async_eval(h)
        if trace is not None:
            trace["final"] = h
        return h

    def step(
        self,
        h: mx.array,
        slots: list,
        offsets: list,
        telem=None,
        tokens: mx.array | None = None,
        ple_states: list | None = None,
    ) -> mx.array:
        """One decode position for B rows (h [B, 1, hc*hid]); `slots[i]` / `offsets[i]` belong to row i."""
        if self.ple is not None:
            h = h + self.ple.step(h, tokens, ple_states)
        mixed, inj = self.hc_attn(h, rows=True)
        blk = self.block.step(mixed, slots, offsets)
        mx.async_eval(blk)
        h = hyper_inject(blk, h, inj, path=self.hc_path)
        mixed, inj = self.hc_mlp(h, rows=True)
        return hyper_inject(self.moe(mixed, telem, rows=True), h, inj, path=self.hc_path)

    def wave(
        self,
        hs: list,
        slots: list,
        offsets: list,
        tokens: list | None = None,
        ple_states: list | None = None,
    ) -> list:
        """One layer of a wave: N independent prefill rows sharing one expert fetch.

        The order (blocks; routers and host syncs; one `segments_multi`; gathers) is the pool's
        use-after-evict invariant and must not be rearranged.
        """
        n = len(hs)
        injs = []
        for r in range(n):
            if self.ple is not None:
                hs[r] = hs[r] + self.ple(hs[r], tokens[r], ple_states[r], row=r)
            mixed, inj = self.hc_attn(hs[r])
            hs[r] = hyper_inject(
                self.block(mixed, slots[r], offsets[r]), hs[r], inj, path=self.hc_path
            )
        if self.wave_eval == "stage":
            mx.eval(*hs)
        mixes, inds_l, scores_l, flats = [], [], [], []
        for r in range(n):
            mixed, inj = self.hc_mlp(hs[r])
            inds, scores = self.moe.route(mixed)
            mixes.append(mixed)
            injs.append(inj)
            inds_l.append(inds)
            scores_l.append(scores)
            flats.append(np.asarray(inds).astype(np.int64, copy=False).reshape(-1))  # the sync
        self.moe.pool.reader_barrier()  # the syncs above drained the previous layer
        # shared experts are built before the fetch and dispatched by the pool while it is in flight
        shs = [self.moe.shared(mixes[r]) for r in range(n)]
        per_row = self.moe.pool.segments_multi(self.moe.layer_idx, flats, overlap=tuple(shs))
        for r in range(n):
            hs[r] = hyper_inject(
                self.moe.apply(mixes[r], inds_l[r], scores_l[r], per_row[r], shared=shs[r]),
                hs[r],
                injs[r],
                path=self.hc_path,
            )
            if self.wave_eval == "stage":
                mixes[r] = inds_l[r] = scores_l[r] = None
                mx.eval(hs[r])
        if self.wave_eval == "layer" or self.moe.sync_each_layer:
            mx.eval(*hs)
        # no barrier here: the gathers above stay pending readers of this layer's slots until the
        # next layer's router sync, where the barrier is
        return hs


class Qwen4ExpForCausalLM:
    """The whole language model over a weights dict of prefix-stripped names (see
    checkpoint/loader.py).

    `kernels_off` names the custom kernels whose load-time checks failed (`kernel_checks.NAMES`);
    the paths that would launch them run the stock ops instead.
    """

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        pool=None,
        sync_each_layer: bool = False,
        wave_eval: str = WAVE_EVAL,
        ple_store=None,
        compute_dtype=mx.float32,
        calibrate: bool = True,
        residual_dtype=None,
        router_dtype=None,
        moe_kernel: str = MOE_KERNEL_DEFAULT,
        moe_kernel_variant: str | None = None,
        hc_path: str = HC_PATH_DEFAULT,
        dense_kernel: str = _dk.DENSE_KERNEL_DEFAULT,
        embed=None,
        kernels_off: frozenset = frozenset(),
    ):
        self.cfg = cfg
        self.hc_path = str(hc_path)
        self.pool = pool
        self.dtype = compute_dtype
        # dtype of the carried residual `h`; matmul inputs stay in the compute dtype, so a wider
        # residual only changes the precision of the residual adds
        self.residual_dtype = residual_dtype or compute_dtype
        self.router_dtype = router_dtype
        self.wave_eval = str(wave_eval)
        self.ple_store = ple_store
        # router-lookahead prefetch distance in layers (0 = off); see `_prefetch_ahead`
        self.prefetch_lookahead = 0
        self.pool = pool
        # `embed`, when given, is the streamed table (`StreamedEmbedding`, bit-identical lookup); only
        # an untied checkpoint streams it (asserted below).
        if embed is not None:
            self.embed = embed
        else:
            bits, gs = cfg.quant_of("model.embed_tokens")
            self.embed = QEmbedding(
                w["model.embed_tokens.weight"],
                w["model.embed_tokens.scales"],
                w.get("model.embed_tokens.biases"),
                gs,
                bits,
            )
        # the n-gram index: the checkpoint's own I64 buffers, asserted equal to the recomputation
        self.ple_layers = [i for i in range(cfg.num_hidden_layers) if cfg.has_ple(i)]
        self.ngram = None
        if self.ple_layers:
            self.ngram = NGramIndex.from_config(
                cfg, w, layer=self.ple_layers[0], ple_layer_index=0, check=True
            )
        self.layers = [
            Qwen4ExpTextDecoderLayer(
                cfg,
                w,
                i,
                pool=pool,
                sync_each_layer=sync_each_layer,
                wave_eval=wave_eval,
                compute_dtype=compute_dtype,
                ple_store=ple_store,
                ngram=self.ngram,
                router_dtype=router_dtype,
                moe_kernel=moe_kernel,
                moe_kernel_variant=moe_kernel_variant,
                hc_path=hc_path,
                kernels_off=kernels_off,
            )
            for i in range(cfg.num_hidden_layers)
        ]
        self.mixer = Qwen4ExpTextGatedResidual(
            cfg,
            w,
            "model.hyper_connection_mixer.",
            with_inject=False,
            compute_dtype=compute_dtype,
            hc_path=hc_path,
            kernels_off=kernels_off,
        )
        if "lm_head.weight" in w:
            self.lm_head = qlinear_from(w, "lm_head", cfg)
        else:  # tied embeddings
            if embed is not None:
                raise ValueError(
                    "a tied checkpoint's embedding is its lm_head and cannot be streamed"
                )
            self.lm_head = self.embed
        # before calibration, which depends on it: each dense linear runs the row-blocked kernel
        # or stock over float32-stored metadata
        self.dense_kernel = str(dense_kernel)
        self.dense_policy = apply_dense_policy(
            self.dense_linears(), compute_dtype, self.dense_kernel
        )
        # the lookahead decode needs every layer speculative; decided after the dense policy,
        # which the fused kernels' `supports` checks read
        self.spec_all = all(layer.spec_eligible(compute_dtype) for layer in self.layers)
        logger.info(
            f"[engine] fused decode: hc {sum(layer.hc_attn.fused_ok and layer.hc_mlp.fused_ok for layer in self.layers)}/{len(self.layers)}, "
            f"gdn {sum(getattr(layer.block, 'fused_ok', False) for layer in self.layers)}, "
            f"moe {sum(layer.moe.fused_ok for layer in self.layers)}, spec {sum(layer.moe.spec_ok for layer in self.layers)}, "
            f"lookahead {'ON' if self.spec_all else 'off'}"
        )
        if calibrate:
            self.calibrate()

    def drop_compiled(self) -> None:
        """See Qwen4ExpTextDecoderLayer.drop_compiled; called by the engine around extension
        release/regrow."""
        for layer in self.layers:
            layer.drop_compiled()

    def dense_linears(self) -> list:
        """Every QLinear a decode step drives with a stack of independent rows."""
        out = [self.lm_head] + self.mixer.linears()
        for layer in self.layers:
            out += layer.linears()
        return out

    def calibrate(self) -> dict:
        """Row-exactness limits at the model dtype (limits depend on dtype as well as shape)."""
        return calibrate_row_limits(self.dense_linears(), dtype=self.dtype)

    def row_limits(self) -> dict:
        """{"<out>x<in>/<bits>b": group} for the readout's linears, published in stats()."""
        out = {}
        for lin in [self.lm_head] + self.mixer.linears():
            out[f"{lin.out_dims}x{lin.in_dims}/{lin.bits}b"] = int(lin.row_limit)
        return out

    def dense_kernel_info(self) -> dict:
        """`stats()["dense_kernel"]`: the option, config table, and served and fallback shapes."""
        return dict(self.dense_policy)

    def hc_info(self) -> dict:
        """`stats()["hc_path"]`: which hyper-connection path ran, and what it does not fuse."""
        return {
            "option": self.hc_path,
            "regions": list(HC_REGIONS) if self.hc_path == "compiled" else [],
            "not_fused": HC_EXCLUDED,
            "basis": HC_BASIS,
            "sites": 2 * len(self.layers) + 1,
        }

    def moe_kernel_info(self) -> dict:
        """`stats()["moe_kernel"]`: option, kernel actually run per layer, variant and configs."""
        per = [layer.moe.kernel_in_use(self.dtype) for layer in self.layers]
        first = self.layers[0].moe if self.layers else None
        configs: dict = {}
        for d in per:
            configs.update(d["configs"])
        kernels = sorted({d["kernel"] for d in per})
        return {
            "option": first.moe_kernel if first else MOE_KERNEL_DEFAULT,
            "kernel": kernels[0] if len(kernels) == 1 else "mixed",
            "variant": first.moe_kernel_variant if (first and "grouped" in kernels) else None,
            "variant_basis": (
                _mk.VARIANT_BASIS
                if (first and first.moe_kernel_variant == _mk.DEFAULT_VARIANT)
                else "selected by option moe_kernel_variant"
            ),
            "configs": configs,
            "config_basis": _mk.CONFIG_BASIS,
            "fallback_layers": [i for i, d in enumerate(per) if d["fallback"]],
            "compute_dtype": _mk.dtype_name(self.dtype),
            "source": _mk.__name__,
        }

    def _embed_streams(self, tokens: mx.array) -> mx.array:
        """[.., S] ids -> [.., S, hc*hidden]: the embedding tiled over the hc streams."""
        return mx.tile(
            self.embed.lookup(tokens).astype(self.residual_dtype), (1, 1, self.cfg.hc_count)
        )

    def head(self, h: mx.array, rows: bool = False) -> mx.array:
        """The terminal hyper-connection mixer straight into lm_head (no final norm)."""
        # the mixer stays stock on both paths: score() reads through `head(rows=True)` and
        # generate through `head()`, and the two must stay bit-identical
        mixed, _ = self.mixer(h, rows=rows)
        return self.lm_head.rows(mixed) if rows else self.lm_head(mixed)

    @staticmethod
    def _snap(slot, ple_state):
        """State a speculative layer build may change, so a rebuild after a miss starts clean.

        KV appends in place, so only `offset` and the block-key cache are saved (holding the buffer
        would force a copy); DeltaNet and PLE arrays are replaced, so their references suffice.
        """
        if hasattr(slot, "keys"):
            sn = ("kv", slot.offset, dict(slot.idx_bk))
        else:
            sn = ("delta", slot.conv, slot.ssm)
        pl = None if ple_state is None else (ple_state.context, ple_state.conv)
        return sn, pl

    @staticmethod
    def _restore(slot, ple_state, snap):
        sn, pl = snap
        if sn[0] == "kv":
            slot.offset, slot.idx_bk = sn[1], sn[2]
        else:
            slot.conv, slot.ssm = sn[1], sn[2]
        if pl is not None:
            ple_state.context, ple_state.conv = pl

    def forward_hidden(self, tokens: mx.array, state: RequestState, telem=None):
        """The solo decode step (`_forward_lookahead`, its logits bit for bit), also returning the
        stack output before the terminal mixer, [1, hc*hid], for the MTP head."""
        return self._forward_lookahead(tokens, state, telem, want_hidden=True)

    def _forward_lookahead(
        self, tokens: mx.array, state: RequestState, telem=None, want_hidden: bool = False
    ):
        """One decode position through every layer with a one-layer lookahead.

        Layer L+1 is built and dispatched before syncing on layer L's router. On a miss, layer L is
        corrected and L+1 rebuilt from it; the discarded speculative kernels finish harmlessly.
        """
        offset = state.tokens
        h = self._embed_streams(tokens)
        n = len(self.layers)

        def build(i, hin):
            layer, slot, pst = self.layers[i], state.layers[i], state.ple.get(i)
            snap = self._snap(slot, pst)
            rec = layer.build_spec(
                hin,
                slot,
                offset,
                tokens=tokens,
                ple_state=pst,
                telem=telem,
            )
            return rec, snap

        cur, _ = build(0, h)
        for i in range(n):
            nxt, moe, ctx = self._begin_settle(i, n, cur, build)
            if ctx["missed"] and nxt is not None:
                self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt[1])
                held: list = []
                _DEFER[0] = held
                try:
                    nb = build(i + 1, ctx["h3"])
                finally:
                    _DEFER[0] = None
                try:
                    h_out, missed = moe.spec_settle_finish(ctx)
                except BaseException:
                    self._restore(state.layers[i + 1], state.ple.get(i + 1), nb[1])
                    raise
                if not ctx.get("pred_ok", True):
                    # a record did not land where begin said: what was built on it is discarded
                    self._restore(state.layers[i + 1], state.ple.get(i + 1), nb[1])
                    nb = build(i + 1, h_out)
                    cur = nb[0]
                    continue
                self._dispatch_held(held)
                cur = nb[0]
                continue
            h_out, missed = moe.spec_settle_finish(ctx)
            nxt = self._rebuild_lookahead_on_miss(state, i, nxt, build, h_out, missed)
            cur = nxt[0] if nxt is not None else None
        state.tokens += 1
        if want_hidden:
            return self.head(h_out[:, -1, :]), h_out[:, -1, :]
        return self.head(h_out[:, -1, :])

    def _rebuild_lookahead_on_miss(
        self, state: RequestState, i: int, nxt: tuple | None, build, h_out: mx.array, missed: bool
    ) -> tuple | None:
        """If layer i missed, undo the lookahead layer's state and rebuild it on the new residual."""
        if missed and nxt is not None:
            # the lookahead layer was built on a wrong residual: undo its state, rebuild
            self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt[1])
            nxt = build(i + 1, h_out)
        return nxt

    def _begin_settle(self, i: int, n: int, cur: dict, build) -> tuple:
        """Build the lookahead layer, then begin layer i's router settle chained to it."""
        nxt = build(i + 1, cur["h3"]) if i + 1 < n else None
        moe = self.layers[i].moe
        ctx = moe.spec_settle_begin(cur, chain=nxt[0] if nxt is not None else None)
        return nxt, moe, ctx

    def _dispatch_held(self, held: list) -> None:
        """Dispatch the kernels held back while the lookahead layer was rebuilt."""
        for arrs in held:
            mx.async_eval(*arrs)

    def verify2(self, t: int, d: int, state: RequestState, telem=None):
        """Run positions p = state.tokens (token t) and p + 1 (draft d) of one sequence in one pass.

        Per layer, position p is built on the layer's state, then p + 1 on the state p produced, then
        both settle. Each position runs the solo decode step's arithmetic, so its logits are
        bit-identical to a solo step's. Returns (logits [2, V], hidden [2, hc*hid] before the terminal
        mixer, snaps), where `snaps` restores every layer to "p consumed, p + 1 not" (`rollback1`);
        `state.tokens` += 2.
        """
        offset = state.tokens
        ta = mx.array([[int(t)]], dtype=mx.int32)
        da = mx.array([[int(d)]], dtype=mx.int32)
        h0, h1 = self._embed_streams(ta), self._embed_streams(da)
        return self._verify2_lookahead(h0, h1, ta, da, offset, state, telem)

    def _verify2_lookahead(self, h0, h1, ta, da, offset: int, state: RequestState, telem=None):  # noqa: C901, PLR0915
        """`verify2` with the solo step's one-layer lookahead over both positions.

        Layer i+1 of both is built on layer i's speculative outputs before layer i settles. A settle
        that corrected position p invalidates i+1 for both (p+1's build read p's layer-(i+1) state);
        one that corrected only p+1 invalidates p+1's alone.
        """
        n = len(self.layers)
        snaps = [None] * n

        def build2(i, a, b, which=(0, 1)):
            layer, slot, pst = self.layers[i], state.layers[i], state.ple.get(i)
            out = {}
            if which == (0, 1):
                out["pre"] = self._snap(slot, pst)

                def on_row0(pl=None):
                    sn, pl_now = self._snap(slot, pst)
                    snaps[i] = (sn, pl if pl is not None else pl_now)

                out[0], out[1] = layer.build_spec2(a, b, slot, offset, ta, da, pst, on_row0, telem)
                return out
            held = {0: [], 1: []}
            joint = _DEFER[0] is None
            if 0 in which:
                pre = self._snap(slot, pst)
                if joint:
                    _DEFER[0] = held[0]
                try:
                    out[0] = layer.build_spec(a, slot, offset, tokens=ta, ple_state=pst)
                finally:
                    if joint:
                        _DEFER[0] = None
                out["pre"] = pre
                snaps[i] = self._snap(slot, pst)
            if 1 in which:
                if joint:
                    _DEFER[0] = held[1]
                try:
                    out[1] = layer.build_spec(b, slot, offset + 1, tokens=da, ple_state=pst)
                finally:
                    if joint:
                        _DEFER[0] = None
            if joint:
                arrs = []
                for j in range(max(len(held[0]), len(held[1]))):
                    for r in (0, 1):
                        if j < len(held[r]):
                            arrs.extend(held[r][j])
                if arrs:
                    mx.async_eval(*arrs)
            return out

        _pc = time.perf_counter
        V = VSTATS
        t_ = _pc()
        cur = build2(0, h0, h1)
        V["build"] += _pc() - t_
        for i in range(n):
            t_ = _pc()
            nxt = build2(i + 1, cur[0]["h3"], cur[1]["h3"]) if i + 1 < n else None
            t1 = _pc()
            moe = self.layers[i].moe
            ch = nxt is not None
            # both positions' reads in flight together (one begin over both rows' pairs)
            pc = moe.spec_settle_pair_begin([cur[0], cur[1]], [nxt[0], nxt[1]] if ch else None)
            pm0, pm1 = pc["outs"][0][1], pc["outs"][1][1]
            if nxt is not None and (pm0 or pm1):
                # layer i+1 on the corrections, built while the reads are in flight, held
                self._restore(
                    state.layers[i + 1],
                    state.ple.get(i + 1),
                    nxt["pre"] if pm0 else snaps[i + 1],
                )
                held: list = []
                _DEFER[0] = held
                try:
                    if pm0:
                        nb = build2(i + 1, pc["outs"][0][0], pc["outs"][1][0])
                    else:
                        nb = dict(nxt)
                        nb[1] = build2(i + 1, None, pc["outs"][1][0], which=(1,))[1]
                finally:
                    _DEFER[0] = None
                (h0, m0), (h1, m1) = moe.spec_settle_pair_finish(pc)
                V["settle0"] += _pc() - t1
                V["build"] += t1 - t_
                if pc.get("pred_ok", True):
                    for arrs in held:
                        mx.async_eval(*arrs)
                    V["prebuilt"] += 1
                    cur = nb
                    continue
                # a record moved after all: discard the prebuild and rebuild from the final outputs
                self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt["pre"])
                cur = build2(i + 1, h0, h1)
                continue
            (h0, m0), (h1, m1) = moe.spec_settle_pair_finish(pc)
            t2 = _pc()
            t3 = _pc()
            V["build"] += t1 - t_
            V["settle0"] += t2 - t1
            V["settle1"] += t3 - t2
            if nxt is not None and m0:
                self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt["pre"])
                nxt = build2(i + 1, h0, h1)
                V["rebuild2"] += 1
            elif nxt is not None and m1:
                self._restore(state.layers[i + 1], state.ple.get(i + 1), snaps[i + 1])
                nxt[1] = build2(i + 1, None, h1, which=(1,))[1]
                V["rebuild1"] += 1
            V["rebuild_t"] += _pc() - t3
            cur = nxt
        V["n"] += 1
        state.tokens += 2
        hid = mx.concatenate([h0[:, -1, :], h1[:, -1, :]], axis=0)
        return self.head(hid, rows=True), hid, snaps

    def rollback1(self, state: RequestState, snaps: list) -> None:
        """Undo the second position of `verify2`: every layer back to "p consumed"."""
        for i, sn in enumerate(snaps):
            self._restore(state.layers[i], state.ple.get(i), sn)
        state.tokens -= 1

    def forward(
        self,
        tokens: mx.array,
        state: RequestState,
        logits: str = "last",
        trace: list | None = None,
        telem=None,
    ) -> mx.array | None:
        """tokens [1, S] int32; advances `state` by S positions.

        logits: "none" (returns None), "last" ([1, V]), "all" ([1, S, V]) or "hidden" ([1, S,
        hc*hidden], the stack output before the terminal mixer). `trace`, when a list, receives one
        dict of intermediates per layer.
        """
        if (
            trace is None
            and logits == "last"
            and tokens.shape[1] == 1
            and tokens.shape[0] == 1
            and self.pool is not None
            and self.spec_all
            and not self.prefetch_lookahead
        ):
            return self._forward_lookahead(tokens, state, telem)
        offset = state.tokens
        h = self._embed_streams(tokens)
        pf = self.prefetch_lookahead if (self.pool is not None and tokens.shape[1] == 1) else 0
        h = self._run_layers(tokens, state, h, offset, pf, trace, telem)
        state.tokens += tokens.shape[1]
        if logits == "none":
            return None
        if logits == "hidden":
            return h
        if logits == "last":
            h = h[:, -1, :]
        return self.head(h)

    def _run_layers(
        self,
        tokens: mx.array,
        state: RequestState,
        h: mx.array,
        offset: int,
        pf: int,
        trace: list | None,
        telem,
    ) -> mx.array:
        """Run every layer over h, prefetching ahead when pf is set and appending each layer's trace."""
        for i, (layer, slot) in enumerate(zip(self.layers, state.layers)):
            t = {} if trace is not None else None
            h = layer(
                h, slot, offset, tokens=tokens, ple_state=state.ple.get(i), trace=t, telem=telem
            )
            if pf:
                self._prefetch_ahead(i, h, pf)
            if trace is not None:
                trace.append(t)
        return h

    def _prefetch_ahead(self, i: int, h: mx.array, k: int) -> None:
        """Prefetch experts that layers i+1..i+k are predicted to use into reserve slots.

        Decode only. A wrong guess costs bytes and a reserve slot, never an eviction a demand
        fetch needs, because `prefetch()` never evicts.
        """
        lazy, pool = [], None
        for d in range(1, k + 1):
            j = i + d
            if j >= len(self.layers):
                break
            tgt = self.layers[j]
            moe = getattr(tgt, "moe", None)
            if moe is None or getattr(moe, "pool", None) is None:
                continue
            try:
                # router input: the grouped norm's mean over the hc streams, a cheap stand-in for
                # the full hyper-connection mix (the router needs [.., hid], not [.., hc*hid])
                hcm = tgt.hc_mlp
                nb = hcm.gnorm(h).astype(hcm.dtype)
                nb = nb.reshape(*nb.shape[:-1], hcm.hc, hcm.hid)
                inds, _ = moe.route(mx.mean(nb, axis=-2))
            except Exception:  # noqa: BLE001  a prediction must never be able to fail a real step
                return
            pool = moe.pool
            lazy.append(inds.reshape(-1) + j * pool.num_experts)
        if not lazy:
            return
        try:
            # one host transfer for the whole group: concatenate on device, sync once
            g = np.asarray(mx.concatenate(lazy) if len(lazy) > 1 else lazy[0]).astype(np.int64)
            pool.prefetch(g)
        except Exception:  # noqa: BLE001  a prefetch is a hint; the demand read still happens
            return

    def wave_forward(self, chunks: list, states: list) -> None:
        """Advance N prefill rows layer-synchronously, reading each layer's expert union once."""
        offsets = [s.tokens for s in states]
        hs = [self._embed_streams(t) for t in chunks]
        for li, layer in enumerate(self.layers):
            hs = layer.wave(
                hs,
                [s.layers[li] for s in states],
                offsets,
                tokens=chunks,
                ple_states=[s.ple.get(li) for s in states],
            )
        for s, t in zip(states, chunks):
            s.tokens += t.shape[1]

    def decode_step(self, tokens: mx.array, states: list, telem=None) -> mx.array:
        """tokens [B, 1] int32, B >= 2 -> logits [B, V]; advances every state by one position."""
        offsets = [s.tokens for s in states]
        h = self._embed_streams(tokens)
        for li, layer in enumerate(self.layers):
            h = layer.step(
                h,
                [s.layers[li] for s in states],
                offsets,
                telem,
                tokens=tokens,
                ple_states=[s.ple.get(li) for s in states],
            )
        for s in states:
            s.tokens += 1
        return self.head(h[:, -1, :], rows=True)
