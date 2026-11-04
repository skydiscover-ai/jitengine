"""Qwen3.8-Flash-Next (`qwen4_exp`) forward pass.

Gated DeltaNet and sparse-attention blocks, each followed by a 512-expert MoE, wired through
hyper-connections over a [B, S, hc * hidden] residual, with a hashed n-gram embedding (PLE).

Every weight-derived array is evaluated in the constructor that creates it: MLX binds a lazy node
to the stream of the thread that built it, and the model is built on one thread and run on another.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as _mk
from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as _dk
from jitengine_qwen3_8_flash_next_m4pro.kernels import router_combine as _mf
from jitengine_qwen3_8_flash_next_m4pro.kvcache.linear_state import LinearStateSlot
from jitengine_qwen3_8_flash_next_m4pro.kvcache.state import RequestState
from jitengine_qwen3_8_flash_next_m4pro.linear import (
    QEmbedding,
    apply_dense_policy,
    calibrate_row_limits,
    qlinear_from,
)
from jitengine_qwen3_8_flash_next_m4pro.model import gdn as _gdn
from jitengine_qwen3_8_flash_next_m4pro.model.attention import Qwen4ExpTextAttention
from jitengine_qwen3_8_flash_next_m4pro.model.common import _DEFER, _IN_SPEC_BUILD, _dispatch
from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig
from jitengine_qwen3_8_flash_next_m4pro.model.gdn import LazyRow0, Qwen4ExpTextGatedDeltaNet
from jitengine_qwen3_8_flash_next_m4pro.model.hyper_connections import (
    HC_BASIS,
    HC_EXCLUDED,
    HC_PATH_DEFAULT,
    HC_REGIONS,
    Qwen4ExpTextGatedResidual,
    hyper_inject,
)
from jitengine_qwen3_8_flash_next_m4pro.model.moe import (
    MOE_KERNEL_DEFAULT,
    Qwen4ExpTextSparseMoeBlock,
)
from jitengine_qwen3_8_flash_next_m4pro.model.ple import NGramIndex, Qwen4ExpTextPLELayer

logger = logging.getLogger(__name__)

# When a wave's activations are evaluated ("none", "layer" or "stage"); never what is computed.
WAVE_EVAL = "none"

# Whether `build_spec2` replays its graph as compiled pieces (the PLE and the attention block stay
# eager) or rebuilds it eagerly per call. Host cost only: never what is computed.
PAIR_COMPILED = True


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

    def _router_half_body(self, h, conv, ssm):
        """Up to the router: (inds, scores, sh, shg, h2, inj2, x2, new_conv, new_ssm)."""
        mixed, inj, _ = self.hc_attn.fused(h)
        s = LinearStateSlot.__new__(LinearStateSlot)
        s.conv, s.ssm = conv, ssm
        out = self.block.fused_decode(mixed, s)
        mixed, inj2, h2 = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        moe = self.moe
        inds, scores, sh, shg = moe.router_and_shared(mixed)
        return inds, scores, sh, shg, h2, inj2, mixed.reshape(-1, mixed.shape[-1]), s.conv, s.ssm

    def _router_half_lazy_body(self, h, conv, pre, q0, k0, v0, g0, beta0):
        """`_router_half_body` on a slot holding a lazy rollback point (a p + 1-only rebuild): the
        state before the pass and row 0's recurrence inputs in place of the state after row 0, which
        the block advances by the pair launch (`LazyRow0.advance`)."""
        return self._router_half_body(h, conv, LazyRow0(self.block, pre, (q0, k0, v0, g0, beta0)))

    def _expert_half_body(self, x2, inds, scores, sh, shg, h2, inj2, tab, dep):
        """The speculative gather + combine: (y_spec, h3). `dep` is the pool's `encode_dep`, an
        input so the compiled build never holds a stale one."""
        moe = self.moe
        k = moe.top_k
        _, _, _, ar_np, _ = moe._spec_consts(k)
        slots_dev = mx.take(tab, inds + int(moe.layer_idx * moe.pool.num_experts))
        y_spec = moe._gather_dev(x2, slots_dev, k, dep)
        sc = mx.where(slots_dev >= 0, scores, mx.zeros_like(scores))
        h3 = _mf.moe_combine(y_spec, None, ar_np, sc, sh, h2, inj2, moe.hc, moe.hid, sh_gate=shg)
        return y_spec, h3.reshape(h2.shape)

    def _pre2_body(self, h):
        """The pair build before an attention block, h [2, hc*hid]: (mixed [2, 1, hid] as `decode2`
        takes it, inject [2, hc])."""
        mixed, inj, _ = self.hc_attn.fused_rows(h)
        return mixed.reshape(2, 1, -1), inj

    def _post2_body(self, h, out, inj):
        """The pair build from the block's output (2 * hid elements) to the routers, on two-row
        arrays: (inds [2k], scores [2k], sh [2, hid], shg0, shg1, x2 [2, hid], h2 [2, hc*hid],
        inject [2, hc])."""
        x2, inj2, h2 = self.hc_mlp.fused_rows(h, block_out=out, inject_in=inj)
        return (*self.moe.router_pair(x2), x2, h2, inj2)

    def _router_half2_body(self, h, conv, ssm):
        """`hc_attn`, the DeltaNet block and `_post2_body` as one graph on h [2, hc*hid]:
        `_post2_body`'s outputs, then the state after position 0 (the rollback point) and after
        position 1 ([8:12]). Lazy (`_lazy_row0`), the rollback point is row 0's conv state and
        recurrence inputs ([8:14]) and the state after position 1 is [14:16]."""
        mixed, inj, _ = self.hc_attn.fused_rows(h)
        out, st0, st1 = self.block.fused_decode2_rows(mixed, conv, ssm, lazy=self._lazy_row0())
        return (*self._post2_body(h, out, inj), *st0, *st1)

    def _lazy_row0(self) -> bool:
        """Whether the compiled pair build of this DeltaNet layer keeps its rollback point lazy."""
        return self.block.rows_lazy()

    def _expert_half2_body(self, x2, inds, scores, sh, g0, g1, h2, inj2, tab, dep):
        """`spec_pair_tail` (the 2k pairs' gather, both positions' combine) on the two-row arrays:
        (y [2k, hid], h3 [2, hc*hid]). `tab` and `dep` are inputs, as in `_expert_half_body`."""
        return self.moe.spec_pair_tail(x2, inds, scores, sh, g0, g1, h2, inj2, tab, dep)

    def drop_compiled(self) -> None:
        """Forget compiled builds; they hold the pool's arrays (extension included) as constants."""
        for a in (
            "_router_half",
            "_router_half_lazy",
            "_expert_half",
            "_router_half2",
            "_pre2",
            "_post2",
            "_expert_half2",
            "_comp_key",
            "_pair_modes",
        ):
            if hasattr(self, a):
                delattr(self, a)

    def _compiled_fns(self):
        pool = self.moe.pool
        key = (
            bool(pool.ext_active),
            id(getattr(pool, "gate_proj_ext", None)),
            bool(getattr(pool, "ann_active", False)),
            id(getattr(pool, "gate_proj_ann", None)),
            bool(_gdn.GDN_LAZY),  # the pair build's outputs differ (`_router_half2_body`)
        )
        if getattr(self, "_comp_key", None) != key:
            self._router_half = mx.compile(self._router_half_body)
            self._router_half_lazy = mx.compile(self._router_half_lazy_body)
            self._expert_half = mx.compile(self._expert_half_body)
            self._router_half2 = mx.compile(self._router_half2_body)
            self._pre2 = mx.compile(self._pre2_body)
            self._post2 = mx.compile(self._post2_body)
            self._expert_half2 = mx.compile(self._expert_half2_body)
            self._comp_key = key
        return self._router_half, self._expert_half

    def _compiled_fns2(self):
        """The pair build's compiled pieces (router half, pre, post, expert half), made and dropped
        with `_compiled_fns`'s; each traces on its first call."""
        self._compiled_fns()
        return self._router_half2, self._pre2, self._post2, self._expert_half2

    def _pair_mode(self, dtype) -> str:
        """How `build_spec2` builds this layer: "solo" (two `build_spec` calls, `_pair_ok` fails),
        "gdn" (the DeltaNet block inside the compiled router half), "attention" (the block eager
        between `_pre2_body` and `_post2_body`) or "eager" (the eager body: the two-row router or
        the DeltaNet block's flat rows are not available). Cached per dtype: every flag it reads is
        set when the layer is built."""
        modes = self.__dict__.setdefault("_pair_modes", {})
        m = modes.get(dtype)
        if m is None:
            if not self._pair_ok(dtype):
                m = "solo"
            elif not _mf.supports_pair(self.moe.pool.num_experts, self.moe.top_k):
                m = "eager"
            elif self.is_attention:
                m = "attention"
            else:
                m = "gdn" if self.block.rows_ok() else "eager"
            modes[dtype] = m
        return m

    def _pair_ok(self, dtype) -> bool:
        """Whether `build_spec2` can share this layer's launches between two positions: the fused
        hyper-connections, the fused shared expert, the speculative gather and a block with a
        two-position decode (GDN `fused_decode2`, attention `decode2`)."""
        block_ok = (
            self.block.decode2_ok()
            if self.is_attention
            else bool(getattr(self.block, "fused_ok", False))
        )
        return (
            block_ok
            and self.hc_attn.fused_ok
            and self.hc_mlp.fused_ok
            and self.moe.fused_ok
            and self.moe.spec_ok
            and self.moe.shared_fused_ok
            and self.moe.grouped_for(dtype)
        )

    def build_spec2(
        self, h0, h1, slot, offset: int, ta, da, ple_state, on_row0, telem=None, pair=None
    ):
        """Build positions offset (h0, token ta) and offset + 1 (h1, token da) of one sequence,
        without a host sync: (rec0, rec1), each the record `build_spec` gives that position.

        Launches whose rows are computed exactly as one row is are shared: both hyper-connections,
        the block's input projections and output projection, the shared expert and the routed
        experts. The PLE, DeltaNet conv and recurrence, attention, router and combine run per
        position in order (position 1 reads the state position 0 left). `on_row0(pl)` is called
        once position 0's state is in the slot (the rollback point), `pl` being the PLE state after
        position 0 or None. A layer that cannot share falls back to two `build_spec` calls.
        `pair`, when given, is [h0; h1] as one [2, hc*hid] array (the previous layer's two-row h3,
        `SpeculativeGatherMixin.pair_rows`), which the compiled build takes as it is.
        """
        mode = self._pair_mode(h0.dtype)
        if mode == "solo":
            r0 = self.build_spec(h0, slot, offset, tokens=ta, ple_state=ple_state, telem=telem)
            on_row0(None)
            r1 = self.build_spec(h1, slot, offset + 1, tokens=da, ple_state=ple_state, telem=telem)
            return r0, r1
        if PAIR_COMPILED and mode != "eager":
            return self._build_spec2_compiled(
                mode, h0, h1, slot, offset, ta, da, ple_state, on_row0, telem, pair
            )
        pl0 = None
        if self.ple is not None:
            # each position's PLE advances the PLE state; the rollback point keeps it after row 0
            h0 = h0 + self.ple(h0, ta, ple_state, None)
            pl0 = (ple_state.context, ple_state.conv)
            h1 = h1 + self.ple(h1, da, ple_state, None)
        h = mx.concatenate([h0, h1], axis=0)  # [2, 1, hc*hid]
        mixed, inj, _ = self.hc_attn.fused(h)
        _IN_SPEC_BUILD[0] = True
        try:
            if self.is_attention:
                out = self.block.decode2(mixed, slot, offset, lambda: on_row0(pl0))
            else:
                out = self.block.fused_decode2(mixed, slot, lambda: on_row0(pl0))
        finally:
            _IN_SPEC_BUILD[0] = False
        mixed, inj2, h2 = self.hc_mlp.fused(h, block_out=out, inject_in=inj)
        rt = self.moe.router_and_shared2(mixed)
        recs = self.moe.spec_build_pair(
            [mixed[0:1], mixed[1:2]],
            [rt[0][0], rt[1][0]],
            [rt[0][1], rt[1][1]],
            [rt[0][2], rt[1][2]],
            [rt[0][3], rt[1][3]],
            [h2[0:1], h2[1:2]],
            [inj2[0:1], inj2[1:2]],
        )
        for r in recs:
            r["telem"] = telem
        return recs[0], recs[1]

    def _build_spec2_compiled(
        self, mode, h0, h1, slot, offset, ta, da, ple_state, on_row0, telem, pair
    ):
        """The eager body below as compiled pieces on two-row arrays, around what stays eager: the
        PLE (host-side n-gram reads) and the attention block (per-token KV shapes). Then the
        routers' dispatch and the compiled expert half, as in `build_spec`. The same launches on
        the same values, except that both positions' top-k and combine share one launch each
        (`topk_softmax2`, `moe_combine2`, each row the one-row kernel's bits), and no launch
        between them only reshapes or copies a row: the host replays traced graphs of about half
        the nodes the eager body evaluates."""
        router_half2, pre2, post2, expert_half2 = self._compiled_fns2()
        pl0 = None
        if self.ple is not None:
            h0 = h0 + self.ple(h0, ta, ple_state, None)
            pl0 = (ple_state.context, ple_state.conv)
            h1 = h1 + self.ple(h1, da, ple_state, None)
            pair = None
        h = pair if pair is not None else mx.concatenate([h0, h1], axis=0).reshape(2, -1)
        if mode == "gdn" and self._lazy_row0():
            pre = slot.ssm
            o = router_half2(h, slot.conv, pre)
            # the rollback point as what makes it: on_row0's snapshot holds it, a restore makes it
            slot.conv, slot.ssm = o[8], LazyRow0(self.block, pre, o[9:14])
            on_row0(pl0)
            slot.conv, slot.ssm = o[14], o[15]
        elif mode == "gdn":
            o = router_half2(h, slot.conv, slot.ssm)
            slot.conv, slot.ssm = o[8], o[9]
            on_row0(pl0)
            slot.conv, slot.ssm = o[10], o[11]
        else:
            mixed, inj = pre2(h)
            _IN_SPEC_BUILD[0] = True
            try:
                out = self.block.decode2(mixed, slot, offset, lambda: on_row0(pl0))
            finally:
                _IN_SPEC_BUILD[0] = False
            o = post2(h, out, inj)
        moe = self.moe
        inds, scores, sh, g0, g1, x2, h2, inj2 = o[:8]
        tab, snap = moe.spec_pair_prelude(inds, scores)
        y, h3 = expert_half2(x2, inds, scores, sh, g0, g1, h2, inj2, tab, moe.pool.encode_dep)
        recs = moe.spec_pair_records(x2, inds, scores, sh, g0, g1, h2, inj2, y, h3, snap)
        _dispatch(h3)
        for r in recs:
            r["telem"] = telem
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
        lazy = isinstance(getattr(slot, "ssm", None), LazyRow0) and slot.ssm.pending
        if lazy:
            self.block.lazy_consumed += 1  # the pair launch advances it (`LazyRow0.advance`)
        if self._compiled_ok():
            router_half, expert_half = self._compiled_fns()
            if lazy:
                router_half = self._router_half_lazy
                st = (slot.ssm.pre, *slot.ssm.row0)
            else:
                st = (slot.ssm.state() if isinstance(slot.ssm, LazyRow0) else slot.ssm,)
            inds, scores, sh, shg, h2, inj2, x2, slot.conv, slot.ssm = router_half(
                h, slot.conv, *st
            )
            _dispatch(inds, scores)  # the router's own command buffer
            pool = self.moe.pool
            if pool._inflight and _DEFER[0] is None:
                pool.harvest_ready()
            tab = pool.device_slot_table()
            snap = pool.layer_snapshot(self.idx)
            y_spec, h3 = expert_half(x2, inds, scores, sh, shg, h2, inj2, tab, pool.encode_dep)
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
    the paths that would launch them run the stock ops instead. `embed`, when given, is the
    streamed token-embedding table (`embed_store.StreamedEmbedding`, a bit-identical lookup), and
    `w` then need not hold `model.embed_tokens.*`; an untied checkpoint's only use of it is `lookup`.
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
        kernels_off: frozenset = frozenset(),
        embed=None,
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
        self.ple_prefetch = True  # engine option `ple_prefetch` (`set_ple_switches`)
        self.gdn_row1_rebuilds = 0  # verify2's p + 1-only rebuilds of DeltaNet layers
        self.pool = pool
        bits, gs = cfg.quant_of("model.embed_tokens")
        self.embed = embed
        if self.embed is None:
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
        elif embed is not None:
            raise ValueError(
                "a tied checkpoint reads lm_head from the embedding: it stays resident"
            )
        else:  # tied embeddings
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
        """`stats()["moe_kernel"]`: option, kernel actually run per layer, variant and configs
        (`solo_vecs`: the VECS the one-row gather takes in place of each config's)."""
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
            "solo_vecs": _mk.SOLO_VECS,
            "solo_vecs_basis": _mk.SOLO_VECS_BASIS,
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
        would force a copy); DeltaNet and PLE arrays are replaced, so their references suffice. A
        lazy pair build's rollback point (`LazyRow0` in the slot while `on_row0` runs) is saved as
        it is, "delta_lazy"; `_restore` puts it back as it is.
        """
        if hasattr(slot, "keys"):
            sn = ("kv", slot.offset, dict(slot.idx_bk))
        elif isinstance(slot.ssm, LazyRow0):
            sn = ("delta_lazy", slot.conv, slot.ssm)
        else:
            sn = ("delta", slot.conv, slot.ssm)
        pl = None if ple_state is None else (ple_state.context, ple_state.conv)
        return sn, pl

    @staticmethod
    def _restore(slot, ple_state, snap):
        sn, pl = snap
        if sn[0] == "kv":
            # a copy: the slot's block-key cache is updated in place by the next build, and the
            # snapshot must survive that (verify2 restores `snaps[i]` and may restore it again in
            # `rollback1`)
            slot.offset, slot.idx_bk = sn[1], dict(sn[2])
        elif sn[0] == "delta_lazy":
            # as it is: a p + 1-only rebuild advances it, `rollback1` makes it
            slot.conv, slot.ssm = sn[1], sn[2]
        else:
            slot.conv, slot.ssm = sn[1], sn[2]
        if pl is not None:
            ple_state.context, ple_state.conv = pl

    def verify_ok(self) -> bool:
        """Whether the solo decode step is `_forward_lookahead`, the step `forward_hidden` and
        `verify2` reproduce: an expert pool, every layer speculative, no router prefetch."""
        return self.pool is not None and bool(self.spec_all) and not self.prefetch_lookahead

    def forward_hidden(self, tokens: mx.array, state: RequestState, telem=None):
        """The solo decode step (`_forward_lookahead`, its logits bit for bit), also returning the
        stack output before the terminal mixer, [1, hc*hid], which the MTP head drafts from."""
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
        try:
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
        except BaseException:
            self._settle_fetches()
            raise
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

    def _settle_fetches(self) -> None:
        """A decode loop raised: settle the layer fetch it left between `spec_settle_begin` and its
        finish, so its landing slots are not lost to the step."""
        settle = getattr(self.pool, "abandon_open", None)  # a stand-in pool may have none
        if settle is not None:
            settle()

    def _dispatch_held(self, held: list) -> None:
        """Dispatch the kernels held back while the lookahead layer was rebuilt."""
        for arrs in held:
            mx.async_eval(*arrs)

    # ------------------------------------------------------------ the MTP verify pass
    def verify2(self, t: int, d: int, state: RequestState, telem=None):  # noqa: PLR0915
        """Positions p = state.tokens (token t) and p + 1 (the draft d) of one sequence in one pass.

        Each position runs the solo decode step's arithmetic, so each row of the logits is
        bit-identical to a solo step's at that position. Returns (logits [2, V], hidden [2, hc*hid]
        before the terminal mixer, snaps); `snaps` restores every layer to "p consumed, p + 1 not"
        (`rollback1`). `state.tokens` += 2.
        """
        offset = state.tokens
        self._ple_prefetch(state, (int(t), int(d)))
        ta = mx.array([[int(t)]], dtype=mx.int32)
        da = mx.array([[int(d)]], dtype=mx.int32)
        h0, h1 = self._embed_streams(ta), self._embed_streams(da)
        n = len(self.layers)
        snaps: list = [None] * n

        def build2(i, a, b, which=(0, 1), pair=None):
            """Layer i for both positions (from the state before p), or for p + 1 alone (from the
            state after p, `snaps[i]`); returns {0: rec0, 1: rec1, "pre": state before p}. `pair`:
            [a; b] as one two-row array, when the previous layer's pair build left it."""
            layer, slot, pst = self.layers[i], state.layers[i], state.ple.get(i)
            out: dict = {}
            if which == (0, 1):
                out["pre"] = self._snap(slot, pst)

                def on_row0(pl):
                    sn, pl_now = self._snap(slot, pst)
                    snaps[i] = (sn, pl if pl is not None else pl_now)

                out[0], out[1] = layer.build_spec2(
                    a, b, slot, offset, ta, da, pst, on_row0, telem, pair=pair
                )
                return out
            out[1] = layer.build_spec(b, slot, offset + 1, tokens=da, ple_state=pst)
            return out

        cur = build2(0, h0, h1)
        try:
            for i in range(n):
                moe = self.layers[i].moe
                nxt = (
                    build2(i + 1, cur[0]["h3"], cur[1]["h3"], pair=moe.pair_rows(cur, "h3_rows"))
                    if i + 1 < n
                    else None
                )
                # both positions' reads in flight together: one begin over both positions' pairs
                pc = moe.spec_settle_pair_begin([cur[0], cur[1]], [nxt[0], nxt[1]] if nxt else None)
                pm0, pm1 = pc["outs"][0][1], pc["outs"][1][1]
                if nxt is not None and (pm0 or pm1):
                    # layer i+1 rebuilt on the corrections while the reads are in flight, held back
                    # until they land: both positions if p was corrected (p + 1 read p's layer i+1
                    # state), else p + 1 alone from the state p left
                    self._restore(
                        state.layers[i + 1],
                        state.ple.get(i + 1),
                        nxt["pre"] if pm0 else snaps[i + 1],
                    )
                    if not pm0 and snaps[i + 1][0][0] != "kv":
                        self.gdn_row1_rebuilds += 1
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
                    try:
                        (h0, _m0), (h1, _m1) = moe.spec_settle_pair_finish(pc)
                    except BaseException:
                        self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt["pre"])
                        raise
                    if pc.get("pred_ok", True):
                        self._dispatch_held(held)
                        cur = nb
                        continue
                    # a record did not land where begin said: what was built on it is discarded
                    self._restore(state.layers[i + 1], state.ple.get(i + 1), nxt["pre"])
                    cur = build2(i + 1, h0, h1)
                    continue
                (h0, _m0), (h1, _m1) = moe.spec_settle_pair_finish(pc)
                cur = nxt
        except BaseException:
            self._settle_fetches()
            raise
        state.tokens += 2
        hid = mx.concatenate([h0[:, -1, :], h1[:, -1, :]], axis=0)
        return self._head2(hid), hid, snaps

    def _ple_prefetch(self, state: RequestState, tokens) -> None:
        """Issue the PLE layers' row reads for the coming single-token calls on `tokens` (host ints, in
        order) now: they land while the layers before the PLE layer build and run, and the calls take
        them from the memo (`Qwen4ExpTextPLELayer.prefetch`; the same rows, read once each)."""
        if not self.ple_prefetch:
            return
        for i in self.ple_layers:
            pst = state.ple.get(i)
            if pst is not None:
                self.layers[i].ple.prefetch(pst, tokens)

    def set_ple_switches(self, prefetch: bool, host_context: bool) -> None:
        """The PLE switches (engine options `ple_prefetch`, `ple_host_context`). Neither changes a
        value, so they may be flipped between requests; both off is the plain PLE path."""
        self.ple_prefetch = bool(prefetch)
        for i in self.ple_layers:
            self.layers[i].ple.host_context = bool(host_context)

    def ple_switches(self) -> dict:
        """{"ple_prefetch", "ple_host_context"} as set; host_context is None without a PLE layer."""
        hc = [bool(self.layers[i].ple.host_context) for i in self.ple_layers]
        return {"ple_prefetch": bool(self.ple_prefetch), "ple_host_context": hc[0] if hc else None}

    def _head2(self, hid: mx.array) -> mx.array:
        """`head` for two rows [2, hc*hid] -> [2, V], each row bit-identical to `head` on it alone:
        the mixer runs per row (its solo compiled body), lm_head once over both rows through
        `QLinear.rows`, which splits wherever the shape's launch is not row-exact."""
        mixed = mx.concatenate([self.mixer(hid[r : r + 1])[0] for r in range(2)], axis=0)
        return self.lm_head.rows(mixed)

    def gdn_lazy_stats(self) -> dict:
        """The lazy rollback state's counters for `/stats` (under "mtp"): `gdn_lazy` and
        `gdn_lazy_layers`, whether and on how many DeltaNet layers the compiled pair build runs
        lazy; `gdn_lazy_materialized`, the row-0 states made (one per lazy layer per rejected
        pass, by `rollback1`); `gdn_lazy_consumed`, the lazy points a p + 1-only rebuild advanced
        by the pair launch instead; `gdn_row1_rebuilds`, the p + 1-only rebuilds of DeltaNet
        layers. Served, materialized = rejected passes x layers and consumed = rebuilds."""
        gdn = [L for L in self.layers if not L.is_attention]
        lazy = [
            L
            for L in gdn
            if PAIR_COMPILED and L._pair_mode(self.residual_dtype) == "gdn" and L._lazy_row0()
        ]
        return {
            "gdn_lazy": bool(lazy),
            "gdn_lazy_layers": len(lazy),
            "gdn_lazy_materialized": sum(L.block.lazy_materialized for L in gdn),
            "gdn_lazy_consumed": sum(L.block.lazy_consumed for L in gdn),
            "gdn_row1_rebuilds": int(self.gdn_row1_rebuilds),
        }

    def rollback1(self, state: RequestState, snaps: list) -> None:
        """Undo `verify2`'s second position: every layer back to "p consumed, p + 1 not". A lazy
        rollback point is made here (`LazyRow0.state`, the stored build's row-0 launch, run with the
        next step that reads it), so no slot holds one after the pass."""
        for i, sn in enumerate(snaps):
            slot = state.layers[i]
            self._restore(slot, state.ple.get(i), sn)
            if isinstance(getattr(slot, "ssm", None), LazyRow0):
                slot.ssm = slot.ssm.state()
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
