"""The MoE block: the router, the routed experts (resident, or the offload pool's slots) and the
gated shared expert."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_solo as _ee
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.elementwise import swiglu
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import router_topk, topk
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import Qwen3_5MoeConfig
from jitengine_qwen3_6_35b_a3b_m4pro.model.kernel_choice import KernelChoice, Strategies
from jitengine_qwen3_6_35b_a3b_m4pro.model.weight import FUSIONS, _fuse, _qlin


def _router_ok(logits, k) -> bool:
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import router_supported

    return router_supported(logits, k)


EE_ROWS_FUSED = True  # row stacks of 4 rows or more go through the fused rows kernels


EE_ROWS_MAX = 16  # row stacks up to this size use the exact expert kernels (16-row verify windows)


class Qwen3_5MoeSparseMoeBlock:
    """Router + 256 routed experts + the gated shared expert.

    With a `pool`, the expert weights are the pool's `[slots, ...]` arrays indexed by slot id;
    everything else is the resident code, unchanged.
    """

    def __init__(
        self,
        cfg: Qwen3_5MoeConfig,
        w: dict,
        p: str,
        pool=None,
        layer_idx: int = 0,
        fuse: frozenset = FUSIONS,
    ):
        gs = cfg.q_group_size
        self.paths = KernelChoice()
        self.strategies = Strategies()
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk = cfg.norm_topk_prob
        self.pool = pool
        self.layer_idx = int(layer_idx)
        self.router = (
            _fuse(w, [p + "gate", p + "shared_expert_gate"], gs) if "router" in fuse else None
        )
        if self.router is None:
            self.gate = _qlin(w, p + "gate", gs)
            self.shared_gate = _qlin(w, p + "shared_expert_gate", gs)
        self.shared_gu = (
            _fuse(w, [p + "shared_expert.gate_proj", p + "shared_expert.up_proj"], gs)
            if "shared" in fuse
            else None
        )
        if self.shared_gu is None:
            self.shared_gate_proj = _qlin(w, p + "shared_expert.gate_proj", gs)
            self.shared_up_proj = _qlin(w, p + "shared_expert.up_proj", gs)
        self.shared_down_proj = _qlin(w, p + "shared_expert.down_proj", gs)
        self.gate_up = None
        self.shared_in_stack = False
        if pool is None:
            self.gate_up = (
                _fuse(w, [p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj"], gs)
                if "experts" in fuse
                else None
            )
            if self.gate_up is None:
                self.gate_proj = _qlin(w, p + "switch_mlp.gate_proj", gs)
                self.up_proj = _qlin(w, p + "switch_mlp.up_proj", gs)
            self.down_proj = _qlin(w, p + "switch_mlp.down_proj", gs)
            self._append_shared_expert()
            if self.shared_in_stack:
                # the appended stacks are copies: free the originals now, not after the build
                for suffix in (".weight", ".scales", ".biases"):
                    w.pop(p + "switch_mlp.down_proj" + suffix, None)
        else:
            self.gate_proj = pool.gate_proj
            self.up_proj = pool.up_proj
            self.down_proj = pool.down_proj

    def _append_shared_expert(self) -> None:
        """Append the shared expert as expert index E when its shapes match the routed experts'.

        The single-token kernels then run it as a ninth expert; stock gathers index 0..E-1 only and
        keep using the separate shared QLinears.
        """
        self.shared_in_stack = False
        gu, sgu, d, sd = self.gate_up, self.shared_gu, self.down_proj, self.shared_down_proj
        if gu is None or sgu is None or self.router is None:
            return
        g, sg_ = gu.lin, sgu.lin
        if (
            g.weight.shape[1:] != sg_.weight.shape
            or d.weight.shape[1:] != sd.weight.shape
            or g.bits != sg_.bits
            or d.bits != sd.bits
            or (g.biases is None) != (sg_.biases is None)
        ):
            return
        cat = lambda a, b: mx.concatenate([a, b[None]], axis=0)  # noqa: E731
        g.weight, g.scales, g.biases = (
            cat(g.weight, sg_.weight),
            cat(g.scales, sg_.scales),
            cat(g.biases, sg_.biases),
        )
        gu.parts = []  # views of the pre-append buffer would pin a second copy; gathers use `lin`
        d.weight, d.scales, d.biases = (
            cat(d.weight, sd.weight),
            cat(d.scales, sd.scales),
            cat(d.biases, sd.biases),
        )
        mx.eval(g.weight, g.scales, g.biases, d.weight, d.scales, d.biases)
        self.shared_in_stack = True

    def with_residual(
        self, x: mx.array, resid: mx.array, telem=None, rows: bool = False, pre=None
    ) -> mx.array:
        """resid + moe(x)."""
        return resid + self(x, telem, rows, pre_in=pre)

    def experts(self, x: mx.array, inds: mx.array, telem=None) -> mx.array:
        """x [B, S, D], inds [B, S, k] -> [B, S, k, D]; every selected expert is applied.

        Pairs are sorted by expert, as in mlx_lm's SwitchGLU, so each distinct expert's weights are
        read once per call; a single row with fewer than 64 pairs skips the sort.
        """
        if self.pool is not None:
            return self._experts_pooled(x, inds, telem)
        xe = mx.expand_dims(x, (-2, -3))  # [B, S, 1, 1, D]
        do_sort = inds.shape[0] > 1 or inds.size >= 64
        if do_sort:
            xe, idx, inv_order = self._sort_pairs(xe, inds)
        else:
            idx = inds
        P = idx.size
        seg = self._use_seg_prefill(xe, do_sort, P)
        y = self._gather_experts(xe, idx, do_sort, P, seg)
        if do_sort:
            y = mx.unflatten(y[inv_order], 0, inds.shape)
        self._note_expert_use(telem, do_sort, idx, inds)
        return y.squeeze(-2)

    def _sort_pairs(self, xe: mx.array, inds: mx.array) -> tuple:
        """Order the (row, expert) pairs by expert; return the gathered inputs, the sorted indices and the inverse order."""
        k = inds.shape[-1]
        flat = inds.flatten()
        order = mx.argsort(flat)
        inv_order = mx.argsort(order)
        xe = xe.flatten(0, -3)[order // k]  # [N*k, 1, D]
        idx = flat[order]
        return xe, idx, inv_order

    def _use_seg_prefill(self, xe: mx.array, do_sort: bool, P: int) -> bool:
        """Whether the pairs take the tiled segment GEMM: 3-bit, group-64 stacks and enough pairs per expert."""
        seg = (
            self.paths.seg_prefill
            and do_sort
            and self.gate_up is not None
            and xe.shape[-2] == 1
            and P >= 16
            and P // self.gate_up.lin.weight.shape[0] >= 4  # stock's gather_qmm_rhs case
            and self.gate_up.lin.bits == 3
            and self.down_proj.bits == 3
            and self.gate_up.lin.group_size == 64
            and self.down_proj.group_size == 64
        )
        return seg

    def _gather_experts(
        self, xe: mx.array, idx: mx.array, do_sort: bool, P: int, seg: bool
    ) -> mx.array:
        """Gate, up and down projections of the (sorted) pairs, by the segment GEMM or the gather kernels."""
        if seg:
            # the same tile GEMM as stock's gather_qmm_rhs, one expert segment per threadgroup
            from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gather_qmm_prefill import seg_gather_qmm

            lin, dn = self.gate_up.lin, self.down_proj
            D = xe.shape[-1]
            gu = seg_gather_qmm(xe.reshape(P, D), lin, idx).reshape(P, 1, -1)
            gate, up = mx.split(gu, self.gate_up.cuts, axis=-1)
            h = swiglu(gate, up)
            y = seg_gather_qmm(h.reshape(P, -1), dn, idx).reshape(P, 1, -1)
        elif self.gate_up is not None:
            gate, up = self.gate_up.gather(xe, idx, sorted_indices=do_sort)
            y = self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=do_sort)
        else:
            up = self.up_proj.gather(xe, idx, sorted_indices=do_sort)
            gate = self.gate_proj.gather(xe, idx, sorted_indices=do_sort)
            y = self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=do_sort)
        return y

    def _note_expert_use(self, telem, do_sort: bool, idx: mx.array, inds: mx.array) -> None:
        """Report the pair and distinct-expert counts of this call to the telemetry collector."""
        if telem is not None and telem.collect:
            f = idx.flatten() if do_sort else mx.sort(inds.flatten())
            telem.note(int(f.size), 1 + (f[1:] != f[:-1]).sum())

    def _experts_pooled(self, x: mx.array, inds: mx.array, telem=None) -> mx.array:
        """The same dispatch against the slot pool: identical arithmetic, identical kernel.

        Adds one host sync per layer to learn which experts to fetch. Pairs are sorted by slot id,
        as `sorted_indices=True` needs ascending indices. Solo decode and prefill stay bit-exact
        with the resident engine; a batched step may differ at near-ties.
        """
        inds_np = np.asarray(inds).astype(np.int64, copy=False)  # the per-layer host sync
        k = inds.shape[-1]
        flat = inds_np.reshape(-1)
        segments = self.pool.segments(self.layer_idx, flat)
        do_sort = inds.shape[0] > 1 or inds.size >= 64
        if len(segments) == 1 and not do_sort:
            idx = mx.array(segments[0][1].reshape(inds.shape).astype(np.uint32))
            xe = mx.expand_dims(x, (-2, -3))
            up = self.up_proj.gather(xe, idx, sorted_indices=False)
            gate = self.gate_proj.gather(xe, idx, sorted_indices=False)
            y = self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=False)
            self._note_pooled_use(telem, flat)
            return y.squeeze(-2)

        xe = mx.expand_dims(x, (-2, -3)).flatten(0, -3)  # [N, 1, D]
        multi = len(segments) > 1
        outs, order = [], []
        for pos, slots in segments:
            srt = np.argsort(slots, kind="stable")
            pos, slots = pos[srt], slots[srt]
            xw = xe[mx.array(pos // k)]
            idx = mx.array(slots.astype(np.uint32))
            up = self.up_proj.gather(xw, idx, sorted_indices=True)
            gate = self.gate_proj.gather(xw, idx, sorted_indices=True)
            y = self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=True)
            if multi:
                mx.eval(y)  # this wave must finish before the next fetch reuses its slots
            outs.append(y)
            order.append(pos)
        y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        order = np.concatenate(order) if len(order) > 1 else order[0]
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)
        y = mx.unflatten(y[mx.array(inv)], 0, tuple(inds.shape))
        self._note_pooled_use(telem, flat)
        return y.squeeze(-2)

    def _note_pooled_use(self, telem, flat: np.ndarray) -> None:
        """Report the pair and distinct-expert counts of this call to the telemetry collector."""
        if telem is not None and telem.collect:
            telem.note(int(flat.size), int(np.unique(flat).size))

    def __call__(
        self, x: mx.array, telem=None, rows: bool = False, gate_outside: bool = False, pre_in=None
    ):
        """MoE block output; with `rows=True`, x [B, 1, D] rows each reproduce their solo result.

        `gate_outside=True` returns (y, shared-gate logit, shared) for callers under mx.compile,
        whose JIT-fused sigmoid-times-x is not bit-identical to MLX's precompiled kernels.
        """
        dense = self._dense_for(rows)
        logits, sg_logit = self._route_logits(x, dense)
        inds, scores = self._route_top_k(logits)
        pre, shared = self._shared_expert(x, inds, rows, dense, pre_in)
        if (
            self.pool is not None
            and self.strategies.speculative
            and inds.shape[0] == 1
            and inds.size < 64
        ):
            sg = mx.sigmoid(sg_logit)
            e = self._experts_pooled_spec(x, inds, telem, extra=(scores, shared, sg))
            return (e * scores[..., None]).sum(axis=-2) + sg * shared
        if rows and self.paths.rows_exact and x.shape[0] > 1 and self._rows_ok():
            y = self._rows_experts(x, inds, scores)
        elif pre is not None:
            y = self._prescaled_experts(x, inds, scores, pre, telem)
        elif rows and 2 <= x.shape[0] <= EE_ROWS_MAX and self._ee_rows_ok(x):
            y = self._exact_rows_experts(x, inds, scores, telem)
        elif self._use_exact(x, inds):
            y = self._exact_experts(x, inds, scores, telem)
        else:
            e = self.experts(x, inds, telem)
            y = (e * scores[..., None]).sum(axis=-2)
        if gate_outside:
            return y, sg_logit, shared
        return y + mx.sigmoid(sg_logit) * shared

    def _dense_for(self, rows: bool) -> object:
        """The projection selector: each projection's solo-row path when `rows`, else the projection itself."""
        dense = (lambda lin: lin.rows) if rows else (lambda lin: lin)
        return dense

    def _route_logits(self, x: mx.array, dense) -> tuple:
        """The router logits and the shared expert's gate logit, from the fused router or the two gates."""
        if self.router is not None:
            logits, sg_logit = dense(self.router)(x)
        else:
            logits, sg_logit = dense(self.gate)(x), dense(self.shared_gate)(x)
        return logits, sg_logit

    def _route_top_k(self, logits: mx.array) -> tuple:
        """The top-k expert indices and their scores (renormalized when `norm_topk`), fused when it can."""
        k = self.top_k
        if self.paths.fused_router and self.paths.exact_topk and _router_ok(logits, k):
            inds, scores = router_topk(logits, k, self.norm_topk)  # the five ops below, one launch
        else:
            gates = mx.softmax(logits, axis=-1, precise=True)
            inds = topk(gates, k, kernel=self.paths.exact_topk)
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk:
                scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def _shared_expert(self, x: mx.array, inds: mx.array, rows: bool, dense, pre_in) -> tuple:
        """The shared expert's output, and the activation pre-scale when it is shared with the routed gate/up."""
        # the shared expert does not depend on routing: built first so the speculative path can
        # dispatch it with the gather, while the host waits on the router (same ops, same values)
        pre = None
        if (
            self.paths.moe_prescaled
            and (not rows or x.shape[0] == 1)
            and self.shared_gu is not None
            and self._use_exact(x, inds)
            and self.gate_up.lin.bits == 3
            and self.shared_gu.lin.prescaled_ok_solo()
            and self._rows_ok()
        ):
            # one activation pre-scale shared by the shared expert's gate/up and the routed gate/up
            from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import prescale

            x2 = x.reshape(1, -1)
            pre = pre_in if pre_in is not None else prescale(x2)
            sgp, sup = mx.split(
                self.shared_gu.lin._prescaled(x2, pre=pre).reshape(*x.shape[:-1], -1),
                self.shared_gu.cuts,
                axis=-1,
            )
            shared = dense(self.shared_down_proj)(swiglu(sgp, sup))
        elif self.shared_gu is not None:
            sgp, sup = dense(self.shared_gu)(x)
            shared = dense(self.shared_down_proj)(swiglu(sgp, sup))
        else:
            sgp, sup = dense(self.shared_gate_proj)(x), dense(self.shared_up_proj)(x)
            shared = dense(self.shared_down_proj)(swiglu(sgp, sup))
        return pre, shared

    def _rows_experts(self, x: mx.array, inds: mx.array, scores: mx.array) -> mx.array:
        """The weighted routed output for a batch of rows, each reproducing its solo result."""
        B = x.shape[0]
        if self.paths.rows_prescaled:
            # dedup gate+up (the prescaled arithmetic, fused SwiGLU) + one-simdgroup-per-expert down
            from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_qmv_prescaled import (
                expert_gate_up_prescaled_rows,
            )

            fi = inds.reshape(-1).astype(mx.uint32)
            h = expert_gate_up_prescaled_rows(
                x.reshape(B, -1),
                self.gate_up.lin,
                inds.reshape(B, -1),
                rps=16 if B >= 4 else 8,
                dedup=False,
            )
            y = _ee.expert_down_exact_par(
                h, self.down_proj, fi, scores.reshape(-1), rows=B, rps=8
            ).reshape(x.shape)
        else:
            from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_rows as _er

            y = _er.experts_rows_exact(
                x.reshape(B, -1),
                self.gate_up.lin,
                self.down_proj,
                inds.reshape(B, -1),
                scores.reshape(B, -1),
            ).reshape(x.shape)
        return y

    def _prescaled_experts(
        self, x: mx.array, inds: mx.array, scores: mx.array, pre, telem
    ) -> mx.array:
        """The weighted routed output of one row with the shared activation pre-scale."""
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_qmv_prescaled import (
            expert_gate_up_prescaled,
        )

        fi = inds.reshape(-1).astype(mx.uint32)
        h = expert_gate_up_prescaled(x.reshape(1, -1), self.gate_up.lin, fi, pre=pre)
        y = _ee.expert_down_exact_par(h, self.down_proj, fi, scores.reshape(-1), rps=8).reshape(
            x.shape
        )
        if telem is not None and telem.collect:
            telem.note(int(fi.size), int(fi.size))
        return y

    def _exact_rows_experts(self, x: mx.array, inds: mx.array, scores: mx.array, telem) -> mx.array:
        """The weighted routed output of a small row stack, each row bitwise its solo exact-kernel result."""
        # a small row stack (the speculative verify's positions) through the solo exact expert
        # kernels, one row at a time, each row bitwise its own solo MoE, in place of the sorted
        # stock gather
        B = x.shape[0]
        if EE_ROWS_FUSED and B >= 4:
            # all rows in two launches (gate+up+SwiGLU rows kernel, then the one-simdgroup-per-expert
            # down kernel); per row the same bits as the per-row loop below
            fi = inds.reshape(B, -1).astype(mx.uint32)
            h = _ee.expert_gate_up_exact_rows(x.reshape(B, -1), self.gate_up.lin, fi)
            y = _ee.expert_down_exact_par(
                h, self.down_proj, fi.reshape(-1), scores.reshape(-1), rows=B
            ).reshape(x.shape)
        else:
            ys = []
            for r in range(B):
                # r:r+1 slices are views; x[r] / inds[r] / scores[r] would each compile to a Gather launch
                fi = inds[r : r + 1].reshape(-1)
                h = _ee.expert_gate_up_exact(x[r : r + 1].reshape(-1), self.gate_up.lin, fi)
                ys.append(
                    _ee.expert_down_exact(h, self.down_proj, fi, scores[r : r + 1].reshape(-1))
                )
            y = mx.stack(ys).reshape(x.shape)
        if telem is not None and telem.collect:
            f = mx.sort(inds.reshape(-1))
            telem.note(int(f.size), 1 + (f[1:] != f[:-1]).sum())
        return y

    def _exact_experts(self, x: mx.array, inds: mx.array, scores: mx.array, telem) -> mx.array:
        """The weighted routed output of one resident decode row by the exact expert kernels."""
        fi = inds.reshape(-1)
        h = _ee.expert_gate_up_exact(x.reshape(-1), self.gate_up.lin, fi)
        y = _ee.expert_down_exact(h, self.down_proj, fi, scores.reshape(-1)).reshape(x.shape)
        if telem is not None and telem.collect:
            telem.note(int(fi.size), int(fi.size))
        return y

    _exact_ok = None

    def _ee_rows_ok(self, x: mx.array) -> bool:
        if not self.paths.exact_experts or self.pool is not None or self.gate_up is None:
            return False
        if self._exact_ok is None:
            self._exact_ok = _ee.supported(self.gate_up.lin, self.down_proj, x.dtype)
        return self._exact_ok

    def _use_exact(self, x: mx.array, inds: mx.array) -> bool:
        """One resident decode row whose stacks meet expert_solo.supported (checked once)."""
        if (
            not self.paths.exact_experts
            or self.pool is not None
            or self.gate_up is None
            or inds.shape[:-1] != (1, 1)
        ):
            return False
        if self._exact_ok is None:
            self._exact_ok = _ee.supported(self.gate_up.lin, self.down_proj, x.dtype)
        return self._exact_ok

    def prefill_parts(self, x: mx.array):
        """__call__'s router and shared expert for x [1, S, D]: (inds, scores, sg_logit, shared)."""
        logits, sg_logit = self.gate(x), self.shared_gate(x)
        k = self.top_k
        if self.paths.fused_router and self.paths.exact_topk and _router_ok(logits, k):
            inds, scores = router_topk(logits, k, self.norm_topk)
        else:
            gates = mx.softmax(logits, axis=-1, precise=True)
            inds = topk(gates, k, kernel=self.paths.exact_topk)
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk:
                scores = scores / scores.sum(axis=-1, keepdims=True)
        if self.shared_gu is not None:
            sgp, sup = self.shared_gu(x)
        else:
            sgp, sup = self.shared_gate_proj(x), self.shared_up_proj(x)
        shared = self.shared_down_proj(swiglu(sgp, sup))
        return inds, scores, sg_logit, shared

    @staticmethod
    def prefill_combine(e: mx.array, scores, sg_logit, shared):
        """__call__'s tail: e [1, S, k, D] routed outputs -> the block output."""
        y = (e * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(sg_logit) * shared

    def prefill_experts_merged(self, xs: list, inds: list) -> list:
        """Routed experts for several chunks in one sorted segment gather (seg_gather_qmm).

        MMA rows are independent, so each chunk's output equals `experts()` on that chunk alone.
        """
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gather_qmm_prefill import seg_gather_qmm

        k = self.top_k
        D = xs[0].shape[-1]
        sizes = [int(i.size) for i in inds]
        flat = mx.concatenate([i.reshape(-1) for i in inds])
        xe = mx.concatenate([x.reshape(-1, D) for x in xs], axis=0)  # [rows, D]
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        idx = flat[order].astype(mx.uint32)
        xs_sorted = xe[order // k]
        lin, dn = self.gate_up.lin, self.down_proj
        P = int(idx.size)
        gu = seg_gather_qmm(xs_sorted, lin, idx).reshape(P, 1, -1)
        gate, up = mx.split(gu, self.gate_up.cuts, axis=-1)
        h = swiglu(gate, up)
        y = seg_gather_qmm(h.reshape(P, -1), dn, idx)[inv]  # [P, D]
        outs, o = [], 0
        for ii, n in zip(inds, sizes):
            outs.append(y[o : o + n].reshape(*ii.shape, -1))
            o += n
        return outs

    def merged_ok(self, total_pairs: int) -> bool:
        return (
            self.paths.seg_prefill
            and self.pool is None
            and self.gate_up is not None
            and self.router is None
            and total_pairs >= 16
            and total_pairs // self.gate_up.lin.weight.shape[0] >= 4
            and self.gate_up.lin.bits == 3
            and self.down_proj.bits == 3
        )

    _rows_ok_v = None

    def _rows_ok(self) -> bool:
        if self._rows_ok_v is None:
            from jitengine_qwen3_6_35b_a3b_m4pro.kernels import expert_rows as _er

            self._rows_ok_v = (
                self.pool is None
                and self.gate_up is not None
                and _er.supported(self.gate_up.lin, self.down_proj)
            )
        return self._rows_ok_v

    def _experts_pooled_spec(self, x: mx.array, inds: mx.array, telem=None, extra=()) -> mx.array:
        """The pooled single-row gather, dispatched before the router's host sync.

        On a full hit the result is bit-identical to the synchronous path; on a miss it is discarded
        and recomputed after the fetch. Eviction-safe: `ensure` protects this call's hits, and rows
        that read clamped slot 0 are discarded.
        """
        pool = self.pool
        off = self.layer_idx * pool.num_experts
        mx.async_eval(inds)  # buffer 1: the router
        slots_dev = mx.take(pool.device_slot_table(), inds + off)
        xe = mx.expand_dims(x, (-2, -3))
        up = self.up_proj.gather(xe, slots_dev, sorted_indices=False)
        gate = self.gate_proj.gather(xe, slots_dev, sorted_indices=False)
        y_spec = self.down_proj.gather(swiglu(gate, up), slots_dev, sorted_indices=False).squeeze(
            -2
        )
        mx.async_eval(y_spec, *extra)  # buffer 2: gather + shared expert
        flat = np.asarray(inds).astype(np.int64, copy=False).reshape(-1)  # the host sync
        all_hit = pool.all_resident(flat + off)
        segments = pool.segments(self.layer_idx, flat)  # accounting (+ the fetch on a miss)
        if telem is not None and telem.collect:
            telem.note(int(flat.size), int(np.unique(flat).size))
        pool.spec_calls = getattr(pool, "spec_calls", 0) + 1
        if all_hit:
            return y_spec
        pool.spec_misses = getattr(pool, "spec_misses", 0) + 1
        if len(segments) != 1:
            raise RuntimeError("speculative gather: a single row's experts exceeded the pool")
        idx = mx.array(segments[0][1].reshape(inds.shape).astype(np.uint32))
        up = self.up_proj.gather(xe, idx, sorted_indices=False)
        gate = self.gate_proj.gather(xe, idx, sorted_indices=False)
        return self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=False).squeeze(-2)
