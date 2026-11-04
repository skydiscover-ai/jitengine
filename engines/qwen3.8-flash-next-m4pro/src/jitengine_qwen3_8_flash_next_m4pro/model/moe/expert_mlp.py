"""The routed-expert MLP every MoE path ends in, as a mixin of `Qwen4ExpTextSparseMoeBlock`.

(token, expert) pairs run in canonical order (expert id, then position) through the grouped kernel
(`kernels.grouped_expert_qmv`) or stock `gather_qmm`, over the resident stack or the slot
pool's arrays (main or extension set), so a result never depends on pool size or slot assignment.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as _mk

_ARANGE_CACHE: dict = {}


# ----------------------------------------------------------------------------- fused elementwise
@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


class ExpertMLPMixin:
    """Kernel choice (`grouped_for`, `kernel_in_use`), the expert MLP over pairs in canonical order
    (`_expert_mlp_sorted`) and one wave's gather (`_gather_slots`).
    """

    def _gather_slots(
        self, xe: mx.array, k: int, slots: np.ndarray, inds_shape, experts: np.ndarray | None = None
    ) -> mx.array:
        """Gather one wave of (token, expert) pairs against the slot pool; the single definition.

        The result must not depend on pool size or slot assignment, so pairs are ordered canonically
        (expert id, then position). Stock mode uses `gather_qmm(sorted_indices=False)`: with True,
        MLX picks its kernel by pairs per weight-array expert (the slot count), and the two kernels
        are not bit-identical.
        """
        pos = np.arange(slots.size, dtype=np.int64)
        key = slots if experts is None else experts
        srt = np.argsort(key, kind="stable")
        pos, slots = pos[srt], slots[srt]
        y = self._expert_mlp_sorted(xe, pos, slots, key[srt], k)
        inv = np.empty_like(pos)
        inv[pos] = np.arange(pos.size, dtype=pos.dtype)
        return mx.unflatten(y[mx.array(inv)], 0, tuple(inds_shape))

    def grouped_for(self, dt) -> bool:
        """Whether the grouped kernel (not stock `gather_qmm`) serves this layer at dtype `dt`.

        Compact (pack16) slots have no bf16 metadata to give stock: they are the grouped kernel's.
        """
        if self.moe_kernel != "grouped":
            return False
        return all(
            _mk.supports(
                lin.bits,
                lin.group_size,
                lin.in_dims,
                lin.weight.dtype,
                mx.bfloat16 if _p16_lin(lin) else lin.scales.dtype,
                _p16_lin(lin) or lin.biases is not None,
                dt,
            )
            for lin in (self.gate_proj, self.up_proj, self.down_proj)
        )

    def kernel_in_use(self, dt) -> dict:
        """What `_expert_mlp_sorted` runs at dtype `dt`: kernel, variant, per-projection config."""
        if not self.grouped_for(dt):
            return {
                "kernel": "stock",
                "variant": None,
                "configs": {},
                "fallback": self.moe_kernel == "grouped",
            }
        cfgs = {}
        for name, lin in (
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ):
            cfgs[f"{name} {lin.out_dims}x{lin.in_dims} {_mk.dtype_name(dt)}"] = list(
                _mk.config_for(lin.out_dims, lin.in_dims, dt)
            )
        return {
            "kernel": "grouped",
            "variant": self.moe_kernel_variant,
            "configs": cfgs,
            "fallback": False,
            "codec": "pack16" if _p16_lin(self.gate_proj) else "bf16",
        }

    def _expert_mlp_sorted(  # noqa: PLR0915
        self,
        xe: mx.array,
        pos: np.ndarray,
        slots: np.ndarray,
        keys: np.ndarray,
        k: int,
        _lins=None,
        dep=None,
    ) -> mx.array:
        """Expert MLP for P pairs in canonical order: xe [N, 1, D] -> y [P, 1, D] in that order.

        `keys` are the ascending expert ids, `pos` the flat pair indices (`pos // k` is the
        activation row) and `slots` each pair's index into the weight arrays. On compact slots
        `dep` orders the reads after the encodes that wrote them: the pool's `encode_dep` unless
        the caller passes `pred_dep` for slots still landing.
        """
        pool = self.pool
        if dep is None and pool is not None:
            dep = pool.encode_dep
        extra = (
            pool.extra_sets()
            if (_lins is None and pool is not None and hasattr(pool, "extra_sets"))
            else []
        )
        if extra:
            # extension / annex sets: pairs in each further array set run the same row-independent
            # kernel over that set, so the split is exact; rows are put back in the caller's order
            firsts = np.asarray([lo for lo, _ in extra], dtype=np.int64)
            lins = [(self.gate_proj, self.up_proj, self.down_proj)] + [ls for _, ls in extra]
            base = np.concatenate([[0], firsts])
            which = np.searchsorted(firsts, slots, side="right")  # 0 main, 1.. the further sets
            parts = [(i, np.flatnonzero(which == i)) for i in range(len(lins))]
            parts = [(i, p) for i, p in parts if p.size]
            if len(parts) == 1:
                i = parts[0][0]
                return self._expert_mlp_sorted(
                    xe, pos, slots - base[i], keys, k, _lins=lins[i], dep=dep
                )
            ys = [
                self._expert_mlp_sorted(
                    xe, pos[p], slots[p] - base[i], keys[p], k, _lins=lins[i], dep=dep
                )
                for i, p in parts
            ]
            order = np.concatenate([p for _, p in parts])
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size, dtype=order.dtype)
            return mx.concatenate(ys, axis=0)[mx.array(inv)]
        if _lins is None:
            _lins = (self.gate_proj, self.up_proj, self.down_proj)
        if self.grouped_for(xe.dtype):
            g, u, d = _lins
            _uniq, start, cnt = np.unique(keys, return_index=True, return_counts=True)
            P = int(pos.size)
            mc = int(cnt.max())
            U = int(start.size)
            buf = np.empty(3 * U + P, dtype=np.uint32)
            buf[:U] = slots[start]
            buf[U : 2 * U] = start
            buf[2 * U : 3 * U] = cnt
            buf[3 * U :] = pos // k
            mbuf = mx.array(buf)
            seg_slot = mbuf[:U]
            seg_start = mbuf[U : 2 * U]
            seg_cnt = mbuf[2 * U : 3 * U]
            pair_row = mbuf[3 * U :]
            x2 = xe.reshape(-1, xe.shape[-1])
            dt = x2.dtype
            if _p16_lin(g):
                return self._expert_mlp_p16(
                    x2, (g, u, d), seg_slot, seg_start, seg_cnt, pair_row, P, mc, dep
                )
            if self.moe_kernel_variant == "fused":
                h = _mk.grouped_swiglu(
                    x2,
                    (g.weight, g.scales, g.biases),
                    (u.weight, u.scales, u.biases),
                    seg_slot,
                    seg_start,
                    seg_cnt,
                    pair_row,
                    P,
                    mc,
                    *_mk.config_for(g.out_dims, g.in_dims, dt),
                    group_size=g.group_size,
                )
            else:
                gg = _mk.grouped_qmv(
                    x2,
                    g.weight,
                    g.scales,
                    g.biases,
                    seg_slot,
                    seg_start,
                    seg_cnt,
                    pair_row,
                    P,
                    mc,
                    *_mk.config_for(g.out_dims, g.in_dims, dt),
                    group_size=g.group_size,
                )
                uu = _mk.grouped_qmv(
                    x2,
                    u.weight,
                    u.scales,
                    u.biases,
                    seg_slot,
                    seg_start,
                    seg_cnt,
                    pair_row,
                    P,
                    mc,
                    *_mk.config_for(u.out_dims, u.in_dims, dt),
                    group_size=u.group_size,
                )
                h = swiglu(gg, uu)
            ident = _ARANGE_CACHE.get(P)
            if ident is None:
                ident = mx.arange(P, dtype=mx.uint32)
                mx.eval(ident)
                _ARANGE_CACHE[P] = ident
            y = _mk.grouped_qmv(
                h,
                d.weight,
                d.scales,
                d.biases,
                seg_slot,
                seg_start,
                seg_cnt,
                ident,
                P,
                mc,
                *_mk.config_for(d.out_dims, d.in_dims, dt),
                group_size=d.group_size,
            )
            return y[:, None, :]
        if _p16_lin(_lins[0]):
            raise RuntimeError("compact (pack16) slots are served by the grouped kernel only")
        xw = xe[mx.array(pos // k)]
        idx = mx.array(slots.astype(np.uint32))
        g, u, d = _lins
        up = u.gather(xw, idx, sorted_indices=False)
        gate = g.gather(xw, idx, sorted_indices=False)
        return d.gather(swiglu(gate, up), idx, sorted_indices=False)

    def _expert_mlp_p16(self, x2, lins, seg_slot, seg_start, seg_cnt, pair_row, P, mc, dep):
        """The split grouped calls of `_expert_mlp_sorted` over compact slots; same launches."""
        g, u, d = lins
        dt = x2.dtype
        gg = _mk.grouped_qmv_p16(
            x2,
            g.weight,
            g.words,
            g.exc,
            dep,
            seg_slot,
            seg_start,
            seg_cnt,
            pair_row,
            P,
            mc,
            *_mk.config_for(g.out_dims, g.in_dims, dt),
            group_size=g.group_size,
        )
        uu = _mk.grouped_qmv_p16(
            x2,
            u.weight,
            u.words,
            u.exc,
            dep,
            seg_slot,
            seg_start,
            seg_cnt,
            pair_row,
            P,
            mc,
            *_mk.config_for(u.out_dims, u.in_dims, dt),
            group_size=u.group_size,
        )
        h = swiglu(gg, uu)
        ident = _ARANGE_CACHE.get(P)
        if ident is None:
            ident = mx.arange(P, dtype=mx.uint32)
            mx.eval(ident)
            _ARANGE_CACHE[P] = ident
        y = _mk.grouped_qmv_p16(
            h,
            d.weight,
            d.words,
            d.exc,
            dep,
            seg_slot,
            seg_start,
            seg_cnt,
            ident,
            P,
            mc,
            *_mk.config_for(d.out_dims, d.in_dims, dt),
            group_size=d.group_size,
        )
        return y[:, None, :]


def _p16_lin(lin) -> bool:
    """Whether `lin` is a projection over compact (pack16) slots."""
    return getattr(lin, "codec", None) == "pack16"
