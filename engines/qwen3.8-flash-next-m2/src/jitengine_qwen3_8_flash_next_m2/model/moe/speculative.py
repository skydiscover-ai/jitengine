"""The speculative gather of `Qwen4ExpTextSparseMoeBlock`, as a mixin.

The routed experts are gathered before the router's host sync, with device-side slot
lookups that assume every expert is resident (`spec_build`); the settle syncs, issues the
misses' reads and recomputes only the pairs whose slot moved (`spec_settle_begin`,
`_correction`, `spec_settle_finish`).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.kernels import grouped_expert_qmv as _mk
from jitengine_qwen3_8_flash_next_m2.kernels import router_combine as _mf
from jitengine_qwen3_8_flash_next_m2.model.common import _DEFER, _ROWOF_CACHE, _dispatch
from jitengine_qwen3_8_flash_next_m2.model.moe.expert_mlp import swiglu


def _chain_pred(chains: list, base: int):
    """(gids, weights, ranks) of the miss chain's prediction, one entry per (row, rank).

    Weights are the speculative router's renormalised top-k scores, synced with the indices, so
    reading them adds no GPU wait; a record without scores gets weight 1.0 in index order. They
    feed `OffloadMoeCache.prefetch`'s outcome counters by rank and weight.
    """
    gs, ws, rs = [], [], []
    for c in chains:
        ind = np.asarray(c["inds"]).astype(np.int64).reshape(-1)
        sc = c.get("scores")
        if sc is None:
            w = np.ones(ind.size)
        else:
            w = np.asarray(sc).astype(np.float64).reshape(-1)
            if w.size != ind.size:
                w = np.ones(ind.size)
        order = np.argsort(-w, kind="stable")
        rank = np.empty(ind.size, dtype=np.int64)
        rank[order] = np.arange(ind.size)
        gs.append(ind + base)
        ws.append(w)
        rs.append(rank)
    return np.concatenate(gs), np.concatenate(ws), np.concatenate(rs)


_SPEC_CONST: dict = {}


class SpeculativeGatherMixin:
    """Build before the router's host sync, settle after it: `spec_build` and `spec_settle*`."""

    def _spec_consts(self, k: int):
        c = _SPEC_CONST.get(k)
        if c is None:
            ar = mx.arange(k, dtype=mx.uint32)
            ones = mx.ones((k,), dtype=mx.uint32)
            zeros = mx.zeros((k,), dtype=mx.uint32)
            mx.eval(ar, ones, zeros)
            c = _SPEC_CONST[k] = (
                ar,
                ones,
                zeros,
                np.arange(k, dtype=np.int64),
                np.zeros(k, dtype=np.int64),
            )
        return c

    def _gather_dev(self, x2: mx.array, slots_dev: mx.array, k: int) -> mx.array:
        """k routed pairs of one row, slots resolved on device (-1 = not resident, output unused).

        Each pair is its own segment, so no host sort is needed; the row-independent kernel gives
        the same values as the canonical-order call.
        """
        pool = self.pool
        ar, ones, zeros, _, _ = self._spec_consts(k)
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        ext_active = bool(getattr(pool, "ext_slots", 0)) and pool.ext_active
        main = int(pool.slots_main) if ext_active else (1 << 30)
        eg = (
            (pool.gate_proj_ext.weight, pool.gate_proj_ext.scales, pool.gate_proj_ext.biases)
            if ext_active
            else None
        )
        eu = (
            (pool.up_proj_ext.weight, pool.up_proj_ext.scales, pool.up_proj_ext.biases)
            if ext_active
            else None
        )
        ed = (
            (pool.down_proj_ext.weight, pool.down_proj_ext.scales, pool.down_proj_ext.biases)
            if ext_active
            else None
        )
        dt = x2.dtype
        gg = _mk.grouped_qmv_x(
            x2,
            g.weight,
            g.scales,
            g.biases,
            eg,
            main,
            slots_dev,
            ar,
            ones,
            zeros,
            k,
            1,
            *_mk.config_for(g.out_dims, g.in_dims, dt),
            group_size=g.group_size,
        )
        uu = _mk.grouped_qmv_x(
            x2,
            u.weight,
            u.scales,
            u.biases,
            eu,
            main,
            slots_dev,
            ar,
            ones,
            zeros,
            k,
            1,
            *_mk.config_for(u.out_dims, u.in_dims, dt),
            group_size=u.group_size,
        )
        h = swiglu(gg, uu)
        return _mk.grouped_qmv_x(
            h,
            d.weight,
            d.scales,
            d.biases,
            ed,
            main,
            slots_dev,
            ar,
            ones,
            ar,
            k,
            1,
            *_mk.config_for(d.out_dims, d.in_dims, dt),
            group_size=d.group_size,
        )

    def _off_arr(self) -> mx.array:
        a = getattr(self, "_off_a", None)
        if a is None:
            a = mx.array([int(self.layer_idx * self.pool.num_experts)], dtype=mx.int32)
            mx.eval(a)
            self._off_a = a
        return a

    def _spec_decode(self, x, inds, scores, sh, shg, h2, inject, telem):
        """Speculative gather: dispatched before the router's host sync with device-side slot
        lookups, assuming every expert is resident; a miss recomputes only the missed pairs.

        Returns h3 [1, 1, hc*hid].
        """
        rec = self.spec_build(x, inds, scores, sh, shg, h2, inject, telem)
        return self.spec_settle(rec)[0]

    def spec_build(self, x, inds, scores, sh, shg, h2, inject, telem) -> dict:
        """Dispatch the router, then the speculative all-hit gather + combine; never waits."""
        pool, k, E = self.pool, self.top_k, self.pool.num_experts
        _, _, _, ar_np, _ = self._spec_consts(k)
        x2 = x.reshape(-1, x.shape[-1])
        # two command buffers: the router's is committed first so the host sync waits for it
        # alone, not for the speculative gather + combine behind it
        _dispatch(inds, scores)
        if pool._inflight and _DEFER[0] is None:
            pool.harvest_ready()
        tab = pool.device_slot_table()
        snap = pool.layer_snapshot(self.layer_idx)
        slots_dev = mx.take(tab, inds + int(self.layer_idx * E))
        y_spec = self._gather_dev(x2, slots_dev, k)  # [k, hid], pair order
        # a non-resident pair weighs zero, so the speculative residual is the true one minus the
        # missing experts; all-resident and corrected combines use the router's weights
        sc = mx.where(slots_dev >= 0, scores, mx.zeros_like(scores))
        h3 = _mf.moe_combine(
            y_spec, None, ar_np, sc, sh, h2, inject, self.hc, self.hid, sh_gate=shg
        )
        _dispatch(h3)
        h3 = h3.reshape(h2.shape)
        return {
            "x2": x2,
            "inds": inds,
            "scores": scores,
            "sh": sh,
            "shg": shg,
            "h2": h2,
            "inject": inject,
            "y_spec": y_spec,
            "h3": h3,
            "telem": telem,
            "snap": snap,
        }

    def spec_build_pair(self, xs, inds_l, scores_l, sh_l, shg_l, h2_l, inj_l) -> list:
        """`spec_build` for two rows at once.

        One router dispatch, one device table read and generation snapshot serve both rows, and the
        2k pairs are gathered by `grouped_qmv_u2` (each expert's weights read once, every pair
        bit-identical to `_gather_dev`'s). Each row then gets its own combine, so its record is
        `spec_build`'s.
        """
        pool, k, E = self.pool, self.top_k, self.pool.num_experts
        _, _, _, ar_np, _ = self._spec_consts(k)
        _dispatch(inds_l[0], scores_l[0], inds_l[1], scores_l[1])
        if pool._inflight and _DEFER[0] is None:
            pool.harvest_ready()
        tab = pool.device_slot_table()
        snap = pool.layer_snapshot(self.layer_idx)
        x2 = mx.concatenate([x.reshape(-1, x.shape[-1]) for x in xs], axis=0)  # [2, hid]
        inds2 = mx.concatenate([inds_l[0].reshape(-1), inds_l[1].reshape(-1)], axis=0)
        off = self._off_arr()
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        ext_active = bool(getattr(pool, "ext_slots", 0)) and pool.ext_active
        main = int(pool.slots_main) if ext_active else (1 << 30)
        eg = (
            (pool.gate_proj_ext.weight, pool.gate_proj_ext.scales, pool.gate_proj_ext.biases)
            if ext_active
            else None
        )
        eu = (
            (pool.up_proj_ext.weight, pool.up_proj_ext.scales, pool.up_proj_ext.biases)
            if ext_active
            else None
        )
        ed = (
            (pool.down_proj_ext.weight, pool.down_proj_ext.scales, pool.down_proj_ext.biases)
            if ext_active
            else None
        )
        dt = x2.dtype

        def cfg2(lin):
            kl, nsg, rb = _mk.config_for(lin.out_dims, lin.in_dims, dt)[1:]
            return dict(vecs=2, k_lanes=kl, nsg=nsg, rb=rb, group_size=lin.group_size)

        gg = _mk.grouped_qmv_u2(
            x2, g.weight, g.scales, g.biases, eg, main, inds2, tab, off, k, False, **cfg2(g)
        )
        uu = _mk.grouped_qmv_u2(
            x2, u.weight, u.scales, u.biases, eu, main, inds2, tab, off, k, False, **cfg2(u)
        )
        hmid = swiglu(gg, uu)
        y_all = _mk.grouped_qmv_u2(
            hmid, d.weight, d.scales, d.biases, ed, main, inds2, tab, off, k, True, **cfg2(d)
        )
        recs, h3s = [], []
        for r in range(2):
            y_spec = y_all[r * k : (r + 1) * k]
            slots_dev = mx.take(tab, inds_l[r] + int(self.layer_idx * E))
            sc = mx.where(slots_dev >= 0, scores_l[r], mx.zeros_like(scores_l[r]))
            h3 = _mf.moe_combine(
                y_spec,
                None,
                ar_np,
                sc,
                sh_l[r],
                h2_l[r],
                inj_l[r],
                self.hc,
                self.hid,
                sh_gate=shg_l[r],
            )
            h3s.append(h3)
            recs.append(
                {
                    "x2": xs[r].reshape(-1, xs[r].shape[-1]),
                    "inds": inds_l[r],
                    "scores": scores_l[r],
                    "sh": sh_l[r],
                    "shg": shg_l[r],
                    "h2": h2_l[r],
                    "inject": inj_l[r],
                    "y_spec": y_spec,
                    "h3": h3.reshape(h2_l[r].shape),
                    "telem": None,
                    "pred": None,
                    "snap": snap,
                }
            )
        _dispatch(*h3s)
        return recs

    def spec_settle_pair_begin(self, recs: list, chains: list | None = None) -> dict:
        """Start `spec_settle` for the two positions' records of one layer at once.

        One `segments_begin` covers both rows' 2k pairs (the pool dedups an expert both rows routed
        to) and the miss chain covers the union of both rows' next-layer routers; each row then runs
        its own solo `_correction` from its half of `pred_slots`. Pair with
        `spec_settle_pair_finish`, which returns [(h3_0, missed_0), (h3_1, missed_1)].
        """
        pool, k, E = self.pool, self.top_k, self.pool.num_experts
        flats = [np.asarray(r["inds"]).astype(np.int64, copy=False).reshape(-1) for r in recs]
        flat = np.concatenate(flats)
        gids = flat + self.layer_idx * E
        missing = pool.slot_for_id[gids] < 0
        during = None
        streak = bool(chains) and bool(missing.any())
        pool.chain_last = None
        if streak:
            nl = self.layer_idx + 1

            def during():
                nxt = (
                    np.unique(
                        np.concatenate(
                            [np.asarray(c["inds"]).astype(np.int64).reshape(-1) for c in chains]
                        )
                    )
                    + nl * E
                )
                # the chain's predicted weights and ranks, for the outcome counters
                g_all, w_all, r_all = _chain_pred(chains, nl * E)
                pool.prefetch(g_all, w_all, r_all)
                for r in recs:
                    r["chained"] = nxt
                pool.chain_last = nxt

        row_of = _ROWOF_CACHE.get(("pair", k))
        if row_of is None:
            row_of = _ROWOF_CACHE.setdefault(("pair", k), np.zeros(2 * k, dtype=np.int64))
        sp = pool.segments_begin(self.layer_idx, flat, row_of, during=during)
        if streak and pool.chain_last is None and not (sp.get("pend") or {}).get("during"):
            during()
        telem = recs[0].get("telem")
        if telem is not None and telem.collect:
            telem.note(int(flat.size), int(np.unique(flat).size))
        slots_all = sp["pred_slots"]
        if slots_all is None:
            pool.segments_finish(sp)
            raise RuntimeError(
                "speculative gather: a two-row top-k set exceeded the pool's wave capacity"
            )
        outs = [
            self._correction(
                recs[r], flats[r], slots_all[r * k : (r + 1) * k], missing[r * k : (r + 1) * k]
            )
            for r in range(2)
        ]
        return {"sp": sp, "outs": outs, "recs": recs, "flats": flats, "pred": slots_all}

    def spec_settle_pair_finish(self, ctx: dict) -> list:
        """Wait for the pair's reads and publish them. `ctx["pred_ok"]` is False when a record did not
        land where begin said: the corrected rows are rebuilt from the final slots, and anything
        built on the first corrections must be discarded."""
        k = self.top_k
        segs = self.pool.segments_finish(ctx["sp"])
        final = segs[0][1]
        outs = ctx["outs"]
        if not np.array_equal(final, ctx["pred"]):
            outs = [
                (
                    self._correction(
                        ctx["recs"][r],
                        ctx["flats"][r],
                        final[r * k : (r + 1) * k],
                        np.ones(k, dtype=bool),
                    )[0],
                    True,
                )
                if outs[r][1]
                else outs[r]
                for r in range(2)
            ]
            ctx["pred_ok"] = False
        return outs

    def spec_settle(self, rec: dict, chain: dict | None = None):
        """`spec_settle_begin` then `spec_settle_finish`: (h3, missed)."""
        return self.spec_settle_finish(self.spec_settle_begin(rec, chain))

    def spec_settle_begin(self, rec: dict, chain: dict | None = None) -> dict:
        """Host sync, pool accounting and fetch issue for one layer; returns the settle context.

        The context's h3 is the speculative one if every expert was resident, else a corrected
        combine. Eviction is safe with a lookahead layer in flight: evictions happen only here,
        after a miss, which always discards the lookahead layer; this layer's own slots are
        protected as hits, and a victim is rewritten only by a fetch issued after a later router
        sync, when every earlier command buffer has completed (one in-order GPU queue).
        """
        pool, k, E = self.pool, self.top_k, self.pool.num_experts
        _, _, _, _ar_np, zeros_np = self._spec_consts(k)
        flat = np.asarray(rec["inds"]).astype(np.int64, copy=False).reshape(-1)  # the host sync
        gids = flat + self.layer_idx * E
        missing = pool.slot_for_id[gids] < 0
        during = None
        streak = False
        if chain is not None:
            # miss chaining: while this layer's reads are in flight, prefetch into reserve slots
            # the non-resident experts the speculative layer i+1's router names (never an eviction)
            streak = bool(missing.any())
        pool.chain_last = None
        if streak:
            nl = self.layer_idx + 1

            def during():
                nxt = (
                    np.asarray(chain["inds"]).astype(np.int64).reshape(-1) + nl * E
                )  # waits for i+1's router
                # the chain's predicted weights and ranks, for the outcome counters
                _g, _w, _r = _chain_pred([chain], nl * E)
                pool.prefetch(_g, _w, _r)
                pool.chain_last = nxt

        sp = pool.segments_begin(
            self.layer_idx, flat, zeros_np, during=during
        )  # accounting + the fetch (issued)
        if streak and pool.chain_last is None and not (sp.get("pend") or {}).get("during"):
            during()  # no read of its own to hide behind: issue the next link now
        telem = rec["telem"]
        if telem is not None and telem.collect:
            telem.note(int(flat.size), int(np.unique(flat).size))
        slots_all = sp["pred_slots"]
        if slots_all is None:
            pool.segments_finish(sp)
            raise RuntimeError("speculative gather: a top-k set exceeded the pool's wave capacity")
        h3, missed = self._correction(rec, flat, slots_all, missing)
        return {"sp": sp, "h3": h3, "missed": missed, "rec": rec, "flat": flat, "pred": slots_all}

    def _correction(self, rec: dict, flat: np.ndarray, slots_all: np.ndarray, missing: np.ndarray):
        """(h3, corrected): the speculative h3, or a combine recomputing every pair that moved.

        Built but not dispatched, since the slots may still be landing; the caller dispatches after
        `spec_settle_finish`.
        """
        pool, k = self.pool, self.top_k
        _, _, _, ar_np, _ = self._spec_consts(k)
        snap = rec.get("snap")
        if snap is not None:
            # recompute every pair whose slot or slot generation moved since the build read the
            # table (a miss or a prefetch remap); a missing record's slot is its landing slot
            s0, g0 = snap
            demand = missing
            missing = (s0[flat] != slots_all) | (pool.gen[slots_all] != g0[flat])
            if missing.any():
                pool.spec_redo_layers += 1
                if not demand.any():
                    pool.spec_stale_layers += 1  # redone with no read of its own to hide behind
        if not missing.any():
            return rec["h3"], False
        miss = np.flatnonzero(missing)
        miss = miss[np.argsort(flat[miss], kind="stable")]
        xe = rec["x2"][:, None, :]
        y_miss = self._expert_mlp_sorted(xe, miss, slots_all[miss], flat[miss], k)  # [m, 1, hid]
        sel = ar_np.copy()
        sel[miss] = k + np.arange(miss.size, dtype=np.int64)
        h3 = _mf.moe_combine(
            rec["y_spec"],
            y_miss.reshape(-1, self.hid),
            sel,
            rec["scores"],
            rec["sh"],
            rec["h2"],
            rec["inject"],
            self.hc,
            self.hid,
            sh_gate=rec["shg"],
        )
        return h3.reshape(rec["h2"].shape), True

    def spec_settle_finish(self, ctx: dict):
        """Wait for the layer's reads and publish them; returns (h3, missed).

        If a record landed elsewhere than predicted, the correction is rebuilt and `ctx["pred_ok"]`
        is set False: anything built on the first correction must be discarded.
        """
        if "sp" in ctx:
            segs = self.pool.segments_finish(ctx["sp"])
            if ctx.get("missed") and "flat" in ctx and not np.array_equal(segs[0][1], ctx["pred"]):
                ctx["h3"], _ = self._correction(
                    ctx["rec"], ctx["flat"], segs[0][1], np.ones(ctx["flat"].size, dtype=bool)
                )
                ctx["pred_ok"] = False
        return ctx["h3"], ctx["missed"]
