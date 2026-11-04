"""The prefill path of `Qwen4ExpTextSparseMoeBlock`, as a mixin: the routed experts of a [B, S]
input.

`Qwen4ExpTextSparseMoeBlock.__call__` takes it for prefill, batched decode and unfused solo decode:
the resident stack, or the slot pool with its per-layer host sync, the gather of already-resident
pairs inside the fetch window, and capacity waves when a layer's experts exceed the pool
(`_experts_pooled`). Solo decode's non-speculative combine calls `_experts_pooled` too.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import swiglu

_ROWOF_CACHE: dict = {}


def _inverse(order: np.ndarray) -> np.ndarray:
    """The permutation that puts `order` back into ascending position order."""
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size, dtype=order.dtype)
    return inv


def _require_unmoved(slots: np.ndarray, resident: np.ndarray, pos: np.ndarray, why: str) -> None:
    """Raise InvariantError unless each pre-gathered pair is still in the slot it was gathered from;
    otherwise the overlapped gather read another expert's weights."""
    if not np.array_equal(slots[pos], resident[pos]):
        raise InvariantError(why)


class PrefillMixin:
    """The routed experts of a [B, S] input: `experts`, and `_experts_pooled` against the pool."""

    def experts(self, x: mx.array, inds: mx.array, telem=None, overlap=None) -> mx.array:
        """x [B, S, D], inds [B, S, k] -> [B, S, k, D], every selected expert applied.

        `overlap` is lazy work independent of this layer's fetch, which the pool dispatches while
        the reads are in flight; ignored without a pool.
        """
        if self.pool is not None:
            return self._experts_pooled(x, inds, telem, overlap=overlap)
        if self.grouped_for(x.dtype):
            # the resident stack is a pool whose slot id is the expert id: same canonical order and
            # kernel as `_gather_slots`, so resident == pooled bit for bit
            flat = np.asarray(inds).astype(np.int64, copy=False).reshape(-1)
            xe = mx.expand_dims(x, (-2, -3)).flatten(0, -3)
            y = self._gather_slots(xe, inds.shape[-1], flat, inds.shape, experts=flat)
            if telem is not None and telem.collect:
                telem.note(int(flat.size), int(np.unique(flat).size))
            return y.squeeze(-2)
        xe = mx.expand_dims(x, (-2, -3))  # [B, S, 1, 1, D]
        do_sort = inds.shape[0] > 1 or inds.size >= 64
        if do_sort:
            k = inds.shape[-1]
            flat = inds.flatten()
            order = mx.argsort(flat)
            inv_order = mx.argsort(order)
            xe = xe.flatten(0, -3)[order // k]  # [N*k, 1, D]
            idx = flat[order]
        else:
            idx = inds
        up = self.up_proj.gather(xe, idx, sorted_indices=do_sort)
        gate = self.gate_proj.gather(xe, idx, sorted_indices=do_sort)
        y = self.down_proj.gather(swiglu(gate, up), idx, sorted_indices=do_sort)
        if do_sort:
            y = mx.unflatten(y[inv_order], 0, inds.shape)
        if telem is not None and telem.collect:
            f = idx.flatten() if do_sort else mx.sort(inds.flatten())
            telem.note(int(f.size), 1 + (f[1:] != f[:-1]).sum())
        return y.squeeze(-2)

    def _experts_pooled(
        self, x: mx.array, inds: mx.array, telem=None, overlap=None, parts: bool = False
    ):
        """`experts` against the slot pool, plus the per-layer host sync that lets the pool fetch.

        That sync (`np.asarray(inds)`) also means every kernel of the previous layer has completed,
        which is why `reader_barrier()` may release that layer's slots right after it.
        """
        inds_np = np.asarray(inds).astype(np.int64, copy=False)  # the per-layer host sync
        self.pool.reader_barrier()  # ... and therefore a reader barrier
        k = inds.shape[-1]
        flat = inds_np.reshape(-1)
        row_of = self._row_of_pairs(inds, flat)
        xe = mx.expand_dims(x, (-2, -3)).flatten(0, -3)  # [N, 1, D]

        # Pre-gather: the already-resident pairs depend on nothing this fetch writes, so their
        # gather is built now and dispatched inside the read window. Splitting pairs across calls
        # is exact because the kernel is row-independent, and results are put back in the original
        # pair order before the score-weighted sum.
        pre_pos = pre_y = None
        if len(x.shape) and flat.size:
            pre_pos, pre_y, overlap, resident = self._pregather_resident(
                xe, flat, k, pre_pos, pre_y, overlap
            )

        segments = self.pool.segments(self.layer_idx, flat, row_of, overlap=overlap)
        if parts and len(segments) == 1:
            self._note_unique_experts(telem, flat)
            if pre_pos is not None and pre_pos.size == flat.size:
                _require_unmoved(
                    segments[0][1],
                    resident,
                    pre_pos,
                    f"all-resident layer {self.layer_idx} moved slots inside its own ensure()",
                )
                return [(pre_y, pre_pos)]
            if pre_pos is not None:
                rest, slots_all = self._split_pregathered(flat, segments, resident, pre_pos)
                return [
                    (pre_y, pre_pos),
                    (self._expert_mlp_sorted(xe, rest, slots_all[rest], flat[rest], k), rest),
                ]
            srt = np.argsort(flat, kind="stable")
            return [(self._expert_mlp_sorted(xe, srt, segments[0][1][srt], flat[srt], k), srt)]
        if parts:
            return None  # the multi-segment path keeps its per-wave evals; caller falls back
        if len(segments) == 1:
            if pre_pos is not None and pre_pos.size == flat.size:
                # no miss, so `ensure` evicted nothing and every slot must be where it was
                _require_unmoved(
                    segments[0][1],
                    resident,
                    pre_pos,
                    f"all-resident layer {self.layer_idx} moved slots inside its own ensure()",
                )
                y = mx.unflatten(pre_y[mx.array(_inverse(pre_pos))], 0, tuple(inds.shape))
            elif pre_pos is not None:
                rest, slots_all = self._split_pregathered(flat, segments, resident, pre_pos)
                y, order = self._gather_rest_in_order(xe, rest, slots_all, flat, k, pre_pos, pre_y)
                y = self._restore_pair_order(y, order, inds)
            else:
                y = self._gather_slots(xe, k, segments[0][1], inds.shape, experts=flat)
            self._note_unique_experts(telem, flat)
            self._sync_layer(y)
            return y.squeeze(-2)
        outs, order = [], []
        for pos, slots in segments:
            srt = np.argsort(flat[pos], kind="stable")  # canonical: by expert id
            pos, slots = pos[srt], slots[srt]
            y = self._expert_mlp_sorted(xe, pos, slots, flat[pos], k)
            mx.eval(y)  # this wave must finish before the next fetch reuses its slots
            self.pool.reader_barrier()  # ... which is exactly the barrier that releases them
            outs.append(y)
            order.append(pos)
        y, order = self._join_segments(outs, order)
        y = self._restore_pair_order(y, order, inds)
        self._note_unique_experts(telem, flat)
        self._sync_layer(y)
        return y.squeeze(-2)

    def _row_of_pairs(self, inds: mx.array, flat: np.ndarray) -> np.ndarray:
        """Which unit row asked for each (token, expert) pair, memoised by shape."""
        # which unit row asked for each pair: [B, S, k] -> row index (the fault seam's attribution)
        _rk = (int(inds.shape[0]), int(flat.size))
        row_of = _ROWOF_CACHE.get(_rk)
        if row_of is None:
            row_of = np.repeat(
                np.arange(inds.shape[0], dtype=np.int64), flat.size // max(1, inds.shape[0])
            )
            _ROWOF_CACHE[_rk] = row_of
        return row_of

    def _pregather_resident(
        self, xe: mx.array, flat: np.ndarray, k: int, pre_pos, pre_y, overlap
    ) -> tuple:
        """Build the gather of the already-resident pairs (all of them, or the resident part when some miss)."""
        resident = self.pool.resident_slots(self.layer_idx, flat)
        hp = np.flatnonzero(resident >= 0)
        if 0 < hp.size < flat.size:
            hp = hp[np.argsort(flat[hp], kind="stable")]  # canonical: by expert id
            pre_y = self._expert_mlp_sorted(xe, hp, resident[hp], flat[hp], k)
            pre_pos = hp
            overlap = tuple(overlap or ()) + (pre_y,)
        elif hp.size == flat.size:
            # all experts resident, so nothing is fetched: dispatch the gather now so the GPU
            # works during the host's `segments` bookkeeping (the router sync is already past)
            hp = np.argsort(flat, kind="stable")
            pre_y = self._expert_mlp_sorted(xe, hp, resident[hp], flat[hp], k)
            pre_pos = hp
            # `ensure` skips `overlap` on a no-miss layer, so submit it with the gather
            mx.async_eval(pre_y, *(overlap or ()))
        return pre_pos, pre_y, overlap, resident

    def _note_unique_experts(self, telem, flat: np.ndarray) -> None:
        """Note the layer's pair and distinct-expert counts when telemetry collects."""
        # return the pieces unassembled, each with its rows' original pair indices, for
        # `moe_combine` to read in place
        if telem is not None and telem.collect:
            telem.note(int(flat.size), int(np.unique(flat).size))

    def _sync_layer(self, y: mx.array) -> None:
        """Evaluate `y` now when the layer syncs each step."""
        if self.sync_each_layer:
            mx.eval(y)

    def _split_pregathered(
        self, flat: np.ndarray, segments: list, resident: np.ndarray, pre_pos: np.ndarray
    ) -> tuple:
        """The pairs not pre-gathered, in expert order, and the slots of the layer; checks the pre-gathered
        pairs did not move."""
        rest = np.setdiff1d(np.arange(flat.size, dtype=np.int64), pre_pos, assume_unique=False)
        rest = rest[np.argsort(flat[rest], kind="stable")]
        slots_all = segments[0][1]
        # a pair resident when the pre-gather was built must still be in the same slot,
        # or the overlapped gather read an evicted slot
        _require_unmoved(
            slots_all,
            resident,
            pre_pos,
            f"pre-gathered pairs moved slots during their own fetch window: layer {self.layer_idx}",
        )
        return rest, slots_all

    def _gather_rest_in_order(
        self,
        xe: mx.array,
        rest: np.ndarray,
        slots_all: np.ndarray,
        flat: np.ndarray,
        k: int,
        pre_pos: np.ndarray,
        pre_y: mx.array,
    ) -> tuple:
        """Gather the rest after the pre-gathered pairs; returns all pairs' outputs and their pair order."""
        y_rest = self._expert_mlp_sorted(xe, rest, slots_all[rest], flat[rest], k)
        order = np.concatenate([pre_pos, rest])
        y = mx.concatenate([pre_y, y_rest], axis=0)
        return y, order

    def _restore_pair_order(self, y: mx.array, order: np.ndarray, inds: mx.array) -> mx.array:
        """Put the rows of `y` back into the original pair order and shape them as `inds`."""
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)
        y = mx.unflatten(y[mx.array(inv)], 0, tuple(inds.shape))
        return y

    def _join_segments(self, outs: list, order: list) -> tuple:
        """Concatenate the waves' outputs and their pair positions."""
        y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        order = np.concatenate(order) if len(order) > 1 else order[0]
        return y, order
