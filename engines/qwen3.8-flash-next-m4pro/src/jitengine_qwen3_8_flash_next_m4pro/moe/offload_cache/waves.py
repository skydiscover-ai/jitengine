"""Wave planning for `OffloadMoeCache`, as a mixin: how a layer's (token, expert) pairs are fetched.

`wave_capacity` bounds the distinct experts one gather may use; `segments_multi` fetches the
union of a wave's rows at once (`can_wave`), and `segments_begin` and `segments_finish` split a
decode row's fetch around its reads, with the fault seam and transient flag these share with
`OffloadMoeCache.segments` (which stays in offload_cache.py with the `_Waves` it returns).
"""

from __future__ import annotations

import numpy as np


class WavePlanningMixin:
    """Capacity and the multi-row and split entry points of the pool's fetch."""

    @property
    def wave_capacity(self) -> int:
        """Distinct experts one gather may use: slots minus reserve, landing and released slots."""
        return (
            self.slots
            - self.reserve_n
            - self.landing_n
            - (0 if self.ext_active else self.ext_slots)
            - (self.ann_slots - self.ann_live)
        )

    # ------------------------------------------------------------------ the MoE entry point
    def _check_seam(self, layer: int, gids: np.ndarray, row_of) -> None:
        """Fault seam: fire for a record this call would fetch, before anything is charged."""
        if self.seam is None or not self.seam.armed:
            return
        seam_layer, e = self.seam.key
        if int(seam_layer) != int(layer):
            return
        gid = int(seam_layer) * self.num_experts + int(e)
        if self.slot_for_id[gid] >= 0:
            return  # resident: no read happens, nothing fires
        self.seam.check("expert_record", gids, row_of, self.owners, key=gid)

    def segments_begin(self, layer: int, expert_ids: np.ndarray, row_of=None, during=None) -> dict:
        """Begin `segments` for one decode row, its reads left in flight (`ensure(split=True)`)."""
        gids = expert_ids.astype(np.int64) + layer * self.num_experts
        uniq = np.unique(gids)
        if uniq.size > self.wave_capacity:
            segs = self.segments(layer, expert_ids, row_of, during=during)
            return {"done": segs, "pred_slots": segs[0][1] if len(segs) == 1 else None}
        self._check_seam(layer, gids, row_of)
        idx = np.searchsorted(uniq, gids)
        pend = self.ensure(
            uniq,
            transient=self._transient_now(),
            during=during,
            split=True,
            lookups_add=int(gids.size),
            dedup_add=int(gids.size) - int(uniq.size),
        )
        return {
            "pend": pend,
            "idx": idx,
            "n": int(gids.size),
            "pred_slots": pend["pred_slots"][idx],
        }

    def segments_finish(self, sp: dict) -> list:
        if "done" in sp:
            return sp["done"]
        return [(np.arange(sp["n"], dtype=np.int64), self.ensure_finish(sp["pend"])[sp["idx"]])]

    def _transient_now(self) -> bool:
        return self.transient_prefill and self._prefill_scope

    def segments_multi(self, layer: int, expert_ids_per_row: list, overlap=None) -> list:
        """One layer, N rows: one `ensure` for the union of every row's (token, expert) pairs.

        Returns one slot array per row, aligned with that row's flat expert ids. Safe because the
        caller (`Qwen4ExpTextDecoderLayer.wave`) materialises all N rows' router indices first,
        which forces every gather of the previous layer to complete. Raises if the union exceeds
        `wave_capacity`.
        """
        gid_rows = [
            np.asarray(f, dtype=np.int64) + layer * self.num_experts for f in expert_ids_per_row
        ]
        allg = gid_rows[0] if len(gid_rows) == 1 else np.concatenate(gid_rows)
        uniq = np.unique(allg)
        if self.seam is not None and self.seam.armed:
            row_of = np.concatenate(
                [np.full(g.size, r, dtype=np.int64) for r, g in enumerate(gid_rows)]
            )
            self._check_seam(layer, allg, row_of)
        if uniq.size > self.wave_capacity:
            raise RuntimeError(
                f"wave needs {uniq.size} distinct records of layer {layer} resident at once but the "
                f"pool holds {self.wave_capacity}; `can_wave` should have prevented this wave"
            )
        with self._lock:
            if len(gid_rows) > 1:
                self.wave_multi_calls += 1
        slots = self.ensure(
            uniq,
            transient=self._transient_now(),
            overlap=overlap,
            lookups_add=int(allg.size),
            dedup_add=int(allg.size) - int(uniq.size),
        )
        return [slots[np.searchsorted(uniq, g)] for g in gid_rows]

    @property
    def can_wave(self) -> bool:
        """True when one layer's whole expert set fits in `wave_capacity`, as a wave needs."""
        return self.wave_capacity >= self.num_experts
