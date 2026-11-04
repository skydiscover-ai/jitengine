"""Victim selection for `OffloadMoeCache`, as a mixin: which slot a missing record takes.

Free slots go first, then victims by the pool's policy (plain LRU, decayed use, or
transient prefill records first), never a reserved, disabled or pending-reader slot
(`_reserve`); `_decay` ages the use counts the policy reads, and `reader_barrier` and
`_note_reader` keep the pending-reader invariant.
"""

from __future__ import annotations

import numpy as np

_VICTIM_SENTINEL = np.iinfo(np.int64).max


def _lru_victims(last, protected, k: int, key, idx):
    """The `k` least-recently-used unprotected slots, in O(slots).

    Same slots in the same order as `cand[np.argsort(last[cand], kind="stable")[:k]]` with
    `cand = np.flatnonzero(~protected)`: the key `last * slots + slot` (exact while the clock is
    below 2^63 / slots) carries the stable tie rule, `argpartition` finds the k smallest, and only
    those k are sorted.
    """
    n = last.shape[0]
    np.multiply(last, n, out=key)
    np.add(key, idx, out=key)
    key[protected] = _VICTIM_SENTINEL
    sel = np.argpartition(key, k - 1)[:k] if k < n else np.arange(n)
    return sel[np.argsort(key[sel], kind="stable")]


def _decay_victims(use, last, protected, k: int):
    """The `k` slots a decayed-use pool evicts: least decayed use, then recency, then slot index.

    Same slots in the same order as `cand[np.lexsort((last[cand], use[cand]))[:k]]` without sorting
    every slot. `use` is non-negative, so its float32 bits read as uint32 are an exact order key.
    The caller guarantees at least k unprotected slots, so `thresh` is never the sentinel.
    """
    n = use.shape[0]
    if k >= n:
        cand = np.flatnonzero(~protected)
        return cand[np.lexsort((last[cand], use[cand]))[:k]]
    u = use.view(np.uint32).astype(np.int64)
    u[protected] = _VICTIM_SENTINEL
    if k == 1:
        # one victim (the harvest remap's call): the minimum key, then the least `last` among the slots
        # holding it; `argmin` returns the first (lowest-index) of equal `last` values, lexsort's order
        # without the partition.
        tie = np.flatnonzero(u == u.min())
        return tie[np.argmin(last[tie])].reshape(1)
    sel = np.argpartition(u, k - 1)[:k]
    thresh = int(u[sel].max())
    tie = np.flatnonzero(u <= thresh)
    order = np.lexsort((tie, last[tie], u[tie]))
    return tie[order[:k]]


DECAY_FACTOR = 0.5


class VictimSelectionMixin:
    """Which slots a fetch takes and which it must not: `_reserve`, `_decay`, the reader barrier."""

    def _decay(self) -> None:
        self.use *= DECAY_FACTOR

    # ------------------------------------------------------------ the pending-reader invariant
    def reader_barrier(self) -> None:
        """Clear pending-reader protection: slots handed out so far may be evicted again.

        Call only right after a host sync that drains the in-order GPU stream (`np.asarray(inds)` at
        the top of a layer, or an `mx.eval` of its output); otherwise an in-flight gather's slot
        could be overwritten.
        """
        if self._pending_reader_n:
            self._pending_reader[:] = False
            self._pending_reader_n = 0

    def _note_reader(self, slots: np.ndarray) -> None:
        """`slots` were handed to a caller that dispatched work inside the fetch window."""
        self._pending_reader[slots] = True
        self._pending_reader_n += int(slots.size)

    # Every eviction goes through `_protect` + `_victims`: demand (`_reserve`, `_swap_in`, the
    # PRESELECT begin, the split-inflight `lost` refetch, the wave prefetch, `release_ext`'s
    # replacement) and the speculative landing remap (`_harvest`) alike. The policy only decides
    # residency; what any expert computes never depends on it.
    def _protect(self, protected: np.ndarray) -> np.ndarray:
        """Union, in place, every slot no eviction path may take: the reserve and landing regions,
        victims an outstanding PRESELECT begin already chose (`_promised`), a released extension or
        its unbacked tail, and slots an in-flight gather still reads (`_pending_reader`)."""
        # Unconditional: `_is_reserve` flags both the prefetch reserve and the landing region, and
        # `reserve_n` alone can be 0 while the landing region exists.
        protected[self._is_reserve] = True  # no reserved slot is ever a victim
        protected |= self._promised  # nor a victim an outstanding begin already chose
        if self.ext_slots and (not self.ext_active or self.ext_live < self.ext_slots):
            protected |= self._disabled  # a released extension, or its unbacked tail
        if self._pending_reader_n:
            # pending-reader invariant: a slot an in-flight gather is reading is never a victim
            protected |= self._pending_reader
        return protected

    def _victims(self, protected: np.ndarray, k: int) -> np.ndarray:
        """The `k` slots the served policy evicts among the unprotected ones, in eviction order.

        The caller has already applied `_protect` and checked at least `k` unprotected slots exist.
        While `cyc_tokens` is armed (single-row decode) the cycle-aware rule decides, on every path
        alike; otherwise `decay_tokens` (set per mode by the engine) selects decay, else LRU.
        """
        if self.cyc_tokens:
            victims = self._cyc_victims(protected, k)
            self.evictions_cyc += int(victims.size)
        elif self.decay_tokens:
            victims = _decay_victims(self.use, self.last, protected, k)
        elif self.transient_prefill:
            # transient (prefill-fetched) records go first, LRU within each class
            cand = np.flatnonzero(~protected)
            order = np.lexsort((self.last[cand], self.transient[cand]))
            victims = cand[order[:k]]
        else:
            victims = _lru_victims(self.last, protected, k, self._vkey, self._vidx)
        if self._pending_reader_n and self._pending_reader[victims].any():
            # guards `_protect`: a regression there would corrupt a gather silently
            raise RuntimeError(
                "expert pool chose a victim with a pending reader: a slot that is still being "
                f"read into must not be evicted (slot(s) {victims[self._pending_reader[victims]].tolist()})"
            )
        return victims

    def _cyc_observe(self, gids: np.ndarray) -> None:
        """One decode visit of a layer: p *= (1 - 1/N) over that layer's experts, += 1/N for the
        routed ones, and the layer's tails L (1-p) / p refreshed, so a victim pick only gathers.

        Skipped when `gids` spans more than one layer, or when the same layer is visited again with
        no other layer in between (a redo or split of one visit: single-row decode visits the L
        layers strictly in turn).
        """
        if gids.size == 0:
            return
        E = self.num_experts
        lay = int(gids.min()) // E
        if int(gids.max()) // E != lay:
            return
        if self.num_layers > 1 and lay == self.cyc_layer and self._cyc_last_obs >= 0:
            return
        a = 1.0 / self.cyc_tokens
        lo, hi = lay * E, (lay + 1) * E
        blk = self.p_use[lo:hi]
        blk *= 1.0 - a
        self.p_use[gids] += a
        with np.errstate(divide="ignore"):
            self._cyc_tail[lo:hi] = self.num_layers * (1.0 - blk) / blk  # p == 0 -> +inf
        self.cyc_layer = lay
        self._cyc_last_obs = self.clock
        self._cyc_ver += 1  # invalidates the per-slot cache (`_cyc_far`)
        self.cyc_observations += 1

    def _cyc_far_of(self, g: np.ndarray) -> np.ndarray:
        """d + L (1-p) / p for record ids `g` from the current layer; +inf for -1 / never seen."""
        lay = (g >> self._cyc_shift) if self._cyc_shift >= 0 else self._cyc_lay[g]
        far = self._cyc_tail[g]  # -1 indexes the sentinel (tail +inf)
        far += self._cyc_d[self.cyc_layer][lay]
        return far

    def _cyc_far(self) -> np.ndarray:
        """Expected next use, in layer-steps, of every slot's record from the current layer (a
        fresh array the caller may modify). Between two observations the per-slot values change
        only where `id_of_slot` changed (tails and the current layer move only in `_cyc_observe`,
        which bumps `_cyc_ver`), so a visit's ~2-3 harvest remaps recompute just those slots."""
        g = self.id_of_slot
        if self._cyc_cache_ver != self._cyc_ver:
            self._cyc_cache_far = self._cyc_far_of(g)
            self._cyc_cache_ids = g.copy()
            self._cyc_cache_ver = self._cyc_ver
        else:
            ch = np.flatnonzero(g != self._cyc_cache_ids)
            if ch.size:
                gc = g[ch]
                self._cyc_cache_far[ch] = self._cyc_far_of(gc)
                self._cyc_cache_ids[ch] = gc
        return self._cyc_cache_far.copy()

    def _cyc_victims(self, protected: np.ndarray, k: int) -> np.ndarray:
        """The k unprotected slots whose expected next use is farthest, in eviction order.

        Ties go to the least recent use, then the lowest slot index. Same slots in the same order as
        the full sort

            cand = np.flatnonzero(~protected)
            cand[np.lexsort((cand, last[cand], -far[cand]))[:k]]

        in O(slots): k == 1 is max + argmin over the tie set; k > 1 is an argpartition, then only the
        slots at or beyond the k-th key are sorted.
        """
        far = self._cyc_far()
        far[protected] = -1.0  # every real key is >= 1
        last = self.last
        if k == 1:
            tie = np.flatnonzero(far == far.max())
            return tie[np.argmin(last[tie])].reshape(1)
        n = far.shape[0]
        if k >= n:
            cand = np.flatnonzero(~protected)
            return cand[np.lexsort((cand, last[cand], -far[cand]))[:k]]
        sel = np.argpartition(far, n - k)[n - k :]  # the k largest (no negated copy)
        thresh = far[sel].min()
        tie = np.flatnonzero(far >= thresh)
        return tie[np.lexsort((tie, last[tie], -far[tie]))[:k]]

    def _reserve(self, missing: np.ndarray, protected: np.ndarray) -> np.ndarray:
        """Choose a slot per id in `missing`: free slots first, then victims by the pool's policy.

        Victims lose their index entry here, but the incoming ids stay unpublished until `_publish`,
        so a fetch that raises leaves an empty slot, never a half-written resident one.
        """
        self._protect(protected)
        take = self._take_free_slots(missing, protected)
        k = missing.size - len(take)
        if k > 0:
            victims = self._choose_victims(protected, k, missing)
            self._evict_victims(victims, take)
        slots = self._own_slots(take)
        return slots

    def _take_free_slots(self, missing: np.ndarray, protected: np.ndarray) -> list[int]:
        """Pop free slots for up to `missing.size` ids and protect them from victim selection."""
        take: list[int] = []
        while self._free and len(take) < missing.size:
            take.append(self._free.pop())
        if take:
            protected[np.asarray(take, dtype=np.int64)] = True  # never also pick them as victims
        return take

    def _choose_victims(self, protected: np.ndarray, k: int, missing: np.ndarray) -> np.ndarray:
        """Pick `k` victims by the pool's rule (`_victims`); raises if the pool is too full."""
        n_prot = int(protected.sum())
        if self.slots - n_prot < k:
            raise RuntimeError(
                f"expert pool cannot hold this step: needs {missing.size} more of "
                f"{self.slots} slots with {n_prot} in use"
            )
        victims = self._victims(protected, k)
        return victims

    def _evict_victims(self, victims: np.ndarray, take: list[int]) -> None:
        """Drop each victim's index entry and add the victims to `take`."""
        for v in victims:
            gid = self.id_of_slot[v]
            if gid >= 0:
                self.slot_for_id[gid] = -1
                self.evictions += 1
        take.extend(int(v) for v in victims)

    def _own_slots(self, take: list[int]) -> np.ndarray:
        """The chosen slots as an array, owned by the in-flight fetch and resident for nobody."""
        slots = np.asarray(take, dtype=np.int64)
        self.id_of_slot[slots] = -1  # owned by the in-flight fetch, resident for nobody
        self.use[slots] = 0.0
        self.transient[slots] = False
        return slots

    def _harvest_protected(self, wanted: np.ndarray) -> np.ndarray:
        """The protection set of a harvest remap: `_protect`'s union plus two recency protections.

        The records `wanted` by this call that are already resident (a pre-gather of this layer may
        be reading them), and every slot used within the last token (`last >= clock - num_layers`).
        `harvest_ready` runs at a speculative build without a host sync and with `wanted` empty,
        while the lookahead's speculative gathers of the previous one or two layers, built from the
        device table and so never marked `_pending_reader`, can still be in flight; remapping one of
        their slots sends it to the reserve, where a background read can overwrite it under the
        gather. `spec_settle`'s (slot, generation) check recomputes such a pair, but each hit costs a
        redo layer, and nothing else re-checks such a pair.
        """
        prot = self._protect(np.zeros(self.slots, dtype=bool))
        _w = self.slot_for_id[wanted]
        prot[_w[_w >= 0]] = True
        prot |= self.last >= self.clock - self.num_layers  # used within the last token
        return prot
