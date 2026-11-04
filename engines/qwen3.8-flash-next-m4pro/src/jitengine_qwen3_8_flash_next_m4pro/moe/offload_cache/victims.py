"""Victim selection for `OffloadMoeCache`, as a mixin: which slot a missing record takes.

Free slots go first, then victims by the pool's policy (plain LRU, decayed use, or
transient prefill records first; while `cyc_tokens` is armed, single-row decode, the cycle-aware
expected next use), never a reserved, disabled or pending-reader slot (`_reserve`); `_decay` ages
the use counts the policy reads, and `reader_barrier` and `_note_reader` keep the pending-reader
invariant. A policy decides only which slot holds which record, never a value any expert computes.
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
    sel = np.argpartition(u, k - 1)[:k]
    thresh = int(u[sel].max())
    tie = np.flatnonzero(u <= thresh)
    order = np.lexsort((tie, last[tie], u[tie]))
    return tie[order[:k]]


DECAY_FACTOR = 0.5
# A layer's EMA scale below this is folded into its experts' values (`_cyc_observe`); far above
# float64's least normal, so a folded value loses nothing a victim order can see.
CYC_RESCALE = 1e-200


def evict_period(rule: str) -> int:
    """The single-row decode victim rule's EMA period N: "cyc<N>" (the cycle-aware rule, e.g.
    "cyc64") -> N; "decay32" (the decayed-use rule of every other operating point) -> 0, off."""
    r = str(rule).strip().lower()
    if r == "decay32":
        return 0
    if r.startswith("cyc") and r[3:].isdigit() and int(r[3:]) > 0:
        return int(r[3:])
    raise ValueError(f"unknown eviction rule {rule!r}: cyc<N> (e.g. cyc64) or decay32")


class VictimSelectionMixin:
    """Which slots a fetch takes and which it must not: `_reserve`, `_decay`, the reader barrier."""

    def _decay(self) -> None:
        self.use *= DECAY_FACTOR

    # ------------------------------------------------------------ the cycle-aware rule
    def init_cyc_state(self) -> None:
        """The cycle-aware rule's state, off until `cyc_tokens` is armed.

        Single-row decode visits the layers in a fixed cycle, so a record of layer l used with
        per-visit probability p next comes back after the layer-steps until l comes round, d, plus
        about L (1 - p) / p more. The rule evicts the largest d + L (1 - p) / p, with p an EMA of
        the layer's visits (alpha 1/N). p is kept in a per-layer scale, p(id) =
        `_cyc_w[id] * _cyc_s[layer]`: a visit decays every expert of its layer by scaling
        `_cyc_s[layer]` and credits the routed ones in `_cyc_w`, so it touches the k routed ids only.

        Within one layer d and the scale are shared, so the farthest record is the one with the
        smallest `_cyc_w`; a pick compares one head per layer (`_cyc_head`, its farthest resident),
        and a head is recomputed only when its layer is marked stale: by a visit, by an eviction or a
        batch publish of its records (one published record is compared with the head directly), or
        after unarmed fetches (the `_cyc_seen` gap). Index
        `num_ids`, which an empty slot's id -1 reads, stays 0 (key +inf).
        """
        L, E = self.num_layers, self.num_experts
        self.cyc_tokens = (
            0  # the armed EMA period N; the engine arms `cyc_solo` around b == 1 decode
        )
        self.cyc_solo = 0  # the period the engine arms (0 = the decayed-use rule there too)
        self.cyc_layer = -1  # layer of the last observed visit
        self._cyc_w = np.zeros(self.num_ids + 1, dtype=np.float64)
        self._cyc_s = np.ones(L + 1, dtype=np.float64)  # [L] is the empty slot's (unused) scale
        # layer of an id by shift when E is a power of two; floor division keeps -1 at -1 either way
        self._cyc_shift = (E.bit_length() - 1) if E & (E - 1) == 0 else -1
        j = np.arange(L, dtype=np.int64)
        # _cyc_d[r, c]: layer-steps from a visit of layer r until layer c's next visit; column L
        # is the empty slot's, whose key is +inf whatever it adds
        self._cyc_d = np.zeros((L, L + 1), dtype=np.float64)
        self._cyc_d[:, :L] = ((j[None, :] - j[:, None] - 1) % L) + 1
        self._cyc_head = np.full(L, -1, dtype=np.int64)  # per layer: its farthest resident's slot
        self._cyc_stale = np.ones(L, dtype=bool)  # heads to recompute before the next pick
        # the last `ensure` call the heads account for: a gap means unarmed fetches (a prefill)
        # moved `last`, which orders ties, so every head is recomputed
        self._cyc_seen = -1
        self.cyc_observations = 0
        self.evictions_cyc = 0  # victims chosen by the cycle-aware rule

    def cyc_invalidate(self, gids=None) -> None:
        """Mark the heads of `gids`' layers (default every layer) for recomputation."""
        if gids is None:
            self._cyc_stale[:] = True
        else:
            self._cyc_stale[np.asarray(gids, dtype=np.int64) // self.num_experts] = True

    def _cyc_published(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """`gids` became resident in `slots`. One record (a harvested prefetch, a refetch) is compared
        with its layer's head directly: every publisher stamps its `last` with the current clock
        before the next pick, so (EMA, clock, slot) against the head's key decides. A batch marks
        its layers stale."""
        if gids.size != 1:
            self.cyc_invalidate(gids)
            return
        g, s = int(gids[0]), int(slots[0])
        lay = g // self.num_experts
        if self._cyc_stale[lay]:
            return
        h = int(self._cyc_head[lay])
        if h >= 0:
            wn, wh = self._cyc_w[g], self._cyc_w[self.id_of_slot[h]]
            lh = int(self.last[h])
            if wn > wh or (wn == wh and (self.clock > lh or (self.clock == lh and s > h))):
                return
        self._cyc_head[lay] = s

    def _cyc_observe(self, gids: np.ndarray) -> None:
        """One visit of a layer, at the top of its `ensure`, before the visit's evictions.

        `gids` arrive sorted (`np.unique`, as every caller passes them), so the first and last id
        say whether the call spans one layer; a call that spans several (a wave) or repeats the layer
        observed last (a redo or split of one visit) is not a visit and is skipped. A visit of a
        verify pass carries the union of both positions' experts, so p is per pass. An observation
        is only an input to the victim order: a skipped or misread one costs residency, never a value.
        """
        n = gids.size
        if n == 0:
            return
        if self.calls != self._cyc_seen + 1:
            self._cyc_stale[:] = True  # unarmed fetches since the last visit moved `last` unseen
        self._cyc_seen = self.calls
        E = self.num_experts
        lay = int(gids[0]) // E
        if int(gids[n - 1]) // E != lay or (lay == self.cyc_layer and self.num_layers > 1):
            return
        a = 1.0 / self.cyc_tokens
        sl = float(self._cyc_s[lay]) * (1.0 - a)
        if (
            sl < CYC_RESCALE
        ):  # fold the scale into the layer's values: O(E), every ~29,000 visits at N 64
            lo = lay * E
            self._cyc_w[lo : lo + E] *= sl
            sl = 1.0
        self._cyc_s[lay] = sl
        self._cyc_w[gids] += a / sl
        self._cyc_stale[lay] = True
        self.cyc_layer = lay
        self.cyc_observations += 1

    def _cyc_far_of(self, g: np.ndarray) -> np.ndarray:
        """d + L (1 - p) / p for record ids `g` from the last observed layer; +inf for -1 or an id
        its layer has never routed."""
        lay = (g >> self._cyc_shift) if self._cyc_shift >= 0 else (g // self.num_experts)
        p = self._cyc_w[g] * self._cyc_s[lay]
        with np.errstate(divide="ignore", invalid="ignore"):
            far = self.num_layers * (1.0 - p) / p  # p == 0 -> +inf
        far += self._cyc_d[self.cyc_layer][lay]
        return far

    def _cyc_layer_best(self, lay: int, protected: np.ndarray | None = None) -> int:
        """The slot of layer `lay`'s farthest resident record outside `protected`: the smallest EMA,
        then the least recent use, then the lowest slot; -1 if there is none."""
        E = self.num_experts
        lo = lay * E
        sl = self.slot_for_id[lo : lo + E]
        r = np.flatnonzero(sl >= 0)
        if protected is not None and r.size:
            r = r[~protected[sl[r]]]
        if r.size == 0:
            return -1
        wv = self._cyc_w[lo + r]
        t = r[wv == wv.min()]
        if t.size == 1:
            return int(sl[t[0]])
        ts = sl[t].astype(np.int64)
        return int(ts[np.lexsort((ts, self.last[ts]))[0]])

    def _cyc_pick(self, protected: np.ndarray) -> int:
        """The unprotected resident slot whose record's next use is farthest (ties: the least recent
        use, then the lowest slot), from the layers' heads; -1 if every resident is protected."""
        stale, head = self._cyc_stale, self._cyc_head
        if self.calls != self._cyc_seen:  # fetches since the last observation, none of them seen
            stale[:] = True
            self._cyc_seen = self.calls
        if stale.any():
            for lay in np.flatnonzero(stale).tolist():
                head[lay] = self._cyc_layer_best(lay)
            stale[:] = False
        cand = head
        blk = protected[cand] & (cand >= 0)  # a head this call protects: the layer's next best
        if blk.any():
            cand = head.copy()
            for lay in np.flatnonzero(blk).tolist():
                cand[lay] = self._cyc_layer_best(lay, protected)
        L = self.num_layers  # cand[l] is layer l's: its scale and distance need no lookup
        p = self._cyc_w[self.id_of_slot[cand]] * self._cyc_s[:L]
        with np.errstate(divide="ignore", invalid="ignore"):
            far = L * (1.0 - p) / p
        far += self._cyc_d[self.cyc_layer][:L]
        far[cand < 0] = -1.0  # a layer with no candidate (every real key is >= 1)
        m = far.max()
        if m < 0:
            return -1
        t = np.flatnonzero(far == m)
        if t.size == 1:
            return int(cand[t[0]])
        ts = cand[t]
        return int(ts[np.lexsort((ts, self.last[ts]))[0]])

    def _cyc_victims_full(self, protected: np.ndarray, k: int) -> np.ndarray:
        """The rule's k victims by the full sort over every slot (empty slots included, key +inf):
        the reference `_cyc_victims` reproduces, and its fallback when no resident is a candidate."""
        far = self._cyc_far_of(self.id_of_slot)
        cand = np.flatnonzero(~protected)
        return cand[np.lexsort((cand, self.last[cand], -far[cand]))[:k]]

    def _cyc_victims(self, protected: np.ndarray, k: int) -> np.ndarray:
        """The k unprotected slots whose expected next use is farthest, in eviction order.

        Ties go to the least recent use, then the lowest slot index: the order of the full sort
        (`_cyc_victims_full`) over the resident records, found by k picks of one head per layer
        rather than a pass over every slot. An empty slot outside the free list is never a victim
        here unless no resident is (then the full sort decides). The caller guarantees k
        unprotected slots and owns `protected`, which is not modified.
        """
        out = np.empty(k, dtype=np.int64)
        prot = protected if k == 1 else protected.copy()
        for i in range(k):
            v = self._cyc_pick(prot)
            if v < 0:
                out[i:] = self._cyc_victims_full(prot, k - i)
                break
            out[i] = v
            if i + 1 < k:
                prot[v] = True  # its layer's next best comes from the protected-head path
        self.evictions_cyc += int(k)
        return out

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

    def _reserve(self, missing: np.ndarray, protected: np.ndarray) -> np.ndarray:
        """Choose a slot per id in `missing`: free slots first, then victims by the pool's policy.

        Victims lose their index entry here, but the incoming ids stay unpublished until `_publish`,
        so a fetch that raises leaves an empty slot, never a half-written resident one.
        """
        protected = self._protect_unusable_slots(protected)
        take = self._take_free_slots(missing, protected)
        k = missing.size - len(take)
        if k > 0:
            try:
                victims = self._choose_victims(protected, k, missing)
                self._evict_victims(victims, take)
            except BaseException:
                self._free.extend(take)  # popped from the free list, still empty: back they go
                raise
        slots = self._own_slots(take)
        return slots

    def _protect_unusable_slots(self, protected: np.ndarray) -> np.ndarray:
        """Mark the slots no demand fetch may take: the reserve, a released extension, pending readers."""
        # Unconditional: `_is_reserve` flags both the prefetch reserve and the landing region, and
        # `reserve_n` alone can be 0 while the landing region exists.
        protected[self._is_reserve] = True  # no reserved slot is ever a demand victim
        if self.disabled_any():
            protected |= self._disabled  # a released extension or its unbacked tail, the annex
        if self._pending_reader_n:
            # pending-reader invariant: a slot an in-flight gather is reading is never a victim
            protected |= self._pending_reader
        return protected

    def _take_free_slots(self, missing: np.ndarray, protected: np.ndarray) -> list[int]:
        """Pop free slots for up to `missing.size` ids and protect them from victim selection."""
        take: list[int] = []
        while self._free and len(take) < missing.size:
            take.append(self._free.pop())
        if take:
            protected[np.asarray(take, dtype=np.int64)] = True  # never also pick them as victims
        return take

    def _choose_victims(self, protected: np.ndarray, k: int, missing: np.ndarray) -> np.ndarray:
        """Pick `k` victims by the pool's policy (cycle-aware while armed, decay, transient first, or
        LRU); raises if the pool is too full."""
        n_prot = int(protected.sum())
        if self.slots - n_prot < k:
            raise RuntimeError(
                f"expert pool cannot hold this step: needs {missing.size} more of "
                f"{self.slots} slots with {n_prot} in use"
            )
        # `use` is maintained under plain LRU too, so decay needs no second bookkeeping path
        if self.cyc_tokens:
            victims = self._cyc_victims(protected, k)
        elif self.decay_tokens:
            victims = _decay_victims(self.use, self.last, protected, k)
        elif self.transient_prefill:
            # transient (prefill-fetched) records go first, LRU within each class
            cand = np.flatnonzero(~protected)
            order = np.lexsort((self.last[cand], self.transient[cand]))
            victims = cand[order[:k]]
        else:
            victims = _lru_victims(self.last, protected, k, self._vkey, self._vidx)
        return victims

    def _evict_victims(self, victims: np.ndarray, take: list[int]) -> None:
        """Check no victim has a pending reader, drop each victim's index entry and add them to `take`."""
        if self._pending_reader_n and self._pending_reader[victims].any():
            # guards the union above: a regression there would corrupt a gather silently
            raise RuntimeError(
                "expert pool chose a victim with a pending reader; the C1-ASYNC invariant "
                f"is broken for slot(s) {victims[self._pending_reader[victims]].tolist()}"
            )
        for v in victims:
            gid = self.id_of_slot[v]
            if gid >= 0:
                self.slot_for_id[gid] = -1
                self.evictions += 1
                self._cyc_stale[gid // self.num_experts] = True
        take.extend(int(v) for v in victims)

    def _own_slots(self, take: list[int]) -> np.ndarray:
        """The chosen slots as an array, owned by the in-flight fetch and resident for nobody."""
        slots = np.asarray(take, dtype=np.int64)
        self.id_of_slot[slots] = -1  # owned by the in-flight fetch, resident for nobody
        self.use[slots] = 0.0
        self.transient[slots] = False
        return slots
