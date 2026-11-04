"""The read pipeline of `OffloadMoeCache`, as a mixin: how a record's bytes reach a slot.

Demand reads land in the reserved landing region and are published by pointer swap
(`_swap_in`); reads too large for it fill synchronously with one bounded retry (`_fill`),
by `preadv` straight into the pool arrays or by staging plus one scatter (`_landing`
picks); prefetches read into the reserve and are published by index remap (`prefetch`,
`_harvest`).
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m2.engine.errors import InvariantError

logger = logging.getLogger(__name__)

FILL_ATTEMPTS = 2  # one bounded retry of a faulting record read


# /stats breaks the miss chain's speculative reads down by predicted rank and router weight
CHAIN_W_EDGES = (0.02, 0.05, 0.10, 0.20)  # weight buckets: <.02 .02-.05 .05-.10 .10-.20 >=.20


_CHAIN_RANKS = 16  # rank buckets 0..15 (top_k is 10 on the target)


_MX_DTYPE = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}


_NP_STORAGE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.uint16, "F32": np.float32}


_MX_STORAGE = {"U32": mx.uint32, "BF16": mx.uint16, "F16": mx.uint16, "F32": mx.float32}


def _writable_host_view(a: mx.array) -> np.ndarray | None:
    """A numpy array aliasing `a`'s unified-memory buffer, or None if that cannot be had."""
    mx.eval(a)
    v = None
    for get in (lambda: np.from_dlpack(a), lambda: np.array(a, copy=False)):
        try:
            v = get()
            break
        except Exception:  # noqa: BLE001  - dtype or protocol unsupported on this MLX
            continue
    if v is None:
        return None
    if v.flags.writeable:
        return v
    try:  # a read-only export still gives the buffer address
        addr, _ro = v.__array_interface__["data"]
        ctype = np.ctypeslib.as_ctypes_type(v.dtype)
        buf = (ctype * v.size).from_address(addr)
        w = np.frombuffer(buf, dtype=v.dtype).reshape(v.shape)
        return w if w.flags.writeable else None
    except Exception:  # noqa: BLE001
        return None


class ReadPipelineMixin:
    """Landing region, synchronous fill, scatter fallback and prefetch."""

    # ------------------------------------------------------------------ landing path
    def _landing(self) -> str:
        """Prefer zero-copy `preadv` into the pool arrays; prove it with a sentinel or fall back."""
        allf = self.fields + self.fields_ext
        views = [_writable_host_view(f.store) for f in allf]
        if any(v is None for v in views):
            return "scatter"
        for f, v in zip(allf, views):
            f.host = v
        # a host write must show through to the tensor the GPU reads, including a bf16 `view()`
        for f in allf:
            flat = f.host.reshape(f.host.shape[0], -1)
            old = int(flat[0, 0])
            flat[0, 0] = 7 if f.tensor is f.store else 0x4000  # bf16 2.0
            want = 7.0 if f.tensor is f.store else 2.0
            seen = float(mx.sum(f.tensor[0].reshape(-1)[:1]).item())
            flat[0, 0] = old
            if seen != want:
                for g in allf:
                    g.host = None
                return "scatter"
        return "direct"

    def _bufs(self, slot: int) -> list:
        """The nine destination buffers of physical slot `slot` (main or extension array set)."""
        slot = int(slot)
        if slot < self.slots_main:
            return [f.host[slot] for f in self.fields]
        r = slot - self.slots_main
        return [f.host[r] for f in self.fields_ext]

    # ------------------------------------------------------------------ the fetch
    def _fill(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Read records `gids` into the reserved `slots` synchronously, with one bounded retry.

        `read_many` waits for every future before it returns or raises, so no worker is still
        writing afterwards; callers on this path evict before the read, so it must stay synchronous.
        Retry bytes go to `retry_bytes`, never `miss_bytes`.
        """
        last: BaseException | None = None
        for attempt in range(FILL_ATTEMPTS):
            if attempt:
                with self._lock:
                    self.read_retries += 1
                    self.retry_bytes += len(gids) * self.record_bytes
                logger.warning(f"[pool] retrying {len(gids)} expert record(s) after {last!r}")
            try:
                self._fill_once(gids, slots)
                return
            except OSError as exc:  # transient device / short-read faults only
                last = exc
        raise last

    # ------------------------------------------------------------ the reserved landing region
    def _take_landing(self, n: int):
        """`n` landing slots, or None (use the synchronous path) when the region cannot serve.

        Too large a request (a prefill wave) gets None, and so does the scatter fallback, whose
        landing is an MLX call that must run on the model thread.
        """
        if self.landing != "direct" or n <= 0 or n > len(self._landing_free):
            return None
        return np.asarray([self._landing_free.pop() for _ in range(n)], dtype=np.int64)

    def _submit_landing(self, gid: int, slot: int):
        """One demand read into a landing slot; uses the held store so its wrappers still apply."""
        return self.store.submit_demand(gid, self._bufs(slot), self.store.read_into)

    def _park(self, futures: list, slots) -> None:
        """Park the landing slots of already-submitted writes until those writes end."""
        for f, sl in zip(futures, np.asarray(slots, dtype=np.int64).tolist()):
            self._orphans.append((f, int(sl)))
        self.landing_orphaned += len(futures)

    def _sweep_orphans(self) -> None:
        """Return parked slots whose writes have ended to the landing region."""
        keep = []
        for f, sl in self._orphans:
            if f.done():
                self._landing_free.append(sl)
            else:
                keep.append((f, sl))
        self._orphans = keep

    def _swap_in(
        self, gids: np.ndarray, lslots: np.ndarray, protected: np.ndarray, vslots=None
    ) -> None:
        """Publish records landed in `lslots` by swapping each landing slot with a victim slot.

        The landing slot joins the pool holding the record and the victim joins the region empty, so
        pool and region sizes stay fixed. Victims come from `_reserve` unless passed as `vslots`.
        """
        # `_reserve` never returns an `_is_reserve` slot; otherwise one slot would sit in
        # `_landing_free` twice and serve two fetches
        if vslots is None:
            vslots = self._reserve(gids, protected)
        else:
            self._promised[vslots] = False
        if self._is_reserve[vslots].any():
            raise InvariantError(
                f"_reserve returned reserved slot(s) {vslots[self._is_reserve[vslots]].tolist()} "
                "as victims; the landing region would be double-allocated"
            )
        self._is_reserve[lslots] = False
        self._is_reserve[vslots] = True
        self.id_of_slot[vslots] = -1
        self.use[vslots] = 0.0
        self.transient[vslots] = False
        self._landing_free.extend(int(v) for v in vslots.tolist())
        self._publish(gids, lslots)

    def _land_staging(self, staging: np.ndarray, slots: np.ndarray) -> None:
        """Scatter fallback, model thread only: move the staged bytes into the pool tensors."""
        n = staging.shape[0]
        if self.ext_slots and (np.asarray(slots) >= self.slots_main).any():
            raise RuntimeError("scatter landing is not supported with an extension pool")
        idx = mx.array(np.asarray(slots, dtype=np.int32))
        for f in self.fields:
            raw = np.ascontiguousarray(staging[:, f.offset : f.offset + f.nbytes])
            vals = mx.array(raw.view(_NP_STORAGE[f.dtype]).reshape(n, *f.shape))
            if f.tensor is not f.store:
                vals = vals.view(_MX_DTYPE[f.dtype])
            f.tensor[idx] = vals
            mx.eval(f.tensor)

    def _fill_once(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """One synchronous attempt: a nine-buffer `preadv` per record, or staging plus scatter."""
        if self.landing == "direct":
            jobs = [(int(g), self._bufs(int(s))) for g, s in zip(gids, slots)]
            self.store.read_many(jobs)
            return
        n = len(gids)
        staging = np.empty((n, self.record_bytes), dtype=np.uint8)
        jobs = [(int(g), [staging[i]]) for i, g in enumerate(gids)]
        self.store.read_many(jobs)
        self._land_staging(staging, slots)

    def _harvest(self, wanted: np.ndarray) -> None:
        """Publish landed prefetches by index remap, waiting for those in `wanted`.

        Model thread only. The reserve slot holding a record becomes a pool slot and an LRU victim
        becomes reserve; no bytes move. The wait is charged as exposed stall and the whole read
        since submit as `fetch_seconds`.
        """
        need = set(int(x) for x in wanted.tolist()) & set(self._inflight)
        done = [g for g, (_s, f, _t) in self._inflight.items() if f.done()]
        for g in set(done) | need:
            rslot, fut, t_sub = self._inflight.pop(g)
            meta = self._inflight_meta.pop(g, None)
            # `release_ext` clears `_is_reserve` on extension slots holding an in-flight speculative
            # read and disables them. Such a read must never be published or recycled into the
            # reserve, since its bytes went to the dropped arrays: drop it (a needed one becomes a
            # demand miss) and count. A release followed by `regrow_ext` before the harvest is not
            # covered (the slot is enabled again and on `_free`).
            dead = bool(self._disabled[rslot])
            t0 = time.monotonic()
            try:
                fut.result()
            except BaseException:  # noqa: BLE001  a failed prefetch only frees its slot
                if not dead:
                    self._reserve_free.append(rslot)  # slot never left the reserve; nothing to undo
                continue
            if dead:
                with self._lock:
                    self.harvest_dropped_disabled += 1
                continue
            self._note_chain_outcome(g, need, meta)
            self._charge_harvest_wait(t0, t_sub)
            if g not in need and self.slot_for_id[g] >= 0:
                self._reserve_free.append(rslot)  # someone else made it resident; drop ours
                with self._lock:
                    self.prefetch_wasted += 1
                continue
            victim = None
            if self._free:
                victim = self._free.pop()
            else:
                prot = self._harvest_protected(wanted)
                if int(prot.sum()) >= self.slots:
                    self._reserve_free.append(rslot)
                    continue
                victim = self._evict_harvest_victim(prot)
            self._publish_harvested(g, rslot, victim, need)

    def _note_chain_outcome(self, g: int, need: set[int], meta: tuple[int, int] | None) -> None:
        """Count a chain prefetch's outcome (used, wasted or unneeded) by rank and weight bucket."""
        if meta is not None:
            key = "used" if g in need else ("wasted" if self.slot_for_id[g] >= 0 else "unneeded")
            with self._lock:
                self.chain_by_rank[key][meta[0]] += 1
                self.chain_by_weight[key][meta[1]] += 1

    def _charge_harvest_wait(self, t0: float, t_sub: float) -> None:
        """Charge the wait since `t0` as exposed stall and the read since submit as `fetch_seconds`."""
        now = time.monotonic()
        dt, read_dt = now - t0, now - t_sub
        with self._lock:
            self.fetch_seconds += read_dt  # read_dt >= dt, so exposed <= read holds
            if dt > 0:
                self.prefetch_waited += 1
                self.exposed_stall_seconds += dt

    def _evict_harvest_victim(self, prot: np.ndarray) -> int:
        """Take the victim the eviction rule picks outside `prot`, unindexing the record it held."""
        victim = int(self._victims(prot, 1)[0])
        old = self.id_of_slot[victim]
        if old >= 0:
            self.slot_for_id[old] = -1
            self.evictions += 1
            self.evictions_harvest += 1
        return victim

    def _publish_harvested(self, g: int, rslot: int, victim: int, need: set[int]) -> None:
        """Swap the reserve slot into the pool and the victim into the reserve, publish the record, count it."""
        self._is_reserve[rslot] = False
        self._is_reserve[victim] = True
        self._reserve_free.append(int(victim))
        self.id_of_slot[victim] = -1
        self._publish(np.asarray([g], dtype=np.int64), np.asarray([rslot], dtype=np.int64))
        self.last[rslot] = self.clock
        self.use[rslot] = 1.0
        with self._lock:
            if g in need:
                self.prefetch_used += 1

    def harvest_ready(self) -> None:
        """Publish the prefetches that have already landed, waiting for none.

        Safe without a host sync: a slot this evicts under an in-flight gather shows up as a moved
        (slot, generation) at that gather's settle, which recomputes the pair.
        """
        self._harvest(np.empty(0, dtype=np.int64))

    def prefetch(self, gids: np.ndarray, weights=None, ranks=None) -> int:
        """Speculatively read `gids` into reserve slots (model thread); evicts nothing.

        `weights` and `ranks`, aligned with `gids`, are the predicting router's top-k weights and
        ranks; they feed the outcome counters. A duplicated gid keeps its best prediction.
        """
        if self.reserve_n == 0 or gids.size == 0 or not self._reserve_free:
            return 0
        meta = None
        if weights is not None and ranks is not None:
            g0 = gids.astype(np.int64).reshape(-1)
            w0 = np.asarray(weights, dtype=np.float64).reshape(-1)
            r0 = np.asarray(ranks, dtype=np.int64).reshape(-1)
            g, inv = np.unique(g0, return_inverse=True)
            w = np.full(g.size, -1.0)
            np.maximum.at(w, inv, w0)
            r = np.full(g.size, 1 << 30, dtype=np.int64)
            np.minimum.at(r, inv, r0)
            keep = self.slot_for_id[g] < 0
            if self._inflight:
                keep &= np.fromiter(
                    (x not in self._inflight for x in g.tolist()), dtype=bool, count=g.size
                )
            g, w, r = g[keep], w[keep], r[keep]
            rb = np.minimum(r, _CHAIN_RANKS - 1)
            wb = np.searchsorted(np.asarray(CHAIN_W_EDGES), w, side="right")
            meta = (rb, wb)
        else:
            g = np.unique(gids.astype(np.int64))
            g = g[self.slot_for_id[g] < 0]
            if self._inflight:
                g = np.asarray([x for x in g.tolist() if x not in self._inflight], dtype=np.int64)
        if g.size == 0:
            return 0
        n = min(int(g.size), len(self._reserve_free))
        g = g[:n]
        for j, gid in enumerate(g.tolist()):
            rslot = self._reserve_free.pop()
            bufs = self._bufs(int(rslot))
            self._inflight[int(gid)] = (
                int(rslot),
                self.store.submit(int(gid), bufs),
                time.monotonic(),
            )
            if meta is not None:
                a, b = int(meta[0][j]), int(meta[1][j])
                self._inflight_meta[int(gid)] = (a, b)
                with self._lock:
                    self.chain_by_rank["issued"][a] += 1
                    self.chain_by_weight["issued"][b] += 1
        with self._lock:
            self.prefetched += int(g.size)
            self.prefetch_bytes += int(g.size) * self.record_bytes
        return int(g.size)
