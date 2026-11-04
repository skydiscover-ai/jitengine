"""The read pipeline of `OffloadMoeCache`, as a mixin: how a record's bytes reach a slot.

Demand reads land in the reserved landing region and are published by pointer swap
(`_swap_in`); reads too large for it fill synchronously with one bounded retry (`_fill`),
by `preadv` straight into the pool arrays or by staging plus one scatter (`_landing`
picks); prefetches read into the reserve and are published by index remap (`prefetch`,
`_harvest`). With the pack16 codec every read lands in a staging record instead (`_read_dest`)
and each publication dispatches the encode into the slot it publishes (`codec.py`), unless the
records are compact on disk (`disk16`), when they land in the compact slots as bf16 ones do.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError

logger = logging.getLogger(__name__)

FILL_ATTEMPTS = 2  # one bounded retry of a faulting record read


_MX_DTYPE = {
    "U32": mx.uint32,
    "U16": mx.uint16,
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
}


_NP_STORAGE = {
    "U32": np.uint32,
    "U16": np.uint16,
    "BF16": np.uint16,
    "F16": np.uint16,
    "F32": np.float32,
}


_MX_STORAGE = {
    "U32": mx.uint32,
    "U16": mx.uint16,
    "BF16": mx.uint16,
    "F16": mx.uint16,
    "F32": mx.float32,
}


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

    def _read_dest(self, slot: int) -> list:
        """Where a read of the record bound for `slot` lands: the slot's own buffers, or with the
        codec (records not compact on disk) its staging record."""
        return self._dest(slot) if (self.codec and not self.disk16) else self._bufs(slot)

    def _bufs(self, slot: int) -> list:
        """The destination buffers of physical slot `slot` (main, extension or annex array set):
        the nine bf16 fields, or the six compact ones of a `disk16` pool."""
        slot = int(slot)
        if slot < self.slots_main:
            return [f.host[slot] for f in self.fields]
        if slot >= self.ann0:
            return [f.host[slot - self.ann0] for f in self.fields_ann]
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
        return self.store.submit_demand(gid, self._read_dest(slot), self.store.read_into)

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
                if self.codec:
                    self._stage_release(sl)
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
        self._encode_publish(lslots)  # before any reader of them is dispatched

    def _land_staging(self, staging: np.ndarray, slots: np.ndarray) -> None:
        """Scatter fallback, model thread only: move the staged bytes into the pool tensors."""
        n = staging.shape[0]
        if (self.ext_slots or self.ann_slots) and (np.asarray(slots) >= self.slots_main).any():
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
        """One synchronous attempt: a nine-buffer (six, `disk16`) `preadv` per record, or staging plus
        scatter."""
        if self.codec and not self.disk16:
            self._fill_codec(gids, slots)
            return
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

        Model thread only. The reserve slot holding a record becomes a pool slot and a victim (the
        cycle-aware rule's while it is armed, else the least recently used) becomes reserve; no
        bytes move. The wait is charged as exposed stall and the whole read
        since submit as `fetch_seconds`.
        """
        need = set(int(x) for x in wanted.tolist()) & set(self._inflight)
        done = [g for g, (_s, f, _t) in self._inflight.items() if f.done()]
        for g in set(done) | need:
            rslot, fut, t_sub = self._inflight.pop(g)
            t0 = time.monotonic()
            try:
                fut.result()
            except BaseException:  # noqa: BLE001  a failed prefetch only frees its slot
                self._drop_prefetch(rslot)  # slot never left the reserve; nothing to undo
                continue
            self._charge_harvest_wait(t0, t_sub)
            if g not in need and self.slot_for_id[g] >= 0:
                self._drop_prefetch(rslot)  # someone else made it resident; drop ours
                with self._lock:
                    self.prefetch_wasted += 1
                continue
            victim = None
            if self._free:
                victim = self._free.pop()
            else:
                prot = self._protect_harvest_slots(wanted)
                cand = np.flatnonzero(~prot)
                if cand.size == 0:
                    self._drop_prefetch(rslot)
                    continue
                victim = self._evict_harvest_victim(cand, prot)
            self._publish_harvested(g, rslot, victim, need)

    def _drop_prefetch(self, rslot: int) -> None:
        """A landed or failed prefetch that is not published: its slot back to the reserve."""
        if self.codec:
            self._stage_release(rslot)
        self._reserve_free.append(rslot)

    def _charge_harvest_wait(self, t0: float, t_sub: float) -> None:
        """Charge the wait since `t0` as exposed stall and the read since submit as `fetch_seconds`."""
        now = time.monotonic()
        dt, read_dt = now - t0, now - t_sub
        with self._lock:
            self.fetch_seconds += read_dt  # read_dt >= dt, so exposed <= read holds
            if dt > 0:
                self.prefetch_waited += 1
                self.exposed_stall_seconds += dt

    def _protect_harvest_slots(self, wanted: np.ndarray) -> np.ndarray:
        """The slots that must not be victims of a harvest: reserve, disabled, pending readers, the layer's
        residents and the last token's."""
        prot = np.zeros(self.slots, dtype=bool)
        prot[self._is_reserve] = True
        if self.disabled_any():
            prot |= self._disabled
        if self._pending_reader_n:
            prot |= self._pending_reader  # an in-flight gather's slot is not a victim
        # the current layer's resident records: a pre-gather may already be reading them
        _w = self.slot_for_id[wanted]
        prot[_w[_w >= 0]] = True
        prot |= self.last >= self.clock - self.num_layers  # used within the last token
        return prot

    def _evict_harvest_victim(self, cand: np.ndarray, prot: np.ndarray) -> int:
        """Take the victim among the candidates (`cand`, the slots `prot` leaves), unindexing the
        record it held: the cycle-aware rule's while it is armed, else the least recently used."""
        if self.cyc_tokens:
            victim = int(self._cyc_victims(prot, 1)[0])
        else:
            victim = int(cand[np.argmin(self.last[cand])])
        if prot[victim]:
            # guards the rule: a protected victim could be a slot an in-flight gather still reads
            raise InvariantError(
                f"harvest chose protected slot {victim} as its victim (reserve, disabled, pending "
                "reader, this layer's or the last token's)"
            )
        old = self.id_of_slot[victim]
        if old >= 0:
            self.slot_for_id[old] = -1
            self.evictions += 1
            self._cyc_stale[old // self.num_experts] = True
        return victim

    def _publish_harvested(self, g: int, rslot: int, victim: int, need: set[int]) -> None:
        """Swap the reserve slot into the pool and the victim into the reserve, publish the record, count it."""
        self._is_reserve[rslot] = False
        self._is_reserve[victim] = True
        self._reserve_free.append(int(victim))
        self.id_of_slot[victim] = -1
        self._publish(np.asarray([g], dtype=np.int64), np.asarray([rslot], dtype=np.int64))
        self._encode_publish([rslot])
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

    def prefetch(self, gids: np.ndarray) -> int:
        """Speculatively read `gids` into reserve slots (model thread); evicts nothing."""
        if self.reserve_n == 0 or gids.size == 0 or not self._reserve_free:
            return 0
        g = np.unique(gids.astype(np.int64))
        g = g[self.slot_for_id[g] < 0]
        if self._inflight:
            g = np.asarray([x for x in g.tolist() if x not in self._inflight], dtype=np.int64)
        if g.size == 0:
            return 0
        g = g[: len(self._reserve_free)]
        for gid in g.tolist():
            rslot = self._reserve_free.pop()
            bufs = self._read_dest(int(rslot))
            self._inflight[int(gid)] = (
                int(rslot),
                self.store.submit(int(gid), bufs),
                time.monotonic(),
            )
        with self._lock:
            self.prefetched += int(g.size)
            self.prefetch_bytes += int(g.size) * self.record_bytes
        return int(g.size)
