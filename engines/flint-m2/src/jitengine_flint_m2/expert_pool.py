"""Offload mode's expert pool: `slots` routed-expert records in unified memory, read from the checkpoint on demand.

Each of the nine per-expert tensors (gate / up / down x weight / scales / biases) is one `[slots, ...]` array that
`mx.gather_qmm` indexes by slot, so a pooled expert is the same bytes, run by the same kernel, as the resident banks'
expert: the graph backend's outputs do not depend on the budget. A record missing from the pool is read straight from
its nine checkpoint tensors (expert e of a `[256, ...]` tensor is one contiguous slice) with `preadv` and F_NOCACHE,
into the pool's own memory when MLX exposes it to the host ("direct" landing, proved at startup with a sentinel), else
through a staging buffer and an MLX scatter ("scatter").

Overwriting a slot is safe because `slots_for` runs on the worker thread after the layer's router ids were evaluated:
everything queued before them, every earlier read of the pool among it, has finished on the in-order GPU stream. The
victim is the slot with the lowest decayed use count, then the oldest last use, then the lowest slot; the slots the
current layer needs are never victims, so the pool must hold one layer's 256 experts (budget.plan_offload sees to it).

A miss layer's records are read as their 9 m checkpoint slices, all queued on the I/O workers at once, so a layer
missing one record still reads at the drive's queue depth instead of nine serial reads. Victims come from a lazy
min-heap of (use, last, slot, version) entries, O(log slots) per touched slot instead of a sort over every slot.

Annex (decode-time capacity): the budget keeps OFFLOAD_HEADROOM_BYTES free for prefill transients, which decode does not
use. While decoding, the engine opens an annex there: an exclusive victim tier of `annex_max` record rows in host memory.
An evicted record is copied into it (the least recently spilled row is reused when it is full), and a miss found in it is
copied back into its victim slot instead of read from the drive (0.04 ms against ~1.4 ms on the M2's SSD). The GPU only
ever reads the base arrays, so decode runs the same ops on the same bytes; replaying recorded traces gives the
same drive misses as a base pool larger by the annex. Closing it (before a prefill or a history growth) unmaps it.
Direct landing only.
"""

from __future__ import annotations

import fcntl
import heapq
import mmap
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jitengine_flint_m2.loader import QLinear
from jitengine_flint_m2.model.config import FlintConfig
from jitengine_flint_m2.weight_banks import EXPERT_PROJ, QSUFFIX

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
WORKERS = 8  # concurrent record reads; deeper queues lengthen a miss without adding throughput
DECAY_LAYERS = (
    64  # halve the use counts every DECAY_LAYERS x n_layers lookups (about 64 tokens of decode)
)
REBASE_EPOCHS = (
    1024  # renormalise the integer use counts after this many halvings (about 65k tokens of decode)
)


def _writable_host_view(a) -> np.ndarray | None:
    """A numpy array aliasing `a`'s unified-memory buffer, or None if MLX does not expose one."""
    import mlx.core as mx

    mx.eval(a)
    for get in (lambda: np.from_dlpack(a), lambda: np.array(a, copy=False)):
        try:
            v = get()
        except Exception:  # noqa: BLE001  (dtype or protocol unsupported on this MLX)
            continue
        if v.flags.writeable:
            return v
        try:  # a read-only export still gives the buffer's address
            addr, _ = v.__array_interface__["data"]
            buf = (np.ctypeslib.as_ctypes_type(v.dtype) * v.size).from_address(addr)
            w = np.frombuffer(buf, dtype=v.dtype).reshape(v.shape)
            if w.flags.writeable:
                return w
        except Exception:  # noqa: BLE001
            continue
    return None


class _Field:
    """One per-expert tensor as a `[slots, ...]` pool array, and where expert e of layer l sits in the checkpoint."""

    def __init__(self, cfg: FlintConfig, index: dict, proj: str, suffix: str, slots: int):
        import mlx.core as mx

        self.name = f"{proj}.{suffix}"
        first = f"model.layers.{cfg.moe_layers[0]}.mlp.switch_mlp.{proj}.{suffix}"
        _, dtype, shape, _, nbytes = index[first]
        self.nbytes = nbytes // cfg.n_experts  # one expert's slice
        self.src = {}  # layer -> (path, byte offset of expert 0)
        for layer in cfg.moe_layers:
            path, dt, sh, off, nb = index[f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"]
            if dt != dtype or sh != shape or nb != nbytes:
                raise ValueError(
                    f"layer {layer} {self.name} is {dt} {sh}, layer {cfg.moe_layers[0]} {dtype} {shape}"
                )
            self.src[layer] = (path, off)
        if dtype == "U8":
            self.np_dtype = np.uint32
            self.store = mx.zeros((slots, shape[1], shape[2] // 4), dtype=mx.uint32)
            self.tensor = self.store
        else:  # BF16, kept as uint16 so the host can write it, read by the GPU as bfloat16
            self.np_dtype = np.uint16
            self.store = mx.zeros((slots, shape[1], shape[2]), dtype=mx.uint16)
            self.tensor = self.store.view(mx.bfloat16)
        mx.eval(self.store, self.tensor)
        self.host = None


class ExpertPool:
    """`slots` expert records over the global (layer, expert) ids layer * 256 + expert, and up to `annex_max` more in
    the host-memory annex while it is open."""

    def __init__(
        self,
        cfg: FlintConfig,
        index: dict,
        slots: int,
        log=print,
        workers: int = WORKERS,
        annex_max: int = 0,
        speculate: bool = False,
    ):
        if slots < cfg.n_experts:
            raise ValueError(
                f"the expert pool needs at least {cfg.n_experts} slots (one layer), got {slots}"
            )
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.slots = int(slots)
        self.fields = [_Field(cfg, index, p, s, self.slots) for p in EXPERT_PROJ for s in QSUFFIX]
        self.record_bytes = sum(f.nbytes for f in self.fields)
        self.offsets = np.cumsum(
            [0] + [f.nbytes for f in self.fields]
        ).tolist()  # field f at [offsets[f], offsets[f+1])
        self.annex_max = max(0, int(annex_max))
        self.speculate = bool(
            speculate
        )  # decode MoE layers run on device_slots, checked after (model._mlp)
        self._table = (
            None  # device copy of slot_for_id for device_slots; ensure drops it on any change
        )
        self.annex_slots = 0  # rows of the open annex
        self._annex_map = None
        self._annex: np.ndarray | None = None  # [annex_slots, record_bytes] uint8 over the mmap
        self._row_of: OrderedDict = OrderedDict()  # gid -> annex row, least recently spilled first
        self._free_rows: list = []
        self.annex_hits = self.annex_saves = self.annex_opens = self.annex_closes = 0
        self._evicted: list = []
        self.log = log
        by = {f.name: f for f in self.fields}
        self.experts = tuple(
            QLinear(
                by[f"{p}.weight"].tensor,
                by[f"{p}.scales"].tensor,
                by[f"{p}.biases"].tensor,
                cfg.bits,
                cfg.group_size,
            )
            for p in EXPERT_PROJ
        )
        n_ids = cfg.n_layers * cfg.n_experts
        self.slot_for_id = np.full(n_ids, -1, dtype=np.int64)
        self.id_of_slot = np.full(self.slots, -1, dtype=np.int64)
        # use count x 2**epoch as exact ints: a halving is epoch += 1, which keeps every key's order, so the heap stays
        # valid; a use adds 2**epoch. heap holds (use, last, slot, ver); an entry is live iff ver == ver[slot].
        self.use = [0] * self.slots
        self.last = [0] * self.slots
        self.ver = [0] * self.slots
        self.epoch = 0
        self.heap: list = []
        self.free = list(range(self.slots - 1, -1, -1))
        self.clock = 0
        self.fds: dict = {}
        self._io = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="flint-expert-io"
        )
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0
        self.read_bytes = 0
        self.read_s = 0.0
        self.landing = self._landing()
        log(
            f"[pool] {self.slots:,} expert slots x {self.record_bytes:,} B = "
            f"{self.slots * self.record_bytes / 2**30:.2f} GiB in {len(self.fields)} arrays, landing={self.landing}"
        )

    # ------------------------------------------------------------------ landing
    def _landing(self) -> str:
        """ "direct" when a host write into every pool array shows through to the tensor the GPU reads, else "scatter"."""
        import mlx.core as mx

        views = [_writable_host_view(f.store) for f in self.fields]
        ok = all(v is not None for v in views)
        for f, v in zip(self.fields, views if ok else ()):
            flat = v.reshape(v.shape[0], -1)
            old = int(flat[0, 0])
            flat[0, 0] = 7 if f.np_dtype is np.uint32 else 0x4000  # uint32 7, or bf16 2.0
            seen = float(mx.sum(f.tensor[0].reshape(-1)[:1]).item())
            flat[0, 0] = old
            ok = ok and seen == (7.0 if f.np_dtype is np.uint32 else 2.0)
        for f, v in zip(self.fields, views):
            # scatter: the bf16 fields drop their uint16 store, so a scatter into the tensor can reuse its buffer
            f.host = v if ok else None
            if not ok:
                f.store = None
        return "direct" if ok else "scatter"

    # ------------------------------------------------------------------ reads
    def _fd(self, path: str) -> int:
        fd = self.fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            except OSError:
                pass
            self.fds[path] = fd
        return fd

    @staticmethod
    def _pread_full(fd: int, dst: memoryview, off: int) -> None:
        got = 0
        while got < len(dst):
            n = os.preadv(fd, [dst[got:]], off + got)
            if n <= 0:
                raise OSError(f"short read at byte {off + got} (wanted {len(dst) - got} more)")
            got += n

    def _pieces(self, gid: int, dsts: list) -> list:
        """The record's nine reads (fd, destination, checkpoint offset), field f of expert e at base + e * f.nbytes."""
        layer, e = divmod(int(gid), self.n_experts)
        out = []
        for f, dst in zip(self.fields, dsts):
            path, base = f.src[layer]
            out.append((self._fd(path), dst, base + e * f.nbytes))
        return out

    def _read_record(self, gid: int, dsts: list) -> None:
        for fd, dst, off in self._pieces(gid, dsts):
            self._pread_full(fd, dst, off)

    def _read_piece(self, fd: int, dst: memoryview, off: int) -> None:
        self._pread_full(fd, dst, off)

    def _fill(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Read records `gids` into `slots` (reserved, not yet published): every slice of every record is its own job
        on the I/O workers, so even one missing record is read eight slices at a time."""
        import mlx.core as mx

        t0 = time.perf_counter()
        if self.landing == "direct":
            dsts = [[memoryview(f.host[s]).cast("B") for f in self.fields] for s in slots]
            staging = None
        else:
            staging = [np.empty((len(gids), f.nbytes), dtype=np.uint8) for f in self.fields]
            dsts = [[memoryview(st[i]) for st in staging] for i in range(len(gids))]
        jobs = [p for g, d in zip(gids, dsts) for p in self._pieces(g, d)]
        errors = [e for e in self._io.map(lambda j: self._try(*j), jobs) if e is not None]
        if errors:
            raise errors[0]
        if staging is not None:
            idx = mx.array(np.asarray(slots, dtype=np.int32))
            for f, st in zip(self.fields, staging):
                vals = mx.array(st.view(f.np_dtype)).reshape(len(gids), *f.tensor.shape[1:])
                if f.np_dtype is np.uint16:
                    vals = vals.view(mx.bfloat16)
                f.tensor[idx] = vals  # rebinds f.tensor in place, so the QLinear views see it
                mx.eval(f.tensor)
        self.read_s += time.perf_counter() - t0
        self.read_bytes += len(gids) * self.record_bytes

    def _try(self, fd: int, dst: memoryview, off: int):
        try:
            self._read_piece(fd, dst, off)
        except OSError as exc:
            return exc
        return None

    # ------------------------------------------------------------------ residency
    def _touch(self, slot: int, count: bool = True) -> None:
        """Count one use of `slot` (none when `count` is false) at the current clock and push its new heap key."""
        if count:
            self.use[slot] += 1 << self.epoch
        self.last[slot] = self.clock
        self.ver[slot] += 1
        heapq.heappush(self.heap, (self.use[slot], self.clock, slot, self.ver[slot]))

    def _reheap(self) -> None:
        """One live entry per published slot (free slots are not candidates: the free list is used first)."""
        self.heap = [
            (self.use[s], self.last[s], s, self.ver[s])
            for s in np.flatnonzero(self.id_of_slot >= 0).tolist()
        ]
        heapq.heapify(self.heap)

    def _decay(self) -> None:
        self.epoch += 1
        if (
            self.epoch >= REBASE_EPOCHS
        ):  # keep the ints small; counts older than the window drop to 0
            self.use = [u >> self.epoch for u in self.use]
            self.epoch = 0
            self._reheap()

    def _victims(self, k: int, protected: list) -> list:
        """k slots to fill: free ones first, then the published slots with the lowest (use, last, slot), never one in
        `protected` (the slots the current lookup already hits)."""
        skip = set(protected)
        if k > self.slots - len(skip):
            raise RuntimeError(
                f"the expert pool ({self.slots} slots) cannot hold one layer's experts"
            )
        take = [self.free.pop() for _ in range(min(k, len(self.free)))]
        k -= len(take)
        if k > 0:
            held = []
            heap, ver = self.heap, self.ver
            while k > 0:
                if not heap:
                    raise RuntimeError("expert pool heap lost a published slot")
                ent = heapq.heappop(heap)
                v = ent[2]
                if ent[3] != ver[v]:
                    continue  # stale
                if v in skip:
                    held.append(ent)
                    continue
                if self.id_of_slot[v] >= 0:
                    self.slot_for_id[self.id_of_slot[v]] = -1
                    self._evicted.append((v, int(self.id_of_slot[v])))
                self.id_of_slot[v] = -1
                self.evictions += 1
                take.append(v)
                k -= 1
            for ent in held:
                heapq.heappush(heap, ent)
        return take

    def ensure(self, gids: np.ndarray, count: bool = True) -> np.ndarray:
        """Slots of the unique global ids `gids`, reading the missing records first. `count=False` (prefill's
        layer tables) refreshes their last use but adds no use count: a long prompt touches nearly every expert of
        every layer, and counting that once-per-chunk sweep evicts the experts decode keeps returning to
        (simulated on recorded lookup traces: 8192-token prompts miss 17-28% less, 1024-token prompts unchanged)."""
        self.clock += 1
        if self.clock % (DECAY_LAYERS * self.cfg.n_layers) == 0:
            self._decay()
        cur = self.slot_for_id[gids]
        miss = gids[cur < 0]
        n_annex = 0
        if miss.size:
            self._table = None  # slot_for_id changes (evictions, then publishes) only here
            protected = cur[cur >= 0].tolist()
            self._evicted = (
                evicted
            ) = []  # (slot, gid) of every record _victims evicts, for the annex
            slots = np.asarray(self._victims(int(miss.size), protected), dtype=np.int64)
            try:
                drive = np.ones(miss.size, dtype=bool)
                if self._annex is not None:
                    drive, n_annex = self._annex_exchange(miss, slots, evicted)
                if drive.any():
                    self._fill(miss[drive], slots[drive])
            except BaseException:
                self.free.extend(int(s) for s in slots)  # never publish a slot whose read failed
                raise
            self.id_of_slot[slots] = miss
            self.slot_for_id[miss] = slots
            for s in slots.tolist():
                self.use[s] = 0
        out = self.slot_for_id[gids]
        for s in set(out.tolist()):
            self._touch(s, count)
        if len(self.heap) > 4 * self.slots:
            self._reheap()
        with self._lock:
            self.hits += int(gids.size - miss.size + n_annex)
            self.misses += int(miss.size - n_annex)
            self.annex_hits += n_annex
        return out

    # ------------------------------------------------------------------ annex
    def open_annex(self, free_bytes: int) -> int:
        """Open the annex with min(annex_max, free_bytes // record_bytes) rows of anonymous memory (direct landing only:
        records move through the base arrays' host views). Returns the rows opened (0: none, or it is already open)."""
        n = min(self.annex_max, max(0, int(free_bytes)) // self.record_bytes)
        if self._annex is not None or self.landing != "direct" or n <= 0:
            return 0
        self._annex_map = mmap.mmap(-1, n * self.record_bytes)
        self._annex = np.frombuffer(self._annex_map, dtype=np.uint8).reshape(n, self.record_bytes)
        self._row_of.clear()
        self._free_rows = list(range(n - 1, -1, -1))
        self.annex_slots = n
        self.annex_opens += 1
        return n

    def close_annex(self) -> int:
        """Drop every annex record and unmap the annex (its memory is back when this returns). Returns the records
        dropped."""
        if self._annex is None:
            return 0
        held = len(self._row_of)
        self._row_of.clear()
        self._free_rows = []
        self._annex = None
        try:
            self._annex_map.close()
        except BufferError:  # a view outlived its lookup: the map is unmapped when that view goes
            self.log(
                "[pool] annex close: a view of the annex is still alive; it unmaps when the view goes"
            )
        self._annex_map = None
        self.annex_slots = 0
        self.annex_closes += 1
        return held

    def _record_views(self, slot: int) -> list:
        return [
            np.frombuffer(memoryview(f.host[slot]).cast("B"), dtype=np.uint8) for f in self.fields
        ]

    def _annex_exchange(self, miss: np.ndarray, slots: np.ndarray, evicted: list) -> tuple:
        """Copy this lookup's victims into the annex and its annex-held misses into their slots. Returns (mask of the
        misses still to read from the drive, annex hits). Victims are saved before any slot is overwritten; the rows
        of this lookup's hits are taken out first, so no victim is saved over a row still to be copied."""
        rows = [self._row_of.pop(int(g), -1) for g in miss]
        a, off = self._annex, self.offsets
        for v, g in evicted:
            if self._free_rows:
                r = self._free_rows.pop()
            elif self._row_of:
                r = self._row_of.popitem(last=False)[1]
            else:
                break  # every row holds a record this lookup is copying back
            for i, src in enumerate(self._record_views(v)):
                a[r, off[i] : off[i + 1]] = src
            self._row_of[g] = r
            self.annex_saves += 1
        drive = np.ones(miss.size, dtype=bool)
        for j, (s, r) in enumerate(zip(slots.tolist(), rows)):
            if r < 0:
                continue
            for i, dst in enumerate(self._record_views(s)):
                dst[:] = a[r, off[i] : off[i + 1]]
            self._free_rows.append(r)
            drive[j] = False
        return drive, int((~drive).sum())

    def slots_for(self, layer: int, ids):
        """The slot of every routed (token, expert) in `ids` (uint32 [..., k], lazy) of `layer`, as uint32 like ids.
        Evaluates `ids` (the host needs them), then reads the records the pool lacks."""
        import mlx.core as mx

        mx.eval(ids)
        e = np.array(ids).astype(np.int64)
        gids, inv = np.unique(e.reshape(-1) + layer * self.n_experts, return_inverse=True)
        slots = self.ensure(gids)[inv].reshape(e.shape)
        return mx.array(slots.astype(np.uint32))

    def device_slots(self, layer: int, ids):
        """Lazy: the slots `slot_for_id` holds now for `ids` (uint32 [..., k]) of `layer`, taken on the device from a
        copy of the map that `ensure` drops whenever it changes (a missing record reads slot 0). Equal to what
        `slots_for` returns exactly when `check_slots` then finds every record resident."""
        import mlx.core as mx

        if self._table is None:
            self._table = mx.array(np.maximum(self.slot_for_id, 0).astype(np.uint32))
        return mx.take(
            self._table, ids.astype(mx.uint32) + mx.array(layer * self.n_experts, dtype=mx.uint32)
        )

    def check_slots(self, layer: int, ids):
        """The lookup `slots_for` does (evaluates `ids`, the same ensure), for a graph built on `device_slots`:
        None when every record was resident (those slots were right), else (the slots after the reads, uint32 numpy,
        and the positions whose record was not resident, bool numpy, both shaped like `ids`). A resident position's
        slot is the one device_slots gave it: ensure never moves a record the lookup hits."""
        import mlx.core as mx

        mx.eval(ids)
        e = np.array(ids).astype(np.int64)
        gids, inv = np.unique(e.reshape(-1) + layer * self.n_experts, return_inverse=True)
        was = self.slot_for_id[gids] >= 0
        if was.all():
            self.ensure(gids)
            return None
        slots = self.ensure(gids)[inv].reshape(e.shape)
        return slots.astype(np.uint32), ~was[inv].reshape(e.shape)

    def layer_table(self, layer: int, ids) -> tuple:
        """`layer`'s experts as three [256, ...] QLinear tables taken from the pool (row e = expert e; rows of experts
        not in `ids` hold any resident slot and are never read), for a gather indexed by expert id like the banks'.
        MLX chooses gather_qmm's sorted kernel from the rows per expert of the weight array, so a gather over the
        pool's [slots, ...] arrays could take another kernel than the banks' [256, ...] views, and round otherwise.
        Evaluates `ids`; the takes are lazy and finish before the next layer's lookup (it evaluates this layer's
        output), so no later read of the pool overwrites a slot they read."""
        import mlx.core as mx

        if self._annex is not None:  # the engine closes the annex before every prefill
            raise RuntimeError("expert pool: a prefill layer table with the annex open")
        mx.eval(ids)
        e = np.unique(np.array(ids).astype(np.int64))
        slots = self.ensure(e + layer * self.n_experts, count=False)
        table = np.full(self.n_experts, slots[0], dtype=np.int32)
        table[e] = slots
        idx = mx.array(table)
        return tuple(
            QLinear(
                mx.take(q.w, idx, axis=0),
                mx.take(q.s, idx, axis=0),
                mx.take(q.b, idx, axis=0),
                q.bits,
                q.group,
            )
            for q in self.experts
        )

    # ------------------------------------------------------------------ checks, stats, release
    def selfcheck(self) -> None:
        """Read three records through the pool and compare every field with a direct read of the checkpoint."""
        import mlx.core as mx

        moe = self.cfg.moe_layers
        picks = [
            (moe[0], 0),
            (moe[len(moe) // 2], 137 % self.n_experts),
            (moe[-1], self.n_experts - 1),
        ]
        gids = np.array(sorted(layer * self.n_experts + e for layer, e in picks), dtype=np.int64)
        slots = self.ensure(gids)
        for g, s in zip(gids, slots):
            want = [np.empty(f.nbytes, dtype=np.uint8) for f in self.fields]
            self._read_record(int(g), [memoryview(w) for w in want])
            for f, w in zip(self.fields, want):
                t = f.tensor[int(s)]
                got = np.array(t if f.np_dtype is np.uint32 else t.view(mx.uint16)).tobytes()
                if got != w.tobytes():
                    layer, e = divmod(int(g), self.n_experts)
                    raise RuntimeError(
                        f"expert pool self-check: layer {layer} expert {e} {f.name} differs"
                    )
            mx.eval(*[f.tensor for f in self.fields])
        with self._lock:
            self.hits = self.misses = self.evictions = self.annex_hits = self.annex_saves = 0
            self.read_bytes, self.read_s = 0, 0.0

    def resident(self) -> int:
        return int((self.id_of_slot >= 0).sum())

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "slots": self.slots,
                "resident": self.resident(),
                "record_bytes": self.record_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else None,
                "evictions": self.evictions,
                "read_bytes": self.read_bytes,
                "read_s": round(self.read_s, 3),
                "landing": self.landing,
                "annex_slots": self.annex_slots,
                "annex_max": self.annex_max,
                "annex_resident": len(self._row_of),
                "annex_hits": self.annex_hits,
                "annex_saves": self.annex_saves,
                "annex_opens": self.annex_opens,
                "annex_closes": self.annex_closes,
            }

    def close(self) -> None:
        self.close_annex()
        self._io.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        for f in self.fields:
            f.host = f.store = f.tensor = None
        self.experts = None
