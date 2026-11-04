"""Resident expert pool: `slots` expert records in unified memory, filled from the pack on demand.

Each expert tensor is a `[slots, ...]` array that `mx.gather_qmm` indexes by slot id. Overwriting a
slot in place is safe only because `ensure()` runs on the model thread after a host sync on the
layer's router indices (one in-order GPU stream), so no queued kernel can still be reading it.
"""

from __future__ import annotations

import logging
import threading
import time

import mlx.core as mx
import numpy as np

from jitengine_qwen3_6_35b_a3b_m2.linear import QLinear

logger = logging.getLogger(__name__)

# Fallback only; the engine passes the device's own `max_buffer_length`.
MAX_BUFFER_BYTES = int(13.3 * 2**30)
DECAY_TOKENS = 64  # halve use counts every N per-layer calls * layers
DECAY_FACTOR = 0.5
FILL_ATTEMPTS = 2  # one bounded retry of a faulting record read

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
    try:  # a read-only export still gives the buffer's address
        addr, _ro = v.__array_interface__["data"]
        ctype = np.ctypeslib.as_ctypes_type(v.dtype)
        buf = (ctype * v.size).from_address(addr)
        w = np.frombuffer(buf, dtype=v.dtype).reshape(v.shape)
        return w if w.flags.writeable else None
    except Exception:  # noqa: BLE001
        return None


class _Field:
    """One of the nine expert tensors, as a `[slots, ...]` pool array."""

    __slots__ = ("name", "dtype", "shape", "nbytes", "offset", "store", "tensor", "host")

    def __init__(self, spec: dict, slots: int):
        self.name = spec["name"]
        self.dtype = spec["dtype"]
        self.shape = tuple(int(d) for d in spec["shape"])
        self.nbytes = int(spec["bytes"])
        self.offset = int(spec["offset"])
        if self.dtype not in _MX_DTYPE:
            raise ValueError(f"unsupported expert dtype {self.dtype} for {self.name}")
        self.store = mx.zeros((slots, *self.shape), dtype=_MX_STORAGE[self.dtype])
        mx.eval(self.store)
        self.tensor = (
            self.store
            if _MX_STORAGE[self.dtype] is _MX_DTYPE[self.dtype]
            else self.store.view(_MX_DTYPE[self.dtype])
        )
        mx.eval(self.tensor)
        self.host: np.ndarray | None = None


class OffloadMoeCache:
    """`slots` expert records over one global (layer, expert) id space."""

    def __init__(
        self,
        store,
        num_layers: int,
        num_experts: int,
        slots: int,
        fields: list[dict],
        group_size: int,
        max_buffer_bytes: int | None = None,
    ):
        self.store = store
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.num_ids = self.num_layers * self.num_experts
        self.slots = int(slots)
        self.record_bytes = sum(int(f["bytes"]) for f in fields)
        self._lock = threading.Lock()

        if self.slots <= 0:
            raise ValueError("expert pool needs at least one slot")
        limit = int(max_buffer_bytes or MAX_BUFFER_BYTES)
        # checked before allocating: an oversized Metal buffer is a fault, not an error
        biggest = max(self.slots * int(f["bytes"]) for f in fields)
        if biggest > limit:
            raise ValueError(
                f"pool array of {biggest / 2**30:.1f} GiB exceeds MLX's "
                f"{limit / 2**30:.1f} GiB max_buffer_length; reduce slots"
            )
        self.fields = [_Field(f, self.slots) for f in fields]
        self.max_array_bytes = biggest

        self.slot_for_id = np.full(self.num_ids, -1, dtype=np.int32)
        self.id_of_slot = np.full(self.slots, -1, dtype=np.int64)
        self.use = np.zeros(self.slots, dtype=np.float32)  # decayed use count
        self.last = np.zeros(self.slots, dtype=np.int64)  # clock of last use
        self._free = list(range(self.slots - 1, -1, -1))
        self.clock = 0
        self._table_writes = 0  # monotone version of slot_for_id (device table staleness)

        self.hits = 0
        self.misses = 0
        self.miss_bytes = 0
        self.evictions = 0
        self.fetch_seconds = 0.0
        # A retried read is not a new miss, so `miss_bytes == misses * record_bytes` stays exact.
        self.read_retries = 0
        self.retry_bytes = 0
        self.seed_records = 0
        self.seed_bytes = 0
        self.wave_calls = 0
        self.calls = 0

        self.landing = self._landing()
        gs = int(group_size)
        by = {f.name: f for f in self.fields}
        self.gate_proj = QLinear(
            by["gate_proj.weight"].tensor,
            by["gate_proj.scales"].tensor,
            by["gate_proj.biases"].tensor,
            gs,
        )
        self.up_proj = QLinear(
            by["up_proj.weight"].tensor,
            by["up_proj.scales"].tensor,
            by["up_proj.biases"].tensor,
            gs,
        )
        self.down_proj = QLinear(
            by["down_proj.weight"].tensor,
            by["down_proj.scales"].tensor,
            by["down_proj.biases"].tensor,
            gs,
        )
        logger.info(
            f"[pool] {self.slots} slots x {self.record_bytes} B = "
            f"{self.slots * self.record_bytes / 2**30:.2f} GiB in {len(self.fields)} arrays "
            f"(largest {biggest / 2**30:.2f} GiB), landing={self.landing}"
        )

    # ------------------------------------------------------------------ landing path
    def _landing(self) -> str:
        """Prefer zero-copy `preadv` into the pool arrays; prove it with a sentinel or fall back."""
        views = [_writable_host_view(f.store) for f in self.fields]
        if any(v is None for v in views):
            return "scatter"
        for f, v in zip(self.fields, views):
            f.host = v
        # A host write must show through to the tensor the GPU reads, including a bf16 `view()`.
        for f in self.fields:
            flat = f.host.reshape(f.host.shape[0], -1)
            old = int(flat[0, 0])
            flat[0, 0] = 7 if f.tensor is f.store else 0x4000  # bf16 2.0
            want = 7.0 if f.tensor is f.store else 2.0
            seen = float(mx.sum(f.tensor[0].reshape(-1)[:1]).item())
            flat[0, 0] = old
            if seen != want:
                for g in self.fields:
                    g.host = None
                return "scatter"
        return "direct"

    # ------------------------------------------------------------------ residency
    def resident(self) -> int:
        return int((self.slot_for_id >= 0).sum())

    def capacity_fraction(self) -> float:
        return self.slots / float(self.num_ids)

    @property
    def wave_capacity(self) -> int:
        """Distinct experts one gather may reference: the whole pool."""
        return self.slots

    def _decay(self) -> None:
        self.use *= DECAY_FACTOR

    # ------------------------------------------------------------------ the fetch
    def _fill(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Read records `gids` into the reserved `slots`, retrying a failed read once.

        The slots stay unpublished until `ensure` publishes them after this returns, so a
        half-written slot is never served and a retry cannot disturb a reachable slot.
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

    def _fill_once(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """One attempt: nine destinations per record, one `preadv` each (or the scatter fallback)."""
        if self.landing == "direct":
            jobs = [(int(g), [f.host[int(s)] for f in self.fields]) for g, s in zip(gids, slots)]
            self.fetch_seconds += self.store.read_many(jobs)
            return
        n = len(gids)
        staging = np.empty((n, self.record_bytes), dtype=np.uint8)
        jobs = [(int(g), [staging[i]]) for i, g in enumerate(gids)]
        self.fetch_seconds += self.store.read_many(jobs)
        idx = mx.array(np.asarray(slots, dtype=np.int32))
        for f in self.fields:
            raw = np.ascontiguousarray(staging[:, f.offset : f.offset + f.nbytes])
            vals = mx.array(raw.view(_NP_STORAGE[f.dtype]).reshape(n, *f.shape))
            if f.tensor is not f.store:
                vals = vals.view(_MX_DTYPE[f.dtype])
            f.tensor[idx] = vals
            mx.eval(f.tensor)

    def _reserve(self, missing: np.ndarray, protected: np.ndarray) -> np.ndarray:
        """Choose a slot for every id in `missing`: empty -> min decayed use -> oldest use.

        Reserves only: victims lose their index entry, but the incoming ids stay unpublished
        (`id_of_slot` = -1) until `_publish`, so a failed fetch leaves empty slots.
        """
        take: list[int] = []
        while self._free and len(take) < missing.size:
            take.append(self._free.pop())
        if take:
            protected[np.asarray(take, dtype=np.int64)] = True  # never also pick them as victims
        k = missing.size - len(take)
        if k > 0:
            cand = np.flatnonzero(~protected)
            if cand.size < k:
                raise RuntimeError(
                    f"expert pool cannot hold this step: needs {missing.size} more of "
                    f"{self.slots} slots with {int(protected.sum())} in use"
                )
            order = np.lexsort((self.last[cand], self.use[cand]))
            victims = cand[order[:k]]
            for v in victims:
                gid = self.id_of_slot[v]
                if gid >= 0:
                    self.slot_for_id[gid] = -1
                    self.evictions += 1
            self._table_writes += 1
            take.extend(int(v) for v in victims)
        slots = np.asarray(take, dtype=np.int64)
        self.id_of_slot[slots] = -1  # owned by the in-flight fetch, resident for nobody
        self.use[slots] = 0.0
        return slots

    def _publish(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Make `gids` resident in `slots`; call only after `_fill` has returned."""
        self.id_of_slot[slots] = gids
        self.slot_for_id[gids] = slots.astype(np.int32)
        self._table_writes += 1

    def _rollback(self, slots: np.ndarray) -> None:
        """After a failed fetch, return the never-published slots to the free list, empty."""
        self.id_of_slot[slots] = -1
        self.use[slots] = 0.0
        self._free.extend(int(s) for s in slots)

    def ensure(self, gids: np.ndarray) -> np.ndarray:
        """Make every id in `gids` (unique, global ids) resident; returns their slots.

        Model thread only; the lock just keeps `stats()` from reading half-updated counters.
        """
        self.calls += 1
        self.clock += 1
        if self.clock % (DECAY_TOKENS * self.num_layers) == 0:
            self._decay()
        cur = self.slot_for_id[gids]
        hit = cur >= 0
        n_hit = int(hit.sum())
        missing = gids[~hit]
        protected = np.zeros(self.slots, dtype=bool)
        protected[cur[hit]] = True
        if missing.size:
            slots = self._reserve(missing, protected)
            protected[slots] = True
            try:
                self._fill(missing, slots)
            except BaseException:
                # Publish nothing: no id may be served from a slot whose fill did not complete.
                self._rollback(slots)
                raise
            self._publish(missing, slots)
        with self._lock:
            self.hits += n_hit
            self.misses += int(missing.size)
            self.miss_bytes += int(missing.size) * self.record_bytes
        out = self.slot_for_id[gids]
        self.use[out] += 1.0
        self.last[out] = self.clock
        return out.astype(np.int64)

    # ------------------------------------------------------------------ the MoE entry point
    def segments(self, layer: int, expert_ids: np.ndarray):
        """(positions, slots) per wave for one layer's (token, expert) pairs.

        More than one wave only when the pool cannot hold the layer's distinct experts; each pair
        is in exactly one wave. Those waves are fetched one at a time as the caller iterates, so it
        must finish a wave's gather before taking the next: a later fetch may evict its slots.
        """
        gids = expert_ids.astype(np.int64) + layer * self.num_experts
        uniq = np.unique(gids)
        cap = self.wave_capacity
        if uniq.size <= cap:
            return [
                (
                    np.arange(gids.size, dtype=np.int64),
                    self.ensure(uniq)[np.searchsorted(uniq, gids)],
                )
            ]
        return _Waves(self, gids, uniq, cap)

    def prefill_seed(self, per_layer: int) -> None:
        """Warm start: load the first `per_layer` experts of every layer as one sequential read."""
        per_layer = int(min(per_layer, self.num_experts, self.slots // max(1, self.num_layers)))
        if per_layer <= 0:
            return
        t0 = time.monotonic()
        for layer in range(self.num_layers):
            gids = np.arange(per_layer, dtype=np.int64) + layer * self.num_experts
            self.ensure(gids)
        # the seed is not serving traffic: it is a load-time fill, not a cache miss
        with self._lock:
            seeded = per_layer * self.num_layers
            self.seed_records = self.misses
            self.seed_bytes = self.miss_bytes
            self.hits = 0
            self.misses = 0
            self.miss_bytes = 0
        self.use[:] = 0.0
        logger.info(
            f"[pool] seeded {seeded} experts ({per_layer}/layer, "
            f"{seeded * self.record_bytes / 2**30:.2f} GiB) in {time.monotonic() - t0:.0f}s"
        )

    # ------------------------------------------------------------------ device slot table
    def device_slot_table(self) -> mx.array:
        """`slot_for_id` on the GPU as int32 [num_ids], clamped to slot 0 where not resident.

        `gather_qmm` must never see an out-of-range index; callers discard the clamped rows
        (`Qwen3_5MoeSparseMoeBlock._experts_pooled_spec`). Refreshed lazily by scattering only
        changed entries.
        """
        mirror = getattr(self, "_dev_mirror", None)
        if mirror is None:
            self._dev_mirror = self.slot_for_id.copy()
            self._dev_table = mx.array(np.maximum(self._dev_mirror, 0).astype(np.int32))
            mx.eval(self._dev_table)
            self._dev_ver = self._table_writes
            return self._dev_table
        ver = self._table_writes  # bumped by every write to slot_for_id
        if ver == self._dev_ver:
            return self._dev_table
        self._dev_ver = ver
        changed = np.flatnonzero(self.slot_for_id != mirror)
        if changed.size:
            vals = self.slot_for_id[changed]
            tab = self._dev_table
            tab[mx.array(changed.astype(np.int32))] = mx.array(np.maximum(vals, 0).astype(np.int32))
            self._dev_table = tab
            mirror[changed] = vals
        return self._dev_table

    def all_resident(self, gids: np.ndarray) -> bool:
        return bool((self.slot_for_id[gids] >= 0).all())

    # ------------------------------------------------------------------ release (memory/control.py)
    def release(self) -> None:
        """Drop every reference this pool holds to its Metal buffers; the pool is unusable after.

        The caller must synchronize the GPU and drop the model's references first.
        """
        for f in self.fields:
            f.host = None
            f.tensor = None
            f.store = None
        self.gate_proj = self.up_proj = self.down_proj = None
        self.fields = []
        self.slots = 0
        self._dev_table = None
        self._dev_mirror = None

    def lifetime_counters(self) -> dict:
        """Counters that survive a rebuild (summed into the replacement pool's totals)."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "miss_bytes": self.miss_bytes,
                "evictions": self.evictions,
                "fetch_seconds": self.fetch_seconds,
                "read_retries": self.read_retries,
                "retry_bytes": self.retry_bytes,
                "seed_records": self.seed_records,
                "seed_bytes": self.seed_bytes,
                "wave_calls": self.wave_calls,
            }

    def carry_counters(self, prev: dict) -> None:
        """Add a previous pool's totals so `/stats` counters stay monotone across a resize."""
        with self._lock:
            self.hits += prev["hits"]
            self.misses += prev["misses"]
            self.miss_bytes += prev["miss_bytes"]
            self.evictions += prev["evictions"]
            self.fetch_seconds += prev["fetch_seconds"]
            self.read_retries += prev["read_retries"]
            self.retry_bytes += prev["retry_bytes"]
            self.seed_records += prev["seed_records"]
            self.seed_bytes += prev["seed_bytes"]
            self.wave_calls += prev["wave_calls"]

    # ------------------------------------------------------------------ observability
    def counters(self) -> dict:
        with self._lock:
            return {
                "expert_cache_hits": self.hits,
                "expert_cache_misses": self.misses,
                "expert_miss_bytes": self.miss_bytes,
            }

    def stats(self) -> dict:
        c = self.counters()
        total = c["expert_cache_hits"] + c["expert_cache_misses"]
        return {
            **c,
            "expert_cache_hit_rate": (c["expert_cache_hits"] / total) if total else 0.0,
            "expert_pool_slots": self.slots,
            "expert_pool_resident": self.resident(),
            "expert_pool_capacity_fraction": self.capacity_fraction(),
            "expert_pool_bytes": self.slots * self.record_bytes,
            "expert_pool_largest_array_bytes": self.max_array_bytes,
            "expert_pool_evictions": self.evictions,
            "expert_seed_records": self.seed_records,
            "expert_seed_bytes": self.seed_bytes,
            "expert_pool_landing": self.landing,
            "expert_fetch_seconds": self.fetch_seconds,
            "expert_read_retries": self.read_retries,
            "expert_retry_bytes": self.retry_bytes,
            "expert_wave_calls": self.wave_calls,
            "expert_spec_calls": getattr(self, "spec_calls", 0),
            "expert_spec_misses": getattr(self, "spec_misses", 0),
            "expert_io": self.store.stats(),
        }


class _Waves:
    """A layer's waves when its distinct experts exceed the pool, each fetched when reached."""

    def __init__(self, pool: OffloadMoeCache, gids: np.ndarray, uniq: np.ndarray, cap: int):
        self.pool, self.gids, self.uniq, self.cap = pool, gids, uniq, cap

    def __len__(self) -> int:
        return -(-int(self.uniq.size) // self.cap)

    def __iter__(self):
        self.pool.wave_calls += 1
        for s in range(0, self.uniq.size, self.cap):
            wave = self.uniq[s : s + self.cap]
            pos = np.flatnonzero(np.isin(self.gids, wave))
            yield pos, self.pool.ensure(wave)[np.searchsorted(wave, self.gids[pos])]
