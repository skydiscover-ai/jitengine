"""The resident expert pool: `slots` records of unified memory, filled from the pack on demand.

One global pool covers every (layer, expert) id. Nine `[slots, ...]` arrays mirror the expert
tensors, so `mx.gather_qmm` indexes them with slot ids, and a record lands by one `preadv` straight
into those Metal buffers (or, without a writable host view, by staging plus one scatter).

Writing into a live buffer is safe only because MLX runs this work on one in-order GPU stream and
`ensure()` runs on the model thread after a host sync on the layer's router indices, so no kernel
dispatched earlier can still be reading a slot that gets overwritten.

`OffloadMoeCache` keeps the index state, the fetch (`ensure`) and the lookups the MoE block reads;
victim selection (`victims`), wave planning (`waves`), the read pipeline (`reads`), the pack16 slot
codec (`codec`) and the stats (`stats`) are its mixins, the package's other modules. With the codec,
the slot arrays are the compact six (weights and one u16 word per group) and reads land in a staging
set first; `codec.py` says how that keeps the guarantees above. With `disk16` the store's records
are already compact (the SKYP16X1 pack) and land in those six arrays directly, as above.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.budget import GIB
from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.codec import P16Linear, Pack16CodecMixin
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.reads import (
    _MX_DTYPE,
    _MX_STORAGE,
    ReadPipelineMixin,
    _writable_host_view,
)
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.stats import PoolStatsMixin
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.victims import VictimSelectionMixin
from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.waves import WavePlanningMixin

logger = logging.getLogger(__name__)

MAX_BUFFER_BYTES = int(30.15 * 1e9)  # MLX `max_buffer_length` on the M4 Pro target
DECAY_TOKENS = 0  # 0 = plain LRU. > 0 halves use counts every N tokens.
# Library default; the engine passes the periods below.
# Engine defaults: DECAY_TOKENS_SERVED for single-row decode, DECAY_TOKENS_BATCHED for prefill,
# waves and batched decode.
DECAY_TOKENS_SERVED = 32
DECAY_TOKENS_BATCHED = 16
# Single-row decode's victim rule (`expert_pool_evict`): the cycle-aware expected next use with an
# EMA period of 64 visits, for demand and harvest victims alike (victims.py); "decay32" keeps the
# decayed-use rule there. Every other operating point keeps decayed use and LRU harvests.
EVICT_SERVED = "cyc64"
# Reserved landing region, in slots: one decode layer's demand misses. A larger request (a prefill
# wave) takes the synchronous path. Region slots come out of residency, so the region stays small.
LANDING_SLOTS = 16
CHAIN_RESERVE = 32  # the miss chain reads into the prefetch reserve, so it needs one


def reserve_slots_for(options: dict) -> int:
    """The prefetch reserve the engine asks for: 16 slots per lookahead layer (floor 32) with the
    router lookahead, else the miss chain's `CHAIN_RESERVE`; the `prefetch_reserve` option wins."""
    look = max(0, int(options.get("prefetch_lookahead", 0)))
    default = max(CHAIN_RESERVE, 16 * look) if look else CHAIN_RESERVE
    return max(0, int(options.get("prefetch_reserve", default)))


def landing_slots_for(options: dict) -> int:
    """The landing region the engine asks for: `LANDING_SLOTS`, or the `landing_slots` option."""
    return max(0, int(options.get("landing_slots", LANDING_SLOTS)))


class _Field:
    """One of the nine expert tensors, as a `[slots, ...]` pool array."""

    __slots__ = ("name", "dtype", "shape", "nbytes", "offset", "store", "tensor", "host")

    def __init__(self, spec: dict, slots: int, reserve_slots: int = 0):
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


class OffloadMoeCache(
    VictimSelectionMixin, WavePlanningMixin, ReadPipelineMixin, Pack16CodecMixin, PoolStatsMixin
):
    """`slots` expert records over one global (layer, expert) id space."""

    def __init__(
        self,
        store,
        num_layers: int,
        num_experts: int,
        slots: int,
        fields: list[dict],
        group_size: int,
        decay_tokens: int = DECAY_TOKENS,
        transient_prefill: bool = False,
        bits: int = 4,
        reserve_slots: int = 0,
        landing_slots: int = LANDING_SLOTS,
        ext_slots: int = 0,
        codec=None,
        ann_slots: int = 0,
        disk16: bool = False,
    ):
        """`fields` are the pack's record fields; `codec` is None (bf16 slots) or the pack16
        (Geometry, escape table) pair, whose compact fields the slot arrays then hold. `disk16`: the
        store's records are those compact fields already (`fields` describes the source record)."""
        if disk16 and codec is not None:
            fields = codec[0].compact_fields()  # what one read now brings: the slot's own bytes
        self.init_settings(
            store,
            num_layers,
            num_experts,
            slots,
            fields,
            decay_tokens,
            transient_prefill,
            bits,
        )
        self.slot_bytes = int(codec[0].slot_bytes) if codec is not None else self.record_bytes
        slot_fields = codec[0].compact_fields() if codec is not None else fields
        biggest = self.init_slot_arrays(slot_fields, group_size, ext_slots, ann_slots)

        self.init_index()
        self.init_reserve(reserve_slots)
        self.init_landing(landing_slots)
        self.init_codec(codec, disk16)
        self.init_eviction_state()

        self.init_counters()

        self.init_scope()

        self.landing = (
            self._codec_landing() if (self.codec and not self.disk16) else self._landing()
        )
        if self.disk16 and self.landing != "direct":
            raise InvariantError(
                "compact records from disk need a writable host view of the slots; none here"
            )
        self.init_projections(group_size)
        logger.debug(
            f"[pool] {self.ann0} slots x {self.slot_bytes} B = "
            f"{self.ann0 * self.slot_bytes / GIB:.2f} GiB in {len(self.fields)} arrays "
            f"(+ an annex of {self.ann_slots} borrowed slots, backed only while lent) "
            f"(largest {biggest / 2**30:.2f} GiB), landing={self.landing}, "
            f"codec={self.codec or 'bf16'}, staging {self.staging_n} x {self.record_bytes} B"
        )

    def init_settings(
        self,
        store,
        num_layers: int,
        num_experts: int,
        slots: int,
        fields: list[dict],
        decay_tokens: int,
        transient_prefill: bool,
        bits: int,
    ) -> None:
        """The store, the pool's geometry (layers, experts, slots, record size) and its settings."""
        self.store = store
        self.bits = int(bits)
        self.transient_prefill = bool(transient_prefill)
        self.decay_tokens = int(decay_tokens)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.num_ids = self.num_layers * self.num_experts
        self.slots = int(slots)
        self.record_bytes = sum(int(f["bytes"]) for f in fields)
        self._lock = threading.Lock()

        if self.slots <= 0:
            raise ValueError("expert pool needs at least one slot")

    def init_slot_arrays(
        self, fields: list[dict], group_size: int, ext_slots: int, ann_slots: int = 0
    ) -> int:
        """The slot arrays, one per checkpoint field: the main set, and the extension set if any;
        returns the largest array's size. The annex (`ann_slots`, ids after the extension's) has no
        arrays until `regrow_ann`: it borrows memory another part of the plan holds, and only while
        that part is idle."""
        # Extension pool: `ext_slots` extra slots in a second array set, carved from budget that
        # single-stream serving leaves idle. Ids [0, slots_main) are main, [slots_main, slots) ext.
        self.slots_main = self.slots
        self.ext_slots = min(max(0, int(ext_slots)), max(0, self.num_ids - self.slots_main))
        if int(ext_slots) > self.ext_slots:
            logger.info(
                f"[pool] extension request {int(ext_slots)} clamped to {self.ext_slots} "
                f"(num_ids {self.num_ids}, main slots {self.slots_main})"
            )
        self.slots = self.slots_main + self.ext_slots
        self.ann0 = self.slots  # the annex's first slot id
        self.ann_slots = min(max(0, int(ann_slots)), max(0, self.num_ids - self.slots))
        self.slots += self.ann_slots
        self.fields_ann: list = []
        self.ann_active = False
        self.ann_live = 0
        self.ann_releases = 0
        self.ann_regrows = 0
        self.ann_dropped = 0  # records the annex held when it was released
        self.gate_proj_ann = self.up_proj_ann = self.down_proj_ann = None
        self.fields = [_Field(f, self.slots_main) for f in fields]
        self.fields_ext = [_Field(f, self.ext_slots) for f in fields] if self.ext_slots else []
        self.ext_active = bool(self.ext_slots)
        self.ext_live = self.ext_slots  # slots currently backed by arrays (see regrow_ext)
        self._ext_spec = list(fields)
        self._ext_gs = int(group_size)
        self.ext_releases = 0
        self.ext_regrows = 0
        biggest = max(self.slots_main * f.nbytes for f in self.fields)
        if biggest > MAX_BUFFER_BYTES:
            raise ValueError(
                f"pool array of {biggest / 2**30:.1f} GiB exceeds MLX's "
                f"{MAX_BUFFER_BYTES / 2**30:.1f} GiB max_buffer_length; reduce slots"
            )
        self.max_array_bytes = biggest
        return biggest

    def init_index(self) -> None:
        """The id-slot index, each slot's use statistics and publication count, and the free list."""
        self.slot_for_id = np.full(self.num_ids, -1, dtype=np.int32)
        self.id_of_slot = np.full(self.slots, -1, dtype=np.int64)
        self.use = np.zeros(self.slots, dtype=np.float32)  # decayed use count
        self.last = np.zeros(self.slots, dtype=np.int64)  # clock of last use
        # with `transient_prefill`, a slot filled during prefill is a first-choice victim
        self.transient = np.zeros(self.slots, dtype=bool)
        # Publication count per slot: `Qwen4ExpTextSparseMoeBlock.spec_settle` compares (slot,
        # generation) at build time with the final ones and recomputes any record remapped in
        # between.
        self.gen = np.zeros(self.slots, dtype=np.int64)
        self.publishes = 0
        self.spec_redo_layers = 0  # settles that recomputed pairs and rebuilt the next layer
        self.spec_stale_layers = 0  # ... of which had no demand miss (a remap after the build)
        # the annex's slots join the free list only when it is regrown
        self._free = list(range(self.ann0 - 1, -1, -1))

    def init_reserve(self, reserve_slots: int) -> None:
        """The prefetch reserve, taken from the free list."""
        # Prefetch reserve: slots never published, so no dispatched kernel reads one and a
        # background thread may write into it. A harvest publishes a reserve slot and returns the
        # evicted pool slot here. It always leaves one layer's expert set to the pool.
        self.reserve_n = min(max(0, int(reserve_slots)), max(0, self.slots - self.num_experts))
        # a pool holding every record has nothing to prefetch; one under 8x the reserve would
        # lose too much residency to it
        if self.slots >= self.num_ids or self.slots < 8 * self.reserve_n:
            self.reserve_n = 0
        self._reserve_free = [self._free.pop() for _ in range(min(self.reserve_n, len(self._free)))]
        self._is_reserve = np.zeros(self.slots, dtype=bool)
        if self._reserve_free:
            self._is_reserve[np.asarray(self._reserve_free, dtype=np.int64)] = True

    def init_landing(self, landing_slots: int) -> None:
        """The landing region for demand reads, taken from the free list, and its counters."""
        # Landing region for demand fetches: a read lands in a slot flagged `_is_reserve` (never a
        # victim, never published), and residency is granted afterwards by pointer swap
        # (`_swap_in`): the landing slot joins the pool holding the record and the victim joins the
        # region empty. An abandoned write therefore lands where nothing can read it (parked in
        # `_orphans` until it ends), so no executor drain is needed. Region slots come out of
        # residency, so a pool must be at least 8x the region to carry one; below that every demand
        # fetch is synchronous.
        want = max(0, int(landing_slots))
        # a pool that holds the whole id space never evicts, so a region would only cost residency
        if self.slots >= self.num_ids:
            want = 0
        self.landing_n = (
            0
            if self.slots < 8 * want
            else min(want, max(0, self.slots - self.num_experts - self.reserve_n), len(self._free))
        )
        self._landing_free = [self._free.pop() for _ in range(self.landing_n)]
        if self._landing_free:
            self._is_reserve[np.asarray(self._landing_free, dtype=np.int64)] = True
        self._orphans: list = []  # (future, slot): an in-flight write nobody is waiting for
        # split fetches returned by `ensure(split=True)` and not yet finished, by id: none is open
        # once a forward returns or a scheduler round ends (`abandon_open` settles any left)
        self._open_pends: dict = {}
        self.landing_fetches = 0  # demand fetches served by the landing region
        self.landing_fallbacks = 0  # demand fetches too large for it (waves), synchronous
        self.landing_orphaned = 0  # slots parked by a partial issue, never silently reused
        self.landing_abandoned = 0  # landing slots settled from fetches whose caller raised

    def init_eviction_state(self) -> None:
        """Victim-selection scratch, the use clock, and the marks that keep a slot a pending gather
        reads from being evicted."""
        # victim-selection scratch, allocated once (see `_lru_victims`)
        self._vkey = np.zeros(self.slots, dtype=np.int64)
        self._vidx = np.arange(self.slots, dtype=np.int64)
        self._disabled = np.zeros(self.slots, dtype=bool)  # extension slots while released
        self._disabled[self.ann0 :] = True  # the annex, until it is regrown
        self.clock = 0
        # Pending-reader invariant: no slot handed out by `ensure` may be chosen as a victim until
        # the gather that reads it has been evaluated. Set for slots returned to a caller that
        # dispatched overlap work, unioned into `protected` by `_reserve`, cleared by
        # `reader_barrier()` at the caller's host sync. If it ever exhausted the pool, `_reserve`
        # would raise rather than corrupt.
        self._pending_reader = np.zeros(self.slots, dtype=bool)
        self._pending_reader_n = 0
        self.init_cyc_state()

    def init_counters(self) -> None:
        """The traffic counters `stats` reports."""
        # counters: hits + misses + dedup_saved == lookups at every batch size. Byte counts are
        # requested reads; F_NOCACHE does not bypass pages already in the buffer cache.
        self.lookups = 0  # (row, layer, expert) requests: rows x layers x top_k per step
        self.hits = 0  # distinct records already resident
        self.misses = 0  # distinct records fetched (miss_bytes == misses * record_bytes)
        self.dedup_saved = 0  # lookups collapsed onto a same-step request another row made
        self.miss_bytes = 0
        self.prefetched = 0  # speculative fetches issued. Their bytes are prefetch_bytes and
        self.prefetch_bytes = 0  # never miss_bytes, so device == miss + prefetch + retry holds.
        self.prefetch_used = 0  # landed and a later demand wanted it
        self.prefetch_wasted = 0  # landed, nothing wanted it, slot returned to the reserve
        self.prefetch_waited = 0  # demand came before it landed (a shortened stall, not a new one)
        self._inflight = {}  # gid -> (reserve slot, Future, submit time)
        self.evictions = 0
        self.fetch_seconds = 0.0
        self.exposed_stall_seconds = 0.0  # model-thread time blocked inside ensure()
        # a retried read costs device bytes but is not a new miss, so it is counted separately
        self.read_retries = 0
        self.retry_bytes = 0
        self.seed_records = 0
        self.seed_bytes = 0
        self.seed_seconds = 0.0
        self.wave_calls = 0  # capacity-partitioned waves inside one layer (0 when it fits)
        self.wave_multi_calls = 0  # multi-row `segments_multi` calls: 48 per prefill wave
        self.calls = 0

    def init_scope(self) -> None:
        """The model thread's current scope: prompt prefill or not, the fault seam, the rows' owners."""
        # True while the model thread runs prompt prefill; only fetches in that scope are transient
        self._prefill_scope = False
        # fault-injection seam (faults.py), checked before a missing record is fetched or counted;
        # `owners` = request ids of the current unit's rows (set by the engine)
        self.seam = None
        self.owners: list | None = None

    def _linears(self, fields: list, gs: int) -> tuple:
        """(gate, up, down) projections over one array set: `QLinear` on bf16 slots, `P16Linear`
        (no scales or biases) on compact ones."""
        by = {f.name: f.tensor for f in fields}
        if self.codec:
            return tuple(
                P16Linear(by[f"{p}.weight"], by[f"{p}.words"], self.exc, gs, self.bits)
                for p in ("gate_proj", "up_proj", "down_proj")
            )
        return tuple(
            QLinear(by[f"{p}.weight"], by[f"{p}.scales"], by[f"{p}.biases"], gs, self.bits)
            for p in ("gate_proj", "up_proj", "down_proj")
        )

    def init_projections(self, group_size: int) -> None:
        """The views over the slot arrays that the MoE block's kernels read, for the main set and the
        extension set."""
        gs = int(group_size)
        self.gate_proj, self.up_proj, self.down_proj = self._linears(self.fields, gs)
        self.gate_proj_ext = self.up_proj_ext = self.down_proj_ext = None
        if self.fields_ext:
            self.gate_proj_ext, self.up_proj_ext, self.down_proj_ext = self._linears(
                self.fields_ext, gs
            )
            logger.debug(
                f"[pool] EXTENSION: {self.ext_slots} extra slots ({self.ext_slots * self.slot_bytes / GIB:.2f} GiB) in a second array set"
            )

    # ------------------------------------------------------------ elastic extension
    def release_ext(self) -> None:
        """Drop the extension's records, slots and arrays (in-flight graphs keep their buffers)."""
        if not self.ext_slots or not self.ext_active:
            return
        self.release_ann()  # the annex is lent only beside a live extension
        self._drop_inflight_ext()  # pack16: no read may land in arrays that are going
        main = self.slots_main
        ext = np.arange(main, self.ann0, dtype=np.int64)
        ids = self.id_of_slot[ext]
        live = ids[ids >= 0]
        self.ext_live = 0
        self.slot_for_id[live] = -1
        self.cyc_invalidate()
        self.id_of_slot[ext] = -1
        self.use[ext] = 0.0
        self.transient[ext] = False
        with self._lock:
            self.evictions += int(live.size)
            self.ext_releases += 1
        self._free = [s for s in self._free if s < main]
        moved_l = [s for s in self._landing_free if s >= main]
        moved_r = [s for s in self._reserve_free if s >= main]
        self._landing_free = [s for s in self._landing_free if s < main]
        self._reserve_free = [s for s in self._reserve_free if s < main]
        self._is_reserve[ext] = False
        self._disabled[ext] = True
        self.ext_active = False
        need = len(moved_l) + len(moved_r)
        if need:
            prot = np.zeros(self.slots, dtype=bool)
            repl = [int(x) for x in self._reserve(np.zeros(need, dtype=np.int64), prot)]
            self._landing_free.extend(repl[: len(moved_l)])
            self._reserve_free.extend(repl[len(moved_l) :])
            self._is_reserve[np.asarray(repl, dtype=np.int64)] = True
        self.fields_ext = []
        self.gate_proj_ext = self.up_proj_ext = self.down_proj_ext = None
        logger.info(
            f"[pool] extension RELEASED: {int(live.size)} records dropped, {self.ext_slots} slots disabled"
        )

    def regrow_ext(self, n: int | None = None) -> None:
        """Re-create the extension with `n` (default all) live slots; the rest stay disabled.

        The arrays are made before any state changes, so a regrow that raises (an allocation
        failure) leaves the extension released, exactly as it was."""
        if not self.ext_slots or self.ext_active:
            return
        n = self.ext_slots if n is None else max(0, min(int(n), self.ext_slots))
        if n == 0:
            return
        fields = [_Field(f, n) for f in self._ext_spec]
        # compact slots are written by the encode kernel, never by the host, unless they come from disk
        if not self.codec or self.disk16:
            views = [_writable_host_view(f.store) for f in fields]
            if any(v is None for v in views):
                logger.warning(
                    "[pool] extension regrow FAILED: no writable host view; staying released"
                )
                return
            for f, v in zip(fields, views):
                f.host = v
        linears = self._linears(fields, self._ext_gs)
        self.ext_live = n
        self.fields_ext = fields
        self.gate_proj_ext, self.up_proj_ext, self.down_proj_ext = linears
        main = self.slots_main
        ext = np.arange(main, main + n, dtype=np.int64)
        self._disabled[ext] = False
        self._disabled[np.arange(main + n, self.ann0, dtype=np.int64)] = True
        self.id_of_slot[ext] = -1
        self.use[ext] = 0.0
        self.last[ext] = 0
        self._free = [int(x) for x in ext[::-1].tolist()] + self._free
        self.ext_active = True
        with self._lock:
            self.ext_regrows += 1
        logger.info(f"[pool] extension REGROWN: {n} of {self.ext_slots} slots live")

    # ------------------------------------------------------------ the annex
    def disabled_any(self) -> bool:
        """Whether some slot is disabled (a released or partly backed extension, or the annex)."""
        ext = bool(self.ext_slots) and (not self.ext_active or self.ext_live < self.ext_slots)
        return ext or (bool(self.ann_slots) and self.ann_live < self.ann_slots)

    def extra_sets(self) -> list:
        """[(first slot id, (gate, up, down))] of each live array set after the main one, ascending:
        the extension, then the annex."""
        out = []
        if self.ext_slots and self.ext_active:
            out.append(
                (self.slots_main, (self.gate_proj_ext, self.up_proj_ext, self.down_proj_ext))
            )
        if self.ann_active:
            out.append((self.ann0, (self.gate_proj_ann, self.up_proj_ann, self.down_proj_ann)))
        return out

    def release_ann(self) -> int:
        """Drop the annex's records, slots and arrays (in-flight graphs keep their buffers): the
        memory it borrowed goes back. Returns the records dropped. The main pool and the extension
        keep theirs, so a short request costs at most the annex's refill, never the extension's."""
        if not self.ann_active:
            return 0
        self._drop_inflight_from(self.ann0)  # pack16: no read may land in arrays that are going
        ann = np.arange(self.ann0, self.slots, dtype=np.int64)
        ids = self.id_of_slot[ann]
        live = ids[ids >= 0]
        self.slot_for_id[live] = -1
        self.cyc_invalidate()  # as `release_ext`: no cycle-aware head may name a dropped record's slot
        self.id_of_slot[ann] = -1
        self.use[ann] = 0.0
        self.transient[ann] = False
        with self._lock:
            self.evictions += int(live.size)
            self.ann_releases += 1
            self.ann_dropped += int(live.size)
        lo = self.ann0
        self._free = [s for s in self._free if s < lo]
        moved_l = [s for s in self._landing_free if s >= lo]
        moved_r = [s for s in self._reserve_free if s >= lo]
        self._landing_free = [s for s in self._landing_free if s < lo]
        self._reserve_free = [s for s in self._reserve_free if s < lo]
        self._is_reserve[ann] = False
        self._disabled[ann] = True
        self.ann_active = False
        self.ann_live = 0
        need = len(moved_l) + len(moved_r)
        if need:  # a pointer swap may have moved landing / reserve slots into the annex
            prot = np.zeros(self.slots, dtype=bool)
            repl = [int(x) for x in self._reserve(np.zeros(need, dtype=np.int64), prot)]
            self._landing_free.extend(repl[: len(moved_l)])
            self._reserve_free.extend(repl[len(moved_l) :])
            self._is_reserve[np.asarray(repl, dtype=np.int64)] = True
        self.fields_ann = []
        self.gate_proj_ann = self.up_proj_ann = self.down_proj_ann = None
        logger.info(
            f"[pool] annex RELEASED: {int(live.size)} records dropped, {self.ann_slots} slots"
        )
        return int(live.size)

    def regrow_ann(self, n: int | None = None) -> int:
        """Back `n` (default all) annex slots with arrays; returns the slots made live. As
        `regrow_ext`, a regrow that raises leaves the annex released."""
        if not self.ann_slots or self.ann_active:
            return 0
        n = self.ann_slots if n is None else max(0, min(int(n), self.ann_slots))
        if n == 0:
            return 0
        fields = [_Field(f, n) for f in self._ext_spec]
        # compact slots are written by the encode kernel, never by the host, unless they come from disk
        if not self.codec or self.disk16:
            views = [_writable_host_view(f.store) for f in fields]
            if any(v is None for v in views):
                logger.warning(
                    "[pool] annex regrow FAILED: no writable host view; staying released"
                )
                return 0
            for f, v in zip(fields, views):
                f.host = v
        linears = self._linears(fields, self._ext_gs)
        self.fields_ann = fields
        self.gate_proj_ann, self.up_proj_ann, self.down_proj_ann = linears
        ann = np.arange(self.ann0, self.ann0 + n, dtype=np.int64)
        self._disabled[ann] = False
        self.id_of_slot[ann] = -1
        self.use[ann] = 0.0
        self.last[ann] = 0
        self._free = [int(x) for x in ann[::-1].tolist()] + self._free
        self.ann_active = True
        self.ann_live = n
        with self._lock:
            self.ann_regrows += 1
        logger.info(f"[pool] annex REGROWN: {n} of {self.ann_slots} slots live")
        return n

    # ------------------------------------------------------------------ the fetch
    def _publish(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Make `gids` resident in `slots`; call only once their bytes have fully landed."""
        self.id_of_slot[slots] = gids
        self.slot_for_id[gids] = slots.astype(np.int32)
        self.gen[slots] += 1
        self.publishes += 1
        self._cyc_published(gids, slots)  # a new resident may be its layer's farthest

    def _rollback(self, slots: np.ndarray) -> None:
        """A fetch failed: return its never-published slots, empty, to the free list."""
        self.id_of_slot[slots] = -1
        self.use[slots] = 0.0
        self._free.extend(int(s) for s in slots)

    @contextmanager
    def _undo_on_error(self, undo):
        """Run `undo` and re-raise if the block raises anything, an interrupt too: the slots a
        fetch took go back where they came from, and nothing it did not finish is published."""
        try:
            yield
        except BaseException:
            undo()
            raise

    def _return_landing(self, slots: np.ndarray) -> None:
        """Give landing slots that no write is filling back to the landing region."""
        if self.codec:
            for sl in slots.tolist():
                self._stage_release(sl)
        self._landing_free.extend(int(sl) for sl in slots.tolist())

    def _park_submitted(self, futures: list, slots: np.ndarray) -> None:
        """A submit loop failed: park the slots of the writes it submitted until they end, and give
        the rest back."""
        self._park(futures, slots[: len(futures)])
        self._return_landing(slots[len(futures) :])

    def _abandon_fetch(self, futs: list, lslots, vslots) -> int:
        """Settle a fetch that will not finish: its submitted reads' landing slots
        are parked until the reads end (`_sweep_orphans` then returns them to the region), its other
        landing slots return at once, and the victims it chose (evicted, never published) rejoin the
        free list empty. Returns the landing slots it held."""
        n = 0
        if lslots is not None:
            self._park_submitted(futs, lslots)
            n = len(lslots)
        if vslots is not None and len(vslots):
            self._rollback(vslots)
        return n

    def _hold(self, pend: dict) -> dict:
        """Register a split fetch until `ensure_finish` takes it."""
        self._open_pends[id(pend)] = pend
        return pend

    def open_fetches(self) -> int:
        """Split fetches begun and not yet finished (0 between scheduler rounds)."""
        return len(self._open_pends)

    def abandon_open(self) -> int:
        """Settle every split fetch whose caller raised between `segments_begin` and
        `segments_finish` (`_abandon_fetch`); none is open once a forward has returned, so the model's
        decode loops call this when they raise and the scheduler at the end of every round. A settled
        fetch can no longer be finished: `ensure_finish` refuses it. Returns the landing slots settled."""
        n = 0
        while self._open_pends:
            _, pend = self._open_pends.popitem()
            pend["abandoned"] = True
            n += self._abandon_fetch(pend.get("futs") or [], pend.get("lslots"), pend.get("vslots"))
        with self._lock:
            self.landing_abandoned += n
        return n

    def ensure(
        self,
        gids: np.ndarray,
        transient: bool = False,
        overlap=None,
        lookups_add: int = 0,
        dedup_add: int = 0,
        during=None,
        split: bool = False,
    ):
        """Make every id in `gids` (unique, global ids) resident; returns their slots.

        Model thread only; the lock just keeps `stats()` from seeing half-updated counters.
        `lookups_add` / `dedup_add` are the routing-side counts for these records, charged with the
        hits and misses in one locked block on success, so a raise charges nothing. `transient`
        marks the slots as first-choice victims. `overlap` holds lazy arrays built before the call
        and independent of this fetch; they are dispatched once the reads are issued, so the blocked
        window (`exposed_stall_seconds`) nests inside the read (`expert_read_seconds`). `during`
        runs while the reads are in flight and may only start reserve-slot reads. `split=True`
        returns a pending dict (with `pred_slots`) for `ensure_finish`, held in `_open_pends` until
        then. A raise anywhere between this call's first landing read and its return (the submit
        loop, `during`, the harvest, the overlap dispatch, the victim choice) settles this call's
        fetch (`_abandon_fetch`), so no landing slot or chosen victim is lost to it.
        """
        self._advance_clock(gids)
        if split and self._inflight:
            pend = self._split_inflight(gids, transient, overlap, lookups_add, dedup_add, during)
            if pend is not None:
                return self._hold(pend)
        held = {"futs": [], "lslots": None, "vslots": None}  # this call's fetch until handed on
        try:
            return self._ensure_body(
                gids, transient, overlap, lookups_add, dedup_add, during, split, held
            )
        except BaseException:
            self._abandon_fetch(held["futs"], held["lslots"], held["vslots"])
            raise

    def _ensure_body(
        self, gids, transient, overlap, lookups_add, dedup_add, during, split, held: dict
    ):
        """`ensure` past the split-in-flight case. `held` names the landing slots, futures and
        victims this call owns while its reads are in flight; it is emptied once ownership passes to
        `ensure_finish` (its own undos) or to `_open_pends`, or once the reads are parked here."""
        early = None
        if self._inflight:
            infl = self._flag_inflight(gids)
            if infl.any():
                other = gids[(self.slot_for_id[gids] < 0) & ~infl]
                if other.size:
                    ls = self._take_early_landing(other)
                    if ls is not None:
                        t_e = time.monotonic()
                        fe: list = []
                        held["futs"], held["lslots"] = fe, ls
                        self._submit_early(fe, other, ls)
                        early = (other, ls, fe, t_e)
                during = self._run_during_once(during)
            self._harvest(gids)
        cur = self.slot_for_id[gids]
        hit = cur >= 0
        n_hit = int(hit.sum())
        missing = gids[~hit]
        protected = np.zeros(self.slots, dtype=bool)
        protected[cur[hit]] = True
        early = self._drop_stale_early(early, missing)
        if early is None:
            held["futs"], held["lslots"] = [], None  # parked by `_drop_stale_early`, or never taken
        if missing.size:
            t_sub, t0, lslots = self._begin_miss_fetch(early, missing)
            if lslots is None:
                slots = self._reserve_for_sync_fill(missing, protected, during)
                # on a failure publish nothing: no id may be served from a slot whose fill did
                # not complete
                with self._undo_on_error(lambda: self._rollback(slots)):
                    self._fill(missing, slots)
                self._publish(missing, slots)
                # `t0` stays `t_sub`: the thread blocked for the whole read, so all of it is stall
            else:
                futs = self._open_landing_fetch(early)
                # on a failure nothing was evicted or published, so the pool is unchanged apart
                # from the parked slots
                held["futs"], held["lslots"] = futs, lslots
                self._submit_missing(futs, early, missing, lslots)
                self._dispatch_overlap(overlap, during)
                pend = self._issue_pending(
                    gids,
                    missing,
                    lslots,
                    futs,
                    protected,
                    t_sub,
                    n_hit,
                    lookups_add,
                    dedup_add,
                    transient,
                    overlap,
                    split,
                )
                held["vslots"] = pend.get("vslots")
                if split:
                    self._predict_slots(pend, cur, hit, lslots)
                    held["futs"], held["lslots"], held["vslots"] = [], None, None
                    return self._hold(pend)
                held["futs"], held["lslots"], held["vslots"] = [], None, None
                return self.ensure_finish(pend)
            t_land = time.monotonic()
            read_dt, dt = t_land - t_sub, t_land - t0
        else:
            read_dt = dt = 0.0
        out = self._charge(
            gids, n_hit, missing, read_dt, dt, lookups_add, dedup_add, transient, overlap
        )
        return {"slots": out, "pred_slots": out} if split else out

    def _advance_clock(self, gids: np.ndarray) -> None:
        """Count the call, tick the clock, feed the cycle-aware rule while it is armed, and decay the
        usage counts on the decay period."""
        self.calls += 1
        self.clock += 1
        if self.cyc_tokens:
            self._cyc_observe(gids)  # before this visit's harvest and demand victims
        if self.decay_tokens and self.clock % (self.decay_tokens * self.num_layers) == 0:
            self._decay()
        if self.codec and self.clock % self.num_layers == 0:
            self._check_encodes()

    def _flag_inflight(self, gids: np.ndarray) -> np.ndarray:
        """Which of `gids` are in flight (a miss chain read them a layer ago), so that the other reads
        can be issued and `during` run before waiting, and the device reads overlap."""
        gl = gids.tolist()
        infl = np.fromiter((g in self._inflight for g in gl), dtype=bool, count=len(gl))
        return infl

    def _take_early_landing(self, other: np.ndarray) -> np.ndarray | None:
        """Sweep the orphans and take landing slots for the early reads; None without a landing region."""
        if self._orphans:
            self._sweep_orphans()
        ls = self._take_landing(int(other.size))
        return ls

    def _submit_early(self, fe: list, other: np.ndarray, ls: np.ndarray) -> None:
        """Submit one landing read per early record, appending each future to `fe`."""
        for g, sl in zip(other.tolist(), ls.tolist()):
            fe.append(self._submit_landing(int(g), int(sl)))

    def _run_during_once(self, during) -> None:
        """Run `during` if there is one, before the wait; returns None, as it has run."""
        if during is not None:
            during()
            during = None
        return during

    def _drop_stale_early(self, early, missing: np.ndarray) -> None:
        """Park the early reads when a harvested read failed and the missing set differs; returns the early tuple or None."""
        if early is not None and not np.array_equal(missing, early[0]):
            # a harvested read failed, so the missing set differs from what was issued: park the
            # early reads (their slots return to the region when they end) and take the plain path
            self._park(early[2], early[1])
            early = None
        return early

    def _begin_miss_fetch(self, early, missing: np.ndarray) -> None:
        """Sweep the orphans, stamp the start time and take (or reuse) the landing slots; returns `t_sub`, `t0`, `lslots`."""
        if self._orphans and early is None:
            self._sweep_orphans()
        t_sub = t0 = early[3] if early is not None else time.monotonic()
        lslots = early[1] if early is not None else self._take_landing(int(missing.size))
        return t_sub, t0, lslots

    def _reserve_for_sync_fill(self, missing: np.ndarray, protected: np.ndarray, during) -> None:
        """No landing region: count the fallback, reserve victims, mark them protected and run `during`; returns the slots."""
        # No landing region (a wave, the scatter fallback, or a small pool): synchronous
        # evict-then-fill with nothing dispatched, so the stall charged is the whole read.
        self.landing_fallbacks += 1
        slots = self._reserve(missing, protected)
        protected[slots] = True
        if during is not None:
            with self._undo_on_error(lambda: self._rollback(slots)):
                during()  # starts reserve-slot reads only; evicts nothing
        return slots

    def _open_landing_fetch(self, early) -> list:
        """Count the landing fetch; returns the futures already issued early (or an empty list)."""
        self.landing_fetches += 1
        futs: list = list(early[2]) if early is not None else []
        return futs

    def _submit_missing(self, futs: list, early, missing: np.ndarray, lslots: np.ndarray) -> None:
        """Submit one landing read per missing record not already issued early, appending each future to `futs`."""
        for g, sl in () if early is not None else zip(missing.tolist(), lslots.tolist()):
            futs.append(self._submit_landing(int(g), int(sl)))

    def _dispatch_overlap(self, overlap, during) -> None:
        """Dispatch the caller's overlap graphs, then run `during` while the reads are in flight."""
        if overlap:
            # Dispatch only: these graphs were built by the caller before `t_sub` and name
            # no slot this fetch is writing (landing slots are published to nobody).
            mx.async_eval(*overlap)
        if during is not None:
            # may only start `prefetch` reads, which publish and evict nothing
            during()

    def _issue_pending(
        self,
        gids: np.ndarray,
        missing: np.ndarray,
        lslots: np.ndarray,
        futs: list,
        protected: np.ndarray,
        t_sub: float,
        n_hit: int,
        lookups_add: int,
        dedup_add: int,
        transient: bool,
        overlap,
        split: bool,
    ) -> dict:
        """The pending fetch for `ensure_finish`, with victims chosen already when `split`."""
        pend = {
            "gids": gids,
            "missing": missing,
            "lslots": lslots,
            "futs": futs,
            "protected": protected,
            "t_sub": t_sub,
            "n_hit": n_hit,
            "lookups_add": lookups_add,
            "dedup_add": dedup_add,
            "transient": transient,
            "overlap": overlap,
        }
        if split:
            # choose victims while the reads are in flight, so landing is only a pointer
            # swap; their index entries go now, their bytes stay until the swap
            pend["vslots"] = self._reserve(missing, protected)
        return pend

    def _predict_slots(
        self, pend: dict, cur: np.ndarray, hit: np.ndarray, lslots: np.ndarray
    ) -> None:
        """Record in `pend` where each id will be once `ensure_finish` runs."""
        # return with the reads in flight; `pred_slots` is where each id will be once
        # `ensure_finish` runs, so the caller may build, never dispatch, graphs on them
        ps = cur.astype(np.int64)
        ps[~hit] = lslots
        pend["pred_slots"] = ps

    def _split_inflight(self, gids, transient, overlap, lookups_add, dedup_add, during):
        """Split `ensure` when some records are still in flight from a miss chain (else None).

        Issues the other misses into landing slots and returns every pair's slot-to-be at once;
        `during` runs in `ensure_finish`, before the wait.
        """
        gl = gids.tolist()
        infl = np.fromiter((g in self._inflight for g in gl), dtype=bool, count=len(gl))
        if not infl.any():
            return None
        cur = self.slot_for_id[gids]
        new = (cur < 0) & ~infl
        other = gids[new]
        ls, fe = None, []
        t_e = time.monotonic()
        if other.size:
            if self._orphans:
                self._sweep_orphans()
            ls = self._take_landing(int(other.size))
            if ls is None:
                return None  # no region for them: the plain path
            with self._undo_on_error(lambda: self._park_submitted(fe, ls)):
                for g, sl in zip(other.tolist(), ls.tolist()):
                    fe.append(self._submit_landing(int(g), int(sl)))
            self.landing_fetches += 1
        pred = cur.astype(np.int64)
        if other.size:
            pred[new] = ls
        for j in np.flatnonzero(infl).tolist():
            pred[j] = self._inflight[gl[j]][0]
        return {
            "split_inflight": True,
            "gids": gids,
            "missing": other,
            "lslots": ls,
            "futs": fe,
            "t_sub": t_e,
            "transient": transient,
            "overlap": overlap,
            "lookups_add": lookups_add,
            "dedup_add": dedup_add,
            "during": during,
            "pred_slots": pred,
        }

    def _finish_split_inflight(self, pend: dict) -> np.ndarray:
        gids, missing, ls, futs = pend["gids"], pend["missing"], pend["lslots"], pend["futs"]
        if pend["during"] is not None:
            with self._undo_on_error(lambda: self._park(futs, ls) if missing.size else None):
                pend["during"]()
        t0 = time.monotonic()
        with self._undo_on_error(lambda: self._park(futs, ls) if missing.size else None):
            self._harvest(gids)  # waits for this call's in-flight records, publishes them
            cur = self.slot_for_id[gids]
            protected = np.zeros(self.slots, dtype=bool)
            protected[cur[cur >= 0]] = True
            # a needed prefetch whose read failed was dropped by `_harvest`: fetch it now
            lost = gids[(cur < 0) & ~np.isin(gids, missing)]
            if lost.size:
                sl = self._reserve(lost, protected)
                protected[sl] = True
                with self._undo_on_error(lambda: self._rollback(sl)):
                    self._fill(lost, sl)
                self._publish(lost, sl)
        if missing.size:
            err = self._first_inflight_read_error(futs)
            if err is not None:
                with self._undo_on_error(lambda: self._return_landing(ls)):
                    if not isinstance(err, OSError):
                        raise err
                    with self._lock:
                        self.read_retries += 1
                        self.retry_bytes += int(missing.size) * self.record_bytes
                    self._fill_once(missing, ls)
            with self._undo_on_error(lambda: self._return_landing(ls)):
                self._swap_in(missing, ls, protected)
        t_land = time.monotonic()
        fetched = np.concatenate([missing, lost]) if lost.size else missing
        return self._charge(
            gids,
            int(gids.size - fetched.size),
            fetched,
            t_land - pend["t_sub"],
            t_land - t0,
            pend["lookups_add"],
            pend["dedup_add"],
            pend["transient"],
            pend["overlap"],
        )

    def _first_inflight_read_error(self, futs: list) -> BaseException | None:
        """Wait for every landing read of the split; returns the first failure, else None."""
        err: BaseException | None = None
        for f in futs:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001
                if err is None:
                    err = exc
        return err

    def ensure_finish(self, pend: dict) -> np.ndarray:
        """Second half of a split `ensure`: wait, publish, charge, and return the slots."""
        if "slots" in pend:
            return pend["slots"]
        self._open_pends.pop(id(pend), None)
        if pend.get("abandoned"):
            raise InvariantError(
                "ensure_finish of an expert fetch that abandon_open settled: its landing slots and "
                "victims have gone back, so it cannot be published"
            )
        if pend.get("split_inflight"):
            return self._finish_split_inflight(pend)
        gids, missing, lslots, futs = pend["gids"], pend["missing"], pend["lslots"], pend["futs"]
        protected = pend["protected"]
        t0 = time.monotonic()
        err = self._first_read_error(futs)
        if err is not None:

            def undo():
                self._return_landing(lslots)
                if pend.get("vslots") is not None:
                    self._free.extend(int(v) for v in pend["vslots"].tolist())

            # every future has been awaited, so no writer is live: retry the whole set
            # synchronously into the same landing slots, which this fetch still owns
            with self._undo_on_error(undo):
                if not isinstance(err, OSError):
                    raise err
                with self._lock:
                    self.read_retries += 1
                    self.retry_bytes += int(missing.size) * self.record_bytes
                logger.warning(f"[pool] retrying {missing.size} expert record(s) after {err!r}")
                self._fill_once(missing, lslots)
        # if `_reserve` refuses, the landing slots return to the region holding unread bytes
        with self._undo_on_error(lambda: self._return_landing(lslots)):
            self._swap_in(missing, lslots, protected, vslots=pend.get("vslots"))
        protected[lslots] = True
        t_land = time.monotonic()
        read_dt, dt = t_land - pend["t_sub"], t_land - t0
        return self._charge(
            gids,
            pend["n_hit"],
            missing,
            read_dt,
            dt,
            pend["lookups_add"],
            pend["dedup_add"],
            pend["transient"],
            pend["overlap"],
        )

    def _first_read_error(self, futs: list) -> BaseException | None:
        """Wait for every read future; returns the first failure (an OSError is retried by the caller), else None."""
        err: BaseException | None = None
        for f in futs:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001
                if err is None:
                    err = exc
        return err

    def _charge(
        self, gids, n_hit, missing, read_dt, dt, lookups_add, dedup_add, transient, overlap
    ):
        with self._lock:
            # one atomic charge after the fetch succeeded: the routing-side and residency-side
            # counters describe the same records, so `stats()` never sees a partial charge
            self.lookups += int(lookups_add)
            self.dedup_saved += int(dedup_add)
            self.hits += n_hit
            self.misses += int(missing.size)
            self.miss_bytes += int(missing.size) * self.record_bytes
            # the read spans issue to publication and the stall only the blocked wait; the regions
            # nest (t_sub <= t0 <= t_land), so 0 <= exposed stall <= read time
            self.fetch_seconds += read_dt
            self.exposed_stall_seconds += dt
        out = self.slot_for_id[gids]
        self.use[out] += 1.0
        self.last[out] = self.clock
        # A demand-served record is never transient, whatever put it here; only a wave fetch marks.
        self.transient[out] = transient
        out = out.astype(np.int64)
        if overlap:
            # overlap work is in flight: these slots have a pending reader until the next barrier
            self._note_reader(out)
        return out

    @contextmanager
    def prefill_scope(self):
        """Mark fetches inside as prefill traffic, which `transient_prefill` evicts first.

        Model thread only, around one unit of prompt work.
        """
        prev = self._prefill_scope
        self._prefill_scope = True
        try:
            yield self
        finally:
            self._prefill_scope = prev

    # ------------------------------------------------------------------ the MoE entry point
    def device_slot_table(self) -> mx.array:
        """`slot_for_id` mirrored on the device as int32 [num_ids], for speculative gathers.

        Updated lazily by diffing against a host mirror and scattering only the changed entries, so
        every write path is covered without per-site hooks.
        """
        mirror = getattr(self, "_dev_mirror", None)
        if mirror is None:
            self._dev_mirror = self.slot_for_id.copy()
            self._dev_table = mx.array(self._dev_mirror)
            mx.eval(self._dev_table)
            return self._dev_table
        # every write to `slot_for_id` moves one of these counters (publish <- misses / harvest,
        # eviction, extension release/regrow), so an unchanged version skips the compare
        ver = (
            self.misses,
            self.evictions,
            self.ext_releases,
            self.ext_regrows,
            self.ann_releases,
            self.ann_regrows,
            self.prefetched,
            self.publishes,
        )
        if ver == getattr(self, "_dev_ver", None):
            return self._dev_table
        self._dev_ver = ver
        changed = np.flatnonzero(self.slot_for_id != mirror)
        if changed.size:
            vals = self.slot_for_id[changed]
            tab = self._dev_table
            tab[mx.array(changed.astype(np.int32))] = mx.array(vals)  # lazy scatter to a new array
            self._dev_table = tab
            mirror[changed] = vals
        return self._dev_table

    def layer_snapshot(self, layer: int):
        """(slot, generation) of every expert of `layer` right now, -1 where not resident."""
        s = self.slot_for_id[layer * self.num_experts : (layer + 1) * self.num_experts].astype(
            np.int64
        )
        return s, np.where(s >= 0, self.gen[np.maximum(s, 0)], -1)

    def resident_slots(self, layer: int, expert_ids: np.ndarray) -> np.ndarray:
        """Read-only: the slot holding each (layer, expert) pair now, -1 where not resident.

        Charges nothing, so `ensure` stays the only site that classifies hits and misses. Safe to
        read during a fetch: fetches write only landing slots, and `ensure` protects these hits from
        eviction.
        """
        gids = expert_ids.astype(np.int64) + layer * self.num_experts
        return self.slot_for_id[gids]

    def segments(self, layer: int, expert_ids: np.ndarray, row_of=None, overlap=None, during=None):
        """(positions, slots) per wave for one layer's (token, expert) pairs.

        Usually one wave; when the layer's distinct experts exceed `wave_capacity` they are split
        into waves so every pair is still computed exactly once. Only the first wave gets `overlap`.
        Those waves are fetched one at a time as the caller iterates, so it must finish a wave's
        gather (`mx.eval`, then `reader_barrier`) before taking the next.
        """
        gids = expert_ids.astype(np.int64) + layer * self.num_experts
        uniq = np.unique(gids)
        self._check_seam(layer, gids, row_of)
        # every (row, expert) pair is one lookup and each pair beyond the distinct set one
        # dedup_saved; `ensure` charges them atomically with the hits and misses
        tr = self._transient_now()
        cap = self.wave_capacity
        if uniq.size <= cap:
            return [
                (
                    np.arange(gids.size, dtype=np.int64),
                    self.ensure(
                        uniq,
                        transient=tr,
                        overlap=overlap,
                        during=during,
                        lookups_add=int(gids.size),
                        dedup_add=int(gids.size) - int(uniq.size),
                    )[np.searchsorted(uniq, gids)],
                )
            ]
        return _Waves(self, gids, uniq, cap, tr, overlap)

    def prefill_seed(self, per_layer: int) -> None:
        """Warm start: read `per_layer` experts per layer, layer-major (sequential in the pack)."""
        per_layer = int(min(per_layer, self.num_experts, self.slots // max(1, self.num_layers)))
        if per_layer <= 0:
            return
        t0 = time.monotonic()
        for layer in range(self.num_layers):
            gids = np.arange(per_layer, dtype=np.int64) + layer * self.num_experts
            self.ensure(gids)
            self.reader_barrier()
        with self._lock:
            seeded = per_layer * self.num_layers
            self.seed_records = self.misses
            self.seed_bytes = self.miss_bytes
            self.seed_seconds = self.exposed_stall_seconds
            # the seed is a load-time fill: it gets its own keys and serving counters restart at 0
            self.lookups = 0
            self.hits = 0
            self.misses = 0
            self.dedup_saved = 0
            self.miss_bytes = 0
            self.exposed_stall_seconds = 0.0
            self.fetch_seconds = 0.0
        self.use[:] = 0.0
        logger.debug(
            f"[pool] seeded {seeded} experts ({per_layer}/layer, "
            f"{seeded * self.record_bytes / GIB:.2f} GiB) in {time.monotonic() - t0:.0f}s"
        )


class _Waves:
    """A layer's waves when its experts exceed `wave_capacity`, each fetched when reached."""

    def __init__(self, pool: OffloadMoeCache, gids, uniq, cap: int, transient: bool, overlap):
        self.pool, self.gids, self.uniq, self.cap = pool, gids, uniq, cap
        self.transient, self.overlap = transient, overlap

    def __len__(self) -> int:
        return -(-int(self.uniq.size) // self.cap)

    def __iter__(self):
        self.pool.wave_calls += 1
        # the waves partition `uniq` and their `pos` sets partition `gids`, so per-wave charges
        # add up to what a single-wave call charges
        for s in range(0, self.uniq.size, self.cap):
            wave = self.uniq[s : s + self.cap]
            pos = np.flatnonzero(np.isin(self.gids, wave))
            slots = self.pool.ensure(
                wave,
                transient=self.transient,
                overlap=(self.overlap if s == 0 else None),
                lookups_add=int(pos.size),
                dedup_add=int(pos.size) - int(wave.size),
            )
            yield pos, slots[np.searchsorted(wave, self.gids[pos])]
