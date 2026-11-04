"""The pack16 slot codec of `OffloadMoeCache`, as a mixin: compact slots filled from staged reads.

With `codec` set, a pool slot holds a record's three weight fields and one u16 word per (scale,
bias) group (`kernels/pack16.py`), 2,764,800 B instead of 3,072,000 B on Qwen3.8-Flash-Next. Reads
stay whole, one `preadv` per record, but land in a separate full-size staging set; publication
encodes the staged record into its compact slot on the GPU. The seed's slot bookkeeping is
unchanged: a demand read still targets a landing slot and is published by pointer swap, a
prefetch still targets a reserve slot and is published by remap; only the bytes take a detour.

Ordering. The encode kernel writes pool arrays MLX does not see it write, so every reader takes
`encode_dep`, the join of every dispatched encode's error count, as an input; MLX then puts a
barrier (or a fence, across command encoders) between the write and the read. A graph built on a
slot whose read is still in flight (the speculative correction, on `pred_slots`) takes
`pred_dep(slots)` instead, which includes the not-yet-dispatched encodes of exactly those slots; it
must be dispatched only after `ensure_finish`, as the seed already requires of those graphs.

Staging reuse. A staging record is handed to a new read only after the encode that read it has
completed (`_stage_take` waits on its error count), and free staging is reused first-in first-out,
so the wait is almost always already satisfied.

Read stamps. Each read staged for a slot gets a sequence number, carried in its encode's job; the
pool writes it into `stage_seq[staging]` only when it publishes that read, just before dispatching
the encode. An encode that runs earlier (a `pred_dep` graph evaluated before `ensure_finish`, which
would otherwise copy the record the staging held before, a valid earlier record, with no error)
finds a different stamp, writes nothing and counts `pack16.STALE_UNIT`, so the next check raises.

Escape table. One global table per checkpoint (`pack16_escapes.json`, keyed by the checkpoint's
index sha256), or a census at load for a pack small enough to scan; the encoder's error count is
checked once per token (`_check_encodes`) and at every staging reuse, and a nonzero count raises.

Records already compact on disk (`disk16`: the SKYP16X1 pack, checkpoint/p16_format.py, whose every
record is this encode's output for the record of the same id). Then nothing above happens: a read
lands in the slot's six arrays by one `preadv`, exactly as a bf16 read lands in its nine, and is
published by the same swap or remap with no encode, no staging and no stamp. The seed's argument for
writing a slot from the host holds unchanged (OffloadMoeCache's docstring; `spec_settle_begin`): only
landing and reserve slots are written, and a victim is rewritten only by a read issued after a later
router sync. `encode_dep` stays the initial zeros, so every reader's dep input is still there.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.engine.errors import InvariantError
from jitengine_qwen3_8_flash_next_m4pro.kernels import pack16 as _p16

logger = logging.getLogger(__name__)

CODECS = ("auto", "pack16", "off")
SYNC_STAGING = (
    8  # staging records of the synchronous path (waves, the seed, retries of unstaged slots)
)
CENSUS_MAX_BYTES = 256 * 2**20  # an unknown pack's (scale, bias) bytes a load-time census may read
ESCAPES_FILE = os.path.join(os.path.dirname(__file__), "pack16_escapes.json")


# ----------------------------------------------------------------------------- the decision
def known_escapes(index_sha256: str) -> list | None:
    """The shipped escape table of the checkpoint with this index sha256, or None."""
    with open(ESCAPES_FILE) as f:
        entry = json.load(f).get("tables", {}).get(str(index_sha256))
    return None if entry is None else [int(v) for v in entry["escapes"]]


def staging_slots(landing: int, reserve: int) -> int:
    """Staging records a pack16 pool allocates: one per landing and reserve slot, plus the sync set."""
    return int(landing) + int(reserve) + SYNC_STAGING


def codec_for(
    fields: list,
    group_size: int,
    bits: int,
    index_sha256: str,
    num_records: int,
    moe_kernel: str,
    variant: str,
    option: str = "auto",
) -> tuple[str, str]:
    """(codec, why) from header facts alone, so the budget planner and the engine agree.

    "pack16" needs the grouped kernel's split variant (the stock `gather_qmm` fallback and the fused
    dual kernel read bf16 metadata and are not expanded), the 4-bit affine geometry, and an escape
    table: shipped for a known checkpoint, else a census at load when the pack is small enough.
    """
    opt = str(option or "auto").strip().lower()
    if opt not in CODECS:
        raise ValueError(f"pool_codec {option!r} not in {CODECS}")
    if opt == "off":
        return "off", "pool_codec=off"
    if moe_kernel != "grouped" or variant != "split":
        why = f"moe kernel {moe_kernel}/{variant} reads bf16 metadata"
        if opt == "pack16":
            raise ValueError(f"pool_codec=pack16 refused: {why}")
        return "off", why
    try:
        geom = _p16.Geometry(fields, group_size, bits)
    except ValueError as exc:
        if opt == "pack16":
            raise
        return "off", str(exc)
    if known_escapes(index_sha256) is not None:
        return "pack16", "shipped escape table"
    meta = 2 * 2 * geom.total_groups * int(num_records)
    if opt == "pack16" or meta <= CENSUS_MAX_BYTES:
        return "pack16", f"census at load ({meta} B of scale/bias)"
    return (
        "off",
        f"no shipped escape table and {meta} B of scale/bias is past the load-time census cap",
    )


def census(store, num_records: int, geom: _p16.Geometry) -> list:
    """The escape table of a whole pack, read record by record (a small, unknown pack only)."""
    table: list = []
    buf = np.empty(geom.record_bytes // 4, dtype=np.uint32)
    for r in range(int(num_records)):
        store.read_into(r, [buf])
        u16 = buf.view(np.uint16)
        for (_ow, os_, ob), n in zip(geom.offsets, geom.groups, strict=True):
            _p16.encode_np(u16[2 * os_ : 2 * os_ + n], u16[2 * ob : 2 * ob + n], table)
    return table


class P16Linear:
    """A projection over compact slots: `weight` u32 [S, N, K/8] and `words` u16 [S, N, K/gs].

    It has no `scales` or `biases`, so a reader that expects bf16 metadata fails loudly.
    """

    codec = "pack16"
    __slots__ = ("weight", "words", "exc", "group_size", "bits", "in_dims", "out_dims")

    def __init__(
        self, weight: mx.array, words: mx.array, exc: mx.array, group_size: int, bits: int
    ):
        self.weight, self.words, self.exc = weight, words, exc
        self.group_size, self.bits = int(group_size), int(bits)
        self.in_dims = int(words.shape[-1]) * self.group_size
        self.out_dims = int(weight.shape[-2])


class Pack16CodecMixin:
    """Staging, encode nodes, `encode_dep` / `pred_dep`, and the encode error check."""

    # ------------------------------------------------------------------ setup
    def init_codec(self, codec, disk16: bool = False) -> None:
        """`codec` is None (bf16 slots, the seed's format) or (Geometry, escape table); `disk16`
        says the store's records are already compact (see the module docstring)."""
        self._dep = mx.zeros((1,), dtype=mx.uint32)
        mx.eval(self._dep)
        self.encodes = 0  # encode dispatches
        self.encoded_records = 0
        self.disk16 = False
        if codec is None:
            if disk16:
                raise ValueError("records compact on disk need the pack16 codec")
            self.codec, self.geom, self.staging_n = None, None, 0
            return
        self.geom, table = codec
        self.codec = "pack16"
        self.escape_table = list(table)
        self.exc = _p16.exc_array(self.escape_table)
        if disk16:
            self.disk16 = True
            self.staging_n, self.staging, self.stage_seq = 0, None, None
            self.staging_host = self.stage_seq_host = None
            self._seq = 0
            self._staged_seq, self._staged, self._enc_node, self._enc_group = {}, {}, {}, {}
            self._stage_free, self._stage_last = deque(), []
            self._dep_checked = None
            return
        self.staging_n = staging_slots(self.landing_n, self.reserve_n)
        self.staging = mx.zeros((self.staging_n, self.geom.record_bytes // 4), dtype=mx.uint32)
        # the read stamps (see the module docstring): 0 matches no read, numbers start at 1
        self.stage_seq = mx.zeros((self.staging_n,), dtype=mx.uint32)
        mx.eval(self.staging, self.stage_seq)
        self.stage_seq_host = None
        self._seq = 0
        self._staged_seq: dict = {}  # compact slot -> the sequence number of the read staged for it
        self.staging_host = None
        self._stage_free = deque(range(self.staging_n))
        self._stage_last: list = [None] * self.staging_n  # the last dispatched encode that read it
        self._staged: dict = {}  # compact slot -> staging record holding its unpublished read
        self._enc_node: dict = {}  # compact slot -> its encode's error count, built, not dispatched
        self._enc_group: dict = {}  # id(error count) -> the slots that encode writes (one publication)
        self._dep_checked = None  # `encode_dep` as of one token ago, checked at the next token

    @property
    def encode_dep(self) -> mx.array:
        """The join of every dispatched encode: the input that orders a pool read after them."""
        return self._dep

    # ------------------------------------------------------------------ staging
    def _stage_take(self) -> int:
        """A free staging record whose last encode has completed (FIFO, so it nearly always has)."""
        if not self._stage_free:
            raise InvariantError(
                f"pack16 staging exhausted: {self.staging_n} records, {len(self._staged)} staged"
            )
        st = self._stage_free.popleft()
        last = self._stage_last[st]
        if last is not None:
            self._stage_last[st] = None
            if int(last.item()):  # waits for that encode to complete
                raise InvariantError(_encode_error(int(last.item()), "would have been"))
        return st

    def _dest(self, slot: int) -> list:
        """The destination buffers of a read for compact `slot`: its staging record (assigned on
        first use, kept across a retry)."""
        slot = int(slot)
        st = self._staged.get(slot)
        if st is None:
            st = self._staged[slot] = self._stage_take()
            self._seq = self._seq % 0xFFFFFFFF + 1
            self._staged_seq[slot] = self._seq
        return [self.staging_host[st]]

    def _stage_release(self, slot: int) -> None:
        """A staged read that will not be published (failed, dropped, superseded): free its staging.

        Only after the write into it has ended; any encode built on it was never dispatched, and
        the slots it shared that encode with are dropped from it (they will be rebuilt alone).
        """
        st = self._staged.pop(int(slot), None)
        self._staged_seq.pop(int(slot), None)
        e = self._enc_node.pop(int(slot), None)
        if e is not None:
            for other in self._enc_group.pop(id(e), set()) - {int(slot)}:
                self._enc_node.pop(other, None)
        if st is not None:
            self._stage_free.append(st)

    def _codec_landing(self) -> str:
        """The staging set's host view, proven with a sentinel, else no codec at all."""
        v = _writable_host_view_of(self.staging)
        if v is None:
            raise InvariantError("pack16 needs a writable host view of the staging set; none here")
        old = int(v[0, 0])
        v[0, 0] = 0x5EED
        seen = int(self.staging[0, 0].item())
        v[0, 0] = old
        if seen != 0x5EED:
            raise InvariantError("pack16 staging: a host write does not show through to the GPU")
        self.staging_host = v
        sv = _writable_host_view_of(self.stage_seq)
        if sv is None:
            raise InvariantError("pack16 needs a writable host view of the read stamps; none here")
        sv[0] = 0x5EED
        seen = int(self.stage_seq[0].item())
        sv[0] = 0
        if seen != 0x5EED:
            raise InvariantError(
                "pack16 read stamps: a host write does not show through to the GPU"
            )
        self.stage_seq_host = sv
        return "direct"

    # ------------------------------------------------------------------ encode nodes
    def _set_arrays(self, ext: bool, ann: bool = False) -> list:
        fs = self.fields_ann if ann else self.fields_ext if ext else self.fields
        by = {f.name: f.tensor for f in fs}
        return [by[f"{p}.{t}"] for p in _p16.PROJ for t in ("weight", "words")]

    def _build_encodes(self, slots) -> None:
        """One encode node for the staged slots of `slots` that have none yet (not dispatched).

        The slots must be published together (`_encode_publish` refuses to dispatch a node part of
        whose slots it is not publishing): one demand fetch's landing slots, one prefetch, one chunk.
        """
        todo = [int(s) for s in slots if int(s) in self._staged and int(s) not in self._enc_node]
        if not todo:
            return
        jobs = np.asarray([(self._staged[s], s, self._staged_seq[s]) for s in todo], dtype=np.int64)
        dst = jobs[:, 1]
        in_ext = (dst >= self.slots_main) & (dst < self.ann0)
        ext = self._set_arrays(True) if (self.fields_ext and in_ext.any()) else None
        if in_ext.any() and ext is None:
            raise InvariantError(
                "pack16: an encode targets an extension slot with no extension arrays"
            )
        ann = (
            self._set_arrays(False, ann=True)
            if (self.fields_ann and (dst >= self.ann0).any())
            else None
        )
        if (dst >= self.ann0).any() and ann is None:
            raise InvariantError("pack16: an encode targets an annex slot with no annex arrays")
        if ann is not None and ext is None:
            ext = self._set_arrays(True) if self.fields_ext else self._set_arrays(False)
        err = _p16.encode_records(
            self.geom,
            self.staging,
            self._set_arrays(False),
            ext,
            self.slots_main,
            self.exc,
            len(self.escape_table),
            jobs,
            self._dep,
            self.stage_seq,
            ann=ann,
            ann_slots=self.ann0,
        )
        for s in todo:
            self._enc_node[s] = err
        self._enc_group[id(err)] = set(todo)

    def pred_dep(self, slots) -> mx.array:
        """The dep of a graph built on `slots` before their reads have landed (lazy; see module).

        A slot an in-flight prefetch fills is published by its own harvest, the others (landing
        slots) by the fetch's swap, so each prefetch gets its own encode node.
        """
        if self.codec is None or self.disk16:
            return self._dep
        slots = np.asarray(slots).reshape(-1)
        staged = [int(s) for s in slots.tolist() if int(s) in self._staged]
        if not staged:
            return self._dep
        prefetching = {int(rs) for rs, _f, _t in self._inflight.values()}
        for s in staged:
            if s in prefetching:
                self._build_encodes([s])
        self._build_encodes([s for s in staged if s not in prefetching])
        errs = {id(self._enc_node[s]): self._enc_node[s] for s in staged}
        dep = self._dep
        for e in errs.values():
            dep = mx.maximum(dep, e)
        return dep

    def _encode_publish(self, slots) -> None:
        """Dispatch the encodes of `slots` (staged, landed), join them into `encode_dep`, and free
        their staging once those encodes complete. Call before any reader of `slots` is dispatched."""
        if self.codec is None or self.disk16:
            return
        slots = [int(s) for s in np.asarray(slots).reshape(-1).tolist() if int(s) in self._staged]
        if not slots:
            return
        self._build_encodes(slots)
        errs = {id(self._enc_node[s]): self._enc_node[s] for s in slots}
        publishing = set(slots)
        for k in errs:
            group = self._enc_group[k]
            if not group <= publishing:
                # its other slots' reads may not have landed: this encode would store their garbage
                raise InvariantError(
                    f"pack16: an encode of slots {sorted(group)} dispatched while publishing only "
                    f"{sorted(publishing)}"
                )
        for k in errs:
            del self._enc_group[k]
        for s in slots:
            e = self._enc_node.pop(s)
            st = self._staged.pop(s)
            self.stage_seq_host[st] = self._staged_seq.pop(s)  # published: its encode may now run
            self._stage_last[st] = e
            self._stage_free.append(st)
        dep = self._dep
        for e in errs.values():
            dep = mx.maximum(dep, e)
        mx.async_eval(*errs.values(), dep)
        self._dep = dep
        self.encodes += len(errs)
        self.encoded_records += len(slots)

    def _fill_codec(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """`_fill_once` with the codec: staged slots (a retry) are read again into their staging and
        published later; the others are read in chunks of free staging and encoded at once."""
        staged = np.asarray([int(s) in self._staged for s in slots.tolist()], dtype=bool)
        if staged.any():
            self.store.read_many(
                [(int(g), self._dest(int(s))) for g, s in zip(gids[staged], slots[staged])]
            )
        rest_g, rest_s = gids[~staged], slots[~staged]
        i = 0
        while i < rest_g.size:
            # every free staging record per chunk (at least the SYNC_STAGING set is free here)
            n = max(1, min(len(self._stage_free), int(rest_g.size - i)))
            cg, cs = rest_g[i : i + n], rest_s[i : i + n]
            try:
                self.store.read_many([(int(g), self._dest(int(s))) for g, s in zip(cg, cs)])
            except BaseException:
                for s in cs.tolist():
                    self._stage_release(s)  # read_many waited for every write
                raise
            self._encode_publish(cs)
            i += n

    def _check_encodes(self) -> None:
        """Once per token: the encode error count as of one token ago (long complete), else raise."""
        if self.codec is None or self.disk16:
            return
        prev, self._dep_checked = self._dep_checked, self._dep
        if prev is not None and int(prev.item()):
            raise InvariantError(_encode_error(int(prev.item()), "may have been"))

    def _drop_inflight_ext(self) -> None:
        """Before the extension's arrays go: settle every read that targets an extension slot (an
        in-flight prefetch or a parked landing write), unpublished, with its staging freed; the slot
        rejoins its free list, from which `release_ext` replaces extension slots with main ones."""
        self._drop_inflight_from(self.slots_main)

    def _drop_inflight_from(self, main: int) -> None:
        """`_drop_inflight_ext` for every slot id >= `main` (the annex's release passes its first)."""
        if self.codec is None:
            return
        for g in [g for g, (rs, _f, _t) in self._inflight.items() if rs >= main]:
            rslot, fut, _t = self._inflight.pop(g)
            try:
                fut.result()
            except BaseException:  # noqa: BLE001  a failed prefetch only frees its staging
                pass
            self._stage_release(rslot)
            self._reserve_free.append(rslot)
            with self._lock:
                self.prefetch_wasted += 1
        keep = []
        for f, sl in self._orphans:
            if sl >= main:
                try:
                    f.result()
                except BaseException:  # noqa: BLE001
                    pass
                self._stage_release(sl)
                self._landing_free.append(sl)
            else:
                keep.append((f, sl))
        self._orphans = keep

    def codec_stats(self) -> dict:
        return {
            "expert_pool_codec": self.codec or "bf16",
            "expert_pool_slot_bytes": int(self.slot_bytes),
            "expert_pool_staging_slots": int(self.staging_n),
            "expert_pool_staging_bytes": int(self.staging_n * self.record_bytes),
            "expert_pool_escapes": len(getattr(self, "escape_table", []) or []),
            "expert_pool_encodes": int(self.encodes),
            "expert_pool_encoded_records": int(self.encoded_records),
            "pack16_disk": bool(getattr(self, "disk16", False)),
            "pack16_disk_why": str(getattr(self, "disk16_why", "")),
        }


def _encode_error(count: int, served: str) -> str:
    """The message of a nonzero encode error count (`pack16.stale_and_lossy`)."""
    stale, lossy = _p16.stale_and_lossy(count)
    parts = []
    if lossy:
        parts.append(
            f"{lossy} group(s) outside the codec and the escape table; a lossy word {served} served"
        )
    if stale:
        parts.append(
            f"{stale} encode(s) ran before their read was published (a pred_dep graph evaluated "
            f"before ensure_finish); the slot was left unwritten and another record's bytes {served} served"
        )
    return "pack16 encode reported " + "; ".join(parts)


def _writable_host_view_of(a: mx.array):
    from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.reads import _writable_host_view

    return _writable_host_view(a)
