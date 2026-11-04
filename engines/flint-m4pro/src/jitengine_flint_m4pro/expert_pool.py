"""Offload mode's expert pool: `slots` routed-expert records in unified memory, read from the checkpoint on demand.

Each of the nine per-expert tensors (gate / up / down x weight / scales / biases) is one `[slots, ...]` array that
`mx.gather_qmm` indexes by slot, so a pooled expert is the same bytes, run by the same kernel, as the resident banks'
expert: the graph backend's outputs do not depend on the budget. A record missing from the pool is read straight from
its nine checkpoint tensors (expert e of a `[256, ...]` tensor is one contiguous slice) with `preadv` and F_NOCACHE,
into the pool's own memory when MLX exposes it to the host ("direct" landing, proved at startup with a sentinel), else
through a staging buffer and an MLX scatter ("scatter").

Overwriting a slot is safe because `slots_for` runs on the worker thread after the layer's router ids were evaluated:
everything queued before them, every earlier read of the pool among it, has finished on the in-order GPU stream. The
victim is the slot with the lowest decayed use count, then the oldest last use; the slots the current layer needs
are never victims, so the pool must hold one layer's 256 experts (budget.plan_offload sees to it).
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jitengine_flint_m4pro.loader import QLinear
from jitengine_flint_m4pro.model.config import FlintConfig
from jitengine_flint_m4pro.weight_banks import EXPERT_PROJ, QSUFFIX

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
WORKERS = 8  # concurrent record reads; deeper queues lengthen a miss without adding throughput
DECAY_LAYERS = (
    64  # halve the use counts every DECAY_LAYERS x n_layers lookups (about 64 tokens of decode)
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
    """`slots` expert records over the global (layer, expert) ids layer * 256 + expert."""

    def __init__(
        self, cfg: FlintConfig, index: dict, slots: int, log=print, workers: int = WORKERS
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
        self.use = np.zeros(self.slots, dtype=np.float64)
        self.last = np.zeros(self.slots, dtype=np.int64)
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

    def _read_record(self, gid: int, dsts: list) -> None:
        layer, e = divmod(int(gid), self.n_experts)
        for f, dst in zip(self.fields, dsts):
            path, base = f.src[layer]
            self._pread_full(self._fd(path), dst, base + e * f.nbytes)

    def _fill(self, gids: np.ndarray, slots: np.ndarray) -> None:
        """Read records `gids` into `slots` (reserved, not yet published)."""
        import mlx.core as mx

        t0 = time.perf_counter()
        if self.landing == "direct":
            jobs = [
                (g, [memoryview(f.host[s]).cast("B") for f in self.fields])
                for g, s in zip(gids, slots)
            ]
            staging = None
        else:
            staging = [np.empty((len(gids), f.nbytes), dtype=np.uint8) for f in self.fields]
            jobs = [(g, [memoryview(st[i]) for st in staging]) for i, g in enumerate(gids)]
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

    def _try(self, gid: int, dsts: list):
        try:
            self._read_record(gid, dsts)
        except OSError as exc:
            return exc
        return None

    # ------------------------------------------------------------------ residency
    def _victims(self, k: int, protected: np.ndarray) -> list:
        take = [self.free.pop() for _ in range(min(k, len(self.free)))]
        k -= len(take)
        if k > 0:
            cand = np.flatnonzero(~protected)
            if take:
                cand = np.setdiff1d(cand, np.asarray(take))
            if cand.size < k:
                raise RuntimeError(
                    f"the expert pool ({self.slots} slots) cannot hold one layer's experts"
                )
            order = np.lexsort((self.last[cand], self.use[cand]))
            for v in cand[order[:k]]:
                if self.id_of_slot[v] >= 0:
                    self.slot_for_id[self.id_of_slot[v]] = -1
                self.id_of_slot[v] = -1
                self.evictions += 1
                take.append(int(v))
        return take

    def ensure(self, gids: np.ndarray) -> np.ndarray:
        """Slots of the unique global ids `gids`, reading the missing records first."""
        self.clock += 1
        if self.clock % (DECAY_LAYERS * self.cfg.n_layers) == 0:
            self.use *= 0.5
        cur = self.slot_for_id[gids]
        miss = gids[cur < 0]
        if miss.size:
            protected = np.zeros(self.slots, dtype=bool)
            protected[cur[cur >= 0]] = True
            slots = np.asarray(self._victims(int(miss.size), protected), dtype=np.int64)
            try:
                self._fill(miss, slots)
            except BaseException:
                self.free.extend(int(s) for s in slots)  # never publish a slot whose read failed
                raise
            self.id_of_slot[slots] = miss
            self.slot_for_id[miss] = slots
            self.use[slots] = 0.0
        out = self.slot_for_id[gids]
        self.use[out] += 1.0
        self.last[out] = self.clock
        with self._lock:
            self.hits += int(gids.size - miss.size)
            self.misses += int(miss.size)
        return out

    def slots_for(self, layer: int, ids):
        """The slot of every routed (token, expert) in `ids` (uint32 [..., k], lazy) of `layer`, as uint32 like ids.
        Evaluates `ids` (the host needs them), then reads the records the pool lacks."""
        import mlx.core as mx

        mx.eval(ids)
        e = np.array(ids).astype(np.int64)
        gids, inv = np.unique(e.reshape(-1) + layer * self.n_experts, return_inverse=True)
        slots = self.ensure(gids)[inv].reshape(e.shape)
        return mx.array(slots.astype(np.uint32))

    def layer_table(self, layer: int, ids) -> tuple:
        """`layer`'s experts as three [256, ...] QLinear tables taken from the pool (row e = expert e; rows of experts
        not in `ids` hold any resident slot and are never read), for a gather indexed by expert id like the banks'.
        MLX chooses gather_qmm's sorted kernel from the rows per expert of the weight array, so a gather over the
        pool's [slots, ...] arrays could take another kernel than the banks' [256, ...] views, and round otherwise.
        Evaluates `ids`; the takes are lazy and finish before the next layer's lookup (it evaluates this layer's
        output), so no later read of the pool overwrites a slot they read."""
        import mlx.core as mx

        mx.eval(ids)
        e = np.unique(np.array(ids).astype(np.int64))
        slots = self.ensure(e + layer * self.n_experts)
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
            self.hits = self.misses = self.evictions = 0
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
            }

    def close(self) -> None:
        self._io.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        for f in self.fields:
            f.host = f.store = f.tensor = None
        self.experts = None
