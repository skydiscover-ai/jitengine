"""Prefix store: immutable state snapshots after N prompt tokens, keyed by the exact token tuple.

Entries live in a byte-capped LRU. `lookup` returns the longest entry whose key is a prefix of the
head, so reuse never passes the common prefix. Restore copies an entry into a fresh slot, so an
entry may be evicted while a request restored from it is still running.

With a `StoreSpill` attached, `spill` writes the entries to its scratch file behind the decode and,
once every one is on file, drops their arrays (one-row decode lends that memory to the expert pool's
annex); `restore` takes the store back. While spilled, the keys stay (`covered_len`,
`divergence_points`, `keys` answer as before) but `lookup` and `admit` refuse, so nothing can read or
add an array the store does not hold. After `restore` an entry's arrays are read back when `lookup`
returns it (an admission reads only the entry it resumes from); one that cannot be read is dropped
and the lookup goes on as if it had never been stored.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from collections.abc import Sequence

import mlx.core as mx

MIN_DIVERGENCE_TOKENS = 256
# Largest share of the cap the pinned set may hold; past it a pin is declined and the entry is
# stored unpinned, so pins never fill the store.
PIN_FRACTION = 0.5
# The memory plan passes the real cap; this default only matters to a standalone construction.
DEFAULT_CAP_BYTES = 320_000_000
# Longest an admission waits for one spill write still in flight (`restore`); a stuck volume then
# costs that much, never a hang.
WRITE_WAIT_S = 10.0


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences, by binary search on slices."""
    hi = min(len(a), len(b))
    if hi == 0:
        return 0
    if a[:hi] == b[:hi]:
        return hi
    lo = 0  # invariant: a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def _scan_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """`common_prefix_len` in work proportional to the common prefix: blocks of doubling size (up to
    1,024 tokens) are compared until one differs, and that block is then scanned token by token."""
    hi = min(len(a), len(b))
    k, step = 0, 32
    while k < hi:
        e = min(hi, k + step)
        if a[k:e] != b[k:e]:
            while a[k] == b[k]:
                k += 1
            return k
        k, step = e, min(2 * step, 1024)
    return hi


def group_prefix_points(heads, chunk: int, min_tokens: int) -> list[int]:
    """Per head, the grid boundary at or below its longest common prefix with any other head.

    0 when that boundary is below `min_tokens` or not strictly inside the head. Flooring (never
    rounding up) keeps the point resumable on the absolute grid and inside the shared prefix.

    In lexicographic order the common prefix of two heads is the shortest one between neighbours
    from the first to the second, so a head's longest common prefix with any other is the longer of
    those with its two neighbours: one sort and P - 1 neighbour scans, not P^2 pair comparisons
    (this runs on the model thread over every pending head).
    """
    heads = [tuple(h) for h in heads]
    order = sorted(range(len(heads)), key=heads.__getitem__)
    best = [0] * len(heads)
    for i, j in itertools.pairwise(order):
        n = _scan_prefix_len(heads[i], heads[j])
        best[i] = max(best[i], n)
        best[j] = max(best[j], n)
    out = []
    for h, b in zip(heads, best):
        a = (b // chunk) * chunk
        out.append(a if a >= min_tokens and a < len(h) else 0)
    return out


class PrefixCache:
    """Byte-capped LRU of immutable `Entry` objects keyed by their token tuple."""

    def __init__(
        self, cap_bytes: int = DEFAULT_CAP_BYTES, min_divergence: int = MIN_DIVERGENCE_TOKENS
    ):
        self.cap_bytes = int(cap_bytes)
        self.min_divergence = int(min_divergence)
        self._entries: OrderedDict[tuple[int, ...], object] = OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.admissions = 0
        self.rejected = 0  # entries larger than the cap, never stored
        # keys protected from eviction while any unpinned entry remains
        self._pinned: set = set()
        self.pins = 0  # entries admitted with a granted pin
        self.pinned_evictions = 0  # times the cap forced a pinned entry out anyway
        self.spiller = None  # a StoreSpill, when the engine lends the store's memory
        self.spilled = False
        self.spill_freed = 0  # bytes the last spill dropped (what the annex may borrow)
        self.spill_kept = 0  # bytes the last spill could not drop (an entry whose write failed)

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return tuple(key) in self._entries

    def keys(self) -> list[tuple[int, ...]]:
        with self._lock:
            return list(self._entries.keys())

    def lookup(self, head: Sequence[int], resumable=None):
        """Longest stored entry whose key is a prefix of `head` (refreshing it), or None on a miss.

        `resumable(k)` optionally restricts key lengths to those the caller can resume from exactly.
        """
        head = tuple(head)
        n = len(head)
        best_key = None
        if self.spilled:
            raise RuntimeError("prefix store lookup while its entries are spilled; restore first")
        with self._lock:
            for key in self._entries:
                k = len(key)
                if (
                    0 < k <= n
                    and (best_key is None or k > len(best_key))
                    and (resumable is None or resumable(k))
                    and head[:k] == key
                ):
                    best_key = key
            if best_key is None:
                self.misses += 1
                return None
            entry = self._entries[best_key]
            if not entry.resident and not self._read_back(entry):
                lost = True
            else:
                lost = False
                self._entries.move_to_end(best_key)
                self.hits += 1
        if lost:  # dropped: answer as if it had never been stored
            return self.lookup(head, resumable)
        return entry

    def _read_back(self, entry) -> bool:
        """Read a spilled entry's arrays back (caller holds the lock); False drops the entry."""
        if self.spiller.read_back(entry):
            return True
        key = tuple(entry.key)
        self._entries.pop(key, None)
        self._pinned.discard(key)
        self.bytes -= int(entry.nbytes)
        self.evictions += 1
        return False

    def covered_len(self, head: Sequence[int], resumable=None) -> int:
        """Tokens of `head` a stored entry would supply; changes no counter and no recency."""
        head = tuple(head)
        n = len(head)
        best = 0
        with self._lock:
            for key in self._entries:
                k = len(key)
                if (
                    0 < k <= n
                    and k > best
                    and (resumable is None or resumable(k))
                    and head[:k] == key
                ):
                    best = k
        return best

    def divergence_points(self, head: Sequence[int], hit_len: int) -> list[int]:
        """Sorted divergence points p = lcp(head, key) worth a snapshot.

        Requires p < len(key), p - hit_len >= min_divergence and no entry keyed on head[:p].
        """
        head = tuple(head)
        out = set()
        with self._lock:
            keys = list(self._entries.keys())
        for key in keys:
            p = common_prefix_len(head, key)
            if (
                p - hit_len >= self.min_divergence
                and p < len(key)
                and head[:p] not in self._entries
            ):
                out.add(p)
        return sorted(out)

    # ------------------------------------------------------------------ updates
    def admit(self, entry, pin: bool = False) -> bool:
        """Insert `entry` (key = entry.key, size = entry.nbytes). Returns True if it is stored.

        `pin` protects a shared-prefix snapshot from eviction by the more recent per-row entries.
        Pins are capped at `PIN_FRACTION` of the store (beyond it the entry is stored unpinned),
        eviction takes pinned entries only when no unpinned one is left, and a pin never changes
        what `lookup` returns.
        """
        key = tuple(entry.key)
        size = int(entry.nbytes)
        if self.spilled:
            raise RuntimeError("prefix store admit while its entries are spilled; restore first")
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                if pin and self._may_pin(size, key):
                    self._pinned.add(key)
                return True
            if size > self.cap_bytes:
                self.rejected += 1
                return False
            while self._entries and self.bytes + size > self.cap_bytes:
                victim = self._victim()
                old = self._entries.pop(victim)
                self._pinned.discard(victim)
                self.bytes -= int(old.nbytes)
                self.evictions += 1
            self._entries[key] = entry
            self.bytes += size
            self.admissions += 1
            if pin and self._may_pin(size, key):
                self._pinned.add(key)
                self.pins += 1
            return True

    def _may_pin(self, size: int, key) -> bool:
        """Would pinning this entry keep the pinned set inside its share of the cap?"""
        held = sum(
            int(self._entries[k].nbytes) for k in self._pinned if k in self._entries and k != key
        )
        return held + size <= self.cap_bytes * PIN_FRACTION

    def _victim(self):
        """LRU among unpinned entries; falls back to the LRU pinned one so the cap always binds."""
        for k in self._entries:  # OrderedDict iterates least-recent first
            if k not in self._pinned:
                return k
        self.pinned_evictions += 1
        return next(iter(self._entries))

    # ------------------------------------------------------------------ spill
    def spill(self) -> int | None:
        """Drop the entries' arrays once each is on the spill file; returns the bytes dropped, 0
        without a spiller or when already spilled, and None while writes are still in flight (the
        first call starts them, behind the decode; call again on a later step). An entry whose write
        failed keeps its arrays (`spill_kept`)."""
        if self.spiller is None or self.spilled:
            return 0
        with self._lock:
            entries = list(self._entries.values())
        if not self.spiller.write_behind(entries):
            return None
        freed = kept = 0
        for e in entries:
            if not e.resident:
                continue
            if e.spill_at is None:  # its write failed: it stays, and the annex is smaller
                kept += int(e.nbytes)
                continue
            freed += int(e.nbytes)
            e.drop_arrays()
        self.spiller.trim(entries)
        self.spill_freed, self.spill_kept = freed, kept
        self.spilled = True
        return freed

    def restore(self) -> None:
        """Take the store back from the annex: lookups and admissions are served again, and each
        entry's arrays are read back when a lookup first returns it (`_read_back`). Writes still in
        flight (a decode shorter than its spill) finish first, so an entry an admission evicts never
        stays held by its write beside the store's cap."""
        if self.spiller is not None and self.spiller._pending:
            self.spiller.wait(timeout=WRITE_WAIT_S)
        self.spilled = False

    def clear(self) -> None:
        with self._lock:
            self.evictions += len(self._entries)
            self._entries.clear()
            self._pinned.clear()
            self.bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "pinned_entries": len(self._pinned),
                "pinned_bytes": sum(
                    int(self._entries[k].nbytes) for k in self._pinned if k in self._entries
                ),
                "pins": self.pins,
                "pinned_evictions": self.pinned_evictions,
                "bytes": self.bytes,
                "cap_bytes": self.cap_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "admissions": self.admissions,
                "evictions": self.evictions,
                "rejected": self.rejected,
                "entry_tokens": [len(k) for k in self._entries],
                "spilled": bool(self.spilled),
                "spill_kept_bytes": int(self.spill_kept),
                **(self.spiller.stats() if self.spiller is not None else {}),
            }


class Entry:
    """Immutable snapshot of a RequestState after `tokens` positions, keyed by those token ids."""

    __slots__ = ("key", "tokens", "layers", "ple", "nbytes", "resident", "spill_at", "spill_failed")

    def __init__(self, key: tuple, layers: list, ple: list | None = None):
        self.key = tuple(int(t) for t in key)
        self.tokens = len(self.key)
        # per model layer: ("kv", keys, values, idx) or ("delta", conv, ssm); every array a copy
        self.layers = layers
        self.ple = list(ple or [])  # ("ple", layer_idx, context, conv)
        self.nbytes = sum(a.nbytes for a in self.arrays())
        self.resident = True  # False while spilled (`drop_arrays` / `install_arrays`)
        self.spill_at = None  # [(offset, nbytes, shape, dtype)] per array once on the spill file
        self.spill_failed = False  # its spill write failed: it is never dropped

    def arrays(self) -> list[mx.array]:
        out = [a for entry in self.layers for a in entry[1:]]
        out += [a for _, _, c, v in self.ple for a in (c, v)]
        return out

    def drop_arrays(self) -> None:
        """Release the arrays (the spill file holds their bytes); `nbytes` keeps their size."""
        self.layers = [(entry[0],) + (None,) * (len(entry) - 1) for entry in self.layers]
        self.ple = [(tag, li, None, None) for tag, li, _c, _v in self.ple]
        self.resident = False

    def install_arrays(self, arrs: list) -> None:
        """Put back arrays in `arrays()` order (the inverse of `drop_arrays`)."""
        it = iter(arrs)
        self.layers = [(entry[0],) + tuple(next(it) for _ in entry[1:]) for entry in self.layers]
        self.ple = [(tag, li, next(it), next(it)) for tag, li, _c, _v in self.ple]
        self.resident = True
