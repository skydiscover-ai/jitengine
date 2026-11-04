"""Prefix store: immutable state snapshots after N prompt tokens, keyed by the exact token tuple.

Entries live in a byte-capped LRU. `lookup` returns the longest entry whose key is a prefix of the
head, so reuse never passes the common prefix. Restore copies an entry into a fresh slot, so an
entry may be evicted while a request restored from it is still running.
"""

from __future__ import annotations

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


def group_prefix_points(heads, chunk: int, min_tokens: int) -> list[int]:
    """Per head, the grid boundary at or below its longest common prefix with any other head.

    0 when that boundary is below `min_tokens` or not strictly inside the head. Flooring (never
    rounding up) keeps the point resumable on the absolute grid and inside the shared prefix.
    """
    heads = [tuple(h) for h in heads]
    out = []
    for i, h in enumerate(heads):
        best = 0
        for j, other in enumerate(heads):
            if i != j:
                best = max(best, common_prefix_len(h, other))
        a = (best // chunk) * chunk
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
            self._entries.move_to_end(best_key)
            self.hits += 1
            return self._entries[best_key]

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
            }


class Entry:
    """Immutable snapshot of a RequestState after `tokens` positions, keyed by those token ids."""

    __slots__ = ("key", "tokens", "layers", "ple", "nbytes")

    def __init__(self, key: tuple, layers: list, ple: list | None = None):
        self.key = tuple(int(t) for t in key)
        self.tokens = len(self.key)
        # per model layer: ("kv", keys, values, idx) or ("delta", conv, ssm); every array a copy
        self.layers = layers
        self.ple = list(ple or [])  # ("ple", layer_idx, context, conv)
        self.nbytes = sum(a.nbytes for a in self.arrays())

    def arrays(self) -> list[mx.array]:
        out = [a for entry in self.layers for a in entry[1:]]
        out += [a for _, _, c, v in self.ple for a in (c, v)]
        return out
