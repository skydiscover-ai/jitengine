"""Prefix store: immutable snapshots of a request's state after N prompt tokens, keyed by the exact
token tuple, held in a byte-capped LRU.

Restore copies an entry into a fresh slot, so evicting an entry while a request restored from it is
still running is safe without pinning.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence

import mlx.core as mx

MIN_DIVERGENCE_TOKENS = 256
PROTECT_SKIPS = 4  # refusals a protected entry survives without a hit, plus one per hit
PROTECT_SKIPS_MAX = 32


def protect_limit(hits: int) -> int:
    """Refusals a protected entry with `hits` hits survives since its last hit."""
    return min(PROTECT_SKIPS + int(hits), PROTECT_SKIPS_MAX)


DEFAULT_CAP_BYTES = 512 * 2**20


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences (binary search on slices)."""
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
    """Per head, the snapshot point of the prefix it shares with another head in the group, or 0.

    The longest common prefix is floored to the absolute chunk grid, so a warm run partitions its
    forward passes exactly as a cold run does; the point is 0 when below `min_tokens` or not
    strictly inside the head.
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
        self.rejected_protected = 0  # unshared entries refused to keep a shared/hit entry
        self._meta: dict = {}  # key -> {"hits", "shared", "skips"}
        self._last_refused: tuple | None = None
        self.promoted = 0  # refused keys admitted on an immediate re-offer

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return tuple(key) in self._entries

    def keys(self) -> list[tuple[int, ...]]:
        with self._lock:
            return list(self._entries.keys())

    def lookup(self, head: Sequence[int], resumable=None):
        """Return the longest entry whose key prefixes `head` (refreshing it), or None on a miss.

        `resumable(k)` lets the caller accept only key lengths it can resume from bit-identically.
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
            m = self._meta.get(best_key)
            if m is not None:
                m["hits"] += 1
                m["skips"] = 0
            return self._entries[best_key]

    def peek_len(self, head: Sequence[int], resumable=None) -> int:
        """Length of the entry `lookup` would return, without refreshing it or counting a hit/miss."""
        head = tuple(head)
        n = len(head)
        best = 0
        with self._lock:
            for key in self._entries:
                k = len(key)
                if best < k <= n and (resumable is None or resumable(k)) and head[:k] == key:
                    best = k
        return best

    def divergence_points(self, head: Sequence[int], hit_len: int) -> list[int]:
        """Return the sorted positions where `head` diverges from a stored key, as snapshot points.

        p = lcp(head, key) qualifies when p < len(key), p - hit_len >= min_divergence and no entry
        is keyed exactly on head[:p].
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

    def _protected(self, key) -> bool:
        m = self._meta.get(key)
        return bool(m and (m["shared"] or m["hits"] > 0) and m["skips"] < protect_limit(m["hits"]))

    def admit(self, entry, shared: bool = False, pending_uses=lambda key: False) -> bool:
        """Insert `entry` (key = entry.key, size = entry.nbytes); return True if it is stored.

        Shared entries (divergence or group boundaries) and entries with hits are protected: an
        unshared entry that would evict one is refused, and each refusal costs every protected entry
        a skip until `protect_limit(hits)` lifts the protection. The key refused last, offered again
        at once, is admitted as shared unless `pending_uses(k)` says a queued request needs a
        protected entry k it would displace. A shared entry evicts in plain LRU order.
        """
        key = tuple(entry.key)
        size = int(entry.nbytes)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._mark_shared(key, shared)
                return True
            if size > self.cap_bytes:
                self.rejected += 1
                return False
            shared = self._promote_reoffer(key, shared, pending_uses)
            if not shared:
                free_able = self._free_able_bytes()
                if self.bytes - free_able + size > self.cap_bytes:
                    self._refuse_protected(key)
                    return False
            self._evict_for(size, shared)
            if self.bytes + size > self.cap_bytes:
                self.rejected += 1
                return False
            parent = getattr(entry, "parent", None)
            if parent is not None and self._entries.get(parent.key) is not parent:
                # the parent left the store: storing the child would keep its arrays alive uncounted
                self.rejected += 1
                return False
            self._entries[key] = entry
            self._meta[key] = {"hits": 0, "shared": bool(shared), "skips": 0}
            self.bytes += size
            self.admissions += 1
            return True

    def _promote_reoffer(self, key: tuple, shared: bool, pending_uses) -> bool:
        """An immediate re-offer of the refused key is admitted as shared unless a queued request needs what it would displace."""
        if (
            not shared
            and key == self._last_refused
            and not any(self._protected(k) and pending_uses(k) for k in self._entries)
        ):
            # an immediate re-offer of the refused key (a session re-sending its prompt) is
            # proven reuse, unless a queued request will restore what it would displace
            self._last_refused = None
            shared = True
            self.promoted += 1
        return shared

    def _evict_for(self, size: int, shared: bool) -> None:
        """Evict least recently used entries until `size` bytes fit; an unshared offer never evicts a protected entry."""
        while self._entries and self.bytes + size > self.cap_bytes:
            victim = next((k for k in self._entries if shared or not self._protected(k)), None)
            if victim is None:
                break
            self._drop(victim)

    def _mark_shared(self, key: tuple, shared: bool) -> None:
        """A re-offered resident entry becomes shared when the offer is shared."""
        if shared:
            self._meta.setdefault(key, {"hits": 0, "shared": True, "skips": 0})["shared"] = True

    def _free_able_bytes(self) -> int:
        """The bytes held by entries that are not protected."""
        free_able = sum(int(e.nbytes) for k, e in self._entries.items() if not self._protected(k))
        return free_able

    def _refuse_protected(self, key: tuple) -> None:
        """Record the refusal of an unshared offer: every protected entry loses a skip and the key is remembered."""
        for k in self._entries:
            if self._protected(k):
                self._meta[k]["skips"] += 1
        self.rejected_protected += 1
        self._last_refused = key

    def resize(self, cap_bytes: int) -> int:
        """Set a new byte cap (0 allowed), evicting LRU entries to fit; return the bytes evicted."""
        if isinstance(cap_bytes, bool) or not isinstance(cap_bytes, int) or cap_bytes < 0:
            raise ValueError(f"cap_bytes must be a non-negative integer, got {cap_bytes!r}")
        evicted = 0
        with self._lock:
            self.cap_bytes = int(cap_bytes)
            while self._entries and self.bytes > self.cap_bytes:
                evicted += self._drop(next(iter(self._entries)))
        return evicted

    def _drop(self, key) -> int:
        """Remove `key` and every entry whose KV suffix sits on it (a child must not outlive its
        parent in the store: it would hold the parent's arrays alive without counting them).
        Caller holds the lock. Returns the bytes removed."""
        old = self._entries.pop(key)
        self._meta.pop(key, None)
        freed = int(old.nbytes)
        self.bytes -= freed
        self.evictions += 1
        for k in [k for k, e in self._entries.items() if getattr(e, "parent", None) is old]:
            freed += self._drop(k)
        return freed

    def clear(self) -> None:
        with self._lock:
            self.evictions += len(self._entries)
            self._entries.clear()
            self._meta.clear()
            self._last_refused = None
            self.bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.bytes,
                "cap_bytes": self.cap_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "admissions": self.admissions,
                "evictions": self.evictions,
                "rejected": self.rejected,
                "rejected_protected": self.rejected_protected,
                "promoted": self.promoted,
                "protected_entries": sum(1 for k in self._entries if self._protected(k)),
                "entry_tokens": [len(k) for k in self._entries],
            }


class Entry:
    """Immutable snapshot of a RequestState after `tokens` positions, keyed by those token ids."""

    __slots__ = ("key", "tokens", "layers", "nbytes", "parent")

    def __init__(self, key: tuple, layers: list, parent: Entry | None = None):
        self.key = tuple(int(t) for t in key)
        self.tokens = len(self.key)
        # per model layer: ("kv", keys, values), ("delta", conv, ssm), or ("kvs", keys, values) =
        # only the positions past `parent.tokens`, the rest being the parent's own "kv" layer
        self.layers = layers
        self.parent = parent
        self.nbytes = sum(x.nbytes + y.nbytes for _, x, y in layers)  # own arrays only

    def arrays(self) -> list[mx.array]:
        return [a for _, x, y in self.layers for a in (x, y)]
