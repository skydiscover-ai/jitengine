"""Context-copy drafts (prompt lookup): propose the tokens that followed an earlier match of the tail.

The verify pass draws every emitted token itself, so outputs are unchanged; only the number of
tokens per verify pass moves. One index per request over prompt + committed tokens (3-gram keys,
longest backward match up to 64 among the 32 most recent occurrences), following oMLX's ContextCopy.
"""

from __future__ import annotations

MAX_MATCH = 64
ENTER_MATCH = 12
MIN_COPY = 2
CANDIDATES = 32
MISSES_BEFORE_QUIET = 4
QUIET_CYCLES = 8


# oMLX's wide rule: 15 tokens when the tail repeats the source verbatim for the whole 64-token match
# and the previous copy was accepted whole (text being copied out), on a 16-row verify
WIDE_COPY = 15


class ContextCopy:
    """Per-request copy proposer: a pause of QUIET_CYCLES rounds follows MISSES_BEFORE_QUIET zero-accept copies."""

    def __init__(self, history, max_copy: int = 7, wide: bool = False, wide_copy: int = WIDE_COPY):
        self.max_copy = int(max_copy)
        self.wide = bool(wide)
        self.wide_copy = int(wide_copy)
        self.last_whole = False
        self.ids: list[int] = []
        self.index: dict[tuple, list[int]] = {}
        self.misses = 0
        self.quiet = 0
        self.extend(history)

    def extend(self, toks) -> None:
        """Append `toks` to the history and index every 3-gram that ends at one of them."""
        ids, index = self.ids, self.index
        for t in toks:
            ids.append(int(t))
            n = len(ids)
            if n >= 3:
                index.setdefault((ids[n - 3], ids[n - 2], ids[n - 1]), []).append(n)

    def propose(self, room: int) -> list[int]:
        """Up to min(max_copy, room) tokens to draft, or [] (the MTP head drafts instead)."""
        if room < MIN_COPY:
            return []
        if self.quiet > 0:
            self.quiet -= 1
            return []
        ids = self.ids
        n = len(ids)
        if n < ENTER_MATCH + 1:
            return []
        ends = self.index.get((ids[n - 3], ids[n - 2], ids[n - 1]))
        if not ends or len(ends) < 2:
            return []
        best_len, best_end = 0, -1
        for e in reversed(ends[-(CANDIDATES + 1) :]):
            if e >= n:  # the tail itself
                continue
            L = 3
            while L < MAX_MATCH and e - L - 1 >= 0 and ids[e - L - 1] == ids[n - L - 1]:
                L += 1
            if L > best_len:
                best_len, best_end = L, e
                if L >= MAX_MATCH:
                    break
        if best_len < ENTER_MATCH:
            return []
        cap = (
            self.wide_copy
            if (self.wide and self.last_whole and best_len >= MAX_MATCH)
            else self.max_copy
        )
        out = ids[best_end : best_end + min(cap, room)]
        return out if len(out) >= MIN_COPY else []

    def observe(self, accepted: int, drafted: int) -> None:
        """Record how many of the `drafted` tokens the verify pass `accepted`; repeated misses pause
        copying for QUIET_CYCLES rounds."""
        self.last_whole = accepted == drafted
        if accepted == 0:
            self.misses += 1
            if self.misses >= MISSES_BEFORE_QUIET:
                self.quiet, self.misses = QUIET_CYCLES, 0
        else:
            self.misses = 0
