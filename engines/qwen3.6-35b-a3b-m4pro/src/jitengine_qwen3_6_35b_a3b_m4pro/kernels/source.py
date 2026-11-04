"""Building one Metal kernel's source from another's."""

from __future__ import annotations


def substitute(source: str, *edits: tuple[str, str]) -> str:
    """`source` with each (old, new) edit applied in turn; each `old` must occur exactly once, so a
    change to the source it edits fails loudly instead of leaving the derived kernel different."""
    for old, new in edits:
        found = source.count(old)
        if found != 1:
            raise ValueError(
                f"kernel source edit expects one occurrence, found {found}: {old[:80]!r}"
            )
        source = source.replace(old, new)
    return source
