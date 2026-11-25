"""What the engine stops a request with when one of its own invariants fails."""

from __future__ import annotations


class InvariantError(RuntimeError):
    """An engine invariant does not hold: a bug in the engine, not a bad request. The step that hit
    it fails its rows, the engine keeps serving, and `/stats` counts it under `error_types`."""
