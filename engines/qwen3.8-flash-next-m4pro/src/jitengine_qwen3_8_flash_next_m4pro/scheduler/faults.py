"""Fault-injection seam, inert unless `create_engine(path, fault_inject={...})` arms it.

    fault_inject={"ple_row": <global row>            [, "request_id": <id>]}
    fault_inject={"expert_record": [layer, expert]   [, "request_id": <id>]}

One fault fires, once, on the first read of that PLE row or expert record made for `request_id`
(any request if omitted), as a `RowFaultError` naming the rows that asked for it. The engine fails
those requests, rolls the other rows back with `StepGuard` and re-issues the unit without them. The
check runs before any byte counter is charged.
"""

from __future__ import annotations

import threading

import numpy as np


class RowFaultError(OSError):
    """A read failed on behalf of `rows` (indices into the current unit's row list)."""

    def __init__(self, rows, msg: str):
        super().__init__(msg)
        self.rows = set(int(r) for r in rows)


class FaultSeam:
    """One armed fault; `check` raises `RowFaultError` for the rows it fires on, then disarms."""

    def __init__(self, kind: str, key: int, request_id=None):
        if kind not in ("ple_row", "expert_record"):
            raise ValueError(f"fault_inject kind {kind!r} (expected ple_row or expert_record)")
        self.kind = kind
        self.key = tuple(int(k) for k in key) if isinstance(key, (list, tuple)) else int(key)
        self.request_id = None if request_id is None else str(request_id)
        self.armed = True
        self.fired = 0
        self._lock = threading.Lock()

    @classmethod
    def from_option(cls, fi) -> FaultSeam | None:
        if not fi:
            return None
        if not isinstance(fi, dict):
            raise ValueError("fault_inject must be a dict")
        rid = fi.get("request_id")
        if fi.get("ple_row") is not None:
            row = fi["ple_row"]
            if isinstance(row, (list, tuple)):
                if len(row) != 1:
                    raise ValueError("fault_inject arms exactly ONE fault: give one ple_row")
                row = row[0]
            return cls("ple_row", int(row), rid)
        if fi.get("expert_record") is not None:
            rec = fi["expert_record"]
            if not (isinstance(rec, (list, tuple)) and len(rec) == 2):
                raise ValueError("fault_inject expert_record must be [layer, expert]")
            return cls("expert_record", (int(rec[0]), int(rec[1])), rid)
        raise ValueError("fault_inject names neither ple_row nor expert_record")

    def check(
        self, kind: str, keys: np.ndarray, row_of: np.ndarray | None, owners, key=None
    ) -> None:
        """Raise `RowFaultError` and disarm if the armed fault applies to this read of `keys`.

        `row_of[i]` is the row (an index into `owners`, the unit's request ids) that asked for
        `keys[i]`; None means row 0.
        """
        if not self.armed or kind != self.kind:
            return
        keys = np.asarray(keys)
        want = self.key if key is None else int(key)
        hit = np.flatnonzero(keys == want)
        if hit.size == 0:
            return
        rows = {0} if row_of is None else {int(r) for r in np.asarray(row_of)[hit]}
        if self.request_id is not None:
            rows = {
                r
                for r in rows
                if owners is not None
                and r < len(owners)
                and owners[r] is not None
                and str(owners[r]) == self.request_id
            }
            if not rows:
                return
        with self._lock:
            if not self.armed:
                return
            self.armed = False
            self.fired += 1
        raise RowFaultError(
            rows, f"injected {kind} read fault on {self.key} for rows {sorted(rows)}"
        )


class StepGuard:
    """Snapshot of the per-row bookkeeping a unit of work advances, to undo it after a
    `RowFaultError`.

    KV buffers are not referenced, only `tokens` and offsets: a second reference would defeat MLX's
    in-place donation and copy every KV buffer on the next append. `restore` empties each KV slot's
    block-key cache, which the undone unit may have extended. The DeltaNet and PLE arrays are
    replaced, not mutated, each step, so holding them is free. Call `release()` before the unit is
    evaluated.
    """

    __slots__ = ("_items",)

    def __init__(self, states):
        from jitengine_qwen3_8_flash_next_m4pro.kvcache.kv_cache import KVSlot

        items = []
        for st in states:
            per = []
            for s in st.layers:
                if isinstance(s, KVSlot):
                    per.append(("kv", s, s.offset))
                else:
                    per.append(("delta", s, s.conv, s.ssm))
            ple = [(li, p, p.context, p.conv) for li, p in st.ple.items()]
            items.append((st, st.tokens, per, ple))
        self._items = items

    def restore(self) -> None:
        for st, tokens, per, ple in self._items:
            st.tokens = tokens
            for e in per:
                if e[0] == "kv":
                    e[1].offset, e[1].idx_bk = e[2], {}
                else:
                    e[1].conv, e[1].ssm = e[2], e[3]
            for _li, p, ctx, conv in ple:
                p.context, p.conv = ctx, conv

    def release(self) -> None:
        self._items = []
