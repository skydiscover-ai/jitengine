"""Idle-only memory reconfiguration: pure orchestration over the `ResizeTarget` protocol.

The new plan validates before anything is freed, and the old pool is released before the new one
is allocated, so two pools never coexist. A failed allocation retries the previous geometry (when
growing) and then the minimum geometry, once each.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from jitengine_qwen3_6_35b_a3b_m2.memory.budget import (
    BudgetError,
    Capacity,
    Limits,
    Plan,
    make_plan,
)

SERVING, VALIDATING, REBUILDING, FAILED = "serving", "validating", "rebuilding", "failed"


@dataclass(frozen=True)
class ResizeRequest:
    process_bytes: int
    reason: str = "manual"


def _result(status: str, req: ResizeRequest, t0: float, **kw) -> dict:
    out = {
        "status": status,
        "requested_bytes": int(req.process_bytes),
        "reason": req.reason,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    out.update(kw)
    return out


def plan_for(
    capacity: Capacity, limits: Limits, process_bytes: int, mode: str, trim_optional: bool = False
) -> Plan:
    """Return the plan `process_bytes` would produce; `trim_optional` zeroes the optional caches."""
    c = capacity
    if trim_optional:
        c = replace(c, prefix_bytes=0, allocator_cache_bytes=0)
    return make_plan(c, replace(limits, process_bytes=int(process_bytes)), mode)


class ResizeTarget:
    """What `run_resize` needs from an engine; the engine implements it on its worker thread."""

    capacity: Capacity
    limits: Limits
    ceiling_bytes: int  # the configured process ceiling; resizes never exceed it
    plan: Plan
    state: str

    def idle(self) -> bool: ...
    def synchronize(self) -> None: ...
    def detach_pool(self): ...  # returns the old pool, model no longer refers to it
    def release_pool(self, pool) -> None: ...  # frees its buffers; clears the allocator cache
    def allocate_pool(self, slots: int): ...  # may raise (MemoryError / RuntimeError / ValueError)
    def attach_pool(self, pool, carried: dict | None) -> None: ...
    def pool_counters(self, pool) -> dict: ...
    def apply_plan(self, plan: Plan, trimmed: bool) -> None: ...  # limits, caches, prefix cap
    def mark_failed(self, exc: BaseException) -> None: ...


def run_resize(t: ResizeTarget, req: ResizeRequest) -> dict:
    """Run one resize on the worker thread; returns a result dict rather than raising."""
    t0 = time.monotonic()
    before = {
        "mode": t.plan.mode,
        "pool_slots": t.plan.pool_slots,
        "planned_process_bytes": t.plan.planned_process_bytes,
    }
    if t.state == FAILED:
        return _result(
            "unavailable", req, t0, error="the engine failed an earlier rebuild", before=before
        )
    if (
        not isinstance(req.process_bytes, int)
        or isinstance(req.process_bytes, bool)
        or req.process_bytes <= 0
    ):
        return _result(
            "rejected", req, t0, error="budget_bytes must be a positive integer", before=before
        )
    if req.process_bytes > t.ceiling_bytes:
        return _result(
            "rejected",
            req,
            t0,
            before=before,
            error=f"{req.process_bytes} B is above the configured ceiling of "
            f"{t.ceiling_bytes} B; restart with a larger budget to raise it",
        )
    if not t.idle():
        return _result(
            "busy", req, t0, before=before, error="requests are in flight; retry when idle"
        )

    t.state = VALIDATING
    try:
        if t.plan.mode == "resident":
            trimmed = False
            try:
                new = plan_for(t.capacity, t.limits, req.process_bytes, "resident")
            except BudgetError:
                try:
                    new = plan_for(
                        t.capacity, t.limits, req.process_bytes, "resident", trim_optional=True
                    )
                    trimmed = True
                except BudgetError as exc:
                    t.state = SERVING
                    return _result(
                        "rejected",
                        req,
                        t0,
                        before=before,
                        requires_offload_restart=True,
                        error=f"a resident model cannot fit {req.process_bytes} B: {exc}",
                    )
            t.apply_plan(new, trimmed)
            t.state = SERVING
            return _result(
                "applied", req, t0, before=before, after=_brief(new), trimmed=trimmed, rebuilt=False
            )
        try:
            new = plan_for(t.capacity, t.limits, req.process_bytes, "offload")
        except BudgetError as exc:
            t.state = SERVING
            return _result("rejected", req, t0, before=before, error=str(exc))
        if new.pool_slots == t.plan.pool_slots:
            t.apply_plan(new, False)
            t.state = SERVING
            return _result("unchanged", req, t0, before=before, after=_brief(new), rebuilt=False)

        carried, attempts, errors = _free_pool_and_plan_attempts(t, new)
        for plan, label in attempts:
            try:
                pool = t.allocate_pool(plan.pool_slots)
            except (MemoryError, RuntimeError, ValueError, OSError) as exc:
                errors.append(f"{label} ({plan.pool_slots} slots): {exc!r}")
                t.synchronize()
                continue
            t.attach_pool(pool, carried)
            t.apply_plan(plan, label == "minimum")
            t.state = SERVING
            status = "applied" if label == "target" else "failed"
            return _result(
                status,
                req,
                t0,
                before=before,
                after=_brief(plan),
                rebuilt=True,
                restored=None if label == "target" else label,
                errors=errors or None,
                error=None
                if label == "target"
                else f"target geometry could not be allocated; serving on the {label} geometry",
            )
        exc = RuntimeError("no expert pool geometry could be allocated: " + "; ".join(errors))
        t.state = FAILED
        t.mark_failed(exc)
        return _result("failed", req, t0, before=before, error=str(exc), unhealthy=True)
    except BaseException as exc:  # noqa: BLE001  an unexpected fault must never leave "rebuilding"
        if t.state in (VALIDATING,):
            t.state = SERVING
            return _result("rejected", req, t0, before=before, error=f"validation failed: {exc!r}")
        t.state = FAILED
        t.mark_failed(exc)
        return _result("failed", req, t0, before=before, error=repr(exc), unhealthy=True)


def _free_pool_and_plan_attempts(t: ResizeTarget, new: Plan) -> tuple:
    """Free the old pool before allocating: drop it, keep its counters, and list the geometries to try (target, previous if larger, minimum)."""
    # ---- commit: free before allocate
    t.state = REBUILDING
    t.synchronize()
    old = t.detach_pool()
    carried = t.pool_counters(old)
    old_slots = t.plan.pool_slots
    t.release_pool(old)
    del old
    attempts = [(new, "target")]
    if new.pool_slots > old_slots:
        attempts.append(
            (
                plan_for(t.capacity, t.limits, t.plan.planned_process_bytes, "offload"),
                "previous",
            )
        )
    attempts.append((minimum_plan(t.capacity, t.limits), "minimum"))
    errors = []
    return carried, attempts, errors


def minimum_plan(capacity: Capacity, limits: Limits) -> Plan:
    """The smallest serving geometry: optional caches at zero, exactly `minimum_slots` experts."""
    c = replace(capacity, prefix_bytes=0, allocator_cache_bytes=0)
    need = c.fixed_mlx_bytes + c.minimum_slots * c.expert_slot_bytes + c.host_side_bytes
    return make_plan(c, replace(limits, process_bytes=need), "offload")


def _brief(plan: Plan) -> dict:
    return {
        "mode": plan.mode,
        "pool_slots": plan.pool_slots,
        "planned_process_bytes": plan.planned_process_bytes,
        "planned_mlx_bytes": plan.planned_mlx_bytes,
    }
