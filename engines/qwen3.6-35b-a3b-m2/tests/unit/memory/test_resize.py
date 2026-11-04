"""Idle-only resize against a fake allocator: validation first, free before allocate, rollback."""

from __future__ import annotations

from dataclasses import replace

from jitengine_qwen3_6_35b_a3b_m2.memory.budget import Capacity, Limits, make_plan
from jitengine_qwen3_6_35b_a3b_m2.memory.resize import FAILED, SERVING, ResizeRequest, run_resize

CAP = Capacity(
    dense_bytes=100,
    all_expert_bytes=400,
    expert_slot_bytes=10,
    total_slots=40,
    minimum_slots=4,
    state_bytes=20,
    kv_bytes=30,
    prefix_bytes=10,
    allocator_cache_bytes=10,
    scratch_bytes=40,
    host_bytes=20,
    safety_bytes=20,
)
LIM = Limits(process_bytes=400, physical_bytes=10**6, os_reserve_bytes=0, metal_bytes=10**6)


class FakeTarget:
    def __init__(self, busy=False, fail_sizes=(), mode="auto", limits=LIM):
        self.capacity, self.limits = CAP, limits
        self.ceiling_bytes = limits.process_bytes
        self.plan = make_plan(CAP, limits, "offload" if mode == "auto" else mode)
        self.state = SERVING
        self.busy = busy
        self.fail_sizes = set(fail_sizes)
        self.log = []
        self.live_bytes = self.plan.pool_slots * 10  # "GPU" bytes held by pools
        self.peak_live = self.live_bytes
        self.pool = ("pool", self.plan.pool_slots)
        self.bound = self.pool
        self.failed = None

    def idle(self):
        return not self.busy

    def synchronize(self):
        self.log.append("sync")

    def detach_pool(self):
        self.log.append("detach")
        p, self.bound = self.pool, None
        return p

    def release_pool(self, pool):
        self.log.append(("release", pool[1]))
        self.live_bytes -= pool[1] * 10
        self.pool = None

    def allocate_pool(self, slots):
        self.log.append(("allocate", slots))
        if slots in self.fail_sizes:
            raise MemoryError(f"no room for {slots}")
        self.live_bytes += slots * 10
        self.peak_live = max(self.peak_live, self.live_bytes)
        self.pool = ("pool", slots)
        return self.pool

    def pool_counters(self, pool):
        return {"hits": 5}

    def attach_pool(self, pool, carried):
        self.log.append(("attach", pool[1], carried))
        self.bound = pool

    def apply_plan(self, plan, trimmed):
        self.log.append(("plan", plan.pool_slots, trimmed))
        self.plan, self.limits = plan, plan.limits

    def mark_failed(self, exc):
        self.failed = exc


def test_no_allocation_before_validation():
    t = FakeTarget()
    r = run_resize(t, ResizeRequest(150))  # below the fixed floor: impossible
    assert r["status"] == "rejected"
    assert not any(isinstance(x, tuple) and x[0] in ("allocate", "release") for x in t.log)
    assert t.plan.pool_slots == 15 and t.state == SERVING


def test_busy_does_not_mutate():
    t = FakeTarget(busy=True)
    r = run_resize(t, ResizeRequest(300))
    assert r["status"] == "busy" and t.log == [] and t.plan.pool_slots == 15


def test_above_ceiling_rejected():
    r = run_resize(FakeTarget(), ResizeRequest(401))
    assert r["status"] == "rejected" and "ceiling" in r["error"]


def test_shrink_frees_before_allocating_and_rebinds():
    t = FakeTarget()
    r = run_resize(t, ResizeRequest(300))
    assert r["status"] == "applied" and r["after"]["pool_slots"] == 5
    kinds = [x if isinstance(x, str) else x[0] for x in t.log]
    assert kinds.index("release") < kinds.index("allocate") < kinds.index("attach")
    assert t.peak_live == 150  # never old + new at once
    assert t.bound == ("pool", 5) and t.state == SERVING
    assert ("attach", 5, {"hits": 5}) in t.log  # lifetime counters carried over


def test_grow_then_shrink_round_trip():
    t = FakeTarget()
    assert run_resize(t, ResizeRequest(300))["after"]["pool_slots"] == 5
    assert run_resize(t, ResizeRequest(400))["after"]["pool_slots"] == 15
    assert t.peak_live == 150


def test_unchanged_target_does_not_rebuild():
    t = FakeTarget()
    r = run_resize(t, ResizeRequest(400))
    assert r["status"] == "unchanged" and not any(
        isinstance(x, tuple) and x[0] == "allocate" for x in t.log
    )


def test_failed_grow_restores_previous_geometry():
    t = FakeTarget()
    run_resize(t, ResizeRequest(300))  # 5 slots
    t.fail_sizes = {15}
    r = run_resize(t, ResizeRequest(400))
    assert r["status"] == "failed" and r["restored"] == "previous"
    assert t.bound == ("pool", 5) and t.state == SERVING and t.failed is None


def test_failed_shrink_falls_back_to_minimum_then_unhealthy():
    t = FakeTarget(fail_sizes={5})
    r = run_resize(t, ResizeRequest(300))
    assert r["status"] == "failed" and r["restored"] == "minimum" and t.bound == ("pool", 4)
    t2 = FakeTarget(fail_sizes={5, 4})
    r2 = run_resize(t2, ResizeRequest(300))
    assert (
        r2["status"] == "failed"
        and r2.get("unhealthy")
        and t2.state == FAILED
        and t2.failed is not None
    )
    assert run_resize(t2, ResizeRequest(300))["status"] == "unavailable"


def test_unexpected_fault_never_leaves_rebuilding():
    t = FakeTarget()

    def boom(slots):
        raise KeyError("not an allocation error")

    t.allocate_pool = boom
    r = run_resize(t, ResizeRequest(300))
    assert r["status"] == "failed" and t.state == FAILED


def test_resident_resize_trims_or_requires_restart():
    t = FakeTarget(mode="resident", limits=replace(LIM, process_bytes=10**4))
    assert run_resize(t, ResizeRequest(650))["status"] == "applied"
    r = run_resize(t, ResizeRequest(632))  # fits only without the optional caches
    assert r["status"] == "applied" and r["trimmed"]
    r = run_resize(t, ResizeRequest(500))
    assert r["status"] == "rejected" and r["requires_offload_restart"]
