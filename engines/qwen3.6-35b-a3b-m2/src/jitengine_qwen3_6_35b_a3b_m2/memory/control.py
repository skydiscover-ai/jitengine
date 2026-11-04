"""Memory control for `Engine`, as a mixin: the worker's maintenance between rounds, the resize
protocol `memory.resize` drives, the memory monitor, and the thread-safe controls in front of them.
"""

from __future__ import annotations

import gc
import logging
import queue

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.engine.errors import MemoryPressureError
from jitengine_qwen3_6_35b_a3b_m2.scheduler.batch import Job

logger = logging.getLogger(__name__)


class MemoryControlMixin:
    """Maintenance, the resize protocol, the monitor, and the controls in front of them."""

    # maintenance (worker thread only)
    def _run_maintenance(self, cmd) -> dict:
        from jitengine_qwen3_6_35b_a3b_m2.memory.resize import ResizeRequest, run_resize

        if isinstance(cmd, ResizeRequest):
            res = run_resize(self, cmd)
            res["mlx_active_bytes_after"] = int(mx.get_active_memory())
            res["pool_generation"] = int(self.pool_generation)
            if self.monitor is not None:
                res["observed_after"] = self.monitor.sample_now()
            self.last_resize = res
            logger.info(
                f"[maintenance] resize -> {res['status']}: {res.get('after') or res.get('error')}"
            )
            return res
        kind, arg = cmd
        if kind == "trim":  # safe between steps: entries are immutable, cache is free
            evicted = self.store.resize(0)
            mx.set_cache_limit(0)
            mx.clear_cache()
            return {"status": "applied", "action": "trim", "prefix_evicted_bytes": evicted}
        if kind == "untrim":
            self.store.resize(int(self.capacity.prefix_bytes))
            mx.set_cache_limit(int(self.capacity.allocator_cache_bytes))
            return {"status": "applied", "action": "untrim"}
        if kind == "abort":
            n = self._abort_active(MemoryPressureError(str(arg)))
            return {"status": "applied", "action": "abort", "aborted": n}
        raise ValueError(f"unknown maintenance command {kind!r}")

    def _abort_active(self, exc: BaseException) -> int:
        """Finish the in-flight step (tokens discarded), then fail every row and pending request."""
        n = 0
        if self._batch is not None:
            try:
                mx.eval(self._y)
            except Exception:  # noqa: BLE001
                pass  # the step may have failed; the rows are failed below either way
            self._batch, self._y, self._telem = None, None, None
        for row in list(self._rows):
            if not row.retired:
                self._fail(row, exc)
                n += 1
        self._prefilling = None
        for job in list(self._pending):
            job.out.put(("error", exc))
            n += 1
        self._pending = []
        self._pending_dirty = True
        mx.clear_cache()
        return n

    # the maintenance.ResizeTarget protocol (worker thread only)
    @property
    def ceiling_bytes(self) -> int:
        return int(self.configured_limits.process_bytes)

    def idle(self) -> bool:
        return (
            self._batch is None
            and not self._rows
            and not self._pending
            and not self._scores
            and self._prefilling is None
        )

    def synchronize(self) -> None:
        sync = getattr(mx, "synchronize", None)
        if sync is not None:
            sync()
        gc.collect()
        mx.clear_cache()

    def detach_pool(self):
        pool = self.expert_pool
        self.model.bind_expert_pool(None)
        self.expert_pool = None
        return pool

    def release_pool(self, pool) -> None:
        if pool is not None:
            pool.release()
        gc.collect()
        mx.clear_cache()

    def allocate_pool(self, slots: int):
        try:
            return self._new_pool(int(slots))
        except BaseException:
            gc.collect()
            mx.clear_cache()
            raise

    def pool_counters(self, pool) -> dict:
        return pool.lifetime_counters()

    def attach_pool(self, pool, carried: dict | None) -> None:
        if carried:
            pool.carry_counters(carried)
        self.expert_pool = pool
        self.model.bind_expert_pool(pool)
        self.pool_generation += 1

    def apply_plan(self, plan, trimmed: bool) -> None:
        self.plan = plan
        self.limits = plan.limits
        mx.set_memory_limit(int(plan.mlx_cap_bytes))
        self.memory_limit_bytes = int(plan.mlx_cap_bytes)
        self.limit_status["memory_limit_bytes"] = int(plan.mlx_cap_bytes)
        wired = min(
            int(plan.planned_mlx_bytes), int(self.device_info["max_recommended_working_set_size"])
        )
        try:
            mx.set_wired_limit(wired)
            self.limit_status["wired_limit_bytes"] = wired
        except Exception as exc:  # noqa: BLE001
            self.limit_status["errors"].append(f"set_wired_limit({wired}): {exc!r}")
        prefix = 0 if trimmed else int(plan.capacity.prefix_bytes)
        cache = 0 if trimmed else int(plan.capacity.allocator_cache_bytes)
        self.store.resize(prefix)
        mx.set_cache_limit(cache)
        self.limit_status["cache_limit_bytes"] = cache

    def mark_failed(self, exc: BaseException) -> None:
        self._fail_everything(exc)

    def reconfigure_memory(
        self, process_bytes: int, reason: str = "manual", timeout_s: float = 900.0
    ) -> dict:
        """Thread-safe idle-only resize to `process_bytes`; returns run_resize's result dict."""
        from jitengine_qwen3_6_35b_a3b_m2.memory.resize import ResizeRequest

        if isinstance(process_bytes, bool) or not isinstance(process_bytes, int):
            return {"status": "rejected", "error": "process_bytes must be an integer"}
        job = self._submit(Job("maintenance", payload=ResizeRequest(int(process_bytes), reason)))
        try:
            kind, payload = job.out.get(timeout=timeout_s)
        except queue.Empty:
            return {
                "status": "unavailable",
                "error": f"no answer from the model worker in {timeout_s}s",
            }
        if kind == "error":
            return {"status": "unavailable", "error": repr(payload)}
        return payload

    def maintenance_command(self, kind: str, arg=None, timeout_s: float = 60.0) -> dict:
        """trim / untrim / abort, executed by the worker at its next safe point."""
        job = self._submit(Job("maintenance", payload=(kind, arg)))
        try:
            k, payload = job.out.get(timeout=timeout_s)
        except queue.Empty:
            return {"status": "unavailable", "error": "timeout"}
        return payload if k == "done" else {"status": "unavailable", "error": repr(payload)}

    def set_admission(self, paused: str | None) -> None:
        """Open (None) or pause new generate admissions with a reason (thread-safe)."""
        with self._outstanding_lock:
            self._admission_paused = paused

    def memory_snapshot(self) -> dict:
        """MLX counters, the plan and (when a monitor runs) the observed process footprint."""
        out = {
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
            "plan": self.plan.to_dict(),
            "limits_applied": dict(self.limit_status),
            "kv_reserved_bytes": int(self.pool.kv_live),
            "kv_allocated_bytes": int(self.pool.kv_allocated),
            "prefix_store_bytes": int(self.store.bytes),
            "prefix_store_cap_bytes": int(self.store.cap_bytes),
            "resident_floor_bytes": int(self.resident_floor_bytes()),
            "admission": self._admission_paused or "open",
            "memory_policy": self.memory_policy,
            "pool_generation": int(self.pool_generation),
            "last_resize": self.last_resize,
        }
        if self.monitor is not None:
            out["observed"] = self.monitor.snapshot()
        if self.controller is not None:
            out["pressure"] = self.controller.snapshot()
        return out

    def _start_monitor(self) -> None:
        """The 1 Hz monitor; with `memory_policy=adaptive` it drives the pressure controller."""
        from jitengine_qwen3_6_35b_a3b_m2.memory.monitor import MemoryMonitor
        from jitengine_qwen3_6_35b_a3b_m2.memory.pressure import PressureController, PressurePolicy

        callback = None
        if self.memory_policy == "adaptive":
            from jitengine_qwen3_6_35b_a3b_m2.memory.resize import minimum_plan

            floor = minimum_plan(self.capacity, self.configured_limits).planned_process_bytes
            policy = PressurePolicy(
                self.ceiling_bytes,
                floor,
                self.plan.mode,
                **(self.options.get("policy_overrides") or {}),
            )
            self.controller = PressureController(self, policy)
            callback = self.controller
        self.monitor = MemoryMonitor(
            float(self.options.get("monitor_interval_s") or 1.0), callback
        ).start()

    def resident_floor_bytes(self) -> int:
        """Bytes this configuration cannot release without a restart."""
        c = self.capacity
        experts = c.all_expert_bytes if not self.offload else c.minimum_slots * c.expert_slot_bytes
        return c.dense_bytes + experts + c.state_bytes + c.kv_bytes + c.scratch_bytes
