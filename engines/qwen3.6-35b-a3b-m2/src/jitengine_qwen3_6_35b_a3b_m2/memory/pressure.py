"""The memory-pressure policy (pure) and the controller that applies its decisions.

`PressurePolicy.observe` turns memory samples into `Decision`s with hysteresis; it never allocates,
calls MLX or submits requests. The thresholds are provisional defaults, each overridable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from jitengine_qwen3_6_35b_a3b_m2.memory.monitor import MemorySample

logger = logging.getLogger(__name__)

GIB = 1 << 30
MIB = 1 << 20

WARN_AVAILABLE = 2 * GIB
WARN_SAMPLES = 3
CRIT_AVAILABLE = 1 * GIB
CRIT_SAMPLES = 2
CEILING_MARGIN = 512 * MIB
SWAP_GROWTH = 256 * MIB
SWAP_WINDOW_S = 30.0
RECOVER_AVAILABLE = 3 * GIB
RECOVER_HOLD_S = 30.0
GROW_STEP = 256 * MIB
SHRINK_STEP = 512 * MIB
RESIZE_SPACING_S = 30.0
GROW_COOLDOWN_S = 60.0
STALE_S = 5.0

ACTIONS = ("keep", "trim", "pause", "abort", "shrink", "resume", "grow")


@dataclass(frozen=True)
class Decision:
    action: str
    target_bytes: int | None
    reason: str


class PressurePolicy:
    def __init__(self, process_limit_bytes: int, minimum_bytes: int, mode: str, **overrides):
        if mode not in ("offload", "resident"):
            raise ValueError("mode must be offload or resident")
        self.process_limit_bytes = int(process_limit_bytes)  # the configured ceiling
        self.minimum_bytes = int(minimum_bytes)
        self.mode = mode
        self.t = {
            k: overrides.get(k, v)
            for k, v in dict(
                warn_available=WARN_AVAILABLE,
                warn_samples=WARN_SAMPLES,
                crit_available=CRIT_AVAILABLE,
                crit_samples=CRIT_SAMPLES,
                ceiling_margin=CEILING_MARGIN,
                swap_growth=SWAP_GROWTH,
                swap_window_s=SWAP_WINDOW_S,
                recover_available=RECOVER_AVAILABLE,
                recover_hold_s=RECOVER_HOLD_S,
                grow_step=GROW_STEP,
                shrink_step=SHRINK_STEP,
                resize_spacing_s=RESIZE_SPACING_S,
                grow_cooldown_s=GROW_COOLDOWN_S,
                stale_s=STALE_S,
            ).items()
        }
        unknown = set(overrides) - set(self.t)
        if unknown:
            raise ValueError(f"unknown policy overrides {sorted(unknown)}")
        self.history: deque = deque()
        self.state = "normal"  # normal | warning | critical
        self.paused = False
        self.trimmed = False
        self.degraded = False
        self.recover_since: float | None = None
        self.last_resize_t = -1e18
        self.grow_blocked_until = -1e18
        self.last_decision = Decision("keep", None, "start")
        self.counts = {a: 0 for a in ACTIONS}

    def _recent(self, n: int) -> list:
        return list(self.history)[-n:]

    def _swap_growth(self, now: float) -> int | None:
        window = [
            s
            for s in self.history
            if now - s.timestamp <= self.t["swap_window_s"] and s.swap_used_bytes is not None
        ]
        if len(window) < 2:
            return None
        return window[-1].swap_used_bytes - min(s.swap_used_bytes for s in window)

    def note_result(self, decision: Decision, ok: bool, now: float | None = None) -> None:
        """The controller reports whether an action took effect (a failed grow -> cooldown)."""
        now = time.monotonic() if now is None else now
        if decision.action in ("shrink", "grow"):
            self.last_resize_t = now
        if decision.action == "grow" and not ok:
            self.grow_blocked_until = now + self.t["grow_cooldown_s"]

    def _decide(self, action: str, target: int | None, reason: str) -> Decision:
        d = Decision(action, target, reason)
        self.last_decision = d
        self.counts[action] += 1
        return d

    def observe(self, sample: MemorySample, *, idle: bool, current_limit_bytes: int) -> Decision:
        now = self._trim_history(sample)

        valid = self._sample_complete(sample)
        if not valid:
            self.degraded = True
            self.recover_since = None
            return self._decide(
                "keep",
                None,
                f"telemetry unavailable ({sample.error or 'missing fields'}); "
                "static plan kept, automatic resizing disabled",
            )
        self.degraded = False
        fp, avail, swap_up, low_crit, low_warn, over_ceiling, near_ceiling, swap_storm = (
            self._read_signals(sample, now)
        )

        if low_crit or over_ceiling or swap_storm:
            self.state = "critical"
            self.recover_since = None
            why = self._critical_why(avail, fp, swap_up, low_crit, over_ceiling, swap_storm)
            if over_ceiling and not idle:
                return self._decide("abort", None, f"critical: {why}; stopping running requests")
            if not self.paused:
                self.paused = True
                return self._decide("pause", None, f"critical: {why}; admissions paused")
            if not self.trimmed:
                self.trimmed = True
                return self._decide("trim", None, f"critical: {why}; trimming optional caches")
            if (
                self.mode == "offload"
                and idle
                and now - self.last_resize_t >= self.t["resize_spacing_s"]
                and current_limit_bytes > self.minimum_bytes
            ):
                target = max(self.minimum_bytes, current_limit_bytes - self.t["shrink_step"])
                self.last_resize_t = now  # spaced from the proposal, even if unreported
                return self._decide("shrink", target, f"critical: {why}; shrinking the expert pool")
            return self._decide(
                "keep",
                None,
                f"critical: {why}; nothing further to release now"
                + (
                    " (resident floor: restart in offload mode or close applications)"
                    if self.mode == "resident"
                    else ""
                ),
            )

        if low_warn or near_ceiling:
            self.state = "warning"
            self.recover_since = None
            if not self.trimmed:
                self.trimmed = True
                why = self._warning_why(low_warn, avail, fp)
                return self._decide("trim", None, f"warning: {why}; trimming optional caches")
            return self._decide("keep", None, "warning: holding (growth stopped)")

        healthy = self._is_healthy(avail, swap_up, fp)
        if not healthy:
            self.recover_since = None
            self.state = "normal" if not self.paused else self.state
            return self._decide("keep", None, "no pressure; recovery window not open")
        held = self._recovery_held(now)
        if held < self.t["recover_hold_s"]:
            return self._decide(
                "keep", None, f"recovering ({held:.0f}/{self.t['recover_hold_s']:.0f}s)"
            )
        self.state = "normal"
        if self.paused or self.trimmed:
            self.paused = False
            self.trimmed = False
            return self._decide(
                "resume", None, f"recovered for {held:.0f}s; admissions reopened, caches restored"
            )
        if (
            self.mode == "offload"
            and idle
            and current_limit_bytes < self.process_limit_bytes
            and now - self.last_resize_t >= self.t["resize_spacing_s"]
            and now >= self.grow_blocked_until
        ):
            target = min(self.process_limit_bytes, current_limit_bytes + self.t["grow_step"])
            self.last_resize_t = now
            return self._decide(
                "grow", target, f"recovered for {held:.0f}s; growing the expert pool"
            )
        return self._decide("keep", None, "normal")

    def _trim_history(self, sample: MemorySample) -> float:
        """Record `sample` and drop samples older than the longest window; return its timestamp."""
        now = sample.timestamp
        self.history.append(sample)
        horizon = max(self.t["swap_window_s"], self.t["recover_hold_s"]) + 5
        while self.history and now - self.history[0].timestamp > horizon:
            self.history.popleft()
        return now

    def _sample_complete(self, sample: MemorySample) -> bool:
        """Whether the sample carries no error and both the footprint and the available memory."""
        valid = (
            sample.error is None
            and sample.footprint_bytes is not None
            and sample.available_bytes is not None
        )
        return valid

    def _read_signals(self, sample: MemorySample, now: float) -> tuple:
        """The pressure signals of this sample: footprint, available memory, swap growth and the five flags."""
        fp, avail = sample.footprint_bytes, sample.available_bytes
        swap_up = self._swap_growth(now)
        crit_n = self._recent(self.t["crit_samples"])
        warn_n = self._recent(self.t["warn_samples"])
        low_crit = len(crit_n) >= self.t["crit_samples"] and all(
            s.available_bytes is not None and s.available_bytes < self.t["crit_available"]
            for s in crit_n
        )
        low_warn = len(warn_n) >= self.t["warn_samples"] and all(
            s.available_bytes is not None and s.available_bytes < self.t["warn_available"]
            for s in warn_n
        )
        over_ceiling = fp > self.process_limit_bytes
        near_ceiling = fp >= self.process_limit_bytes - self.t["ceiling_margin"]
        swap_storm = swap_up is not None and swap_up > self.t["swap_growth"]
        return fp, avail, swap_up, low_crit, low_warn, over_ceiling, near_ceiling, swap_storm

    def _critical_why(
        self,
        avail: int,
        fp: int,
        swap_up: float | None,
        low_crit: bool,
        over_ceiling: bool,
        swap_storm: bool,
    ) -> str:
        """The reasons a sample is critical, joined for the decision text."""
        why = ", ".join(
            w
            for w, c in (
                (f"available {avail / GIB:.2f} GiB", low_crit),
                (f"footprint {fp / GIB:.2f} GiB above the ceiling", over_ceiling),
                (
                    f"system swap +{(swap_up or 0) / MIB:.0f} MiB in "
                    f"{self.t['swap_window_s']:.0f}s",
                    swap_storm,
                ),
            )
            if c
        )
        return why

    def _warning_why(self, low_warn: bool, avail: int, fp: int) -> str:
        """The reason a sample is a warning, for the decision text."""
        why = (
            f"available {avail / GIB:.2f} GiB"
            if low_warn
            else f"footprint {fp / GIB:.2f} GiB within {self.t['ceiling_margin'] / MIB:.0f} MiB of the ceiling"
        )
        return why

    def _is_healthy(self, avail: int, swap_up: float | None, fp: int) -> bool:
        """Neither warning nor critical: whether the sample is healthy enough to count toward recovery."""
        healthy = (
            avail >= self.t["recover_available"]
            and not (swap_up and swap_up > 0)
            and fp + self.t["grow_step"] <= self.process_limit_bytes
        )
        return healthy

    def _recovery_held(self, now: float) -> float:
        """Start the recovery window if it is not open; return how long it has been open."""
        if self.recover_since is None:
            self.recover_since = now
        held = now - self.recover_since
        return held

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "paused": self.paused,
            "trimmed": self.trimmed,
            "degraded": self.degraded,
            "last_decision": self.last_decision.__dict__,
            "counts": dict(self.counts),
            "process_limit_bytes": self.process_limit_bytes,
            "minimum_bytes": self.minimum_bytes,
            "mode": self.mode,
            "thresholds": dict(self.t),
        }


class PressureController:
    """Applies policy decisions on the monitor thread via the engine's thread-safe entry points."""

    def __init__(self, engine, policy: PressurePolicy):
        self.engine = engine
        self.policy = policy
        self._lock = threading.Lock()
        self.applied: deque = deque(maxlen=32)

    def __call__(self, sample: MemorySample) -> None:
        eng = self.engine
        with self._lock:
            d = self.policy.observe(
                sample,
                idle=bool(eng.idle()),
                current_limit_bytes=int(eng.plan.planned_process_bytes),
            )
            if d.action == "keep":
                return
            ok, detail = True, None
            if d.action == "pause":
                eng.set_admission("memory_pressure")
            elif d.action == "trim":
                detail = eng.maintenance_command("trim")
            elif d.action == "abort":
                eng.set_admission("memory_pressure")
                detail = eng.maintenance_command("abort", d.reason)
            elif d.action == "resume":
                detail = eng.maintenance_command("untrim")
                eng.set_admission(None)
            elif d.action in ("shrink", "grow"):
                detail = eng.reconfigure_memory(int(d.target_bytes), reason=f"pressure:{d.action}")
                ok = detail.get("status") in ("applied", "unchanged")
            self.policy.note_result(d, ok, sample.timestamp)
            self.applied.append(
                {
                    "t": sample.timestamp,
                    "action": d.action,
                    "target_bytes": d.target_bytes,
                    "reason": d.reason,
                    "ok": ok,
                    "status": (detail or {}).get("status") if isinstance(detail, dict) else None,
                }
            )
            logger.info(f"[pressure] {d.action}: {d.reason}")

    def snapshot(self) -> dict:
        with self._lock:
            out = self.policy.snapshot()
            out["applied"] = list(self.applied)
            return out
