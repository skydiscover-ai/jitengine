"""Memory plan: split a process ceiling into integer-byte buckets and size the expert pool.

`capacity_for` and `limits_for` take the plan's inputs from the checkpoint's headers, the options
and the device; `make_plan` and the rest are pure: no I/O and no floats in any decision. None of it
imports MLX, so `inspect` plans without Metal. The ceilings are admission targets, not reservations
(`mx.set_memory_limit` is only a guideline inside MLX).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.inventory import inventory
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import load_config
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.batch import delta_bytes_for, kv_bytes_for

GIB = 1 << 30
MIB = 1 << 20

# Default allowances; an observed overrun shrinks capacities, never raises the process ceiling.
DEFAULT_SCRATCH_BYTES = 384 * MIB
DEFAULT_HOST_BYTES = 768 * MIB
DEFAULT_SAFETY_BYTES = 512 * MIB
DEFAULT_PREFIX_BYTES = 768 * MIB


def host_default_prefix_bytes() -> int:
    """Prefix-store budget for this host: 1152 MiB where unified memory is >= 40 GiB, else 768 MiB."""
    import os

    try:
        mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except ValueError, OSError:
        mem = 0
    # Sized so the store fills within about 18 short-prompt entries: memory must stop growing early in a run of
    # distinct prompts, and a larger store keeps growing past that point.
    return 1152 * MIB if mem >= 40 * GIB else DEFAULT_PREFIX_BYTES


DEFAULT_ALLOCATOR_CACHE_BYTES = 128 * MIB


def host_default_budget_bytes() -> int:
    """Default process budget for this host: 34 GiB where unified memory is >= 40 GiB (so the 4-bit checkpoint plans
    resident), else 18 GiB. MLX-free (sysconf)."""
    import os

    try:
        mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except ValueError, OSError:
        mem = 0
    return 34 * GIB if mem >= 40 * GIB else 18 * GIB


MODES = ("auto", "resident", "offload")


class BudgetError(ValueError):
    """A plan that cannot be satisfied. Raised before anything is allocated."""


def _nonneg_int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise BudgetError(f"{name} must be an integer number of bytes, got {v!r}")
    if v < 0:
        raise BudgetError(f"{name} must be >= 0, got {v}")
    return v


@dataclass(frozen=True)
class Capacity:
    dense_bytes: int
    all_expert_bytes: int
    expert_slot_bytes: int
    total_slots: int
    minimum_slots: int
    state_bytes: int
    kv_bytes: int
    prefix_bytes: int
    allocator_cache_bytes: int
    scratch_bytes: int
    host_bytes: int
    safety_bytes: int

    def __post_init__(self):
        for k, v in asdict(self).items():
            _nonneg_int(k, v)

    @property
    def fixed_mlx_bytes(self) -> int:
        return (
            self.dense_bytes
            + self.state_bytes
            + self.kv_bytes
            + self.prefix_bytes
            + self.allocator_cache_bytes
            + self.scratch_bytes
        )

    @property
    def host_side_bytes(self) -> int:
        return self.host_bytes + self.safety_bytes


@dataclass(frozen=True)
class Limits:
    process_bytes: int
    physical_bytes: int
    os_reserve_bytes: int
    metal_bytes: int

    def __post_init__(self):
        for k, v in asdict(self).items():
            _nonneg_int(k, v)

    @property
    def process_cap(self) -> int:
        return max(0, min(self.process_bytes, self.physical_bytes - self.os_reserve_bytes))


@dataclass(frozen=True)
class Plan:
    mode: str  # "resident" | "offload", never "auto"
    pool_slots: int  # 0 when resident
    planned_process_bytes: int
    planned_mlx_bytes: int
    limits: Limits
    capacity: Capacity | None = None
    process_cap_bytes: int = 0
    mlx_cap_bytes: int = 0

    @property
    def pool_bytes(self) -> int:
        return self.pool_slots * (self.capacity.expert_slot_bytes if self.capacity else 0)

    @property
    def expert_fraction(self) -> float:
        if self.mode == "resident" or not self.capacity or not self.capacity.total_slots:
            return 1.0
        return self.pool_slots / self.capacity.total_slots

    def buckets(self) -> dict:
        c = self.capacity
        if c is None:
            return {}
        experts = c.all_expert_bytes if self.mode == "resident" else self.pool_bytes
        return {
            "dense": c.dense_bytes,
            "experts": experts,
            "state": c.state_bytes,
            "kv": c.kv_bytes,
            "prefix_store": c.prefix_bytes,
            "allocator_cache": c.allocator_cache_bytes,
            "scratch": c.scratch_bytes,
            "host": c.host_bytes,
            "safety": c.safety_bytes,
        }

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pool_slots": self.pool_slots,
            "total_slots": self.capacity.total_slots if self.capacity else None,
            "expert_fraction": self.expert_fraction,
            "planned_process_bytes": self.planned_process_bytes,
            "planned_mlx_bytes": self.planned_mlx_bytes,
            "process_cap_bytes": self.process_cap_bytes,
            "mlx_cap_bytes": self.mlx_cap_bytes,
            "limits": asdict(self.limits),
            "buckets": self.buckets(),
        }

    def describe(self) -> str:
        g = GIB
        b = self.buckets()
        parts = " + ".join(f"{k} {v / g:.3f}" for k, v in b.items())
        extra = (
            f" ({self.pool_slots} of {self.capacity.total_slots} experts, "
            f"{100 * self.expert_fraction:.1f}%)"
            if self.mode == "offload" and self.capacity
            else ""
        )
        return (
            f"{self.mode}: {parts} = {self.planned_process_bytes / g:.3f} GiB planned process "
            f"(cap {self.process_cap_bytes / g:.3f} GiB; MLX {self.planned_mlx_bytes / g:.3f} of "
            f"{self.mlx_cap_bytes / g:.3f} GiB){extra}"
        )


def make_plan(capacity: Capacity, limits: Limits, mode: str = "auto") -> Plan:
    """Charge every MLX bucket once against the MLX ceiling, or raise BudgetError with the reason.

    `offload` uses a pool even when every expert fits, so the pool can be resized while idle.
    """
    if mode not in MODES:
        raise BudgetError(f"mode must be one of {MODES}, got {mode!r}")
    c = capacity
    process_cap = limits.process_cap
    mlx_cap = min(process_cap - c.host_side_bytes, limits.metal_bytes)
    if mlx_cap <= 0:
        raise BudgetError(
            f"process ceiling {process_cap} B leaves nothing for MLX after the "
            f"{c.host_side_bytes} B host+safety allowance"
        )
    fixed = c.fixed_mlx_bytes
    if fixed > mlx_cap:
        raise BudgetError(
            f"the non-expert floor ({fixed / GIB:.2f} GiB: dense {c.dense_bytes / GIB:.2f}, state "
            f"{c.state_bytes / GIB:.2f}, KV {c.kv_bytes / GIB:.2f}, caches "
            f"{(c.prefix_bytes + c.allocator_cache_bytes) / GIB:.2f}, scratch {c.scratch_bytes / GIB:.2f}) "
            f"exceeds the MLX ceiling {mlx_cap / GIB:.2f} GiB"
        )

    def done(m: str, slots: int, experts: int) -> Plan:
        mlx = fixed + experts
        return Plan(
            mode=m,
            pool_slots=slots,
            planned_process_bytes=mlx + c.host_side_bytes,
            planned_mlx_bytes=mlx,
            limits=limits,
            capacity=c,
            process_cap_bytes=process_cap,
            mlx_cap_bytes=mlx_cap,
        )

    resident_fits = fixed + c.all_expert_bytes <= mlx_cap
    if mode == "resident" or (mode == "auto" and resident_fits):
        if not resident_fits:
            need = fixed + c.all_expert_bytes + c.host_side_bytes
            raise BudgetError(
                f"resident mode needs {need / GIB:.2f} GiB of process ceiling, "
                f"{process_cap / GIB:.2f} GiB is available; use auto or offload"
            )
        return done("resident", 0, c.all_expert_bytes)

    if c.expert_slot_bytes <= 0:
        raise BudgetError("expert slot size is zero: this checkpoint cannot be offloaded")
    slots = min((mlx_cap - fixed) // c.expert_slot_bytes, c.total_slots)
    if slots < c.minimum_slots:
        need = fixed + c.minimum_slots * c.expert_slot_bytes + c.host_side_bytes
        raise BudgetError(
            f"an expert pool of {slots} slots is below the {c.minimum_slots} one decode "
            f"step needs; the minimum process ceiling is {need / GIB:.2f} GiB"
        )
    return done("offload", slots, slots * c.expert_slot_bytes)


def parse_bytes(text: str) -> int:
    """Parse '18GiB' (binary), '18.5GB' (decimal) or a bare byte count; raises BudgetError."""
    s = str(text).strip().replace("_", "")
    units = [
        ("KiB", 1 << 10),
        ("MiB", 1 << 20),
        ("GiB", 1 << 30),
        ("TiB", 1 << 40),
        ("KB", 10**3),
        ("MB", 10**6),
        ("GB", 10**9),
        ("TB", 10**12),
        ("B", 1),
    ]
    mult = 1
    for suf, m in units:
        if s.endswith(suf):
            s, mult = s[: -len(suf)].strip(), m
            break
    try:
        if s.isdigit():
            v = int(s) * mult
        else:
            from decimal import Decimal, InvalidOperation

            try:
                d = Decimal(s)
            except InvalidOperation:
                raise BudgetError(f"not a byte quantity: {text!r}") from None
            if not d.is_finite():
                raise BudgetError(f"not a finite byte quantity: {text!r}")
            v = int(d * mult)
    except ValueError:
        raise BudgetError(f"not a byte quantity: {text!r}") from None
    if v < 0:
        raise BudgetError(f"byte quantity must be >= 0: {text!r}")
    return v


KV_STEP = 256  # kv_cache.KV_STEP, duplicated so this module stays MLX-free (checked by a test)


def context_positions(max_context_tokens: int) -> int:
    return max(KV_STEP, -(-int(max_context_tokens) // KV_STEP) * KV_STEP)


def capacity_for(model_path: str, opt: dict, cfg=None, inv=None, scan=None) -> tuple:
    """(Capacity, cfg, inventory, scan, kv_budget_bytes, score_reserve_bytes) for resolved options."""
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.pack import CheckpointScan

    cfg = cfg or load_config(model_path)
    inv = inv or inventory(model_path, cfg)
    scan = scan or CheckpointScan(model_path)
    delta = delta_bytes_for(cfg)
    ctx = context_positions(opt["max_context_tokens"])
    slots = int(opt["max_slots"])
    row = kv_bytes_for(cfg, ctx)
    rows = int(opt.get("kv_budget_rows") or 0)
    kv_budget = (min(slots, rows) if rows > 0 else slots) * row
    score_reserve = delta + kv_bytes_for(cfg, ctx)  # `score` runs outside the decode slots
    cap = Capacity(
        dense_bytes=int(inv.dense_bytes),
        all_expert_bytes=int(inv.expert_bytes),
        expert_slot_bytes=int(scan.record_bytes),
        total_slots=int(scan.num_records),
        minimum_slots=int(cfg.num_hidden_layers * cfg.num_experts_per_tok),
        state_bytes=int(slots * delta + score_reserve),
        kv_bytes=int(kv_budget),
        prefix_bytes=int(opt["prefix_cache_bytes"]),
        allocator_cache_bytes=int(opt["allocator_cache_bytes"]),
        scratch_bytes=int(opt["scratch_bytes"]),
        host_bytes=int(opt["host_bytes"]),
        safety_bytes=int(opt["safety_bytes"]),
    )
    return cap, cfg, inv, scan, kv_budget, score_reserve


def limits_for(opt: dict, device_info: dict | None) -> Limits:
    import os

    physical = int(
        opt.get("physical_bytes") or os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    )
    metal = opt.get("metal_bytes")
    if not metal:
        metal = int((device_info or {}).get("max_recommended_working_set_size") or physical)
    return Limits(
        process_bytes=int(opt["process_budget_bytes"]),
        physical_bytes=physical,
        os_reserve_bytes=int(opt["system_reserve_bytes"]),
        metal_bytes=int(metal),
    )


def device_info_or_none() -> tuple[dict | None, str | None]:
    """mx.device_info() when MLX + Metal import here, else (None, reason)."""
    try:
        import mlx.core as mx

        return {k: (int(v) if isinstance(v, int) else v) for k, v in mx.device_info().items()}, None
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)
