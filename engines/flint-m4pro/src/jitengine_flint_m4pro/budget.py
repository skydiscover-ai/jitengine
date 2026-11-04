"""Memory plan: whether the whole model fits (resident), or the routed experts stream from the checkpoint (offload).

Resident is the tuned engine: every weight in the GPU's wired memory, decoded by the megakernel. Its need is the
weights plus the history plus config.HEADROOM_BYTES (config.resolve_context_limit, unchanged). Without a budget the
engine is resident whenever the GPU wired limit holds that, as it always was.

Offload is for a Mac or a budget that does not hold the 22.6 GB of weights: the non-expert weights (1.3 GB) stay
resident, `pool_slots` of the 12,032 routed-expert records (47 layers x 256, 1,769,472 bytes each) live in an expert
pool (expert_pool.py), and a record not in the pool is read from the checkpoint when the router picks it. Decode runs
on the graph backend (stock MLX ops), since the megakernel reads every expert from resident banks. The budget is the
process's memory: non-expert weights + history + pool + OFFLOAD_HEADROOM_BYTES (prefill transients, MLX's buffer
cache, the Python process). The pool must hold at least one layer's 256 experts, which a prefill chunk can touch.

Everything here is pure arithmetic over the checkpoint headers and the device facts (no MLX), so a plan is printed,
and a refusal raised, before any weight is allocated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from jitengine_flint_m4pro.model.config import (
    CONTEXT_AUTO,
    CONTEXT_AUTO_MIN,
    CONTEXT_AUTO_QUANTUM,
    CONTEXT_LIMIT_MAX,
    HEADROOM_BYTES,
    SYSTEM_RESERVE_BYTES,
    StartupError,
    check_context_limit,
    resolve_context_limit,
)

GIB = 2**30
# prefill transients (a 2,048-token chunk), MLX's buffer cache (capped at ALLOCATOR_CACHE_BYTES in offload) and the
# Python process itself; measured peaks on the 24 GB M2 stayed below the plan with this allowance
OFFLOAD_HEADROOM_BYTES = 2 * GIB
ALLOCATOR_CACHE_BYTES = 256 * 2**20
# `auto` in offload: this much history, less if the budget cannot hold it beside one layer of experts. The history is
# memory the expert pool cannot use, so offload does not size it to the budget as resident does.
OFFLOAD_CONTEXT_AUTO = 32768
# `serve --mode`: auto offloads only when the whole model does not fit; resident refuses instead; offload streams the
# experts even when it fits (a smaller footprint, or a test of the offload path on a big Mac)
MEMORY_MODES = ("auto", "resident", "offload")


@dataclass(frozen=True)
class MemoryPlan:
    mode: str  # "resident" | "offload"
    budget_bytes: (
        int | None
    )  # what the plan fits under (None: resident against the GPU wired limit alone)
    budget_source: str  # "flag" | "default" | "none"
    context_limit: int
    context_limit_source: str  # "auto" | "explicit"
    weights_bytes: int
    dense_bytes: int  # the non-expert weights (bank 0)
    history_bytes: int
    headroom_bytes: int
    record_bytes: int
    total_records: int
    pool_slots: int  # offload only (0 when resident)

    @property
    def pool_bytes(self) -> int:
        return self.pool_slots * self.record_bytes

    @property
    def planned_bytes(self) -> int:
        if self.mode == "resident":
            return self.weights_bytes + self.history_bytes + self.headroom_bytes
        return self.dense_bytes + self.history_bytes + self.pool_bytes + self.headroom_bytes

    @property
    def expert_fraction(self) -> float:
        return 1.0 if self.mode == "resident" else self.pool_slots / self.total_records

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            pool_bytes=self.pool_bytes,
            planned_bytes=self.planned_bytes,
            expert_fraction=round(self.expert_fraction, 4),
        )
        return d

    def line(self) -> str:
        g = lambda n: f"{n / GIB:.2f}"  # noqa: E731
        budget = (
            f"budget {g(self.budget_bytes)} GiB ({self.budget_source})"
            if self.budget_bytes is not None
            else "no budget (the GPU wired limit)"
        )
        if self.mode == "resident":
            return (
                f"resident, {budget}: weights {g(self.weights_bytes)} + history {g(self.history_bytes)} "
                f"({self.context_limit:,} tokens) + headroom {g(self.headroom_bytes)} = "
                f"{g(self.planned_bytes)} GiB"
            )
        return (
            f"offload, {budget}: non-expert weights {g(self.dense_bytes)} + history {g(self.history_bytes)} "
            f"({self.context_limit:,} tokens) + expert pool {g(self.pool_bytes)} ({self.pool_slots:,} of "
            f"{self.total_records:,} experts, {100 * self.expert_fraction:.1f}%) + headroom "
            f"{g(self.headroom_bytes)} = {g(self.planned_bytes)} GiB"
        )


def flag_budget(gib: int | None, gb: int | None) -> int | None:
    """The budget the flags request, in bytes: --memory-budget-gib, or --memory-budget-gb (decimal GB)."""
    if gib is not None and gb is not None:
        raise ValueError(
            "pass the budget once: --memory-budget-gib or --memory-budget-gb, not both"
        )
    return gib if gib is not None else gb


def default_offload_budget(wired_bytes: int, memory_size: int) -> int:
    """The budget offload takes when none is given: the GPU wired limit, at most physical memory less 6 GiB."""
    cap = int(memory_size) - SYSTEM_RESERVE_BYTES if memory_size else int(wired_bytes)
    return max(0, min(int(wired_bytes), cap))


def _offload_context(
    limit, avail: int, bytes_per_token: int, min_pool: int, budget: int, floor: int
) -> tuple:
    """(context limit, source) for offload: an explicit limit must leave `min_pool` of `avail`; `auto` is
    OFFLOAD_CONTEXT_AUTO, rounded down to what leaves `min_pool`, at least CONTEXT_AUTO_MIN."""
    if limit is None or limit == CONTEXT_AUTO:
        fit = max(0, (avail - min_pool) // bytes_per_token)
        n = min(OFFLOAD_CONTEXT_AUTO, fit // CONTEXT_AUTO_QUANTUM * CONTEXT_AUTO_QUANTUM)
        if n < CONTEXT_AUTO_MIN:
            need = floor + CONTEXT_AUTO_MIN * bytes_per_token
            raise StartupError(
                f"a memory budget of {budget / GIB:.2f} GiB is too small for Flint: offload needs the non-expert "
                f"weights, one layer's 256 experts, {CONTEXT_AUTO_MIN:,} tokens of history and the headroom; "
                f"declare at least {_ceil_gib(need)} GiB"
            )
        return n, "auto"
    if isinstance(limit, str):
        raise StartupError(
            f"context limit must be {CONTEXT_AUTO!r} or an integer in 1..{CONTEXT_LIMIT_MAX}, got {limit!r}"
        )
    n = check_context_limit(limit)
    if avail - n * bytes_per_token < min_pool:
        need = floor + n * bytes_per_token
        raise StartupError(
            f"context limit {n:,} does not fit a memory budget of {budget / GIB:.2f} GiB in offload: its history "
            f"({n * bytes_per_token / GIB:.2f} GiB) leaves less than one layer's 256 experts; declare at least "
            f"{_ceil_gib(need)} GiB, or a smaller --context-limit"
        )
    return n, "explicit"


def _ceil_gib(nbytes: int) -> str:
    return f"{-(-int(nbytes) * 100 // GIB) / 100:.2f}"


def plan_offload(
    budget: int,
    budget_source: str,
    limit,
    *,
    weights_bytes: int,
    dense_bytes: int,
    record_bytes: int,
    total_records: int,
    min_slots: int,
    max_slots: int,
    bytes_per_token: int,
    wired_bytes: int,
) -> MemoryPlan:
    """The offload plan for `budget` (capped at the GPU wired limit, which must hold everything the GPU reads); the
    pool holds at most `max_slots` (one pool array is slots x the largest per-expert field, below the GPU's largest
    buffer) and at most every record."""
    cap = min(int(budget), int(wired_bytes))
    min_pool = min_slots * record_bytes
    floor = dense_bytes + OFFLOAD_HEADROOM_BYTES + min_pool
    avail = cap - dense_bytes - OFFLOAD_HEADROOM_BYTES
    n, src = _offload_context(limit, avail, bytes_per_token, min_pool, cap, floor)
    hist = n * bytes_per_token
    slots = min(total_records, int(max_slots), (avail - hist) // record_bytes)
    return MemoryPlan(
        "offload",
        cap,
        budget_source,
        n,
        src,
        weights_bytes,
        dense_bytes,
        hist,
        OFFLOAD_HEADROOM_BYTES,
        record_bytes,
        total_records,
        int(slots),
    )


def plan_memory(
    budget: int | None,
    limit,
    *,
    weights_bytes: int,
    dense_bytes: int,
    record_bytes: int,
    total_records: int,
    min_slots: int,
    max_slots: int,
    bytes_per_token: int,
    wired_bytes: int,
    memory_size: int,
    profile: str = "main",
    offload_ok: bool = True,
    resident_ok: bool = True,
    mode: str = "auto",
) -> MemoryPlan:
    """Resident when the budget (or, without one, the GPU wired limit) holds the whole model and the history;
    otherwise offload, when `offload_ok` (profile main on the graph backend), else the resident refusal.
    resident_ok=False (a weight bank larger than the GPU's largest buffer) plans offload directly. mode "resident"
    refuses where "auto" would offload; mode "offload" streams the experts even when the whole model fits."""
    if mode not in MEMORY_MODES:
        raise StartupError(f"unknown memory mode {mode!r}; one of {', '.join(MEMORY_MODES)}")
    if mode == "offload" and not offload_ok:
        raise StartupError(
            "memory mode 'offload' streams the experts with stock MLX ops: it needs profile 'main', decode "
            "backend 'auto' or 'graph' and no pinned grid"
        )
    offload_ok = offload_ok and mode != "resident"
    cap = int(wired_bytes) if budget is None else min(int(budget), int(wired_bytes))
    resident_err = None
    if mode != "offload":
        try:
            if not resident_ok:
                raise StartupError(
                    "a routed-expert weight bank is larger than this GPU's largest buffer"
                )
            n, src = resolve_context_limit(
                limit,
                weights_bytes=weights_bytes,
                bytes_per_token=bytes_per_token,
                wired_bytes=cap,
                memory_size=memory_size,
                profile=profile,
                budget=cap < int(wired_bytes),
            )
        except StartupError as exc:
            if not offload_ok:
                raise
            resident_err = exc
        else:
            return MemoryPlan(
                "resident",
                None if budget is None else cap,
                "none" if budget is None else "flag",
                n,
                src,
                weights_bytes,
                dense_bytes,
                n * bytes_per_token,
                HEADROOM_BYTES,
                record_bytes,
                total_records,
                0,
            )
    source = "flag"
    if budget is None:
        budget, source = default_offload_budget(wired_bytes, memory_size), "default"
    try:
        return plan_offload(
            budget,
            source,
            limit,
            weights_bytes=weights_bytes,
            dense_bytes=dense_bytes,
            record_bytes=record_bytes,
            total_records=total_records,
            min_slots=min_slots,
            max_slots=max_slots,
            bytes_per_token=bytes_per_token,
            wired_bytes=wired_bytes,
        )
    except StartupError as exc:
        if resident_err is None:
            raise
        raise StartupError(f"{exc} (the whole model does not fit either: {resident_err})") from None
