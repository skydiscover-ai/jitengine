"""The memory plan (budget.py) on Flint's real geometry (tests/data headers): resident exactly when the whole model
fits, offload below it, refusals with what would fit."""

import pytest

from jitengine_flint_m2.budget import (
    OFFLOAD_CONTEXT_AUTO,
    OFFLOAD_HEADROOM_BYTES,
    default_offload_budget,
    flag_budget,
    plan_memory,
)
from jitengine_flint_m2.model.config import (
    HEADROOM_BYTES,
    StartupError,
    load_config,
    resolve_context_limit,
    tensor_index,
)
from jitengine_flint_m2.weight_banks import BankPlan

GIB = 2**30


@pytest.fixture
def geometry(flint_headers):
    cfg = load_config(flint_headers)
    bp = BankPlan(cfg, tensor_index(flint_headers, cfg))
    return dict(
        weights_bytes=bp.total_bytes,
        dense_bytes=bp.bank_sizes[0],
        record_bytes=bp.expert_stride,
        total_records=len(cfg.moe_layers) * cfg.n_experts,
        min_slots=cfg.n_experts,
        max_slots=10**9,
        bytes_per_token=cfg.history_bytes_per_token_bits(16),
    )


def plan(geometry, budget, limit="auto", wired=40 * GIB, memory=48 * GIB, **kw):
    return plan_memory(budget, limit, wired_bytes=wired, memory_size=memory, **{**geometry, **kw})


def test_resident_without_a_budget_is_the_old_rule(geometry):
    """No budget and a wired limit that holds the model: resident, the context resolve_context_limit gives."""
    for wired in (40 * GIB, 37 * GIB, 26 * GIB):
        p = plan(geometry, None, wired=wired)
        want = resolve_context_limit(
            "auto",
            weights_bytes=geometry["weights_bytes"],
            bytes_per_token=geometry["bytes_per_token"],
            wired_bytes=wired,
            memory_size=48 * GIB,
        )
        assert (p.mode, p.context_limit, p.context_limit_source) == ("resident", *want)
        assert p.budget_bytes is None and p.pool_slots == 0 and p.expert_fraction == 1.0
        assert p.planned_bytes == geometry["weights_bytes"] + p.history_bytes + HEADROOM_BYTES


def test_offload_when_the_model_does_not_fit(geometry):
    """A 24 GB M2 (wired limit 22 GiB): offload in the default budget, physical memory less 6 GiB, with 32,768
    tokens of history; the plan fills the budget and every bucket is accounted for."""
    p = plan(geometry, None, wired=22 * GIB, memory=24 * GIB)
    assert p.mode == "offload" and p.budget_source == "default" and p.budget_bytes == 18 * GIB
    assert (p.context_limit, p.context_limit_source) == (OFFLOAD_CONTEXT_AUTO, "auto")
    used = geometry["dense_bytes"] + p.history_bytes + OFFLOAD_HEADROOM_BYTES
    assert p.pool_slots == (18 * GIB - used) // geometry["record_bytes"]
    assert 18 * GIB - geometry["record_bytes"] < p.planned_bytes <= 18 * GIB
    assert 0 < p.expert_fraction < 1 and "offload" in p.line() and f"{p.pool_slots:,}" in p.line()
    assert default_offload_budget(22 * GIB, 24 * GIB) == 18 * GIB
    assert default_offload_budget(16 * GIB, 24 * GIB) == 16 * GIB  # the wired limit is the cap


def test_a_budget_chooses_the_mode(geometry):
    """On the M4 Pro: a budget that holds the model stays resident (its context sized to the budget); a smaller one
    offloads with the pool shrinking as the budget does; the pool never exceeds the records or max_slots."""
    p = plan(geometry, 30 * GIB)
    assert p.mode == "resident" and p.budget_bytes == 30 * GIB and p.budget_source == "flag"
    assert p.planned_bytes <= 30 * GIB < p.planned_bytes + 1024 * geometry["bytes_per_token"]
    slots = [plan(geometry, b * GIB).pool_slots for b in (20, 16, 12, 8)]
    assert slots == sorted(slots, reverse=True) and all(s > 0 for s in slots)
    assert plan(geometry, 20 * GIB, max_slots=1000).pool_slots == 1000
    # the wired limit caps the budget, and the pool caps at every record
    p = plan(geometry, 64 * GIB, wired=24 * GIB, memory=64 * GIB, limit=4096)
    assert p.mode == "offload" and p.budget_bytes == 24 * GIB
    assert p.pool_slots == geometry["total_records"]


def test_offload_context(geometry):
    """An explicit limit is kept when one layer of experts still fits beside it, else refused with the budget that
    would hold it; `auto` gives less than 32,768 tokens only when the budget needs it."""
    p = plan(geometry, 12 * GIB, limit=131072)
    assert (p.mode, p.context_limit, p.context_limit_source) == ("offload", 131072, "explicit")
    with pytest.raises(StartupError) as err:
        plan(geometry, 6 * GIB, limit=262144)
    msg = str(err.value)
    assert "context limit 262,144 does not fit" in msg and "declare at least" in msg, msg
    small = geometry["dense_bytes"] + OFFLOAD_HEADROOM_BYTES + 256 * geometry["record_bytes"]
    p = plan(geometry, small + 8192 * geometry["bytes_per_token"])
    assert p.context_limit == 8192 and p.pool_slots == 256


def test_refusals(geometry):
    """Below one layer of experts and 4,096 tokens: refused with the smallest budget that works. kv8 or the megakernel
    (offload_ok False) keep the resident refusal; a bank above the largest buffer plans offload directly."""
    with pytest.raises(StartupError) as err:
        plan(geometry, 2 * GIB)
    msg = str(err.value)
    assert "too small for Flint" in msg and "declare at least" in msg, msg
    need = float(msg.split("declare at least ")[1].split(" GiB")[0])
    assert plan(geometry, int(need * GIB)).mode == "offload"
    with pytest.raises(StartupError) as err:
        plan(geometry, None, wired=22 * GIB, offload_ok=False)
    msg = str(err.value)
    assert "context limit auto: Flint needs" in msg, msg
    assert plan(geometry, None, resident_ok=False).mode == "offload"
    assert (
        flag_budget(None, None) is None and flag_budget(3, None) == 3 and flag_budget(None, 5) == 5
    )
    with pytest.raises(ValueError, match="once"):
        flag_budget(1, 2)


def test_mode(geometry):
    """--mode: resident refuses where auto offloads; offload streams the experts even when the whole model fits, and
    needs offload_ok; an unknown mode is refused."""
    assert plan(geometry, None, mode="resident").mode == "resident"
    with pytest.raises(StartupError, match="context limit auto: Flint needs"):
        plan(geometry, None, wired=22 * GIB, mode="resident")
    with pytest.raises(StartupError, match="larger than this GPU's largest buffer"):
        plan(geometry, None, resident_ok=False, mode="resident")
    p = plan(geometry, None, mode="offload")
    assert (p.mode, p.budget_source, p.context_limit) == (
        "offload",
        "default",
        OFFLOAD_CONTEXT_AUTO,
    )
    p = plan(geometry, 30 * GIB, mode="offload")
    assert (p.mode, p.budget_bytes) == ("offload", 30 * GIB) and p.planned_bytes <= 30 * GIB
    with pytest.raises(StartupError, match="memory mode 'offload' streams the experts"):
        plan(geometry, None, offload_ok=False, mode="offload")
    with pytest.raises(StartupError) as err:
        plan(geometry, 2 * GIB, mode="offload")  # no resident plan was tried, so none is quoted
    assert "too small for Flint" in str(err.value) and "does not fit either" not in str(err.value)
    with pytest.raises(StartupError, match="unknown memory mode 'swap'"):
        plan(geometry, None, mode="swap")


def test_a_budget_refusal_names_the_budget(geometry):
    """When the budget, not the GPU wired limit, is what cannot hold the whole model, the refusal says so and asks
    for a larger budget, not a sysctl that would change nothing."""
    with pytest.raises(StartupError) as err:
        plan(geometry, 12 * GIB, offload_ok=False)
    msg = str(err.value)
    assert (
        "the memory budget is 12.00 GiB" in msg and "raise --memory-budget-gib to at least" in msg
    ), msg
    assert "sysctl" not in msg, msg
    with pytest.raises(StartupError) as err:
        plan(geometry, 2 * GIB)
    msg = str(err.value)
    assert (
        "declare at least" in msg and "the memory budget is 2.00 GiB" in msg and "sysctl" not in msg
    ), msg
    with pytest.raises(StartupError, match=r"sudo sysctl iogpu\.wired_limit_mb="):
        plan(geometry, 30 * GIB, wired=22 * GIB, offload_ok=False)  # the wired limit is the cap
