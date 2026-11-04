"""The memory planner, replayed on the recorded 16 GB plan's geometry (no MLX needed)."""

import re
from decimal import Decimal
from types import SimpleNamespace

import pytest


@pytest.fixture
def budget(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro import budget as m

    # Qwen3.8-Flash-Next-4bit, float32, as the recorded plan states it
    monkeypatch.setattr(m, "dense_fp32_meta_bytes", lambda *a: 41745920)
    monkeypatch.setattr(m, "gdn_state_bytes", lambda *a: 118038544)
    monkeypatch.setattr(m, "ple_state_bytes", lambda *a: 0)
    monkeypatch.setattr(m, "kv_bytes_per_position", lambda *a: 55296)
    monkeypatch.setattr(m, "wave_geometry", lambda *a: 32768)
    monkeypatch.setattr(m, "moe_kernel_for", lambda *a: "grouped")
    return m


GB = 10**9


def _plan(m, nbytes):
    scan = SimpleNamespace(
        dense_bytes=3123406360,
        record_bytes=3072000,
        num_layers=48,
        num_records=24576,
        field_bytes=[0] * 9,
        field_dtype=["U32"] * 9,
    )
    cfg = SimpleNamespace(num_experts_per_tok=10, vocab_size=248320)
    return m.plan_budget(nbytes, scan, cfg, {"compute_dtype": "float32", "wave_tokens": 8192})


def test_the_recorded_16_gb_plan(budget):
    p = _plan(budget, 16 * GB)
    assert (p.pool_slots, p.wave_tokens, p.scratch_bytes) == (1519, 8192, 4035144000)


def test_a_budget_near_the_floor_steps_the_wave_down(budget):
    # the 8192-token wave leaves 475 records here, 5 short of one decode step; a smaller one fits
    p = _plan(budget, 12.66 * GB)
    assert p.wave_tokens < 8192 and p.pool_slots >= p.experts_per_step


def test_the_smallest_budget_a_refusal_names_fits(budget):
    with pytest.raises(budget.BudgetError, match="Declare at least") as exc:
        _plan(budget, 8 * GB)
    gib = re.search(r"Declare at least ([0-9.]+) GiB", str(exc.value)).group(1)
    p = _plan(budget, int(Decimal(gib) * budget.GIB))
    assert p.pool_slots >= p.experts_per_step


def test_a_budget_declared_in_decimal_gb_converts_to_the_byte(budget):
    assert budget.declared_budget_bytes({}) == 40 * GB  # the measured operating point
    assert budget.declared_budget_bytes({"memory_budget_gb": 12.66}) == 12_660_000_000
    assert budget.declared_budget_bytes({"memory_budget_bytes": 37 * budget.GIB}) == 37 * 2**30
    with pytest.raises(ValueError, match="not both"):
        budget.declared_budget_bytes({"memory_budget_bytes": 1, "memory_budget_gb": 1})


def test_one_row_decode_lends_the_store_cap_and_the_plan_declares_it(budget, monkeypatch):
    """The annex: the store's cap in whole slots, declared as borrowed from the store's bucket;
    `store_spill` off (or no store, or bf16 slots) is the plan without it, bucket for bucket."""
    scan = SimpleNamespace(
        dense_bytes=3123406360,
        record_bytes=3072000,
        num_layers=48,
        num_records=24576,
        field_bytes=[0] * 9,
        field_dtype=["U32"] * 9,
    )
    cfg = SimpleNamespace(num_experts_per_tok=10, vocab_size=248320)
    opts = {"compute_dtype": "float32", "wave_tokens": 8192}
    bf16 = budget.plan_budget(40 * GB, scan, cfg, dict(opts))  # no field table: bf16 slots
    assert bf16.pool_codec != "pack16" and bf16.store_spill_slots == 0
    assert bf16.as_dict()["borrowed"] == []
    # compact slots, as the checkpoint's census gives them (2,764,800 B per slot)
    monkeypatch.setattr(budget, "_pool_codec", lambda *a: ("pack16", "test", 2764800, 0))
    on = budget.plan_budget(40 * GB, scan, cfg, dict(opts, store_spill="on"))
    off = budget.plan_budget(40 * GB, scan, cfg, dict(opts, store_spill="off"))
    assert (
        budget.plan_budget(40 * GB, scan, cfg, dict(opts)).store_spill_slots == on.store_spill_slots
    )  # the default
    none = budget.plan_budget(
        40 * GB, scan, cfg, dict(opts, store_spill="on", prefix_store_entries=0)
    )
    assert on.store_spill_slots == on.prefix_store_bytes // on.slot_bytes > 0
    (b,) = on.as_dict()["borrowed"]
    assert b["bytes"] == on.store_spill_slots * on.slot_bytes <= on.prefix_store_bytes
    assert b["sources"] == {"prefix store cap": b["bytes"]}
    assert [n for n, _ in on.buckets() if n.startswith("prefix store cap")]  # the named source
    assert off.store_spill_slots == 0 and off.as_dict()["borrowed"] == []
    assert none.store_spill_slots == 0
    assert (on.pool_slots, on.planned_bytes) == (off.pool_slots, off.planned_bytes)
    with pytest.raises(ValueError, match="store_spill"):
        budget.plan_budget(40 * GB, scan, cfg, dict(opts, store_spill="maybe"))
