"""The memory planner, replayed on the recorded 16 and 18 GB plans' geometry (no MLX needed)."""

import re
import sys
import types
from decimal import Decimal
from types import SimpleNamespace

import pytest

EMBED_BYTES = 397_312_000  # the streamed embedding table: 248,320 rows x 1,600 B


@pytest.fixture
def budget(monkeypatch):
    from jitengine_qwen3_8_flash_next_m2 import budget as m

    # plan_budget imports embed_store (which needs MLX) when it runs; a stub stands in for it
    embed = types.ModuleType("jitengine_qwen3_8_flash_next_m2.model.embed_store")
    embed.embed_streamed_bytes = lambda scan: EMBED_BYTES
    monkeypatch.setitem(sys.modules, "jitengine_qwen3_8_flash_next_m2.model.embed_store", embed)
    # Qwen3.8-Flash-Next-4bit, float32, as the recorded plans state it
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


@pytest.mark.parametrize("gb, pool", [(16.0, 1702), (18.0, 2327)])
def test_the_recorded_plans(budget, gb, pool):
    p = _plan(budget, round(gb * GB))
    assert (p.pool_slots, p.wave_tokens, p.scratch_bytes) == (pool, 8192, 4035144000)


def test_a_budget_near_the_floor_steps_the_wave_down(budget):
    # the 8192-token wave leaves 476 records here, 4 short of one decode step; a smaller one fits
    p = _plan(budget, round(12.08 * GB))
    assert p.wave_tokens < 8192 and p.pool_slots >= p.experts_per_step


def test_the_smallest_budget_a_refusal_names_fits(budget):
    with pytest.raises(budget.BudgetError, match="Declare at least") as exc:
        _plan(budget, 8 * GB)
    gib = re.search(r"Declare at least ([0-9.]+) GiB", str(exc.value)).group(1)
    p = _plan(budget, int(Decimal(gib) * budget.GIB))
    assert p.pool_slots >= p.experts_per_step


def test_a_budget_declared_in_decimal_gb_converts_to_the_byte(budget):
    assert budget.declared_budget_bytes({}) == 18 * GB  # the measured operating point
    assert budget.declared_budget_bytes({"memory_budget_gb": 12.66}) == 12_660_000_000
    assert budget.declared_budget_bytes({"memory_budget_bytes": 37 * budget.GIB}) == 37 * 2**30
    with pytest.raises(ValueError, match="not both"):
        budget.declared_budget_bytes({"memory_budget_bytes": 1, "memory_budget_gb": 1})
