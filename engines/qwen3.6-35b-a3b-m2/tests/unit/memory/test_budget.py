"""The integer-byte memory plan. Pure: no MLX."""

from __future__ import annotations

from jitengine_qwen3_6_35b_a3b_m2.memory.budget import (
    GIB,
    MIB,
    BudgetError,
    Capacity,
    Limits,
    make_plan,
    parse_bytes,
)


def _raises(fn, needle):
    try:
        fn()
    except BudgetError as exc:
        assert needle in str(exc), f"{needle!r} not in {exc!r}"
        return
    raise AssertionError("no BudgetError")


def test_auto_keeps_all_weights_when_complete_plan_fits():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    p = make_plan(c, Limits(700, 1000, 300, 650))
    assert p.mode == "resident"
    assert p.planned_process_bytes == 650
    assert p.planned_mlx_bytes <= 650


def test_small_budget_uses_pool_and_prices_every_bucket():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    p = make_plan(c, Limits(400, 1000, 300, 650))
    assert p.mode == "offload"
    assert p.pool_slots == 15
    assert p.planned_process_bytes == 400


def test_exact_fit_boundary_is_resident_one_byte_less_is_offload():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    assert make_plan(c, Limits(650, 10**6, 0, 10**6)).mode == "resident"
    p = make_plan(c, Limits(649, 10**6, 0, 10**6))
    assert p.mode == "offload" and p.pool_slots == 39


def test_physical_minus_reserve_caps_the_request():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    p = make_plan(c, Limits(10**6, 1000, 600, 10**6))  # 400 B effective
    assert p.process_cap_bytes == 400 and p.mode == "offload" and p.pool_slots == 15


def test_metal_cap_binds_separately():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    p = make_plan(c, Limits(10**6, 10**6, 0, 300))
    assert p.mlx_cap_bytes == 300 and p.pool_slots == 9
    assert p.planned_mlx_bytes <= 300


def test_explicit_resident_refuses_unfit():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    _raises(lambda: make_plan(c, Limits(400, 1000, 300, 650), "resident"), "resident mode needs")


def test_explicit_offload_keeps_a_pool_even_when_everything_fits():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    p = make_plan(c, Limits(10**6, 10**6, 0, 10**6), "offload")
    assert p.mode == "offload" and p.pool_slots == 40


def test_below_minimum_slots_refused_with_the_needed_ceiling():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    _raises(lambda: make_plan(c, Limits(280, 1000, 0, 650)), "minimum process ceiling")


def test_fixed_floor_above_ceiling_refused():
    c = Capacity(100, 400, 10, 40, 4, 20, 30, 10, 10, 40, 20, 20)
    _raises(lambda: make_plan(c, Limits(200, 1000, 0, 650)), "non-expert floor")


def test_non_integer_inputs_rejected():
    _raises(lambda: Capacity(1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), "integer")
    _raises(lambda: Limits(-1, 1, 1, 1), ">= 0")
    _raises(lambda: make_plan(Capacity(*[1] * 12), Limits(10, 10, 0, 10), "sometimes"), "mode")


def test_units_are_distinct():
    assert parse_bytes("18GiB") == 18 * GIB
    assert parse_bytes("18GB") == 18 * 10**9
    assert parse_bytes("512MiB") == 512 * MIB
    assert parse_bytes("19327352832") == 18 * GIB
    assert parse_bytes("1.5GiB") == 3 * GIB // 2
    for bad in ("nan", "inf", "-1GiB", "12 parsecs", ""):
        _raises(lambda b=bad: parse_bytes(b), "")


def test_real_geometry_air_profile_forces_offload_for_4bit():
    """The 4-bit 35B cannot be resident under the 18 GiB profile, so the plan is a fitting pool."""
    dense, experts, rec = 1_389_394_176, 18_119_393_280, 1_769_472
    c = Capacity(
        dense,
        experts,
        rec,
        10240,
        320,
        67_502_080,
        167_772_160,
        128 * MIB,
        128 * MIB,
        1 * GIB,
        768 * MIB,
        512 * MIB,
    )
    p = make_plan(c, Limits(18 * GIB, 24 * GIB, 6 * GIB, 23_622_320_128))
    assert p.mode == "offload"
    assert p.planned_process_bytes <= 18 * GIB
    assert 0.75 < p.expert_fraction < 0.9
    assert sum(p.buckets().values()) == p.planned_process_bytes
