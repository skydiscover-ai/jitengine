"""The load-time kernel checks run at the width of the weights the model serves (no Metal needed)."""

from __future__ import annotations

import pytest


def _checks() -> list:
    """The check functions the engine calls, which the test replaces where they are defined."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_solo import check_experts
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_exact import check_gdn_exact
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import check_gdn_rows
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import check_dense
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import check_sort
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.rms_norm_prescale import check_rms_pre
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import check_router, check_topk
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import check_sdpa, check_sdpa_rows
    from jitengine_qwen3_6_35b_a3b_m4pro.sampler import check_sampler_tail

    return [
        check_sdpa,
        check_sdpa_rows,
        check_gdn_exact,
        check_gdn_rows,
        check_dense,
        check_experts,
        check_router,
        check_topk,
        check_rms_pre,
        check_sort,
        check_sampler_tail,
    ]


def _run(monkeypatch, bits: int, group_size: int, offload: bool = False):
    """`run_load_checks` for a config of that width, every check replaced by one that records its
    arguments and passes."""
    import importlib

    from jitengine_qwen3_6_35b_a3b_m4pro.engine.engine import run_load_checks
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_solo import expert_checks
    from jitengine_qwen3_6_35b_a3b_m4pro.model.config import Qwen3_5MoeConfig

    calls = {}

    def recorder(name):
        def check(*args):
            calls[name] = args
            if name == "check_experts":
                return dict.fromkeys(expert_checks(args[2]), True)
            return True

        return check

    for check in _checks():
        monkeypatch.setattr(
            importlib.import_module(check.__module__), check.__name__, recorder(check.__name__)
        )
    cfg = Qwen3_5MoeConfig(q_bits=bits, q_group_size=group_size)
    checks, exp_ok = run_load_checks(cfg, offload)
    return cfg, checks, exp_ok, calls


@pytest.mark.parametrize("group_size", [32, 64])
def test_the_checks_get_the_served_bits_and_group_size(requires_mlx_any, monkeypatch, group_size):
    cfg, checks, exp_ok, calls = _run(monkeypatch, 3, group_size)
    assert calls["check_dense"] == (3, group_size)
    assert calls["check_experts"] == (cfg.moe_intermediate_size, cfg.hidden_size, 3, group_size)
    assert set(exp_ok) == {"exact_experts", "moe_v4", "rows_v4", "seg_prefill"}
    assert {"dense_v4", "moe_v4", "rows_v4", "seg_prefill"} <= set(checks.results)


def test_a_four_bit_model_runs_only_the_checks_that_apply_to_it(requires_mlx_any, monkeypatch):
    """The prescaled GEMV and the prescaled, rows and segmented expert kernels are 3-bit only, so a
    4-bit model neither checks nor reports them; the exact expert kernels are checked at 4 bits."""
    cfg, checks, exp_ok, calls = _run(monkeypatch, 4, 64)
    assert "check_dense" not in calls
    assert calls["check_experts"] == (cfg.moe_intermediate_size, cfg.hidden_size, 4, 64)
    assert set(exp_ok) == {"exact_experts"}
    assert not {"dense_v4", "moe_v4", "rows_v4", "seg_prefill"} & set(checks.results)
    assert all(checks.results.values())


def test_an_offloading_engine_checks_no_expert_kernel(requires_mlx_any, monkeypatch):
    _, _, exp_ok, calls = _run(monkeypatch, 4, 64, offload=True)
    assert "check_experts" not in calls and exp_ok == {}
