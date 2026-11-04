"""The load-time kernel checks pass on this device."""

from __future__ import annotations

# (heads, KV heads, head dim): Qwen3.6-35B-A3B's attention, and the tiny test model's
ATTENTION = ((16, 2, 256), (2, 1, 64))


def test_kernel_selfchecks_pass_on_this_device(requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_solo import check_experts
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import check_dense
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import check_topk
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import check_sdpa

    assert check_dense(3, 64)
    assert all(check_sdpa(*geometry) for geometry in ATTENTION)
    assert check_topk()
    assert all(check_experts(512, 2048, 3, 64).values())
    assert check_experts(512, 2048, 4, 64) == {"exact_experts": True}


def test_sdpa_rows_and_gdn_rows_match_single_row(requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import check_gdn_rows
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import check_sdpa_rows

    assert all(check_sdpa_rows(*geometry) for geometry in ATTENTION)
    assert check_gdn_rows()
