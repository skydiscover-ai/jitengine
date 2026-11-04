"""What the engine sets for the model: which kernel each block runs, once the load-time checks have
passed (`KernelChoice`), and how its expert work is scheduled, from the engine's options
(`Strategies`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelChoice:
    """Which kernel each block runs where a custom one replaces a stock op. All stock until the
    engine has checked the kernels on this device and passes its choice to
    `Qwen3_5MoeForCausalLM.use_kernel_choice`."""

    dense_prescaled: bool = False  # QLinear: qmv_prescaled for one-row and multi-row calls
    exact_experts: bool = False  # the MoE block: expert_solo for one resident decode row
    moe_prescaled: bool = (
        False  # the MoE block: expert_qmv_prescaled gate/up and the parallel down, solo path
    )
    rows_exact: bool = False  # the MoE block: expert_rows for a stack of decode rows
    rows_prescaled: bool = False  # the MoE block: the prescaled kernels for a stack of rows too
    exact_topk: bool = False  # the MoE block: topk_stable for the router top-k
    fused_router: bool = False  # the MoE block: router_topk, the router in one launch
    seg_prefill: bool = False  # the MoE block: gather_qmm_prefill for sorted prefill gathers
    exact_sdpa: bool = False  # the attention: sdpa_decode for single-query decode
    sdpa_rows: bool = False  # the attention: batched decode's rows in one SDPA launch
    gdn_rows: bool = False  # the GDN layer: batched decode's recurrences in one launch
    gdn_exact: bool = False  # the GDN layer: the exact pre/post kernels around the recurrence
    fused_norm: bool = False  # the decoder layer: rms_norm_pre on the solo path


@dataclass(frozen=True)
class Strategies:
    """How the model schedules its expert work where it has a choice, from the engine's options.
    Unlike `KernelChoice`, nothing here is a kernel the engine checks on the device."""

    speculative: bool = False  # the MoE block: the speculative pooled gather (offload, solo path)
    merge_prefill_experts: bool = False  # Model: one routed-expert gather over all prompt chunks
    prefill_async_depth: int = (
        4  # Model: layer-major prefill's async evals in flight (0 = blocking)
    )
    verify_relaxed: bool = (
        False  # Model: the speculative verify's dense rows take MLX's small-M kernel
    )
