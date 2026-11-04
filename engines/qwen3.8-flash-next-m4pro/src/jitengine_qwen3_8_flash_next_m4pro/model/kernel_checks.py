"""The load-time kernel checks (`kernels/selfcheck.py`) at the geometry the model serves.

`run_kernel_checks` runs the check next to each custom kernel this model would run, with the
config's shapes and quantization and the model's launch constants; a kernel the config gives
nothing to run is not checked. The engine runs it before the memory plan, which prices the dense
and routed-expert kernels, and builds the model with every kernel whose check failed switched off.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_decode as gdn_fused
from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_recurrence as gdn_kernel
from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as moe_kernel
from jitengine_qwen3_8_flash_next_m4pro.kernels import hc_decode as hc_kernel
from jitengine_qwen3_8_flash_next_m4pro.kernels import (
    indexer_prefill,
    sdpa_decode_masked,
    selfcheck,
)
from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as dense_kernel
from jitengine_qwen3_8_flash_next_m4pro.kernels import qsa_decode_mask as attn_fused
from jitengine_qwen3_8_flash_next_m4pro.kernels import qsa_indexer_decode as indexer_fused
from jitengine_qwen3_8_flash_next_m4pro.kernels import router_combine as moe_fused
from jitengine_qwen3_8_flash_next_m4pro.model import gdn
from jitengine_qwen3_8_flash_next_m4pro.model.attention import (
    INPUT_PROJS_CFG,
    MASKED_SDPA_CHUNK,
    MASKED_SDPA_HEADS_PER_SIMDGROUP,
)
from jitengine_qwen3_8_flash_next_m4pro.model.common import GDN_FUSED_MULTI
from jitengine_qwen3_8_flash_next_m4pro.model.hyper_connections import (
    HC_FUSED_DOWN,
    HC_FUSED_PRE_NT,
    HC_FUSED_ROWS,
    HC_FUSED_UP,
)

# Every kernel with a check, by the name its module is imported under above: the keys
# `kernel_checks` can hold, which `/stats`, the log and the e2e test use. The last four are the
# entry points of the MTP verify pass (`gdn_decode.qmv_multi_v`,
# `grouped_expert_qmv.grouped_qmv_u2`, `sdpa_decode_masked.decode_masked2`,
# `gdn_recurrence.gdn_recurrence_pair`), which `run_kernel_checks` runs only with an MTP head (the
# last only under `gdn.GDN_LAZY`).
NAMES = (
    "dense_kernel",
    "moe_kernel",
    "moe_fused",
    "hc_kernel",
    "gdn_fused",
    "gdn_kernel",
    "sdpa_decode_masked",
    "attn_fused",
    "indexer_fused",
    "indexer_prefill",
    "qmv_multi_v",
    "grouped_qmv_u2",
    "sdpa_decode_masked2",
    "gdn_pair",
)


class LinearShape(NamedTuple):
    """The facts about a `QLinear` the kernels' `supports` read, and its metadata dtype."""

    out_dims: int
    in_dims: int
    bits: int
    group_size: int
    meta: object = mx.bfloat16
    biases: bool = True

    @property
    def spec(self) -> tuple:
        """(out, in, bits, group_size, metadata dtype), a weight set as `check_glue` takes it."""
        return self.out_dims, self.in_dims, self.bits, self.group_size, self.meta


def run_kernel_checks(
    cfg, act_dtype, dense: str, moe: str, moe_variant: str, mtp: bool = False
) -> selfcheck.Checks:
    """Check each kernel the model would run at `act_dtype` under the engine's `dense` and `moe`
    kernel options; the checks' buffers are returned to the system before the model loads.

    `mtp` says the MTP head is present (`mtp.present(cfg)`), so the lone greedy row may run the
    verify pass, which launches kernels the solo path does not, `qmv_multi_v`, `grouped_qmv_u2`,
    `decode_masked2` and, under `gdn.GDN_LAZY`, `gdn_recurrence_pair`. Each is checked against
    the solo kernel it must reproduce row by row, wherever the model's geometry gives it something
    to run, whatever the options and the solo kernels' results are: the verify pass launches them
    either way."""
    checks = selfcheck.Checks()
    shapes = dense_shapes(cfg)
    if dense == "rb" and dense_kernel.served(shapes, act_dtype):
        checks.run("dense_kernel", dense_kernel.check_dense, shapes, act_dtype)
    served = dense == "rb" and checks.results.get("dense_kernel", True)
    lin = _Linears(cfg, act_dtype, served)
    _check_moe(checks, cfg, act_dtype, moe, moe_variant, mtp)
    _check_hc(checks, cfg, act_dtype, lin)
    calls, glue = _glue_calls(cfg, lin)
    if calls:
        checks.run("gdn_fused", gdn_fused.check_glue, calls, act_dtype, glue)
        if mtp:
            checks.run("qmv_multi_v", gdn_fused.check_multi_v, calls, act_dtype)
    if (
        any(not cfg.is_attention(i) for i in range(cfg.num_hidden_layers))
        and gdn.GDN_IMPL == "kernel"
    ):
        dims = (cfg.linear_num_key_heads, cfg.linear_num_value_heads)
        dims += (cfg.linear_key_head_dim, cfg.linear_value_head_dim)
        checks.run("gdn_kernel", gdn_kernel.check_recurrence, *dims, act_dtype, gdn.GDN_STAGED_TB)
        if mtp and gdn.GDN_LAZY:
            checks.run("gdn_pair", gdn_kernel.check_pair, *dims, act_dtype)
    if any(cfg.is_attention(i) for i in range(cfg.num_hidden_layers)):
        _check_attention(checks, cfg, act_dtype, mtp)
    mx.clear_cache()
    return checks


def dense_shapes(cfg) -> list:
    """(out, in, bits, group_size) of each dense linear `Qwen4ExpForCausalLM.dense_linears` holds,
    with the quantization the config gives its module."""
    h, hc = cfg.hidden_size, cfg.hc_count
    mix = {
        "input_mix_weight_down": (cfg.hc_lowrank, hc * h),
        "input_mix_weight_up": (hc * h, cfg.hc_lowrank),
    }
    mods = {"lm_head": (cfg.vocab_size, h)}
    mods.update({f"model.hyper_connection_mixer.{n}": s for n, s in mix.items()})
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        for site in ("attn_hyper_connection.", "mlp_hyper_connection."):
            mods.update({p + site + n: s for n, s in mix.items()})
            mods[p + site + "block_inject_weight"] = (hc, hc * h)
        mods.update({p + n: s for n, s in _block_shapes(cfg, cfg.is_attention(i)).items()})
        if cfg.has_ple(i):
            mods[p + "ple.key_proj"] = (hc * h, cfg.ple_embed_dim)
            mods[p + "ple.value_proj"] = (h, cfg.ple_embed_dim)
    return [(n, k, *cfg.quant_of(m)) for m, (n, k) in mods.items()]


def _block_shapes(cfg, attention: bool) -> dict:
    """{module under the layer: (out, in)} of a layer's mixer block and MoE dense linears."""
    h, inter = cfg.hidden_size, cfg.shared_expert_intermediate_size
    out = {
        "mlp.gate": (cfg.num_experts, h),
        "mlp.shared_expert_gate": (1, h),
        "mlp.shared_expert.gate_proj": (inter, h),
        "mlp.shared_expert.up_proj": (inter, h),
        "mlp.shared_expert.down_proj": (h, inter),
    }
    if attention:
        heads, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
        idx = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim
        out |= {
            "self_attn.q_proj": (2 * heads, h),
            "self_attn.k_proj": (kv, h),
            "self_attn.v_proj": (kv, h),
            "self_attn.o_proj": (h, heads),
            "self_attn.indexer.index_qk_proj": (idx, h),
        }
    else:
        hv = cfg.linear_num_value_heads
        out |= {
            "linear_attn.in_proj_qkv": (cfg.conv_dim, h),
            "linear_attn.in_proj_z": (cfg.value_dim, h),
            "linear_attn.in_proj_b": (hv, h),
            "linear_attn.in_proj_a": (hv, h),
            "linear_attn.out_proj": (h, cfg.value_dim),
        }
    return out


class _Linears:
    """`LinearShape`s of the first layer of each kind, with the metadata dtype the dense policy leaves
    them in (`linear.apply_dense_policy`): the checkpoint's bfloat16 where the row-blocked
    kernel serves the shape, else float32 under float32 activations."""

    def __init__(self, cfg, act_dtype, served: bool):
        self.cfg, self.act_dtype, self.served = cfg, act_dtype, served
        kinds = [cfg.is_attention(i) for i in range(cfg.num_hidden_layers)]
        self.attention = f"model.layers.{kinds.index(True)}." if True in kinds else None
        self.gdn = f"model.layers.{kinds.index(False)}." if False in kinds else None

    def __call__(self, module: str, out_dims: int, in_dims: int) -> LinearShape:
        bits, gs = self.cfg.quant_of(module)
        shape = (out_dims, in_dims, bits, gs)
        if self.served and dense_kernel.served([shape], self.act_dtype):
            meta = mx.bfloat16
        else:
            meta = mx.float32 if self.act_dtype == mx.float32 else mx.bfloat16
        return LinearShape(*shape, meta)


def _check_moe(checks, cfg, act_dtype, moe: str, variant: str, mtp: bool) -> None:
    h, inter, k = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok
    bits, gs = cfg.quant_of("model.layers.0.mlp.switch_mlp.gate_proj")
    grouped = all(moe_kernel.supports(bits, gs, kk, act_dtype=act_dtype) for kk in (h, inter))
    if moe == "grouped" and grouped:
        checks.run(
            "moe_kernel", moe_kernel.check_grouped, h, inter, bits, gs, act_dtype, k, variant
        )
    if mtp and grouped:
        checks.run("grouped_qmv_u2", moe_kernel.check_grouped_u2, h, inter, bits, gs, act_dtype, k)
    if moe_fused.supports_topk(cfg.num_experts, k):
        args = (cfg.num_experts, k, cfg.norm_topk_prob, cfg.hc_count, h, act_dtype)
        checks.run("moe_fused", moe_fused.check_router_combine, *args)


def _check_hc(checks, cfg, act_dtype, lin: _Linears) -> None:
    hc, h, rank = cfg.hc_count, cfg.hidden_size, cfg.hc_lowrank
    p = "model.layers.0.attn_hyper_connection."
    lins = (
        lin(p + "input_mix_weight_down", rank, hc * h),
        lin(p + "input_mix_weight_up", hc * h, rank),
        lin(p + "block_inject_weight", hc, hc * h),
    )
    bits, gs = lins[0].bits, lins[0].group_size
    if hc_kernel.supports(hc, h, rank, gs, bits, *lins):
        metas = [m.meta for m in lins]
        launch = (HC_FUSED_DOWN, HC_FUSED_UP, HC_FUSED_PRE_NT)
        args = (hc, h, rank, bits, gs, cfg.rms_norm_eps, act_dtype, metas, launch, HC_FUSED_ROWS)
        checks.run("hc_kernel", hc_kernel.check_hc, *args)


def _glue_calls(cfg, lin: _Linears) -> tuple:
    """(the `qmv_multi` calls the model makes, gdn_pre/gdn_post's arguments or None)."""
    calls, glue, h = [], None, cfg.hidden_size
    glue = _gdn_glue(cfg, lin, h, calls, glue)
    _attention_calls(cfg, lin, h, calls)
    _shared_expert_calls(cfg, lin, h, calls)
    return calls, glue


def _gdn_glue(cfg, lin: _Linears, h: int, calls: list, glue) -> tuple | None:
    """Append the GatedDeltaNet projections' qmv_multi call to calls; returns gdn_pre/gdn_post's arguments."""
    if lin.gdn is not None and gdn_fused.supports_gdn(cfg):
        p = lin.gdn + "linear_attn."
        hv = cfg.linear_num_value_heads
        sets = [
            lin(p + "in_proj_qkv", cfg.conv_dim, h),
            lin(p + "in_proj_z", cfg.value_dim, h),
            lin(p + "in_proj_b", hv, h),
            lin(p + "in_proj_a", hv, h),
        ]
        if gdn_fused.supports_multi(sets):
            calls.append(([s.spec for s in sets], GDN_FUSED_MULTI, False))
            dk = cfg.linear_key_head_dim  # Qwen4ExpTextGatedDeltaNet's l2 eps and q, k scales
            glue = (cfg, 1e-6 / dk, 1.0 / dk, dk**-0.5, gdn.GDN_FUSED_NSG)
    return glue


def _attention_calls(cfg, lin: _Linears, h: int, calls: list) -> None:
    """Append the attention projections' qmv_multi call to calls when the kernel supports them."""
    if lin.attention is not None:
        p = lin.attention + "self_attn."
        heads, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
        idx = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim
        sets = [
            lin(p + "q_proj", 2 * heads, h),
            lin(p + "k_proj", kv, h),
            lin(p + "v_proj", kv, h),
            lin(p + "indexer.index_qk_proj", idx, h),
        ]
        if gdn_fused.supports_multi(sets):
            calls.append(([s.spec for s in sets], INPUT_PROJS_CFG, False))


def _shared_expert_calls(cfg, lin: _Linears, h: int, calls: list) -> None:
    """Append the shared expert's gate/up and down qmv_multi calls when the kernel supports them."""
    if moe_fused.supports_topk(cfg.num_experts, cfg.num_experts_per_tok):
        p, inter = "model.layers.0.mlp.shared_expert.", cfg.shared_expert_intermediate_size
        gate_up = [lin(p + "gate_proj", inter, h), lin(p + "up_proj", inter, h)]
        down = [lin(p + "down_proj", h, inter)]
        if gdn_fused.supports_multi(gate_up) and gdn_fused.supports_multi(down):
            calls.append(([s.spec for s in gate_up], GDN_FUSED_MULTI, False))
            calls.append(([s.spec for s in down], GDN_FUSED_MULTI, True))


def _check_attention(checks, cfg, act_dtype, mtp: bool) -> None:
    cr = cfg.indexer_compress_ratio
    topk = cfg.indexer_budget // cr
    heads = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    args = (*heads, act_dtype, MASKED_SDPA_CHUNK, MASKED_SDPA_HEADS_PER_SIMDGROUP, cr, topk)
    checks.run("sdpa_decode_masked", sdpa_decode_masked.check_decode_masked, *args)
    if mtp:
        checks.run("sdpa_decode_masked2", sdpa_decode_masked.check_decode_masked2, *args)
    checks.run("attn_fused", attn_fused.check_qsa_mask, cr, topk)
    nh, hd, rope = cfg.indexer_n_heads, cfg.indexer_head_dim, cfg.rope_dims
    ix = (nh, hd, rope, cr, topk, cfg.rms_norm_eps, cfg.rope_theta, act_dtype)
    if indexer_fused.supports(nh, hd, rope, topk + 1):
        checks.run("indexer_fused", indexer_fused.check_indexer, *ix)
    if indexer_prefill.supports(nh, hd, rope, topk + 1):
        checks.run("indexer_prefill", indexer_prefill.check_prefill_mask, *ix)
