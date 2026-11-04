"""The MoE block: router, routed experts (a resident stack or the slot pool's) and the sigmoid-gated
shared expert, and the routed-expert kernel options.

`Qwen4ExpTextSparseMoeBlock` keeps the router, the shared expert and `__call__`; its paths are
mixins, the package's other modules: solo decode (`decode`), the speculative gather
(`speculative`), batched waves (`waves`), prefill and batched decode (`prefill`), and the
routed-expert MLP they end in (`expert_mlp`).
"""

from __future__ import annotations

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.kernels import gdn_decode as _gf
from jitengine_qwen3_8_flash_next_m4pro.kernels import grouped_expert_qmv as _mk
from jitengine_qwen3_8_flash_next_m4pro.kernels import router_combine as _mf
from jitengine_qwen3_8_flash_next_m4pro.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig
from jitengine_qwen3_8_flash_next_m4pro.model.moe.decode import SoloDecodeMixin
from jitengine_qwen3_8_flash_next_m4pro.model.moe.expert_mlp import ExpertMLPMixin, swiglu
from jitengine_qwen3_8_flash_next_m4pro.model.moe.prefill import PrefillMixin
from jitengine_qwen3_8_flash_next_m4pro.model.moe.speculative import SpeculativeGatherMixin
from jitengine_qwen3_8_flash_next_m4pro.model.moe.waves import BatchedWavesMixin

# Routed experts run on the grouped kernel (`kernels/grouped_expert_qmv.py`) or stock `gather_qmm`, the per-shape
# fallback; variant "fused" reads gate and up in one pass with SwiGLU in fp32, "split" in two calls.
MOE_KERNELS = ("grouped", "stock")


MOE_KERNEL_DEFAULT = "grouped"


MOE_VARIANTS = ("fused", "split")


class Qwen4ExpTextSparseMoeBlock(
    SoloDecodeMixin, SpeculativeGatherMixin, BatchedWavesMixin, PrefillMixin, ExpertMLPMixin
):
    """Router, routed experts and the sigmoid-gated shared expert.

    With a `pool` (an `OffloadMoeCache`), the routed expert weights are the pool's `[slots, ...]`
    arrays, indexed by slot id rather than expert id.
    """

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        p: str,
        pool=None,
        layer_idx: int = 0,
        sync_each_layer: bool = False,
        router_dtype=None,
        moe_kernel: str = MOE_KERNEL_DEFAULT,
        moe_kernel_variant: str | None = None,
        kernels_off: frozenset = frozenset(),
    ):
        self.top_k = cfg.num_experts_per_tok
        self.moe_kernel = str(moe_kernel)
        self.moe_kernel_variant = str(moe_kernel_variant or _mk.DEFAULT_VARIANT)
        if self.moe_kernel not in MOE_KERNELS:
            raise ValueError(f"moe_kernel {self.moe_kernel!r} not in {MOE_KERNELS}")
        if self.moe_kernel_variant not in MOE_VARIANTS:
            raise ValueError(
                f"moe_kernel_variant {self.moe_kernel_variant!r} not in {MOE_VARIANTS}"
            )
        self.norm_topk = cfg.norm_topk_prob
        # diagnostic: cast the router input to this dtype (None = the activation's own); in bf16
        # the logits keep ~3 significant digits and near-tie experts at the top-k boundary flip
        self.router_dtype = router_dtype
        self.pool = pool
        self.layer_idx = int(layer_idx)
        # verification hook: evaluate each layer's gather before the next `ensure()` can evict;
        # a result that changes with it on exposes a pool race
        self.sync_each_layer = bool(sync_each_layer)
        self.gate = qlinear_from(w, p + "gate", cfg)
        self.shared_gate = qlinear_from(w, p + "shared_expert_gate", cfg)
        self.shared_gate_proj = qlinear_from(w, p + "shared_expert.gate_proj", cfg)
        self.shared_up_proj = qlinear_from(w, p + "shared_expert.up_proj", cfg)
        self.shared_down_proj = qlinear_from(w, p + "shared_expert.down_proj", cfg)
        if pool is None:
            self.gate_proj = qlinear_from(w, p + "switch_mlp.gate_proj", cfg)
            self.up_proj = qlinear_from(w, p + "switch_mlp.up_proj", cfg)
            self.down_proj = qlinear_from(w, p + "switch_mlp.down_proj", cfg)
        else:
            self.gate_proj = pool.gate_proj
            self.up_proj = pool.up_proj
            self.down_proj = pool.down_proj
        self.hc = int(cfg.hc_count)
        self.hid = int(cfg.hidden_size)
        self.fused_ok = (
            "moe_fused" not in kernels_off
            and pool is not None
            and _mf.supports_topk(int(cfg.num_experts), self.top_k)
        )
        self.spec_ok = (
            self.fused_ok and self.moe_kernel_variant == "split" and not self.sync_each_layer
        )
        # the shared expert's gate/up as one multi-weight launch and its down with silu(g)*u folded in
        self.shared_fused_ok = (
            self.fused_ok
            and "gdn_fused" not in kernels_off
            and _gf.supports_multi([self.shared_gate_proj, self.shared_up_proj])
            and _gf.supports_multi([self.shared_down_proj])
            and self.shared_gate_proj.out_dims
            == self.shared_up_proj.out_dims
            == self.shared_down_proj.in_dims
        )

    def linears(self) -> list:
        out = [
            self.gate,
            self.shared_gate,
            self.shared_gate_proj,
            self.shared_up_proj,
            self.shared_down_proj,
        ]
        return out

    # ------------------------------------------------------------------ the wave seam
    def route(self, x: mx.array, rows: bool = False):
        """Router: (inds, scores) from an fp32 softmax and top-k, renormalised when `norm_topk`."""
        dense = (lambda lin: lin.rows) if rows else (lambda lin: lin)
        xr = x if self.router_dtype is None else x.astype(self.router_dtype)
        gates = mx.softmax(dense(self.gate)(xr), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def shared(self, x: mx.array, rows: bool = False) -> mx.array:
        dense = (lambda lin: lin.rows) if rows else (lambda lin: lin)
        sh = dense(self.shared_down_proj)(
            swiglu(dense(self.shared_gate_proj)(x), dense(self.shared_up_proj)(x))
        )
        return mx.sigmoid(dense(self.shared_gate)(x)) * sh

    def _route_shared_body(self, x: mx.array):
        inds, scores = self.route(x)
        return inds, scores, self.shared(x)

    def __call__(
        self, x: mx.array, telem=None, rows: bool = False, trace: dict | None = None
    ) -> mx.array:
        if not rows and trace is None and x.shape[-2] == 1 and self.pool is not None:
            if not hasattr(self, "_compiled_rs"):
                self._compiled_rs = mx.compile(self._route_shared_body)
            inds, scores, sh = self._compiled_rs(x)
            y = (self.experts(x, inds, telem, overlap=(sh,)) * scores[..., None]).sum(axis=-2)
            return y + sh
        inds, scores = self.route(x, rows=rows)
        if trace is not None:
            trace["moe_sel"] = inds
            trace["moe_vals"] = scores
        # the shared expert reads only resident weights, so the pool can run it as overlap while
        # this layer's expert reads are in flight
        sh = self.shared(x, rows=rows)
        y = (self.experts(x, inds, telem, overlap=(sh,)) * scores[..., None]).sum(axis=-2)
        return y + sh
