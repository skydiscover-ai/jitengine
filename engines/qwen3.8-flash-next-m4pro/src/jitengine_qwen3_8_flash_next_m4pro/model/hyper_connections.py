"""Hyper-connections: `Qwen4ExpTextGatedResidual` mixes the hc residual streams
into a block's input and `hyper_inject` adds the block's output back. With them: their compiled
elementwise regions, the fused decode launch configs and the path options ("compiled", "stock").
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from jitengine_qwen3_8_flash_next_m4pro.kernels import hc_decode as _hk
from jitengine_qwen3_8_flash_next_m4pro.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m4pro.model.config import Qwen4ExpTextConfig

# Decode-path hyper-connection as three Metal launches; prefill and score keep the stock ops.
# (KL, NSG, RB) of the down-projection launch. KL fixes each row's k-order; NSG and RB move no bit and
# were picked by the dependent-chain latency at the MTP verify pass's 2 rows (x0.75
# of (32, 4, 1) per call at p10 at M=2, x0.89 at M=1)
HC_FUSED_DOWN = (32, 2, 4)


HC_FUSED_UP = (
    2,
    8,
)  # measured (lanes per row, simdgroups) of the up-projection launch; LPR fixes bits


HC_FUSED_PRE_NT = 1024  # measured threads per threadgroup of the inject and norm launch


HC_FUSED_ROWS = 8  # the most batched decode rows that take the fused kernels


HC_PATHS = ("compiled", "stock")


HC_PATH_DEFAULT = "compiled"


HC_REGIONS = ("norm", "silu", "mulmean", "inject")


HC_EXCLUDED = (
    "sigmoid-into-multiply, and the inject gate 2*sigmoid(y/hc): neither is "
    "bit-identical (see layers.py)"
)


HC_BASIS = (
    "bit-identity to the stock path checked per region on real weights: layer 0 in fp32 "
    "and bf16 at 1 and 8 rows, and layers 0-3 on 16- and 48-token prompts in both dtypes"
)


@partial(mx.compile, shapeless=True)
def _hc_norm(xg: mx.array, w1g: mx.array, eps: float) -> mx.array:
    """Grouped `(1 + w)` RMSNorm, all of it in one region. `xg`/`w1g` are grouped [.., hc, hid]."""
    return (mx.fast.rms_norm(xg, None, eps) * w1g).astype(xg.dtype)


@partial(mx.compile, shapeless=True)
def _hc_silu_div(y: mx.array, hc: float) -> mx.array:
    return nn.silu(y / hc)


@partial(mx.compile, shapeless=True)
def _hc_mulmean(wg: mx.array, normed_g: mx.array) -> mx.array:
    """The stream mix without its sigmoid, which stays a stock op (see the note above)."""
    return mx.mean(wg * normed_g, axis=-2)


@partial(mx.compile, shapeless=True)
def _hc_add_outer(hg: mx.array, block_out: mx.array, inject: mx.array) -> mx.array:
    """The injection add, grouped: `hg + block_out[.., None, :] * inject[.., :, None]`."""
    return hg + mx.expand_dims(block_out, -2) * mx.expand_dims(inject, -1)


# ----------------------------------------------------------------------------- hyper-connections
class Qwen4ExpTextGatedResidual:
    """`Qwen4ExpTextGatedResidual`: hyper-connection mix, and inject weights if `block_inject`."""

    def __init__(
        self,
        cfg: Qwen4ExpTextConfig,
        w: dict,
        p: str,
        with_inject: bool = True,
        compute_dtype=mx.float32,
        hc_path: str = HC_PATH_DEFAULT,
        kernels_off: frozenset = frozenset(),
    ):
        self.hc_path = str(hc_path)
        self.hc = int(cfg.hc_count)
        self.hid = int(cfg.hidden_size)
        self.eps = float(cfg.rms_norm_eps)
        self.dtype = compute_dtype  # the linears' input dtype (the residual `h` may be wider)
        self.hc_norm_w1 = 1.0 + w[p + "hc_norm.weight"].astype(mx.float32)  # (1 + w), fp32
        # grouped view for the compiled norm region (a reshape, so the same values)
        self.hc_norm_w1g = self.hc_norm_w1.reshape(self.hc, self.hid)
        self.mix_down = qlinear_from(w, p + "input_mix_weight_down", cfg)  # [lowrank, hc*hid]
        self.mix_up = qlinear_from(w, p + "input_mix_weight_up", cfg)  # [hc*hid, lowrank]
        self.block_inject = qlinear_from(w, p + "block_inject_weight", cfg) if with_inject else None
        mx.eval(self.hc_norm_w1, self.hc_norm_w1g)  # derived weights are evaluated here
        self._compiled = mx.compile(self._decode_body)
        self.fused_ok = "hc_kernel" not in kernels_off and _hk.supports(
            self.hc,
            self.hid,
            self.mix_down.out_dims,
            self.mix_down.group_size,
            self.mix_down.bits,
            self.mix_down,
            self.mix_up,
            self.block_inject,
        )

    def fused(
        self, h: mx.array, block_out: mx.array | None = None, inject_in: mx.array | None = None
    ):
        """Fused path: h [.., 1, hc*hid] -> (mixed [.., hid], inject [.., hc] or None, h2).

        `block_out` [.., hid] and `inject_in` [.., hc], when given, are injected into h first; h2 is
        the injected residual, or h itself.
        """
        shp = h.shape
        mixed, inject, h2 = _hk.hc_fused(
            h.reshape(-1, shp[-1]),
            self.hc_norm_w1,
            self.mix_down,
            self.mix_up,
            self.block_inject,
            self.hc,
            self.hid,
            self.eps,
            block_out=None if block_out is None else block_out.reshape(-1, self.hid),
            inject_in=None if inject_in is None else inject_in.reshape(-1, self.hc),
            down_cfg=HC_FUSED_DOWN,
            up_cfg=HC_FUSED_UP,
            pre=True,
            pre_nt=HC_FUSED_PRE_NT,
        )
        mixed = mixed.reshape(*shp[:-1], self.hid)
        inject = None if inject is None else inject.reshape(*shp[:-1], self.hc)
        return mixed, inject, h2.reshape(shp)

    def fused_rows(
        self, h: mx.array, block_out: mx.array | None = None, inject_in: mx.array | None = None
    ):
        """`fused` on flat rows with no reshapes: h [M, hc*hid] -> (mixed [M, hid], inject [M, hc]
        or None, h2 [M, hc*hid]). `block_out` (M * hid elements) and `inject_in` [M, hc] are read
        flat, as `fused` passes them."""
        return _hk.hc_fused(
            h,
            self.hc_norm_w1,
            self.mix_down,
            self.mix_up,
            self.block_inject,
            self.hc,
            self.hid,
            self.eps,
            block_out=block_out,
            inject_in=inject_in,
            down_cfg=HC_FUSED_DOWN,
            up_cfg=HC_FUSED_UP,
            pre=True,
            pre_nt=HC_FUSED_PRE_NT,
        )

    def _decode_body(self, h: mx.array):
        """The whole hyper-connection as one compiled graph (solo decode); same ops as stock."""
        hc, hid = self.hc, self.hid
        shp = h.shape
        xg = h.reshape(*shp[:-1], hc, hid)
        normed = (mx.fast.rms_norm(xg, None, self.eps) * self.hc_norm_w1g).astype(xg.dtype)
        normed = normed.reshape(shp).astype(self.dtype)
        y = self.mix_down(normed)
        w = nn.silu(y / hc)
        w = mx.sigmoid(self.mix_up(w)).reshape(*w.shape[:-1], hc, hid)
        normed_g = normed.reshape(*normed.shape[:-1], hc, hid)
        mixed = mx.mean(w * normed_g, axis=-2)
        if self.block_inject is None:
            return mixed
        inject = 2.0 * mx.sigmoid(self.block_inject(normed) / hc)
        return mixed, inject

    def linears(self) -> list:
        return [self.mix_down, self.mix_up] + (
            [self.block_inject] if self.block_inject is not None else []
        )

    def gnorm(self, x: mx.array) -> mx.array:
        shp = x.shape
        if self.hc_path == "compiled":
            xg = x.reshape(*shp[:-1], self.hc, self.hid)
            return _hc_norm(xg, self.hc_norm_w1g, self.eps).reshape(shp)
        g = mx.fast.rms_norm(x.reshape(*shp[:-1], self.hc, self.hid), None, self.eps).reshape(shp)
        return (g * self.hc_norm_w1).astype(x.dtype)

    def __call__(self, h: mx.array, rows: bool = False):
        """h [.., hc*hid] -> (mixed [.., hid], inject [.., hc] or None)."""
        if not rows and (h.ndim == 2 or h.shape[-2] == 1):
            out = self._compiled(h)
            return (out, None) if self.block_inject is None else out
        if (
            rows
            and getattr(self, "fused_ok", False)
            and h.shape[-2] == 1
            and h.shape[0] <= HC_FUSED_ROWS
            and self.block_inject is not None
        ):
            # batched decode through the fused kernels: each row is its own grid.z with the solo
            # path's per-row arithmetic, so batched rows match solo rows bit for bit
            mixed, inject, _ = self.fused(h)
            return mixed, inject
        dense = (lambda lin: lin.rows) if rows else (lambda lin: lin)
        hc, hid = self.hc, self.hid
        # the grouped norm runs in fp32 whatever h's dtype; everything downstream is compute dtype
        normed = self.gnorm(h).astype(self.dtype)
        fused = self.hc_path == "compiled"
        y = dense(self.mix_down)(normed)
        w = _hc_silu_div(y, hc) if fused else nn.silu(y / hc)
        w = mx.sigmoid(dense(self.mix_up)(w)).reshape(*w.shape[:-1], hc, hid)
        normed_g = normed.reshape(*normed.shape[:-1], hc, hid)
        mixed = _hc_mulmean(w, normed_g) if fused else mx.mean(w * normed_g, axis=-2)
        if self.block_inject is None:
            return mixed, None
        # the inject gate stays stock on both paths: compiled it is not bit-identical (see above)
        inject = 2.0 * mx.sigmoid(dense(self.block_inject)(normed) / hc)
        return mixed, inject


def hyper_inject(
    block_out: mx.array, h: mx.array, inject: mx.array, path: str = HC_PATH_DEFAULT
) -> mx.array:
    """h + (block_out[..., None, :] * inject[..., :, None]).flatten(-2).

    `path="compiled"` runs the multiply and add as one region; it is bit-identical to stock.
    """
    if path == "compiled":
        hid, hcount = block_out.shape[-1], inject.shape[-1]
        out = _hc_add_outer(h.reshape(*h.shape[:-1], hcount, hid), block_out, inject)
        return out.reshape(*out.shape[:-2], hcount * hid)
    injection = mx.expand_dims(block_out, -2) * mx.expand_dims(inject, -1)
    return h + injection.reshape(*injection.shape[:-2], -1)
