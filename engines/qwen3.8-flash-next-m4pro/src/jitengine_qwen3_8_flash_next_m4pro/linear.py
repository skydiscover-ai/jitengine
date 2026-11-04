"""Quantized weight wrappers over MLX primitives (affine, `weight` U32-packed along the input dim).

`(bits, group_size)` come from the config, never from shapes: `[N, K/8]` U32 with `[N, K/64]`
scales is both (4, 64) and (8, 32). `QLinear.rows` must match applying each row alone bit for bit.
Stock `mx.quantized_matmul` changes accumulation order past a row count that depends on shape and
dtype, so `rows` splits at a calibrated limit; a shape the dense kernel serves has a fixed k-order
and takes the kernel at every M, M = 1 included, so solo and batched decode agree.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as _dk

logger = logging.getLogger(__name__)

PROBE_ROWS = 16  # largest batch the row-limit probe considers
_ROW_LIMIT: dict = {}  # (out_dims, in_dims, bits, group_size, dtype) -> measured safe group size


class QLinear:
    __slots__ = (
        "weight",
        "scales",
        "biases",
        "group_size",
        "bits",
        "in_dims",
        "out_dims",
        "row_limit",
        "kernel_cfg",
        "kernel_dtype",
    )

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array | None,
        group_size: int,
        bits: int = 4,
    ):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.in_dims = scales.shape[-1] * self.group_size
        self.out_dims = weight.shape[-2]
        packed = weight.shape[-1] * 32
        if packed != self.in_dims * self.bits:
            raise ValueError(
                f"quantized weight {tuple(weight.shape)} with scales {tuple(scales.shape)} does not "
                f"match (bits={self.bits}, group_size={self.group_size}): {packed} packed bits per row "
                f"!= {self.in_dims} x {self.bits}"
            )
        self.row_limit = 1  # conservative until calibrated
        self.kernel_cfg = None  # (KL, NSG, RB) once `enable_dense_kernel` serves this shape
        self.kernel_dtype = None  # the one activation dtype that config was chosen for

    def enable_dense_kernel(self, option: str, dtype) -> bool:
        """Select the row-blocked kernel for this shape at `dtype`; returns whether it serves."""
        self.kernel_cfg = None
        self.kernel_dtype = None
        if option != "rb":
            return False
        if not _dk.supports(
            self.bits,
            self.group_size,
            self.in_dims,
            self.biases is not None,
            self.scales.dtype,
            dtype,
        ):
            return False
        cfg = _dk.config_for(self.out_dims, self.in_dims, dtype)
        if cfg is None:
            return False
        self.kernel_cfg = cfg
        self.kernel_dtype = dtype
        return True

    def _rb(self, x: mx.array, m: int) -> mx.array:
        """Row-blocked kernel over m <= MAX_M rows; `vecs` depends on m, the k-order does not.
        Flat rows [m, in] go in and come out [m, out] with no reshape launches."""
        kl, nsg, rb = self.kernel_cfg
        flat = x.ndim == 2
        y = _dk.qmv_wide_rb(
            x if flat else x.reshape(m, self.in_dims),
            self.weight,
            self.scales,
            self.biases,
            vecs=_dk.vecs_for(m, x.dtype),
            k_lanes=kl,
            nsg=nsg,
            rb=rb,
            group_size=self.group_size,
        )
        return y if flat else y.reshape(*x.shape[:-1], self.out_dims)

    def _serves(self, x: mx.array) -> bool:
        return self.kernel_cfg is not None and x.dtype == self.kernel_dtype

    def _rowcount(self, x: mx.array) -> int:
        m = 1
        for d in x.shape[:-1]:
            m *= int(d)
        return m

    def __call__(self, x: mx.array) -> mx.array:
        # Served shapes take the kernel at every M <= MAX_M, M = 1 included: solo decode enters
        # here and must match `rows` bit for bit. Larger M is prompt processing, which stays stock.
        if self._serves(x):
            m = self._rowcount(x)
            if m <= _dk.MAX_M:
                return self._rb(x, m)
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )

    def store_metadata_fp32(self, dtype) -> int:
        """Store 16-bit scales and biases as float32 for float32 activations; returns bytes added.

        Exact (every bf16/f16 value is a float32 value); it removes stock's per-call cast.
        """
        if dtype != mx.float32:
            return 0
        added = 0
        if self.scales is not None and self.scales.dtype in (mx.bfloat16, mx.float16):
            added += self.scales.nbytes
            self.scales = self.scales.astype(mx.float32)
        if self.biases is not None and self.biases.dtype in (mx.bfloat16, mx.float16):
            added += self.biases.nbytes
            self.biases = self.biases.astype(mx.float32)
        if added:
            mx.eval(self.scales, self.biases)
        return added

    def shape_key(self, dtype) -> tuple:
        return (self.out_dims, self.in_dims, self.bits, self.group_size, str(dtype))

    def rows(self, x: mx.array) -> mx.array:
        """Apply to B independent rows ([B, 1, in] or [B, in]), bit-identically to each row alone.

        Batched decode and the scored readout use this; prompt processing uses `__call__`.
        """
        m = x.shape[0]
        # A served shape's k-order is fixed by KL, so it splits only at MAX_M to bound the launch.
        lim = _dk.MAX_M if self._serves(x) else self.row_limit
        if m <= lim:
            return self(x)
        return mx.concatenate([self(x[i : i + lim]) for i in range(0, m, lim)], axis=0)

    def calibrate(self, probe_rows: int = PROBE_ROWS, dtype=mx.float32) -> int:
        """Measure the largest row group whose result equals the one-row result bit for bit.

        Cached per (weight shape, `dtype`): the threshold never depends on the weight values.
        """
        key = self.shape_key(dtype)
        cached = _ROW_LIMIT.get(key)
        if cached is not None:
            self.row_limit = cached
            return cached
        limit = probe_rows
        # Probe both ranks decode uses ([B, 1, in] projections, [B, in] lm_head) so no call site
        # falls outside what was measured.
        for shape in ((probe_rows, 1, self.in_dims), (probe_rows, self.in_dims)):
            x = mx.random.normal(shape, key=mx.random.key(0x5EED)).astype(dtype)
            solo = mx.concatenate([self(x[i : i + 1]) for i in range(probe_rows)], axis=0)
            mx.eval(solo)
            here = 1
            for m in range(2, probe_rows + 1):
                y = self(x[:m])
                mx.eval(y)
                if not bool(mx.all(y == solo[:m]).item()):
                    break
                here = m
            limit = min(limit, here)
        _ROW_LIMIT[key] = limit
        self.row_limit = limit
        return limit

    def gather(self, x: mx.array, rhs_indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Expert-gathered matmul: `weight` carries a leading expert axis, `rhs_indices` picks it."""
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )

    @property
    def nbytes(self) -> int:
        return (
            self.weight.nbytes
            + self.scales.nbytes
            + (self.biases.nbytes if self.biases is not None else 0)
        )


def qlinear_from(w: dict, name: str, cfg) -> QLinear:
    """`QLinear` over `w[name + .weight/.scales/.biases]` with (bits, group) from `cfg.quant_of(name)`."""
    bits, gs = cfg.quant_of(name)
    return QLinear(w[name + ".weight"], w[name + ".scales"], w.get(name + ".biases"), gs, bits)


def apply_dense_policy(linears, dtype, dense_kernel: str) -> dict:
    """Route each dense linear to the row-blocked kernel at `dtype`, else store its metadata fp32.

    Embedding tables are skipped: on a tied checkpoint `lm_head` is the embedding, and doubling its
    metadata costs more than the per-call cast it saves. Returns a summary dict.
    """
    served: dict = {}
    unserved: dict = {}
    converted = 0
    for lin in linears:
        if isinstance(lin, QEmbedding):
            continue
        key = f"{lin.out_dims}x{lin.in_dims}/{lin.bits}b"
        if lin.enable_dense_kernel(dense_kernel, dtype):
            served[key] = served.get(key, 0) + 1
        else:
            unserved[key] = unserved.get(key, 0) + 1
            converted += lin.store_metadata_fp32(dtype)
    out = {
        "option": str(dense_kernel),
        "compute_dtype": str(dtype).split(".")[-1],
        "served_shapes": served,
        "unserved_shapes": unserved,
        "served_linears": sum(served.values()),
        "unserved_linears": sum(unserved.values()),
        "fp32_metadata_bytes_added": int(converted),
        "max_rows_per_launch": _dk.MAX_M,
        "config_basis": _dk.CONFIG_BASIS,
        "configs": {f"{n}x{k} {d}": list(c) for (n, k, d), c in sorted(_dk.CONFIGS.items())},
        "source": _dk.__name__,
    }
    logger.debug(
        f"[engine] dense kernel {dense_kernel!r} at {out['compute_dtype']}: "
        f"{out['served_linears']} linears on the row-blocked kernel over {len(served)} shapes "
        f"(no row splitting below M={_dk.MAX_M}), {out['unserved_linears']} on stock over "
        f"{len(unserved)} shapes with {converted / 1e6:.1f} MB of float32-stored metadata"
    )
    return out


def calibrate_row_limits(linears, dtype=mx.float32) -> dict:
    """Calibrate every distinct weight shape once at `dtype`; returns {(shape, dtype): group}."""
    for lin in linears:
        lin.calibrate(dtype=dtype)
    here = {k: v for k, v in _ROW_LIMIT.items() if k[4] == str(dtype)}
    worst = min(here.values()) if here else 0
    logger.debug(
        f"[engine] dense row-exactness at {dtype}: {len(here)} distinct weight shapes, safe row groups "
        + ", ".join(f"{k[0]}x{k[1]}/{k[2]}b:{v}" for k, v in sorted(here.items()))
        + f" (worst {worst})"
    )
    return dict(here)


class QEmbedding(QLinear):
    def lookup(self, ids: mx.array) -> mx.array:
        biases = self.biases
        return mx.dequantize(
            self.weight[ids],
            scales=self.scales[ids],
            biases=biases[ids] if biases is not None else None,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )
