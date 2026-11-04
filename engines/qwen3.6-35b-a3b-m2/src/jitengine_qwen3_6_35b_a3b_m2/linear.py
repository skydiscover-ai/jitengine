"""Quantized weight wrappers over MLX primitives (affine; bits inferred from the shapes).

`mx.quantized_matmul` changes its accumulation order past a shape-dependent row count, so batched
decode splits rows into groups no larger than that threshold (measured per weight shape at load by
`calibrate_row_limits`) to stay bit-identical to solo decode.
"""

from __future__ import annotations

import itertools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kernels import qmv_prescaled

PROBE_ROWS = 16  # largest batch the row-limit probe considers
_ROW_LIMIT: dict = {}  # (out_dims, in_dims, bits, group_size) -> measured safe group size

# qmv_mrow.qmv_fast_mrow: qmv_fast's arithmetic with MROW_BLOCK input rows per threadgroup;
# bit-identical to solo at every M, so it needs no row_limit split (verified per shape)
MROW_BLOCK = 2
# (results_per_simdgroup, num_simdgroups) of the multi-row kernel: pure scheduling, same bits
MROW_RPS_NSG = (8, 4)
_MROW_OK: dict = {}  # (out_dims, in_dims, bits, group_size) -> verified usable at load time

# The same kernel at MB = 1 in place of stock `mx.quantized_matmul` for one row. rps and nsg only
# schedule work and never change the per-lane add order (`packs_per_thread` would, so it is left
# alone). An mx.fast.metal_kernel call costs a few microseconds more than a native op, so only
# weights of at least SOLO_MIN_BYTES use it.
SOLO_RPS_NSG = (8, 2)  # (results_per_simdgroup, num_simdgroups)
SOLO_MIN_BYTES = 64 * 2**20  # smallest weight for which one call's saving beats the wrapper cost
_SOLO_OK: dict = {}  # shape_key -> the verified (rps, nsg), or None to keep the stock op
_SOLO_SMALL: set = (
    set()
)  # shape_keys declined on size alone, so the log can tell them from failures


# `qmv_prescaled`'s 3-bit GEMV for one-row and multi-row calls of a QLinear whose `prescaled` is set
# (Qwen3_5MoeForCausalLM.use_kernel_choice); each shape is verified on first use
# (`calibrate_prescaled`)
_PRESCALED_OK: dict = {}


# prefill dense projections via qmm_prefill.qmm_t (stock qmm_t arithmetic, verified per shape)
_PF_OK: dict = {}

# per-shape checks that found a difference from stock, {(kernel, shape_key)}, and those that raised
# instead of comparing, {(kernel, shape_key): "Type: message"}; either way the kernel is off for that
# shape, and `dense_kernel_report` says which and why
_DIFFER: set = set()
_RAISED: dict = {}


def _raised(kernel: str, key, exc: Exception) -> None:
    _RAISED[(kernel, key)] = f"{type(exc).__name__}: {exc}"


class _TooSmall(Exception):
    """This weight is too small for the kernel's per-call wrapper cost to be worth paying."""


def _one_row(x: mx.array) -> bool:
    n = 1
    for d in x.shape[:-1]:
        n *= d
    return n == 1


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
        "prescaled",
        "_views",
    )

    def __init__(
        self, weight: mx.array, scales: mx.array, biases: mx.array | None, group_size: int
    ):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.in_dims = scales.shape[-1] * group_size
        self.out_dims = weight.shape[-2]
        self.bits = weight.shape[-1] * 32 // self.in_dims
        self.row_limit = 1  # conservative until calibrated
        self.prescaled = False  # qmv_prescaled for one-row and multi-row calls (use_kernel_choice)
        self._views = None  # `expert_views`, built on first use

    def expert_views(self) -> tuple[mx.array, mx.array, mx.array]:
        """This expert stack's weight, scales and biases as one-expert views of the same buffers.

        The routed-expert kernels index experts by pointer and never read shapes, while MLX
        commits a command buffer once its encoded input bytes pass max_mb_per_buffer: a whole stack
        as a kernel input forced a commit at every launch. The views are built on first use and
        kept here, so they live exactly as long as the stack.
        """
        if self._views is None:
            self._views = (self.weight[:1], self.scales[:1], self.biases[:1])
            mx.eval(*self._views)
        return self._views

    @property
    def shape_key(self) -> tuple:
        return (self.out_dims, self.in_dims, self.bits, self.group_size)

    def _stock(self, x: mx.array) -> mx.array:
        """Stock `mx.quantized_matmul`, the reference for every calibration (never `__call__`)."""
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

    def _prescaled(self, flat: mx.array, pre=None) -> mx.array:
        return qmv_prescaled.qmv(
            flat, self.weight, self.scales, self.biases, self.group_size, pre=pre
        )

    def prescaled_ok_solo(self) -> bool:
        return self.prescaled and self.calibrate_prescaled()

    def calibrate_prescaled(self) -> bool:
        key = self.shape_key
        ok = _PRESCALED_OK.get(key)
        if ok is None:
            ok = _PRESCALED_OK[key] = self.prescaled_matches_stock()
        return ok

    def prescaled_matches_stock(self, probe_rows: int = 8) -> bool:
        """`qmv_prescaled.check` for this weight, when the kernel supports it."""
        try:
            if not qmv_prescaled.supports(self.weight, self.scales, self.biases, self.group_size):
                return False
            if qmv_prescaled.check(
                self.weight, self.scales, self.biases, self.group_size, probe_rows
            ):
                return True
            _DIFFER.add(("prescaled", self.shape_key))
            return False
        except Exception as exc:  # noqa: BLE001  off for this shape; the report says why
            _raised("prescaled", self.shape_key, exc)
            return False

    def calibrate_pf(self) -> bool:
        key = self.shape_key
        ok = _PF_OK.get(key)
        if ok is not None:
            return ok
        ok = False
        try:
            from jitengine_qwen3_6_35b_a3b_m2.kernels import qmm_prefill as dp

            if self.biases is not None and self.scales.dtype == mx.bfloat16:
                ok = True
                for m in (192, 231, 256, 300):
                    if not dp.eligible(m, self.out_dims, self.in_dims, self.bits):
                        continue
                    x = mx.random.normal((m, self.in_dims), key=mx.random.key(0xF0 + m)).astype(
                        mx.bfloat16
                    )
                    y = dp.qmm_t(
                        x, self.weight, self.scales, self.biases, self.bits, self.group_size
                    )
                    if not bool(mx.array_equal(y, self._stock(x)).item()):
                        _DIFFER.add(("prefill", key))
                        ok = False
                        break
        except Exception as exc:  # noqa: BLE001  off for this shape; the report says why
            _raised("prefill", key, exc)
            ok = False
        _PF_OK[key] = ok
        return ok

    def __call__(self, x: mx.array, pre=None) -> mx.array:
        if x.ndim >= 2 and x.dtype == mx.bfloat16:
            from jitengine_qwen3_6_35b_a3b_m2.kernels import qmm_prefill as dp

            m = x.size // self.in_dims
            if dp.eligible(m, self.out_dims, self.in_dims, self.bits) and self.calibrate_pf():
                y = dp.qmm_t(
                    x.reshape(m, self.in_dims),
                    self.weight,
                    self.scales,
                    self.biases,
                    self.bits,
                    self.group_size,
                )
                return y.reshape(*x.shape[:-1], self.out_dims)
        if self.prescaled and _one_row(x) and self.calibrate_prescaled():
            return self._prescaled(x.reshape(1, self.in_dims), pre=pre).reshape(
                *x.shape[:-1], self.out_dims
            )
        if _one_row(x):
            t = _SOLO_OK.get(self.shape_key)
            if t is not None:
                from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_mrow import qmv_fast_mrow

                y = qmv_fast_mrow(
                    x.reshape(1, self.in_dims),
                    self.weight,
                    self.scales,
                    self.biases,
                    mb=1,
                    group_size=self.group_size,
                    bits=self.bits,
                    rps=t[0],
                    nsg=t[1],
                )
                return y.reshape(*x.shape[:-1], self.out_dims)
        return self._stock(x)

    def rows(self, x: mx.array) -> mx.array:
        """Batched decode over [B, 1, in_dims] independent rows, bitwise as each row on its own."""
        m = x.shape[0]
        if m >= 2 and self.prescaled and self.calibrate_prescaled():
            return self._prescaled(x.reshape(m, self.in_dims)).reshape(*x.shape[:-1], self.out_dims)
        if m >= 2 and _MROW_OK.get(self.shape_key):
            from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_mrow import qmv_fast_mrow

            # never dispatch a ragged block (a threadgroup always does MB rows of work): whole
            # blocks go through the kernel, the remainder through the single-row path (same bits)
            flat = x.reshape(m, self.in_dims)
            whole = (m // MROW_BLOCK) * MROW_BLOCK
            y = qmv_fast_mrow(
                flat[:whole],
                self.weight,
                self.scales,
                self.biases,
                mb=MROW_BLOCK,
                group_size=self.group_size,
                bits=self.bits,
                rps=MROW_RPS_NSG[0],
                nsg=MROW_RPS_NSG[1],
            )
            if whole < m:
                y = mx.concatenate([y, self(flat[whole:])], axis=0)
            return y.reshape(*x.shape[:-1], self.out_dims)
        lim = self.row_limit
        if m <= lim:
            return self(x)
        return mx.concatenate([self(x[i : i + lim]) for i in range(0, m, lim)], axis=0)

    def calibrate_mrow(self, mb: int, probe_rows: int = PROBE_ROWS) -> bool:
        """Verify per shape that the multi-row kernel equals per-row `_stock` bit for bit."""
        key = self.shape_key
        cached = _MROW_OK.get(key)
        if cached is not None:
            return cached
        ok = False
        try:
            from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_mrow import qmv_fast_mrow

            vals_per_thread = (8 if self.bits in (3, 4) else 32 // self.bits) * 2
            if (
                self.biases is not None
                and self.bits in (3, 4, 8)
                and self.in_dims % (vals_per_thread * 32) == 0
                and self.in_dims % self.group_size == 0
            ):
                x = mx.random.normal((probe_rows, self.in_dims), key=mx.random.key(0xB10C)).astype(
                    mx.bfloat16
                )
                solo = mx.concatenate(
                    [self._stock(x[i : i + 1]) for i in range(probe_rows)], axis=0
                )
                mx.eval(solo)
                ok = True
                for m in range(2, probe_rows + 1):
                    y = qmv_fast_mrow(
                        x[:m],
                        self.weight,
                        self.scales,
                        self.biases,
                        mb=mb,
                        group_size=self.group_size,
                        bits=self.bits,
                        rps=MROW_RPS_NSG[0],
                        nsg=MROW_RPS_NSG[1],
                    )
                    mx.eval(y)
                    if not bool(mx.all(y == solo[:m]).item()):
                        _DIFFER.add(("multi_row", key))
                        ok = False
                        break
        except Exception as exc:  # noqa: BLE001  any failure means the stock path; the report says why
            _raised("multi_row", key, exc)
            ok = False
        _MROW_OK[key] = ok
        return ok

    def calibrate_solo(self, probe_rows: int = PROBE_ROWS):
        """Verify the single-row kernel bitwise against `_stock`; returns its (rps, nsg) or None."""
        key = self.shape_key
        if key in _SOLO_OK:
            return _SOLO_OK[key]
        got = None
        try:
            from jitengine_qwen3_6_35b_a3b_m2.kernels.qmv_mrow import qmv_fast_mrow

            vals_per_thread = (8 if self.bits in (3, 4) else 32 // self.bits) * 2
            if (
                self.biases is not None
                and self.bits in (3, 4, 8)
                and self.in_dims % (vals_per_thread * 32) == 0
                and self.in_dims % self.group_size == 0
            ):
                if self.nbytes < SOLO_MIN_BYTES:
                    raise _TooSmall
                rps, nsg = SOLO_RPS_NSG
                x = mx.random.normal((probe_rows, self.in_dims), key=mx.random.key(0x50103)).astype(
                    mx.bfloat16
                )
                ok = True
                for i in range(probe_rows):
                    ref = self._stock(x[i : i + 1])
                    y = qmv_fast_mrow(
                        x[i : i + 1],
                        self.weight,
                        self.scales,
                        self.biases,
                        mb=1,
                        group_size=self.group_size,
                        bits=self.bits,
                        rps=rps,
                        nsg=nsg,
                    )
                    mx.eval(ref, y)
                    if not bool(mx.all(y == ref).item()):
                        _DIFFER.add(("single_row", key))
                        ok = False
                        break
                if ok:
                    got = (rps, nsg)
        except _TooSmall:  # not a failure: the wrapper cost would exceed the saving
            _SOLO_SMALL.add(key)
            got = None
        except Exception as exc:  # noqa: BLE001  any failure means the stock op; the report says why
            _raised("single_row", key, exc)
            got = None
        _SOLO_OK[key] = got
        return got

    def calibrate(self, probe_rows: int = PROBE_ROWS) -> int:
        """Largest row group `_stock` computes bit-identically to single rows; cached per shape."""
        cached = _ROW_LIMIT.get(self.shape_key)
        if cached is not None:
            self.row_limit = cached
            return cached
        limit = probe_rows
        # both ranks decode uses ([B, 1, in] for block projections, [B, in] for the lm_head), so
        # no call site falls outside what was measured
        for shape in ((probe_rows, 1, self.in_dims), (probe_rows, self.in_dims)):
            x = mx.random.normal(shape, key=mx.random.key(0x5EED)).astype(mx.bfloat16)
            solo = mx.concatenate([self._stock(x[i : i + 1]) for i in range(probe_rows)], axis=0)
            mx.eval(solo)
            here = 1
            for m in range(2, probe_rows + 1):
                y = self._stock(x[:m])
                mx.eval(y)
                if not bool(mx.all(y == solo[:m]).item()):
                    break
                here = m
            limit = min(limit, here)
        _ROW_LIMIT[self.shape_key] = limit
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


class FusedLinear:
    """Same-input projections fused along the output axis (an exact re-layout) into one launch."""

    __slots__ = ("lin", "sizes", "cuts", "parts")

    def __init__(self, parts: list):
        ref = parts[0]
        self._check_fusable(parts, ref)
        w, s, b = self._concat_parts(parts, ref)
        self.lin = QLinear(w, s, b, ref.group_size)
        acc, cuts = self._set_cut_points(parts)
        self._split_parts(parts, ref, w, s, b, cuts, acc)

    def _check_fusable(self, parts: list, ref) -> None:
        """Raise unless every part shares the first part's bits, group size, input width and bias presence."""
        for p in parts[1:]:
            if (p.bits, p.group_size, p.in_dims) != (ref.bits, ref.group_size, ref.in_dims):
                raise ValueError("fused parts must share bits, group size and input width")
            if (p.biases is None) != (ref.biases is None):
                raise ValueError("fused parts must all have (or all lack) biases")

    def _concat_parts(self, parts: list, ref) -> tuple:
        """Concatenate the parts' weights, scales and biases along the output axis and evaluate them."""
        cat = lambda xs: mx.concatenate(xs, axis=-2)  # noqa: E731  the output axis, 2-D or stacked 3-D
        w = cat([p.weight for p in parts])
        s = cat([p.scales for p in parts])
        b = cat([p.biases for p in parts]) if ref.biases is not None else None
        mx.eval(w, s, *([b] if b is not None else []))
        return w, s, b

    def _set_cut_points(self, parts: list) -> tuple:
        """Record each part's output size and the cumulative cut points between parts; return them with the running total."""
        self.sizes = [p.out_dims for p in parts]
        acc, cuts = 0, []
        for n in self.sizes[:-1]:
            acc += n
            cuts.append(acc)
        self.cuts = cuts
        return acc, cuts

    def _split_parts(
        self, parts: list, ref, w: mx.array, s: mx.array, b: mx.array | None, cuts: list, acc: int
    ) -> None:
        """Build the per-part row-range views of the fused tensors, each with its source part's row limit, and evaluate them."""
        # zero-copy row-range views per part: multi-row inputs run each part unfused, because MLX's
        # multi-row quantized GEMM may split its reduction by the total output width
        edges = [0] + cuts + [acc + self.sizes[-1]]
        self.parts = [
            QLinear(
                w[..., lo:hi, :],
                s[..., lo:hi, :],
                b[..., lo:hi, :] if b is not None else None,
                ref.group_size,
            )
            for lo, hi in itertools.pairwise(edges)
        ]
        for q, p in zip(self.parts, parts):
            q.row_limit = p.row_limit
        # evaluate now: an unevaluated graph is bound to its building thread, not the model worker
        mx.eval(*[t for q in self.parts for t in (q.weight, q.scales, q.biases) if t is not None])

    @staticmethod
    def compatible(parts: list) -> bool:
        ref = parts[0]
        return all(
            (p.bits, p.group_size, p.in_dims, p.biases is None)
            == (ref.bits, ref.group_size, ref.in_dims, ref.biases is None)
            for p in parts
        )

    @staticmethod
    def _one_row(x: mx.array) -> bool:
        return _one_row(x)

    def __call__(self, x: mx.array, pre=None) -> list:
        if self._one_row(x):  # one kernel; bit-exact with the parts at M = 1
            return mx.split(self.lin(x, pre=pre), self.cuts, axis=-1)
        return [p(x) for p in self.parts]

    def rows(self, x: mx.array) -> list:
        if self._one_row(x):
            return mx.split(self.lin(x), self.cuts, axis=-1)
        m = x.shape[0]
        if self.lin.prescaled and self.lin.calibrate_prescaled():
            # one launch over the fused rows: each output row is the same qmv_fast row either way
            y = self.lin._prescaled(x.reshape(m, self.lin.in_dims)).reshape(
                *x.shape[:-1], self.lin.out_dims
            )
            return mx.split(y, self.cuts, axis=-1)
        return [p.rows(x) for p in self.parts]

    def gather(self, x: mx.array, rhs_indices: mx.array, sorted_indices: bool = False) -> list:
        return mx.split(
            self.lin.gather(x, rhs_indices, sorted_indices=sorted_indices), self.cuts, axis=-1
        )


def calibrate_row_limits(linears, solo_extra=()) -> dict:
    """Calibrate every distinct weight shape once; returns {shape_key: safe group size}.

    `solo_extra` holds linears only the single-row path reaches (the fused `FusedLinear.lin`
    buffers), so they are calibrated for the solo kernel only. `dense_kernel_report` gives what
    each per-shape check decided.
    """
    for lin in linears:
        lin.calibrate()
    for lin in list(linears) + list(solo_extra):
        lin.calibrate_solo()
    for lin in linears:
        lin.calibrate_mrow(MROW_BLOCK)
    return dict(_ROW_LIMIT)


def _shape(key) -> str:
    return f"{key[0]}x{key[1]}/{key[2]}b"


def _kernel(table: dict, kind: str) -> dict:
    return {
        "shapes": len(table),
        "on": [_shape(k) for k, v in sorted(table.items()) if v],
        "differ": [_shape(k) for kd, k in sorted(_DIFFER) if kd == kind],
        "raised": {_shape(k): err for (kd, k), err in sorted(_RAISED.items()) if kd == kind},
    }


def dense_kernel_report() -> dict:
    """What the per-shape checks decided so far. For each dense kernel: the shapes it was asked
    about, those it runs on, those where it differs from stock and those whose check raised (the
    rest are shapes it does not take); and each shape's safe row group. The single- and multi-row
    kernels are checked at load, the prescaled and prefill kernels on first use."""
    return {
        "single_row": {
            **_kernel(_SOLO_OK, "single_row"),
            "rps_nsg": list(SOLO_RPS_NSG),
            "below_min_bytes": [_shape(k) for k in sorted(_SOLO_SMALL)],
        },
        "multi_row": {
            **_kernel(_MROW_OK, "multi_row"),
            "mb": MROW_BLOCK,
            "rps_nsg": list(MROW_RPS_NSG),
        },
        "prescaled": _kernel(_PRESCALED_OK, "prescaled"),
        "prefill": _kernel(_PF_OK, "prefill"),
        "row_groups": {_shape(k): v for k, v in sorted(_ROW_LIMIT.items())},
    }


class QEmbedding(QLinear):
    """The quantized token embedding: a QLinear whose rows are looked up and dequantized."""

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
