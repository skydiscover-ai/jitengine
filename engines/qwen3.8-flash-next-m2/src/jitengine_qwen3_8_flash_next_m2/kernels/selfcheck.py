"""Load-time checks that each custom kernel agrees with the stock MLX ops it replaces, on this
device and at the served model's geometry (`gdn_recurrence_staged` against `gdn_recurrence`,
`grouped_qmv_x` against the `grouped_qmv` call, and the two kernels of the MTP verify pass,
`qmv_multi_v` and `grouped_qmv_u2`, against the solo kernels `qmv_multi` and `grouped_qmv_x` whose
rows they must reproduce).

Each module keeps its Metal source as Python strings and compiles it with `mx.fast.metal_kernel` in
a `functools.cache` factory on first launch, so importing a module compiles nothing and a kernel
that does not compile fails inside its check. The kernels and their dense and expert tuning tables
(`CONFIGS`) come from the M4 Pro build, and `CONFIG_BASIS` says the tables were measured on an M4 Pro;
this build was then tuned end to end, and its published numbers measured, on a 24 GB M2. Another GPU
family may compile or schedule the kernels differently, and the tables are keyed by shape and dtype,
not by GPU family or row count. Each kernel module has a check; `Checks` runs them and keeps each result. A
kernel whose check fails, or raises because the kernel does not compile or run here, is switched off
and the model runs its stock path instead; the error of a check that raised is kept to report it.
Source paths below are relative to the package, `src/jitengine_qwen3_8_flash_next_m2/`, tests to
the engine root. `README.md` in this directory tables each kernel's check and test and names its
result key first in `covers`. The published numbers were measured on one 24 GB M2 Mac at an 18 GB
budget with mlx 0.32.2, and `tests/e2e/test_gate.py` expects its greedy-512 hash unless
`JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_SHA` names another.

Call sites. The kernels are launched from `linear.py` (`QLinear`),
`model/hyper_connections.py`, `model/attention.py`,
`model/qsa_indexer.py`, `model/gdn.py` and `model/model.py`, and
from the mixins of the package `model/moe/`: `decode.py` (`topk_softmax`, `moe_combine`,
and `qmv_multi` for the shared expert), `speculative.py` (`grouped_qmv_x`, `moe_combine`),
`expert_mlp.py` (`grouped_qmv`, `grouped_swiglu`, which `prefill.py` and `waves.py` reach through
it). The MTP verify pass (`Qwen4ExpForCausalLM.verify2`, which `DecodingMixin._mtp_round` calls
for a lone greedy row) launches `qmv_multi_v` from `Qwen4ExpTextDecoderLayer.build_spec2` (through
`_gdn2` and `_moe2`) and from `Qwen4ExpTextAttention.decode2`, and `grouped_qmv_u2` from
`spec_build_pair` in `speculative.py`. The memory plan prices the expert kernel from checkpoint headers (`budget.py`,
`moe_kernel_for`, which calls `supports_geometry`), and `Engine.init_model` raises RuntimeError if
the model's choice differs, so `supports` and `supports_geometry` change together.

How the checks run. `Engine.init_kernels` (`engine/engine.py`) loads the config, resolves the MTP head's path
(`mtp.resolve`) and, before the memory plan, calls
`run_kernel_checks(cfg, compute dtype, dense_kernel, moe_kernel, moe_kernel_variant, mtp)`, with
`mtp` true when the head is present
(`model/kernel_checks.py`). It runs the check of each kernel the config and the options
allow the model to launch, at the config's geometry, quantization and metadata dtypes and the
model's launch constants, through `Checks.run`, under its key in `kernel_checks.NAMES`:
`dense_kernel`, `moe_kernel`, `moe_fused`, `hc_kernel`, `gdn_fused`, `gdn_kernel`,
`sdpa_decode_masked`, `attn_fused`, `indexer_fused`, `indexer_prefill`, and with a head
`qmv_multi_v` and `grouped_qmv_u2`. That is a superset of
what the built model launches when every layer of a kind matches the one probed (the first attention
layer and the first GDN layer, layer 0's hyper-connection and experts), with activations at the
compute dtype only, so not under `--residual-dtype`. The dense check runs first: its verdict sets
the metadata dtypes that the hyper-connection and glue checks use. The keys are the names
`kernel_checks.py` imports the modules under (`qsa_decode_mask as attn_fused`,
`gdn_recurrence as gdn_kernel`, ...) and appear in `/stats`, the log and the e2e test, so they do
not follow file renames. A kernel that the config or the options give nothing to run gets no check
and no key: `dense_kernel` under the option `dense_kernel="stock"` or when its table serves no dense
shape; `moe_kernel` under `moe_kernel="stock"` or when its `supports` rejects the experts'
quantization or widths; `moe_fused` when `supports_topk` rejects the router; `hc_kernel`,
`indexer_fused` and `indexer_prefill` when their `supports` rejects the geometry; `gdn_fused` when
`_glue_calls` finds no call to check (it adds the GDN, attention and shared-expert calls where
`supports_multi` takes their weight sets; the GDN call also needs `supports_gdn`, and the shared
expert's two calls need `supports_topk` and go together); `gdn_kernel` without a GDN layer or when
`GDN_IMPL` (`model/gdn.py`, "kernel" by default) is not "kernel"; the attention checks
without an attention layer. The two verify kernels get a key only with an MTP head:
`qmv_multi_v` when `_glue_calls` finds a call (the calls `check_glue` takes), `grouped_qmv_u2` when
`moe_kernel.supports` accepts the experts' quantization and widths. Both run under any `moe_kernel`
option and whether or not `gdn_fused` and `moe_kernel` passed, since the verify pass launches them
either way. Seven functions raise ValueError when `supports` rejects the geometry they
are asked to check: `check_decode_masked`, `check_indexer`, `check_prefill_mask`, `check_hc`,
`check_grouped`, `check_grouped_u2` and `_random_lins` (which `_check_multi`, for `check_glue`, and
`check_multi_v` call). `run_kernel_checks` calls
`check_decode_masked` unguarded, so a geometry it rejects raises and turns `sdpa_decode_masked` off;
it calls the other six only where `supports` (`supports_multi` for the glue and `check_multi_v`) holds, so an
unsupported geometry is not checked and never recorded false. No kernel entry point validates its
operands: each relies on its caller's gate, the flag (`kernels_off`) or option (`dense_kernel`,
`moe_kernel`) its check sets and, for most, the module's `supports*` function. `decode_masked`'s
call site has no `supports` gate (it tests one token, `masked_sdpa_ok` and that the indexer returned
a mask; its `supports` is enforced by its check), and `gdn_recurrence` and `gdn_recurrence_staged`
have no `supports`.

How a failing kernel is switched off. `Checks.run` records False when a check returns false and also
when it raises; the exception's type and message go in `Checks.errors`. The engine keeps
`kernel_checks` (every result), `kernel_check_errors` and `kernels_off` (the keys whose result is
False). A failed `dense_kernel` sets the engine's option `dense_kernel` (`DENSE_KERNELS`, "rb" or
"stock"; default `DENSE_KERNEL_DEFAULT`) to "stock", and a failed `moe_kernel` sets `moe_kernel` to
"stock", before the memory plan prices them; `Qwen4ExpForCausalLM` takes those two as options and
ignores their names in `kernels_off`. `kernels_off` goes to `Qwen4ExpForCausalLM`
(`model/model.py`), whose blocks clear the flags that the other kernels' paths test:
`indexer_fused`, `attn_fused` and `indexer_prefill` clear `Qwen4ExpTextQSAIndexer.indexer_fused_ok`,
`attn_fused_ok` and `indexer_prefill_ok`; `sdpa_decode_masked` clears
`Qwen4ExpTextAttention.masked_sdpa_ok`; `gdn_fused` clears `Qwen4ExpTextAttention.input_projs_ok`,
`Qwen4ExpTextGatedDeltaNet.fused_ok` and `Qwen4ExpTextSparseMoeBlock.shared_fused_ok`; `gdn_kernel`
clears `Qwen4ExpTextGatedDeltaNet.recurrence_kernel_ok`; `hc_kernel` clears
`Qwen4ExpTextGatedResidual.fused_ok`; `moe_fused` clears `Qwen4ExpTextSparseMoeBlock.fused_ok`, and
with it `spec_ok` and `shared_fused_ok`. That `fused_ok` also needs an expert pool and
`supports_topk`, and `spec_ok` needs the variant "split" and no `moe_sync_each_layer`. Each path
then runs the ops in the README's `stock op` column; kernels whose keys are on still run inside it:
`qmv_wide_rb` in `QLinear` for a served shape at the compute dtype and at most `MAX_M` rows,
`gdn_recurrence` in `_decode_body`, the grouped expert kernels in the pooled gather, and `qsa_mask`
in `select_device` while `attn_fused` is on.

MTP. A failed `qmv_multi_v`, `grouped_qmv_u2`, `hc_kernel`, `moe_fused` or `moe_kernel` also keeps
MTP off: `init_kernels` records those keys in `Engine.mtp_kernels_off` (`MTP_KERNELS`),
`init_memory_plan` leaves `mtp_on` false (`/stats` shows `mtp.enabled` false), and a lone greedy row
decodes on the solo path with no draft head loaded. The last three count because
`forward_hidden` and `verify2` build every layer through `build_spec` and `build_spec2` without
testing the layer flags, so with one of them off the solo step runs the stock ops and the verify
pass could not match it. A failed `dense_kernel` (an option both paths read), `gdn_fused`,
`gdn_kernel` or an attention kernel (flags both paths read) leaves MTP on.

What the engine reports. It logs `[engine] kernel self-checks vs stock MLX on this device: {...}`
and, when a kernel is off, `[engine] WARNING: ...` from `report`, which names each kernel that
disagreed and each that raised (with its error), the mlx version and the GPU architecture. On a GPU
family other than `TUNED_GPU`, `tuned_note` then says, as information, which Mac the build was tuned
on and what the checks found, and the ready line ends in "WARNING: some kernels are off (see above)"
when any is. With an MTP head and a key of `MTP_KERNELS` off it also warns `[engine] MTP off: ...
failed the load-time check, so the lone greedy row decodes on the solo path`. `/stats` (`Engine.stats`) carries `kernel_checks` and `kernel_check_errors`, and its
`dense_kernel` and `moe_kernel` entries say which dense shapes and which routed-expert kernel run.
Each carries `configs` (the dense table; for the experts, `config_for` of each projection), the
`compute_dtype`, a `config_basis` note on how the table was picked (`CONFIG_BASIS` of
`kernels/qmv_row_blocked.py` and of `kernels/grouped_expert_qmv.py`) and the dotted name of
that module as `source`. The `moe_kernel` entry also gives the `variant` (null while the experts run
stock), with a `variant_basis` (`VARIANT_BASIS`) while the option `moe_kernel_variant` is
`DEFAULT_VARIANT` ("split"), else "selected by option moe_kernel_variant".
`Qwen4ExpForCausalLM.__init__` logs `[engine] fused decode: hc .../..., gdn ..., moe ..., spec ...,
lookahead ON|off`: `hc` counts the layers whose two `Qwen4ExpTextGatedResidual` have `fused_ok`,
`gdn` the GDN blocks' `fused_ok`, `moe` and `spec` the MoE blocks' `fused_ok` and `spec_ok`, and
`lookahead ON` means every layer is `spec_eligible`.

The criterion. A check compares bit for bit (`mx.array_equal`, or `np.array_equal` on ids and masks
copied to the host) wherever the kernel claims bit-identity: a row computed in a batch and alone,
one launch and another, the ids and masks of a stated rule, a copy of stock's bits (`gdn_pre`'s conv
state). Elsewhere (`prefill_mask` aside, which states its own tolerance on the mask) it holds the
kernel to `no_worse_than_stock(got, stock, exact)`: against a float64 result, the kernel's largest
error is at most `FACTOR` = 8 times stock's, plus 4 * 2^-24 of the result's largest magnitude, plus
a `slack` where a check says so. `topk_softmax` on continuous logits must also pick stock's set of
experts, with non-increasing probabilities on tied logits. `no_worse_than_stock` raises ValueError
when stock itself is further than 1e-3 of that magnitude from the float64 result (2^-6 of it when
stock's result is bfloat16, 2^-9 when float16): then the reference is wrong, not the kernel. It also
raises it when the shapes of `got`, `stock` and `exact` differ. Inside a load-time check either
raise is recorded like any other: the kernel is off and the message is in `Checks.errors`. The two
verify checks use bit equality alone: each row of `qmv_multi_v`, and each pair of `grouped_qmv_u2`,
must equal what the solo kernel (`qmv_multi`, `grouped_qmv_x`) gives that row or pair alone, because
`verify2` promises both positions' logits bit-identical to two solo steps'.

Tests. `tests/unit/kernels/test_selfcheck.py` covers `Checks`, `report`, `no_worse_than_stock` and
the checks' inputs; `tests/unit/kernels/test_verify_checks.py` runs the two verify checks' own logic on CPU
stand-ins; `tests/unit/model/test_kernel_checks.py` runs `run_kernel_checks`
without Metal (MLX's CPU build suffices), with the checks replaced by recorders, and compares the
shapes and metadata dtypes the checks assume with the built tiny model's;
`tests/metal/model/test_kernel_checks.py` runs every check on a Mac's GPU;
`tests/unit/model/test_model.py` builds the tiny model without a pool and with
`dense_kernel` and `moe_kernel` at "stock": with every `NAMES` key in `kernels_off` its prefill,
solo decode and batched decode launch no kernel, and each of `hc_kernel`, `gdn_fused`, `gdn_kernel`,
`sdpa_decode_masked`, `attn_fused`, `indexer_fused` and `indexer_prefill`, switched back on alone,
is launched by the path it guards, while the MoE flags (`fused_ok`, `spec_ok`, `shared_fused_ok`,
with `moe_fused` or `gdn_fused` off) are read on a model given a pool stand-in;
`tests/unit/engine/test_engine.py` covers the engine's handling of failed and raising checks and
the MTP switch;
`tests/metal/engine/test_engine_consistency.py` runs the tiny engine with its real checks and
compares tokens across paths, not with stock; `tests/e2e/test_gate.py` requires `/stats` to show
every `kernel_checks.NAMES` key true on a Mac with the real checkpoint (the verify keys only when
an MTP head is present) (it skips unless
`JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_CHECKPOINT` is set).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

U32 = 2.0**-24  # float32 unit roundoff
# How much larger than stock's error a kernel's may be. Two summation orders of the same products
# have errors of the same size; a wrong index, group or formula is orders of magnitude larger.
FACTOR = 8.0


class Checks:
    """The result of each check run so far, and the error of each one that raised."""

    def __init__(self):
        self.results: dict = {}
        self.errors: dict = {}

    def run(self, name: str, check, *args) -> bool:
        """Run `check(*args)`, which says whether kernel `name` agrees with stock; a raise is a
        no."""
        try:
            ok = bool(check(*args))
        except Exception as exc:  # noqa: BLE001  a kernel that cannot run here is switched off, and said so
            ok = False
            self.errors[name] = f"{type(exc).__name__}: {exc}"
        self.results[name] = ok
        return ok

    def run_many(self, names: tuple, check, *args) -> dict:
        """Run `check(*args)`, which returns {name: agrees} for each of `names`; a raise is a no for
        every one of them."""
        try:
            got = check(*args)
            out = {name: bool(got[name]) for name in names}
        except Exception as exc:  # noqa: BLE001  as in `run`
            out = dict.fromkeys(names, False)
            for name in names:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        self.results.update(out)
        return out


def report(checks: dict, mlx_version: str, gpu: str, errors: dict | None = None) -> str | None:
    """A warning in words naming every kernel whose check failed or raised, or None if none did."""
    errors = errors or {}
    differ = [name for name, ok in checks.items() if not ok and name not in errors]
    raised = [f"{name} ({errors[name]})" for name in checks if name in errors]
    if not differ and not raised:
        return None
    parts = []
    if differ:
        parts.append(f"{', '.join(differ)}: disagree with stock MLX on this Mac")
    if raised:
        parts.append(f"{', '.join(raised)}: could not run on this Mac")
    return (
        f"{'; '.join(parts)} (mlx {mlx_version}, {gpu}), so these kernels are off and the stock "
        f"ops run instead. Decode will be slower than the published numbers, which were measured "
        f"with mlx 0.32.2 on a 24 GB M2."
    )


# the Mac this build was tuned and measured on end to end (its kernel tuning tables were measured on an M4 Pro, see
# CONFIG_BASIS), and its GPU family as `mx.device_info()` names it
TUNED_MAC, TUNED_GPU = "a 24 GB M2 Mac", "applegpu_g14g"


def tuned_note(device_name: str, gpu: str, checks_passed: bool | None = None) -> str | None:
    """What a Mac outside the build's GPU family should know, said as information rather than a
    warning: the family the build was tuned on, what the load-time checks found here
    (`checks_passed`; None before they have run, as in `doctor`), and that speed will differ. None
    on the tuned family."""
    if gpu == TUNED_GPU:
        return None
    if checks_passed is None:
        found = "the engine checks every kernel at load, and stock MLX runs for any that disagrees"
    elif checks_passed:
        found = "every kernel check passed"
    else:
        found = "the kernels whose checks failed are off, and stock MLX runs in their place"
    return (
        f"This build was tuned on {TUNED_MAC} ({TUNED_GPU}). It runs on this "
        f"{device_name.removeprefix('Apple ')} ({gpu}): {found}. Speed will differ from the "
        "published numbers."
    )


# ----------------------------------------------------------------------------- the checks' inputs
def normal(shape, seed: int, dtype=mx.float32) -> mx.array:
    """N(0, 1) values of `dtype`, evaluated."""
    a = mx.random.normal(tuple(shape), key=mx.random.key(seed)).astype(dtype)
    mx.eval(a)
    return a


def quantized(shape, seed: int, group_size: int, bits: int = 4):
    """(weight, scales, biases) of a random affine-quantized [N, K] matrix, evaluated: uniform
    random codes, and bfloat16 scales and biases that give each entry mean 0 and variance about
    1/K, so a matvec of unit-variance activations stays O(1). Drawing the codes directly costs a
    few bytes per entry, not a quantization pass, so the checks can use the served shapes."""
    n, k = (int(d) for d in shape)
    kw, ks, kb = mx.random.split(mx.random.key(seed), 3)
    w = mx.random.randint(0, 1 << 16, (n, k * bits // 16), dtype=mx.uint32, key=kw)
    w = w.astype(mx.uint16).view(mx.uint32)
    levels = 2**bits
    step = (12.0 / ((levels * levels - 1) * k)) ** 0.5  # the code's variance is (L^2 - 1) / 12
    s = step * mx.random.uniform(0.75, 1.25, (n, k // group_size), key=ks)
    b = -s * (levels - 1) / 2 + 0.1 * step * mx.random.normal(s.shape, key=kb)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b


class Linear:
    """A random quantized matrix with the attributes the kernels read from the model's `QLinear`;
    `meta_dtype` is the dtype the model stores its scales and biases in."""

    def __init__(self, out_dims, in_dims, seed, group_size, bits=4, meta_dtype=mx.bfloat16):
        self.weight, s, b = quantized((out_dims, in_dims), seed, group_size, bits)
        self.scales, self.biases = s.astype(meta_dtype), b.astype(meta_dtype)
        mx.eval(self.scales, self.biases)
        self.out_dims, self.in_dims = int(out_dims), int(in_dims)
        self.group_size, self.bits = int(group_size), int(bits)

    def stock(self, x: mx.array) -> mx.array:
        """Stock `quantized_matmul` with the metadata in the activations' dtype."""
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales.astype(x.dtype),
            biases=self.biases.astype(x.dtype),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def exact(self, x64: np.ndarray, rows=None) -> np.ndarray:
        """x64 [M, K] @ W^T in float64, W read exactly as stored (scale * q + bias per group), for
        the output `rows` given (default all)."""
        if rows is None:
            w, s, b = self.weight, self.scales, self.biases
        else:
            idx = mx.array(np.asarray(rows))
            w, s, b = self.weight[idx], self.scales[idx], self.biases[idx]
        return exact_matmul(x64, w, s, b, self.group_size, self.bits)


def sample(n: int, count: int = 64) -> np.ndarray:
    """About `count` indices spread evenly over range(n), the first and last included: the outputs a
    check compares with an exact float64 result when computing every one would cost too much."""
    return np.unique(np.linspace(0, n - 1, min(n, count)).round().astype(np.int64))


def f64(a) -> np.ndarray:
    """An MLX or numpy array as float64 numpy (16-bit floats go through float32, exactly)."""
    if isinstance(a, mx.array):
        a = a if a.dtype in (mx.float32, mx.float64) else a.astype(mx.float32)
    return np.array(a, dtype=np.float64)


def exact_matmul(x64: np.ndarray, w, scales, biases, group_size: int, bits: int) -> np.ndarray:
    """x64 [M, K] @ dequant(w)^T in float64, a slab of rows at a time so a wide matrix stays
    small."""
    n, k = int(w.shape[0]), int(scales.shape[-1]) * int(group_size)
    slab = max(1, (1 << 21) // k)
    out = np.empty((x64.shape[0], n), dtype=np.float64)
    for r0 in range(0, n, slab):
        r1 = min(n, r0 + slab)
        q = mx.dequantize(
            w[r0:r1],
            mx.ones_like(scales[r0:r1]),
            mx.zeros_like(biases[r0:r1]),
            group_size=group_size,
            bits=bits,
        )
        groups = f64(q).reshape(r1 - r0, -1, group_size)
        dq = groups * f64(scales[r0:r1])[..., None] + f64(biases[r0:r1])[..., None]
        out[:, r0:r1] = x64 @ dq.reshape(r1 - r0, k).T
    return out


def no_worse_than_stock(got, stock, exact, factor: float = FACTOR, slack: float = 0.0) -> bool:
    """Whether the kernel's largest error against `exact` is at most `factor` times stock's, plus
    four float32 ulps of the result's scale (for when stock happens to be exact) and `slack` (an
    error bound the kernel has and stock does not, stated where it is used).

    Raises if stock itself is not within 1e-3 of the scale of `exact`, or two ulps of a 16-bit
    result (2^-6 for bfloat16, whose quantized matmul rounds partial sums, and 2^-9 for float16):
    then the reference is wrong, not the kernel.
    """
    ref = REFERENCE.get(getattr(stock, "dtype", None), 1e-3)
    got, stock, exact = f64(got), f64(stock), np.asarray(exact, np.float64)
    if not got.shape == stock.shape == exact.shape:
        raise ValueError(f"shapes differ: kernel {got.shape}, stock {stock.shape}, {exact.shape}")
    e_got = float(np.abs(got - exact).max())
    e_stock = float(np.abs(stock - exact).max())
    scale = float(np.abs(exact).max())
    if e_stock > ref * scale:
        raise ValueError(f"stock is {e_stock:.3g} from the exact result (scale {scale:.3g})")
    return e_got <= factor * e_stock + 4 * U32 * scale + slack


# How far from the exact result stock may be, by its result's dtype, before the reference is wrong.
REFERENCE = {mx.bfloat16: 2.0**-6, mx.float16: 2.0**-9}
