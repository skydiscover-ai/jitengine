"""Load-time checks that each exact custom kernel still matches its reference on this device: the
stock MLX op it replaces, or a kernel already held to it (`gdn_recurrence` for `gdn_rows`, the
exact expert kernels for `moe_v4` and `rows_v4`, `prescale` for `fused_norm`'s pre-scale).

Other GPU generations may dispatch a different stock kernel, so a kernel that disagrees on random
inputs is switched off, and its call site takes its next path: the stock op, or another custom
kernel. A check that raises (a kernel that does not compile or run here) also switches its kernel
off, and `Checks` keeps the error to report it. Paths below are relative to the package,
`src/jitengine_qwen3_6_35b_a3b_m2/`, except `pyproject.toml` and `tests/`, which are the engine
root's. `README.md` in this directory tables each kernel's check, its result and its test.

Call sites and MLX version. The call sites are `model/attention.py`, `model/gdn.py`,
`model/moe.py`, `model/model.py`, `linear.py` and `sampler.py`. `pyproject.toml` pins `mlx==0.32.2`
on Apple silicon, and several kernels copy that version's stock arithmetic or dispatch rules (for
example `single_pass` in `kernels/sdpa_decode.py`, `eligible` in `kernels/qmm_prefill.py`,
`kernels/qmv_mrow.py`, `kernels/rms_norm_prescale.py`); `kernels/expert_solo.py` and
`kernels/gather_qmm_prefill.py` build their source from the installed MLX headers
(`kernels/expert_rows.py` and `kernels/qmm_prefill.py` reuse those headers), so another MLX version
is a case for these checks. The published numbers that `report` cites are the engine README's
Results, measured on an Apple M2 Mac (24 GB, 3-bit) with mlx 0.32.2.

How the checks run. `Engine.init_kernels` (`engine/engine.py`) runs them once the model is built,
each through `Checks`: `run(name, check, *args)` records whether the check returned true, and
`run_many(names, check, *args)` records one result per name from a check that returns a dict. A
check that raises records False for each of its names and keeps "Type: message" in `Checks.errors`.
A check compares with `mx.array_equal` (`QLinear.calibrate_solo` and `calibrate_mrow` use
`mx.all(y == ref)`) on inputs it draws, bf16 activations (`check_gdn_rows` also passes fp32 g and
states, as the model does): "bit-identical" means equal values, so +0 and -0 count as equal.
`check_sdpa` compares past stock's two-pass switch (`SDPA_LENGTHS`) with `decode` calling stock
itself, so it catches a switch set too late, not too early, and covers only the config's head dim.

How a failing kernel is switched off. The results become a `KernelChoice`
(`model/kernel_choice.py`), whose fields are the flags the call sites test.
`exact_sdpa`, `exact_topk`, `sdpa_rows` and `gdn_rows` set the field of that name; `dense_v4` sets
`dense_prescaled`; `radix_sort` sets `Engine.radix_sort`, which the engine passes to each `Sampler`
as `radix`. One `run_many` of `check_experts` gives `exact_experts`, `moe_v4`, `rows_v4` and
`seg_prefill`: they set `exact_experts`, `moe_prescaled`, `rows_exact` and `seg_prefill`, and
`moe_v4` with `rows_v4` sets `rows_prescaled`. In offload mode all of these but `seg_prefill` are
off (offload prefill still runs `seg_gather_qmm`), and so is `fused_norm`, which is set from its
result only together with `moe_prescaled`. `Qwen3_5MoeForCausalLM.use_kernel_choice`
(`model/model.py`) hands the choice to every decoder layer, block and MoE block and
sets each `QLinear.prescaled` from `dense_prescaled`. The engine sets every field from its result,
False when the check failed or raised, so that kernel is never selected (until then every field is
False).

What a call site falls back to. `decode` runs row by row where `decode_rows` does not (`sdpa_rows`
off with `exact_sdpa` on, more than 8 rows, a row that `supported` declines), and `gdn_recurrence`
row by row when `gdn_rows` is off or `supported_rows` declines (outside 2 to 8 rows, or a state that
is not float32). Of the expert kernels, the prescaled solo pair (`moe_prescaled`, with the shared
expert's gate/up on `qmv`) gives way to the exact solo pair (`exact_experts`) and that to stock, and
the prescaled rows kernels (`rows_prescaled`) give way to `experts_rows_exact` (field `rows_exact`,
set from `rows_v4`, whose check runs only the prescaled rows kernels) and that to stock.
`qmv_fast_mrow` runs, on the shapes whose check passed, on one-row calls and batched-decode rows
wherever `qmv` does not (`dense_v4` off, or a shape that fails `supports` or its
`calibrate_prescaled`).

Per-shape dense checks, in `linear.py`. `qmv_fast_mrow` and `qmm_t` have no `KernelChoice`
field. While the model is built, `Qwen3_5MoeForCausalLM.calibrate` runs `calibrate_row_limits`,
which runs `QLinear.calibrate` (stock's safe row group), `QLinear.calibrate_solo` (also on the fused
one-row buffers) and `QLinear.calibrate_mrow`, each once per distinct shape.
`QLinear.calibrate_prescaled` (the prescaled GEMV, `qmv`) runs only once `dense_v4` has passed, on a
weight shape's first call that the kernel could take; for the shapes solo decode reaches, that call
is in the engine's two warm-up requests (`Engine._warmup`), which run after `init_kernels`.
`QLinear.calibrate_pf` (`qmm_t`) runs on a shape's first call with `eligible(...)` true, so at the
first long enough prompt, not at startup; it compares at M = 192, 231, 256 and 300 where `eligible`
holds, so a shape that none of them makes eligible (n of 832 or less) passes uncompared, and one
with n of 896 to 1024 is compared only at M = 300. `qmm_t` serves both from m > 2^18 / n rows, which
the default 256-token prefill chunk never reaches and a larger one can (`prefill_chunk_tokens` in a
profile, the engine option `prefill_chunk`). A shape whose check differs or raises keeps that kernel
off, and `dense_kernel_report()` lists it. `qmv` and `qmv_fast_mrow` have no row cap and `max_slots`
is only required to be positive, so a value above 8 (`qmv`) or 16 (`qmv_fast_mrow`) serves row
counts that no check compares.

What the engine reports. It logs `[engine] kernel self-checks vs stock MLX on this device: {...}`.
When a check failed, `report` builds a warning, which the engine logs as `[engine] WARNING: ...`: it
names each failing result as "not bit-identical to stock MLX on this Mac" (`moe_v4` and `rows_v4`
fail whenever `exact_experts` does, and `gdn_rows` is compared with `gdn_recurrence`), each result
whose check "could not run on this Mac" (with its error), and each weight shape a load-time dense
check (`DENSE_AT_LOAD`: the single- and multi-row ones) turned a kernel off for. On a GPU family
other than `TUNED_GPU`, `tuned_note` then says, as information, which Mac the build was tuned on and
what the checks found. The ready line ends in "WARNING: some kernels are off (see above)" when the
warning was logged. None of them reports `calibrate_prescaled` or
`calibrate_pf`, which run later: a shape they turn off appears only in `/stats`. `/stats` returns
`kernel_checks` (every result), `kernel_check_errors` (the error of each check that raised) and
`dense_kernels` (`dense_kernel_report()`): for each of `single_row`, `multi_row`, `prescaled` and
`prefill`, how many shapes were asked about (`shapes`), those the kernel runs on (`on`), those where
it differs from stock (`differ`) and those whose check raised, with its error (`raised`);
`single_row` also lists `below_min_bytes` (shapes under 64 MiB, declined without a comparison) and
`rps_nsg`, `multi_row` lists `mb` and `rps_nsg`, and `row_groups` gives each shape's safe row group.

Paths. Prefill is `forward`, `forward_layer_major` or `forward_layer_major_multi` on a chunk of
several tokens; a chunk of one token runs exactly as solo decode does. `seg_gather_qmm` serves the
sorted routed experts of prefill: resident, offload (`_experts_pooled`) and merged across chunks
(`prefill_experts_merged`, resident only: the options `merge_prefill_experts`, `layer_major_prefill`
and `group_prefill`); decode would need at least 4 pairs per expert for it, about 128 rows here, and
`max_slots` defaults to 8. The offload speculative gather (`speculative_gather`) changes when the
stock gather is dispatched, not which kernel runs (`tests/metal/engine/test_engine_speculative.py`
holds its tokens and scores to the synchronous path's).

Tests. From the engine root, `pip install '.[dev]'` and
`pytest tests/unit/kernels tests/metal/kernels`: a test that launches a kernel takes the
`requires_mlx` fixture (`tests/conftest.py`), which skips it unless MLX runs on Metal, and those
under `tests/unit/kernels` take `requires_mlx_any` and also run on MLX's CPU build
(`pip install 'mlx[cpu]==0.32.2'` on Linux). `tests/unit/kernels/test_selfcheck.py` covers `Checks`
and `report`; `tests/metal/kernels/test_selfcheck.py` runs the checks on the device; on the real
checkpoint `tests/e2e/test_gate.py::test_greedy_512_hash_kernel_checks_and_decode` asserts that
every `kernel_checks` entry is true and that no dense kernel differs from stock or raised on any
shape, and expects that Mac's greedy-512 hash unless `JITENGINE_QWEN3_6_35B_A3B_M2_E2E_SHA` names
another. The engine tests under `tests/metal/engine/` run the kernels their shapes admit end to end
and compare tokens across paths (batched with solo, a prefix hit with a cold run, offload with
resident), not with stock.
"""

from __future__ import annotations

import mlx.core as mx


def random_quantized(shape, bits, seed):
    """Random bf16 weights of `shape`, quantized (affine, group 64): (weight, scales, biases)."""
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


class Checks:
    """The results of the checks run so far, and the error of each one that raised."""

    def __init__(self):
        self.results: dict = {}
        self.errors: dict = {}

    def run(self, name: str, check, *args) -> bool:
        """Run `check`, which says whether kernel `name` matches its stock op on this device."""
        try:
            ok = bool(check(*args))
        except Exception as exc:  # noqa: BLE001  a kernel that cannot run here is switched off, and said so
            ok = False
            self.errors[name] = f"{type(exc).__name__}: {exc}"
        self.results[name] = ok
        return ok

    def run_many(self, names: tuple, check, *args) -> dict:
        """Run `check`, one check for several kernels that share its inputs: it returns whether each
        of `names` matches its stock op, and so does this."""
        try:
            got = check(*args)
        except Exception as exc:  # noqa: BLE001  as in `run`
            got = dict.fromkeys(names, False)
            for name in names:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        for name in names:
            self.results[name] = bool(got[name])
        return {name: self.results[name] for name in names}


# the dense kernels whose checks run per weight shape at load (linear.dense_kernel_report)
DENSE_AT_LOAD = {
    "single_row": "the single-row dense kernel",
    "multi_row": "the multi-row dense kernel",
}


def report(
    checks: dict, mlx_version: str, gpu: str, errors: dict | None = None, dense: dict | None = None
) -> str | None:
    """A warning in words naming every kernel whose check failed, or None when none did; `dense`
    adds each weight shape a per-shape check turned a dense kernel off for."""
    errors = errors or {}
    differ = [name for name, ok in checks.items() if ok is False and name not in errors]
    raised = [f"{name} ({errors[name]})" for name in checks if name in errors]
    differ, raised = _add_dense_failures(differ, raised, dense)
    if not differ and not raised:
        return None
    parts = _failure_parts(differ, raised)
    return (
        f"{'; '.join(parts)} (mlx {mlx_version}, {gpu}), so these kernels and the ones built on them "
        f"are off and the stock ops run instead. Decode will be slower than the published numbers, "
        f"which were measured with mlx 0.32.2."
    )


def _add_dense_failures(differ: list[str], raised: list[str], dense: dict | None) -> tuple:
    """Add each weight shape a per-shape check turned a dense kernel off for to the two failure lists."""
    for kind, label in DENSE_AT_LOAD.items() if dense else ():
        differ += [f"{label} on {shape}" for shape in dense[kind]["differ"]]
        raised += [f"{label} on {shape} ({err})" for shape, err in dense[kind]["raised"].items()]
    return differ, raised


def _failure_parts(differ: list[str], raised: list[str]) -> list[str]:
    """The warning's clauses: the kernels that differ from stock MLX and the ones that could not run."""
    parts = []
    if differ:
        parts.append(f"{', '.join(differ)}: not bit-identical to stock MLX on this Mac")
    if raised:
        parts.append(f"{', '.join(raised)}: could not run on this Mac")
    return parts


# the Mac this build was tuned and measured on, and its GPU family as `mx.device_info()` names it
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
