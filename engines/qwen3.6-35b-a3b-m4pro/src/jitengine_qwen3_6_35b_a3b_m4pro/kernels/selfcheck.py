"""Load-time checks that each exact custom kernel still matches its reference on this device: the
stock MLX op it replaces, or a kernel already held to it (`gdn_recurrence` for `gdn_rows`, the exact
expert kernels for `moe_v4` and `rows_v4`, `prescale` for `fused_norm`'s pre-scale).

Other GPU generations may dispatch a different stock kernel, so a kernel that disagrees on random
inputs is switched off, and its call site takes its next path: the stock op, or another custom
kernel. A check that raises (a kernel that does not compile or run here) also switches its kernel
off, and `Checks` keeps the error to report it. Paths below are relative to the package,
`src/jitengine_qwen3_6_35b_a3b_m4pro/`, except `pyproject.toml` and `tests/`, which are the engine
root's. `README.md` in this directory tables each kernel's check, its result and its test.

Which kernels run. The exact default runs the exact kernels at the served width, 4 bits. The
3-bit-only kernels (`qmv`, `qmm_t`, the prescaled and rows expert kernels, `seg_gather_qmm`) are
never selected by a 4-bit model: `run_load_checks` does not run their checks (`dense_v4` needs
`covers(bits, group_size)`; `expert_checks(bits)` names only `exact_experts` outside 3 bits) and
`/stats` `kernel_checks` does not list them. `--profile relaxed` adds `kernels/relaxed_qmm.py` and
`kernels/topk_fast.py` to the speculative verify path; they claim no bit-identity and have no
check. The load-time checks and `KernelChoice` are the same in both profiles.

Call sites and MLX version. The call sites are `model/attention.py`,
`model/gdn.py`, `model/moe.py`, `model/model.py`,
`linear.py`, `sampler.py`, `speculative/verify.py` and `speculative/spec_sample.py`.
`pyproject.toml` pins `mlx==0.32.2` on Apple silicon, and several kernels copy that version's stock
arithmetic or dispatch rules (for example `mlx_uses_2pass` in `kernels/sdpa_decode.py`,
`kernels/qmv_mrow.py`, `kernels/rms_norm_prescale.py`); `kernels/expert_solo.py` and
`kernels/gather_qmm_prefill.py` build their source from the installed MLX headers
(`kernels/expert_rows.py` reuses those headers), so another MLX version is a case for these
checks. The published numbers that `report` cites are the engine README's Results, measured on a
48 GB M4 Pro (4-bit) with mlx 0.32.2.

How the checks run. `Engine.init_kernels` (`engine/engine.py`) runs them once the model is built,
through `run_load_checks`, each through `Checks`: `run(name, check, *args)` records whether the
check returned true, and `run_many(names, check, *args)` records one result per name from a check
that returns a dict. A check that raises records False for each of its names and keeps
"Type: message" in `Checks.errors`. A check compares with `mx.array_equal` (`QLinear.calibrate_solo`
and `calibrate_mrow` use `mx.all(y == ref)`, `calibrate_rl3` the `uint16` views) on inputs it
draws, bf16 activations (`check_gdn_rows` also passes fp32 g and states, as the model does):
"bit-identical" means equal values, so +0 and -0 count as equal (but not in `calibrate_rl3`).
`check_sdpa` draws lengths on both sides of the first two-pass switch (`SDPA_LENGTHS`) and covers
only the config's head dim. In offload mode the expert checks do not run.

How a failing kernel is switched off. The results become a `KernelChoice`
(`model/kernel_choice.py`), whose fields are the flags the call sites test.
`exact_sdpa`, `exact_topk`, `sdpa_rows`, `gdn_rows` and `gdn_exact` set the field of that name;
`fused_router` sets its field only together with `exact_topk`; `dense_v4` sets `dense_prescaled`;
`radix_sort` sets `Engine.radix_sort`, which the engine passes to each `Sampler` as `radix`, and
`sampler_tail` sets `Engine.fused_tail`. One `run_many` of `check_experts` gives `exact_experts`
and, at 3 bits, `moe_v4`, `rows_v4` and `seg_prefill`: they set `exact_experts`, `moe_prescaled`,
`rows_exact` and `seg_prefill`, and `moe_v4` with `rows_v4` sets `rows_prescaled`. In offload mode
all of these are off, and so is `fused_norm`, which is set from its result in resident mode (at
3 bits only together with `moe_prescaled`). `Qwen3_5MoeForCausalLM.use_kernel_choice`
(`model/model.py`) hands the choice to every decoder layer, block and MoE block and
sets each `QLinear.prescaled` from `dense_prescaled`. The engine sets every field from its result,
False when the check failed or raised, so that kernel is never selected (until then every field is
False).

What a call site falls back to. `decode` runs row by row where `sdpa_decode_rows` does not (`sdpa_rows`
off with `exact_sdpa` on, more than 8 rows, a row that `supported` declines), `gdn_recurrence` row by
row when `gdn_rows` is off or `supported_rows` declines, and the composed MLX chain runs where
`gdn_exact` is off. `router_topk` gives way to `topk` and that to stock. Of the expert kernels at
4 bits, the exact kernels (`exact_experts`) give way to stock. `qmv_fast_mrow` and
`qmv_fast_mrow_rl4` have no `KernelChoice` field: each is chosen per weight shape by the dense
checks below, `qmv_fast_mrow_rl4` first for 2 or more rows of a 4-bit weight, and stock
`mx.quantized_matmul` where neither is on.

Per-shape dense checks, in `linear.py`. While the model is built,
`Qwen3_5MoeForCausalLM.calibrate` runs `calibrate_row_limits`, which runs `QLinear.calibrate_solo`
and `QLinear.calibrate_mrow` (single- and multi-row `qmv_fast_mrow`) and, for 4-bit weights,
`QLinear.calibrate_rl3` (`qmv_fast_mrow_rl4`), each once per distinct shape. A 3-bit model also runs
`calibrate_prescaled` (the prescaled GEMV, once `dense_v4` has passed) and `calibrate_pf` (`qmm_t`)
on a shape's first call, which a 4-bit model never reaches. A shape whose check differs or raises
keeps that kernel off, and `dense_kernel_report()` lists it. `qmv_fast_mrow` and the rl4 kernel have
no row cap, and `max_slots` is only required to be positive, so a value above 16 serves row counts
that no check compares.

What the engine reports. It logs `[engine] kernel self-checks vs stock MLX on this device: {...}`.
When a check failed, `report` builds a warning, which the engine logs as `[engine] WARNING: ...`: it
names each failing result as "not bit-identical to stock MLX on this Mac", each result whose check
"could not run on this Mac" (with its error), and each weight shape a load-time dense check
(`DENSE_AT_LOAD`: `single_row`, `multi_row`, `rl3`) turned a kernel off for. On a GPU family other
than `TUNED_GPU`, `tuned_note` then says, as information, which Mac the build was tuned on and what
the checks found. The ready line ends in "WARNING: some kernels are off (see above)" when the
warning was logged. None of them reports a check that runs later (`calibrate_prescaled`,
`calibrate_pf`): a shape they turn off appears only in `/stats`. `/stats` returns `kernel_checks`
(every result), `kernel_check_errors` (the error of each check that raised) and `dense_kernels`
(`dense_kernel_report()`): for each dense kernel, how many shapes were asked about, those it runs
on, those where it differs from stock and those whose check raised, with its error.

Tests. From the engine root, `pytest tests/unit/kernels tests/metal/kernels`: a test that launches a
kernel takes the `requires_mlx` fixture (`tests/conftest.py`), which skips it unless MLX runs on
Metal, and those under `tests/unit/kernels` take `requires_mlx_any` and also run on MLX's CPU build
(`pip install 'mlx[cpu]==0.32.2'` on Linux). `tests/unit/kernels/test_selfcheck.py` covers `Checks`
and `report`, `tests/unit/engine/test_kernel_checks.py` which checks run at 3 and at 4 bits, and
`tests/metal/kernels/test_selfcheck.py` runs the checks on the device. On the real checkpoint
`tests/e2e/test_gate.py::test_greedy_512_hash_kernel_checks_and_decode` asserts that every
`kernel_checks` entry is true and that no dense kernel differs from stock or raised on any shape,
and expects the M4 Pro's greedy-512 hash unless `JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_SHA` names
another. The engine tests under `tests/metal/engine/` run the kernels their shapes admit end to end
and compare tokens across paths (batched with solo, a prefix hit with a cold run, offload with
resident), not with stock.
"""

from __future__ import annotations

import mlx.core as mx


def random_quantized(shape, bits, seed, group_size=64):
    """Random bf16 weights of `shape`, quantized (affine, `bits` wide, groups of `group_size`):
    (weight, scales, biases)."""
    w = (mx.random.normal(shape, key=mx.random.key(seed)) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=group_size, bits=bits)
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


# the dense kernels whose checks run per weight shape at load (quant.dense_kernel_report)
DENSE_AT_LOAD = {
    "single_row": "the single-row dense kernel",
    "multi_row": "the multi-row dense kernel",
    "rl3": "the 4-bit multi-row dense kernel",
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
TUNED_MAC, TUNED_GPU = "a 48 GB M4 Pro", "applegpu_g16s"


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
