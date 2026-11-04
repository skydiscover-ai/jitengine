"""The engine's load-time kernel checks, on MLX's CPU build: what it does with each result. The
checks need Metal, so `run_kernel_checks` is replaced by one that fails every kernel the tiny model
reaches (any launch would fail here) and has one of them raise; the dense kernel passes, and its
table serves no tiny shape."""

import logging

import pytest

mx = pytest.importorskip("mlx.core")


class _Lines(logging.Handler):
    """The message of every record it is handed, in order."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture(scope="module")
def engine(tiny_flashnext):
    """(engine, its startup log) on the tiny checkpoint, built once with the replaced checks."""
    from jitengine_qwen3_8_flash_next_m2.engine import engine as engine_module
    from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck
    from jitengine_qwen3_8_flash_next_m2.model import kernel_checks

    def checks(cfg, act_dtype, dense, moe, variant, mtp=False):
        out = selfcheck.Checks()
        for name in kernel_checks.NAMES:
            out.run(name, lambda name=name: name == "dense_kernel")
        out.run("attn_fused", _no_library)
        return out

    lines, package = _Lines(), logging.getLogger(engine_module.__name__.partition(".")[0])
    level = package.level
    model, pack = tiny_flashnext()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernel_checks, "run_kernel_checks", checks)
        package.addHandler(lines)
        package.setLevel(logging.DEBUG)
        try:
            # F_NOCACHE is a macOS read flag; the tiny checkpoint's embedding table is streamed too
            e = engine_module.create_engine(
                model,
                expert_pack_path=pack,
                log_requests=False,
                expert_io_nocache=False,
                embed_io_nocache=False,
            )
        finally:
            package.removeHandler(lines)
            package.setLevel(level)
    return e, "\n".join(lines.lines)


def _no_library():
    raise RuntimeError("Unable to build metal library from source")


def test_failed_and_raising_checks_show_in_stats(engine):
    from jitengine_qwen3_8_flash_next_m2.model.kernel_checks import NAMES

    e, _ = engine
    stats = e.stats()
    want = {name: name == "dense_kernel" for name in NAMES}
    assert stats["kernel_checks"] == want
    assert stats["kernel_check_errors"] == {
        "attn_fused": "RuntimeError: Unable to build metal library from source"
    }


def test_a_failed_kernel_runs_its_stock_path(engine):
    """The options of a failed check fall back to stock, and the model's fused paths are off; the
    dense kernel, which passed, keeps its option."""
    e, log = engine
    stats = e.stats()
    assert (stats["moe_kernel"]["option"], stats["moe_kernel"]["kernel"]) == ("stock", "stock")
    assert stats["dense_kernel"]["option"] == "rb"
    assert "fused decode: hc 0/4, gdn 0, moe 0, spec 0, lookahead off" in log
    assert e.kernels_off == frozenset(stats["kernel_checks"]) - {"dense_kernel"}


def test_one_line_lists_the_results_and_one_warning_names_each_kernel_off(engine):
    _, log = engine
    lines = log.splitlines()
    listed = [x for x in lines if x.startswith("[engine] kernel self-checks vs stock MLX")]
    assert len(listed) == 1 and "'attn_fused': False" in listed[0]
    (warning,) = [x for x in lines if x.startswith("[engine] WARNING: ")]
    assert "attn_fused (RuntimeError: Unable to build metal library from source)" in warning
    for name in ("moe_kernel", "moe_fused", "hc_kernel", "gdn_fused", "gdn_kernel"):
        assert name in warning
    assert "dense_kernel" not in warning


def test_the_engine_serves_on_the_stock_ops(engine):
    from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenRequest

    e, _ = engine
    out = e.generate(GenRequest(prompt_ids=list(range(3, 27)), max_tokens=4))
    assert len(out.token_ids) == 4


def test_the_kernel_reports_name_their_modules(engine):
    """/stats' `source` of the dense and expert kernel reports names a module that exists."""
    import importlib.util

    e, _ = engine
    stats = e.stats()
    for key in ("dense_kernel", "moe_kernel"):
        assert importlib.util.find_spec(stats[key]["source"]) is not None, stats[key]["source"]


def test_stats_give_the_budget_in_bytes_and_gib(engine):
    st = engine[0].stats()
    assert st["memory_budget_bytes"] == 18 * 10**9  # the measured 18 GB, by default
    assert st["memory_budget_gib"] == pytest.approx(16.7638, abs=1e-4)


# ------------------------------------------------------------------------------ the MTP verify pass
VERIFY = ("qmv_multi_v", "grouped_qmv_u2")
MTP_KERNELS = (*VERIFY, "hc_kernel", "moe_fused", "moe_kernel")


def _plan_with_head(tiny_flashnext, monkeypatch, failing=(), head=True):
    """(engine, startup log) after `init_kernels` and `init_memory_plan` on the tiny checkpoint,
    with an MTP head reported present or not, the checks replaced by ones that pass except for the
    kernels in `failing`, and `mtp` (what `run_kernel_checks` was told) recorded as `e.told_mtp`."""
    from jitengine_qwen3_8_flash_next_m2.engine import engine as engine_module
    from jitengine_qwen3_8_flash_next_m2.kernels import selfcheck
    from jitengine_qwen3_8_flash_next_m2.model import kernel_checks, mtp

    told = []

    def checks(cfg, act_dtype, dense, moe, variant, mtp=False):
        told.append(mtp)
        out = selfcheck.Checks()
        for name in kernel_checks.NAMES:
            if mtp or name not in VERIFY:
                out.run(name, lambda name=name: name not in failing)
        return out

    monkeypatch.setattr(kernel_checks, "run_kernel_checks", checks)
    monkeypatch.setattr(mtp, "present", lambda cfg=None: head)
    monkeypatch.setattr(mtp, "resident_bytes", lambda: 1 << 20)
    model, pack = tiny_flashnext()
    options = {"expert_pack_path": pack, "log_requests": False}
    lines, package = _Lines(), logging.getLogger(engine_module.__name__.partition(".")[0])
    level = package.level
    package.addHandler(lines)
    package.setLevel(logging.DEBUG)
    try:
        e = engine_module.Engine.__new__(engine_module.Engine)
        e.init_options(model, options)
        e.init_kernels(model)
        e.init_memory_plan(model, options)
    finally:
        package.removeHandler(lines)
        package.setLevel(level)
    e.told_mtp = told
    return e, "\n".join(lines.lines)


def test_mtp_stays_on_when_the_verify_kernels_pass(tiny_flashnext, monkeypatch):
    e, log = _plan_with_head(tiny_flashnext, monkeypatch)
    assert e.told_mtp == [True]
    assert all(e.kernel_checks[name] for name in VERIFY)
    assert e.mtp_kernels_off == () and e.mtp_on
    assert "MTP off" not in log


@pytest.mark.parametrize(
    "failing",
    [
        ("qmv_multi_v",),
        ("grouped_qmv_u2",),
        VERIFY,
        ("hc_kernel",),
        ("moe_fused",),
        ("moe_kernel",),
    ],
)
def test_a_failed_check_of_a_kernel_the_verify_pass_launches_keeps_mtp_off(
    tiny_flashnext, monkeypatch, failing
):
    """The lone row then decodes on the solo path, which is exact; the other kernels are as they
    were (a failed `moe_kernel` also turns its option to stock, as without MTP), and /stats shows
    the failed check."""
    e, log = _plan_with_head(tiny_flashnext, monkeypatch, failing)
    assert e.kernels_off == frozenset(failing)
    assert e.mtp_kernels_off == tuple(n for n in MTP_KERNELS if n in failing)
    assert not e.mtp_on and e.mtp_bytes == 0
    assert (e.moe_kernel, e.dense_kernel) == (
        "stock" if "moe_kernel" in failing else "grouped",
        "rb",
    )
    assert {name for name, ok in e.kernel_checks.items() if not ok} == set(failing)
    assert sum("[engine] MTP off: " in line for line in log.splitlines()) == 1
    assert ", ".join(e.mtp_kernels_off) + " failed the load-time check" in log


@pytest.mark.parametrize("failing", [("dense_kernel",), ("gdn_kernel",), ("attn_fused",)])
def test_a_failed_check_of_a_kernel_with_a_flag_or_option_leaves_mtp_on(
    tiny_flashnext, monkeypatch, failing
):
    """These kernels' stock paths are what the solo step and the verify pass both run, so MTP
    stays on."""
    e, log = _plan_with_head(tiny_flashnext, monkeypatch, failing)
    assert e.kernels_off == frozenset(failing) and e.mtp_kernels_off == ()
    assert e.mtp_on and "MTP off" not in log


def test_without_a_head_the_verify_kernels_are_not_checked_and_mtp_stays_off(
    tiny_flashnext, monkeypatch
):
    e, log = _plan_with_head(tiny_flashnext, monkeypatch, head=False)
    assert e.told_mtp == [False]
    assert not set(VERIFY) & set(e.kernel_checks)
    assert not e.mtp_on and "MTP off" not in log
