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
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as engine_module
    from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck
    from jitengine_qwen3_8_flash_next_m4pro.model import kernel_checks

    def checks(cfg, act_dtype, dense, moe, variant):
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
            # F_NOCACHE is a macOS read flag
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
    from jitengine_qwen3_8_flash_next_m4pro.model.kernel_checks import NAMES

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
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenRequest

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
    assert st["memory_budget_bytes"] == 40 * 10**9  # the measured 40 GB, by default
    assert st["memory_budget_gib"] == pytest.approx(37.2529, abs=1e-4)
