"""The command line, run in process (`main([...])`) so coverage sees it: help and usage errors,
the serve flags and the engine options they become, `inspect` on the tiny checkpoint, `doctor`
with the device replaced, `status` with `urlopen` replaced, `pack` with the builders replaced, and
the budget policy's log lines. Nothing here needs Metal, downloads anything or starts a server."""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import sys
import urllib.error
import urllib.request

import pytest

from jitengine_qwen3_8_flash_next_m4pro import cli

GIB = cli.GIB
PROG = "jitengine-qwen3.8-flash-next-m4pro"
COMMANDS = ["doctor", "inspect", "pack", "serve", "status"]


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """`serve` configures the package's logger; put it back so other tests see the default."""
    pkg = logging.getLogger("jitengine_qwen3_8_flash_next_m4pro")
    state = (list(pkg.handlers), pkg.level, pkg.propagate)
    yield
    pkg.handlers[:] = state[0]
    pkg.setLevel(state[1])
    pkg.propagate = state[2]


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[tuple[str, str]] = []

    def emit(self, record):
        self.lines.append((record.levelname, record.getMessage()))


@pytest.fixture
def cli_log():
    """The (level, message) of what the CLI's logger is handed, whatever the package's logger does."""
    handler, logger = _Lines(), logging.getLogger(cli.__name__)
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler.lines
    logger.removeHandler(handler)
    logger.setLevel(level)


def _run(argv, capsys):
    rc = cli.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _exit_code(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return e.value.code, capsys.readouterr()


# ---------------------------------------------------------------- help, version, usage errors
def test_version_prints_the_package_version(capsys):
    code, cap = _exit_code(["--version"], capsys)
    assert code == 0 and cap.out.startswith(PROG + " ")


def test_no_command_prints_the_help_and_succeeds(capsys):
    rc, out, _ = _run([], capsys)
    assert rc == 0 and all(cmd in out for cmd in COMMANDS)


@pytest.mark.parametrize("cmd", COMMANDS)
def test_each_command_has_help_that_names_it(cmd, capsys):
    code, cap = _exit_code([cmd, "--help"], capsys)
    assert code == 0 and f"{PROG} {cmd}" in cap.out


def test_serve_help_lists_every_option_that_becomes_an_engine_option(capsys):
    _, cap = _exit_code(["serve", "--help"], capsys)
    hidden = {
        "moe_kernel_variant",
        "prefetch_lookahead",
        "prefetch_reserve",
        "transient_prefill",
        "moe_sync_each_layer",
    }
    for name in cli.SERVE_OPTIONS + cli.SERVE_SWITCHES:
        flag = "--" + name.replace("_", "-")
        assert (flag in cap.out) == (name not in hidden), flag
    assert "--wave-chunk" not in cap.out
    for flag in (
        "--memory-budget-gib",
        "--memory-budget-gb",
        "--force",
        "--log-level",
    ):
        assert flag in cap.out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["serve", "--model-path", "m", "--compute-dtype", "fp8"], "invalid choice"),
        (["serve", "--model-path", "m", "--moe-kernel", "fast"], "invalid choice"),
        (["serve", "--model-path", "m", "--dense-kernel", "fast"], "invalid choice"),
        (["serve", "--model-path", "m", "--moe-kernel-variant", "x"], "invalid choice"),
        (["serve", "--model-path", "m", "--residual-dtype", "x"], "invalid choice"),
        (["serve", "--model-path", "m", "--log-level", "loud"], "invalid choice"),
        (["serve", "--model-path", "m", "--max-slots", "1.5"], "invalid int value"),
        (["serve", "--model-path", "m", "--port", "http"], "invalid int value"),
        (["serve", "--model-path", "m", "--kv-budget-mb", "lots"], "invalid float value"),
        (["serve", "--model-path", "m", "--memory-budget-gb", "-3"], "non-negative"),
        (["inspect", "--model-path", "m", "--memory-budget-gib", "abc"], "not a number"),
        (["serve"], "--model-path"),
        (["inspect"], "--model-path"),
        (["pack"], "--model-path"),
        (["bogus"], "invalid choice"),
    ],
)
def test_usage_errors_exit_2_and_say_why(argv, message, capsys):
    code, cap = _exit_code(argv, capsys)
    assert code == 2 and message in cap.err


def test_serve_defaults_leave_every_engine_option_unset():
    a = cli.build_parser().parse_args(["serve", "--model-path", "m"])
    assert (a.host, a.port, a.force, a.log_level) == ("127.0.0.1", 8080, False, "info")
    assert all(getattr(a, name) is None for name in cli.SERVE_OPTIONS)
    assert not any(getattr(a, name) for name in cli.SERVE_SWITCHES)
    assert a.expert_pack_path is None and a.ple_pack_path is None


def test_wave_chunk_is_an_alias_of_the_prefix_grid():
    for flag in ("--prefix-grid-tokens", "--wave-chunk"):
        a = cli.build_parser().parse_args(["serve", "--model-path", "m", flag, "512"])
        assert a.prefix_grid_tokens == 512


def test_both_budget_units_are_refused_with_a_message(capsys):
    argv = ["serve", "--model-path", "m", "--memory-budget-gib", "1", "--memory-budget-gb", "1"]
    a = cli.build_parser().parse_args(argv)
    with pytest.raises(SystemExit) as e:
        cli._requested_budget(a)
    assert e.value.code == 2
    assert "not both" in capsys.readouterr().err


# ---------------------------------------------------------------- the budget policy
def test_the_default_budget_is_62_percent_of_memory_to_a_hundredth_of_a_gib(monkeypatch):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 64 * GIB)
    assert cli._default_budget_bytes() == int(round(64 * 0.62, 2) * GIB)
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 36 * GIB)
    assert cli._default_budget_bytes() == int(round(36 * 0.62, 2) * GIB)


def test_a_low_budget_warns_how_little_is_left_for_the_expert_pool(monkeypatch, cli_log):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 48 * GIB)
    assert cli._budget_bytes(12 * 10**9, force=False) == 12 * 10**9
    levels = {level: msg for level, msg in cli_log}
    assert "low hit rate" in levels["WARNING"] and "about 0.93 GiB" in levels["WARNING"]
    assert (
        "engine budget 11.18 GiB" in levels["INFO"] and "unified memory 48.00 GiB" in levels["INFO"]
    )


def test_a_budget_below_the_model_floor_leaves_nothing_for_the_pool(monkeypatch, cli_log):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 48 * GIB)
    cli._budget_bytes(5 * 10**9, force=False)
    assert any("about 0.00 GiB for the expert pool" in msg for _, msg in cli_log)


def test_forcing_a_budget_over_80_percent_warns(monkeypatch, cli_log):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 48 * GIB)
    assert cli._budget_bytes(44 * 10**9, force=True) == 44 * 10**9
    assert any("(--force)" in msg for level, msg in cli_log if level == "WARNING")


def test_a_comfortable_budget_only_logs_the_plan(monkeypatch, cli_log):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 48 * GIB)
    cli._budget_bytes(30 * 10**9, force=False)
    assert [level for level, _ in cli_log] == ["INFO"]


def test_gib_formats_none_and_sizes():
    assert cli._gib(None) == "?" and cli._gib(GIB) == "1.00 GiB"
    assert cli._unified_memory_bytes() > 0


# ---------------------------------------------------------------- inspect
@pytest.fixture
def checkpoint(tiny_flashnext):
    pytest.importorskip("mlx.core")  # building the tiny checkpoint needs MLX
    return tiny_flashnext()[0]


def test_inspect_prints_the_checkpoint_and_the_plan(checkpoint, capsys):
    rc, out, _ = _run(["inspect", "--model-path", checkpoint, "--memory-budget-gib", "2"], capsys)
    assert rc == 0
    assert out.splitlines()[0] == f"{checkpoint}: 4 layers, 32 expert records"
    assert "budget 2.00 GiB" in out and "32 of 32 experts" in out


def test_inspect_json_gives_the_budget_in_bytes_and_the_plan(checkpoint, capsys):
    rc, out, _ = _run(
        ["inspect", "--model-path", checkpoint, "--json", "--memory-budget-gb", "3"], capsys
    )
    rep = json.loads(out)
    assert rc == 0 and rep["budget_bytes"] == 3 * 10**9
    plan = rep["plan"]
    assert (plan["num_records"], plan["pool_slots"], plan["max_slots"]) == (32, 32, 8)
    assert plan["budget_bytes"] == 3 * 10**9 and plan["compute_dtype"] == "float32"
    assert plan["moe_kernel"] == "grouped" and plan["dense_kernel"] == "rb"


def test_inspect_without_a_budget_uses_the_default_one(checkpoint, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 8 * GIB)
    _, out, _ = _run(["inspect", "--model-path", checkpoint, "--json"], capsys)
    assert json.loads(out)["budget_bytes"] == cli._default_budget_bytes()


def test_a_smaller_budget_gives_a_smaller_expert_pool(checkpoint, capsys):
    pools = {}
    for gib in ("0.5", "2"):
        _, out, _ = _run(
            ["inspect", "--model-path", checkpoint, "--json", "--memory-budget-gib", gib], capsys
        )
        pools[gib] = json.loads(out)["plan"]["pool_slots"]
    assert 0 < pools["0.5"] <= pools["2"] == 32


def test_inspect_says_when_a_budget_does_not_fit_and_exits_1(checkpoint, capsys):
    rc, out, err = _run(
        ["inspect", "--model-path", checkpoint, "--memory-budget-gb", "0.001"], capsys
    )
    assert rc == 1 and out == "" and "does not fit" in err and err.startswith(PROG)


# ---------------------------------------------------------------- doctor
@pytest.fixture
def fake_mlx(monkeypatch):
    """`mx.device_info` and `mx.metal.is_available` replaced; returns the dict to edit."""
    mx = pytest.importorskip("mlx.core")
    state = {"info": {"device_name": "Apple M4 Pro", "architecture": "applegpu_g16s",
                      "max_recommended_working_set_size": 36 * GIB}, "metal": True}  # fmt: skip
    monkeypatch.setattr(mx, "device_info", lambda: dict(state["info"]))
    monkeypatch.setattr(mx.metal, "is_available", lambda: state["metal"])
    return state


def test_doctor_without_metal_is_unsupported_with_the_reason(fake_mlx, capsys):
    fake_mlx["metal"] = False
    rc, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 1 and rep["supported"] is False
    assert rep["metal"] == {"error": "Metal is not available"}
    assert rep[PROG] and rep["python"] == sys.version.split()[0]
    for key in ("platform", "machine", "physical_bytes", "packages"):
        assert key in rep
    assert set(rep["packages"]) == {"mlx", "mlx-metal", "numpy", "transformers"}
    assert "tuned_on" not in rep


def test_doctor_text_without_metal(fake_mlx, capsys):
    fake_mlx["metal"] = False
    rc, out, _ = _run(["doctor"], capsys)
    assert rc == 1 and "Metal: unavailable (Metal is not available)" in out
    assert "supported for serving: False" in out and "packages:" in out


def test_doctor_reports_a_failing_mlx_as_unsupported(fake_mlx, monkeypatch, capsys):
    mx = pytest.importorskip("mlx.core")

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(mx, "device_info", boom)
    _, out, _ = _run(["doctor", "--json"], capsys)
    assert json.loads(out)["metal"] == {"error": "RuntimeError('no device')"}


def test_doctor_on_an_apple_silicon_mac_is_supported(fake_mlx, monkeypatch, capsys):
    import platform
    import subprocess

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    class Done:
        stdout, returncode = "Apple M4 Pro\n", 0

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Done())
    rc, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 0 and rep["supported"] is True and rep["chip"] == "Apple M4 Pro"
    assert rep["metal"]["device_name"] == "Apple M4 Pro"
    assert rep["tuned_on"]["gpu"] and rep["tuned_on"]["note"]
    rc, out, _ = _run(["doctor"], capsys)
    assert rc == 0 and "Metal: Apple M4 Pro applegpu_g16s, recommended working set 36.00 GiB" in out
    assert "supported for serving: True" in out and "(Apple M4 Pro)" in out


def test_doctor_survives_a_missing_sysctl(fake_mlx, monkeypatch, capsys):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(subprocess, "run", boom)
    _, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rep["chip"] is None and "sysctl" in rep["chip_error"]


def test_tuned_on_has_a_note_for_the_tuned_family_and_for_another():
    from jitengine_qwen3_8_flash_next_m4pro.kernels import selfcheck

    same = cli._tuned_on({"device_name": "Apple M4 Pro", "architecture": selfcheck.TUNED_GPU})
    other = cli._tuned_on({"device_name": "Apple M2", "architecture": "applegpu_g14g"})
    assert same["mac"] == other["mac"] == selfcheck.TUNED_MAC
    assert same["note"] and other["note"] and same["note"] != other["note"]


# ---------------------------------------------------------------- status
def _urlopen_returning(routes):
    def urlopen(url, timeout=None):
        for path, result in routes.items():
            if url.endswith(path):
                if isinstance(result, Exception):
                    raise result
                return io.BytesIO(json.dumps(result).encode())
        raise AssertionError(url)

    return urlopen


def test_status_of_an_unreachable_server_reports_the_error_and_exits_1(capsys):
    argv = ["status", "--url", "http://127.0.0.1:1", "--timeout", "0.5", "--json"]
    rc, out, _ = _run(argv, capsys)
    rep = json.loads(out)
    assert rc == 1 and rep["url"] == "http://127.0.0.1:1"
    assert "error" in rep["health"] and "error" in rep["stats"]
    rc, out, _ = _run(argv[:-1], capsys)
    assert rc == 1 and out.startswith("http://127.0.0.1:1: ") and "plan: None" in out


def test_status_of_a_healthy_server_exits_0_and_prints_its_plan(monkeypatch, capsys):
    routes = {"/health": {"status": "ok"}, "/stats": {"memory_plan": "budget 40 GB"}}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning(routes))
    rc, out, _ = _run(["status", "--url", "http://host:1/"], capsys)
    assert rc == 0 and "http://host:1/: ok" in out and "plan: budget 40 GB" in out
    rc, out, _ = _run(["status", "--url", "http://host:1/", "--json"], capsys)
    assert rc == 0 and json.loads(out)["stats"] == {"memory_plan": "budget 40 GB"}


def test_status_records_an_http_error_body_and_exits_1(monkeypatch, capsys):
    err = urllib.error.HTTPError("http://h/health", 503, "busy", {}, io.BytesIO(b"draining"))
    monkeypatch.setattr(
        urllib.request, "urlopen", _urlopen_returning({"/health": err, "/stats": {"a": 1}})
    )
    rc, out, _ = _run(["status", "--url", "http://h", "--json"], capsys)
    assert rc == 1 and json.loads(out)["health"] == {"http_status": 503, "body": "draining"}


# ---------------------------------------------------------------- pack
def test_pack_builds_the_expert_pack_then_the_ple_pack_with_the_verify_count(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_experts, build_ple

    calls = []
    monkeypatch.setattr(build_experts, "main", lambda argv: calls.append(("experts", argv)) or 0)
    monkeypatch.setattr(build_ple, "main", lambda argv: calls.append(("ple", argv)) or 0)
    assert cli.main(["pack", "--model-path", "/m", "--verify", "128"]) == 0
    flags = ["--model", "/m", "--verify", "128"]
    assert calls == [("experts", flags), ("ple", flags)]
    calls.clear()
    cli.main(["pack", "--model-path", "/m"])
    assert calls[0][1][-1] == "64"


def test_pack_stops_at_the_first_builder_that_fails(monkeypatch):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_experts, build_ple

    monkeypatch.setattr(build_experts, "main", lambda argv: 3)
    monkeypatch.setattr(build_ple, "main", lambda argv: pytest.fail("the PLE pack was built"))
    assert cli.main(["pack", "--model-path", "/m"]) == 3


def test_pack_refuses_a_checkpoint_that_is_not_the_real_geometry(checkpoint):
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import PackError

    with pytest.raises(PackError, match="expected 48"):
        cli.main(["pack", "--model-path", checkpoint])


# ---------------------------------------------------------------- serve
class _RecordingEngine:
    def __init__(self, path, **opts):
        self.path, self.opts, self.served = path, opts, None

    def serve(self, host, port, allowed_hosts=()):
        self.served = (host, port)
        self.allowed_hosts = tuple(allowed_hosts)


@pytest.fixture
def served(monkeypatch, tmp_path):
    """A checkpoint directory that passes the config.json check, `create_engine` replaced by a
    recorder, and the startup SIGTERM handler not installed. Returns (path, engines made)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as engine_module

    (tmp_path / "config.json").write_text("{}")
    made = []

    def create(path, **opts):
        made.append(_RecordingEngine(path, **opts))
        return made[-1]

    monkeypatch.setattr(engine_module, "create_engine", create)
    monkeypatch.setattr(cli, "_install_startup_term_handler", lambda: None)
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: 64 * GIB)
    return str(tmp_path), made


def test_serve_of_a_directory_without_config_json_is_refused(tmp_path, capsys):
    rc, _, err = _run(["serve", "--model-path", str(tmp_path)], capsys)
    assert rc == 2 and "no checkpoint at" in err and err.startswith(f"{PROG}: error:")


def test_serve_builds_the_engine_with_the_default_budget_and_serves(served, capsys):
    path, made = served
    rc, _, _ = _run(["serve", "--model-path", path, "--port", "9123", "--host", "0.0.0.0"], capsys)
    assert rc == 0
    (engine,) = made
    assert engine.path == path and engine.served == ("0.0.0.0", 9123)
    assert engine.opts == {"memory_budget_bytes": cli._default_budget_bytes()}


def test_serve_forwards_the_flags_that_were_given_as_engine_options(served, capsys):
    path, made = served
    argv = [
        "serve", "--model-path", path, "--memory-budget-gb", "40", "--max-slots", "4",
        "--kv-budget-mb", "256.5", "--wave-chunk", "512", "--prefix-min-divergence-tokens", "32",
        "--compute-dtype", "bfloat16", "--residual-dtype", "float32", "--moe-kernel", "stock",
        "--moe-kernel-variant", "fused", "--dense-kernel", "stock", "--wave-width", "2",
        "--admit-warm-quiet-s", "0.5", "--prefetch-lookahead", "2", "--prefetch-reserve", "64",
        "--transient-prefill", "--moe-sync-each-layer", "--log-level", "error",
    ]  # fmt: skip
    assert _run(argv, capsys)[0] == 0
    assert made[0].opts == {
        "memory_budget_bytes": 40 * 10**9,
        "max_slots": 4,
        "kv_budget_mb": 256.5,
        "prefix_grid_tokens": 512,
        "prefix_min_divergence_tokens": 32,
        "compute_dtype": "bfloat16",
        "residual_dtype": "float32",
        "moe_kernel": "stock",
        "moe_kernel_variant": "fused",
        "dense_kernel": "stock",
        "wave_width": 2,
        "admit_warm_quiet_s": 0.5,
        "prefetch_lookahead": 2,
        "prefetch_reserve": 64,
        "transient_prefill": True,
        "moe_sync_each_layer": True,
    }
    assert logging.getLogger("jitengine_qwen3_8_flash_next_m4pro").level == logging.ERROR


def test_serve_expands_the_pack_paths_and_keeps_a_zero_option(served, capsys, monkeypatch):
    path, made = served
    monkeypatch.setenv("HOME", "/home/someone")
    argv = ["serve", "--model-path", path, "--expert-pack-path", "~/e.experts",
            "--ple-pack-path", "/abs/p.ple", "--prefetch-lookahead", "0"]  # fmt: skip
    assert _run(argv, capsys)[0] == 0
    opts = made[0].opts
    assert opts["expert_pack_path"] == os.path.expanduser("~/e.experts")
    assert opts["ple_pack_path"] == "/abs/p.ple" and opts["prefetch_lookahead"] == 0


def test_serve_refuses_a_budget_over_80_percent_unless_forced(served, capsys):
    path, made = served
    with pytest.raises(SystemExit, match="more than 80%"):
        cli.main(["serve", "--model-path", path, "--memory-budget-gib", "60"])
    assert made == []
    assert (
        _run(["serve", "--model-path", path, "--memory-budget-gib", "60", "--force"], capsys)[0]
        == 0
    )
    assert made[0].opts["memory_budget_bytes"] == 60 * GIB


def test_the_startup_handler_exits_zero_on_sigterm_and_serve_replaces_it():
    before = signal.getsignal(signal.SIGTERM)
    try:
        cli._install_startup_term_handler()
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not before
        with pytest.raises(SystemExit) as e:
            handler(signal.SIGTERM, None)
        assert e.value.code == 0
    finally:
        signal.signal(signal.SIGTERM, before)


def test_the_startup_handler_is_skipped_outside_the_main_thread():
    import threading

    box = []
    t = threading.Thread(target=lambda: box.append(cli._install_startup_term_handler()))
    t.start()
    t.join(10)
    assert box == [None]  # signal.signal raised ValueError, which the CLI ignores


def test_serve_passes_every_allowed_host_to_the_server(served, capsys):
    path, made = served
    argv = ["serve", "--model-path", path, "--allowed-host", "mac.local",
            "--allowed-host", "api.internal"]  # fmt: skip
    assert _run(argv, capsys)[0] == 0
    assert made[0].allowed_hosts == ("mac.local", "api.internal")
    assert _run(["serve", "--model-path", path], capsys)[0] == 0
    assert made[1].allowed_hosts == ()
