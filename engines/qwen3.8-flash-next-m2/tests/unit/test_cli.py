"""The CLI's memory-budget policy, without MLX."""

import pytest

from jitengine_qwen3_8_flash_next_m2 import cli

M2_24GB = 24 * cli.GIB  # unified memory as macOS reports it
GB = 10**9


def test_the_default_budget_is_the_m2s_measured_18_gb(monkeypatch):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: M2_24GB)
    assert cli._budget_bytes(None, force=False) == 18 * GB  # 16.76 GiB


def test_the_measured_16_and_18_gb_budgets_run_on_a_24_gb_mac(monkeypatch):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: M2_24GB)
    assert cli._budget_bytes(16 * GB, force=False) == 16 * GB
    assert cli._budget_bytes(18 * GB, force=False) == 18 * GB


def test_a_budget_over_80_percent_needs_force(monkeypatch):
    monkeypatch.setattr(cli, "_unified_memory_bytes", lambda: M2_24GB)
    with pytest.raises(SystemExit, match="more than 80%"):
        cli._budget_bytes(22 * GB, force=False)  # 85% of 25.8 GB
    assert cli._budget_bytes(22 * GB, force=True) == 22 * GB


@pytest.mark.parametrize(
    ("flags", "nbytes"),
    [
        ([], None),
        (["--memory-budget-gib", "16.75"], 16.75 * 2**30),
        # the published budgets were measured in decimal GB, and this flag gives them to the byte
        (["--memory-budget-gb", "18"], 18 * GB),
    ],
)
def test_the_budget_flags_convert_to_bytes_exactly(flags, nbytes):
    for cmd in ("inspect", "serve"):
        a = cli.build_parser().parse_args([cmd, "--model-path", "m", *flags])
        assert cli._requested_budget(a) == nbytes, cmd


def test_a_budget_in_both_units_is_refused():
    flags = ["--memory-budget-gib", "30", "--memory-budget-gb", "32"]
    a = cli.build_parser().parse_args(["serve", "--model-path", "m", *flags])
    with pytest.raises(SystemExit):
        cli._requested_budget(a)


@pytest.mark.parametrize("text", ["x", "-1", "nan", "inf"])
def test_a_budget_that_is_not_a_finite_size_is_a_usage_error(text):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["serve", "--model-path", "m", "--memory-budget-gib", text])
    assert exc.value.code == 2


def test_help_does_not_import_mlx():
    import os
    import subprocess
    import sys

    src = os.path.dirname(os.path.dirname(os.path.abspath(cli.__file__)))
    code = (
        f"import sys; from {cli.__name__} import main\n"
        "try:\n    main(['serve', '--help'])\nexcept SystemExit:\n    pass\n"
        "print('MLX' if 'mlx.core' in sys.modules else 'NO_MLX')"
    )
    env = dict(os.environ, PYTHONPATH=src)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip().endswith("NO_MLX"), r.stdout + r.stderr


def test_every_command_of_the_template_is_there_in_its_order():
    """The engines' shared CLI template: the same commands, listed in the same order."""
    commands = ["doctor", "inspect", "pack", "serve", "status"]
    assert list(cli.build_parser()._subparsers._group_actions[0].choices) == commands
    for cmd in commands:
        with pytest.raises(SystemExit) as exc:
            cli.main([cmd, "--help"])
        assert exc.value.code == 0, cmd


def test_doctor_reports_a_sysctl_that_times_out(monkeypatch, capsys):
    import argparse
    import json
    import subprocess

    run = subprocess.run

    def hung(cmd, *a, **k):
        if cmd[:1] != ["sysctl"]:
            return run(cmd, *a, **k)
        raise subprocess.TimeoutExpired(cmd, k.get("timeout"))

    monkeypatch.setattr(subprocess, "run", hung)
    cli.cmd_doctor(argparse.Namespace(json=True))
    rep = json.loads(capsys.readouterr().out)
    assert rep["chip"] is None and "TimeoutExpired" in rep["chip_error"]
