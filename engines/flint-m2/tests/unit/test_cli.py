"""The command line's engine options, shared by serve and doctor, and serve's refusals before any weight."""

import argparse
import json

import pytest

from jitengine_flint_m2 import cli


def test_engine_options(tmp_path):
    """--context-limit auto|N (default auto), --decode-backend auto|mk|graph (default auto), --mk-grid G (default
    none: the device's plan), --simulate JSON or @FILE; serve and doctor take the same ones."""
    defaults = ("auto", "auto", None, None)
    for cmd in ("serve", "doctor"):
        a = cli._parser().parse_args([cmd, "--model-path", "/m"])
        assert (a.context_limit, a.decode_backend, a.mk_grid, a.simulate) == defaults
        assert a.mode == "auto"
        assert (
            cli._parser().parse_args([cmd, "--model", "/m", "--mode", "offload"]).mode == "offload"
        )
        with pytest.raises(SystemExit):
            cli._parser().parse_args([cmd, "--model", "/m", "--mode", "swap"])
    assert cli.context_limit_arg("auto") == "auto" and cli.context_limit_arg("4096") == 4096
    for bad in ("4k", "", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.context_limit_arg(bad)
    f = tmp_path / "sim.json"
    f.write_text(json.dumps({"fail_compile": True}))
    sim = cli.simulate_arg('{"fail_compile": true}')
    assert cli.simulate_arg(f"@{f}") == sim == {"fail_compile": True}
    for bad in ("{not json", f"@{tmp_path / 'missing.json'}"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.simulate_arg(bad)
    argv = "doctor --model /m --mk-grid 48 --json --tune --context-limit 8192".split()
    a = cli._parser().parse_args(argv)
    assert (a.mk_grid, a.json, a.tune, a.context_limit, a.smoke_tokens) == (48, "-", True, 8192, 16)
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["serve", "--model-path", "/m", "--decode-backend", "bogus"])
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["serve"])  # the checkpoint is required


def test_serve_refuses_before_weights(fake_gpu, capsys):
    """A simulation without --allow-fault-injection, a grid outside the domain, a context limit the wired limit
    cannot hold on the megakernel, a memory budget too small for offload, or two budgets: serve exits 2 with one
    line, before any weight is read or a port is bound."""
    for extra, needle in (
        (["--simulate", '{"fail_compile": true}'], "needs allow_fault_injection"),
        (["--mk-grid", "62"], "mk_grid must be a multiple of 4"),
        (["--decode-backend", "graph", "--profile", "kv8"], "profile 'kv8' runs on the megakernel"),
        (
            '--allow-fault-injection --simulate {"wired_limit_mb":26624} --context-limit 262144 '
            "--decode-backend mk".split(),
            "the most that fits is 73,728 tokens",
        ),
        (["--memory-budget-gib", "2"], "too small for Flint"),
        (["--memory-budget-gib", "2", "--memory-budget-gb", "2"], "pass the budget once"),
        (["--mode", "offload", "--profile", "kv8"], "memory mode 'offload' streams the experts"),
    ):
        code = cli.main(["serve", "--model-path", fake_gpu.model, "--port", "1", *extra])
        err = capsys.readouterr().err
        assert code == 2 and "jitengine-flint-m2: refusing to start: " in err, err
        assert needle in err, err
    assert fake_gpu.MK.built == []


def test_version_and_queue_bound(capsys):
    """--version prints the version pyproject.toml declares; serve --max-queued N (default the engine's MAX_QUEUE)
    takes an integer >= 0."""
    import os
    import tomllib

    import jitengine_flint_m2
    from jitengine_flint_m2.engine.engine import MAX_QUEUE

    with pytest.raises(SystemExit) as stop:
        cli._parser().parse_args(["--version"])
    assert stop.value.code == 0
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"), "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert declared == jitengine_flint_m2.__version__
    assert capsys.readouterr().out.strip() == f"jitengine-flint-m2 {declared}"
    a = cli._parser().parse_args(["serve", "--model", "/m"])
    assert a.max_queued == cli.DEFAULT_MAX_QUEUED == MAX_QUEUE
    assert cli._parser().parse_args(["serve", "--model", "/m", "--max-queued", "0"]).max_queued == 0
    for bad in ("-1", "two"):
        with pytest.raises(SystemExit):
            cli._parser().parse_args(["serve", "--model", "/m", "--max-queued", bad])


def test_serve_hands_its_options_to_the_engine(fake_gpu, monkeypatch, capsys):
    """serve builds the engine with --max-queued and --mode, as /health then reports them, and logs at --log-level:
    info prints the startup lines, warning none of them on a Mac where nothing goes wrong."""
    from jitengine_flint_m2.engine.engine import Engine

    seen = {}
    monkeypatch.setattr(cli, "_watchdog", lambda engine: None)
    monkeypatch.setattr(
        Engine, "serve", lambda self, host, port, allowed_hosts=(): seen.update(self.health())
    )
    argv = ["serve", "--model-path", fake_gpu.model, "--context-limit", "4096"]
    assert cli.main([*argv, "--max-queued", "3", "--mode", "resident"]) == 0
    assert (seen["max_queued"], seen["memory_mode_requested"]) == (3, "resident")
    assert seen["memory"]["mode"] == "resident"
    err = capsys.readouterr().err
    assert "[engine] memory plan: resident" in err and "[server] profile=main" in err, err
    assert cli.main([*argv, "--log-level", "warning"]) == 0
    assert capsys.readouterr().err == ""
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["serve", "--model", "/m", "--log-level", "loud"])
