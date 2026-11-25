"""The command line's engine options, shared by serve and doctor, and serve's refusals before any weight."""

import argparse
import json

import pytest

from jitengine_flint import cli, pinned
from jitengine_runtime import hub


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
    assert (
        cli._parser().parse_args(["serve"]).model == pinned.CHECKPOINT
    )  # serve defaults to the pinned checkpoint
    for cmd in ("doctor", "inspect"):
        with pytest.raises(SystemExit):
            cli._parser().parse_args([cmd])  # the others need the checkpoint named


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
        assert code == 2 and "jitengine-flint: refusing to start: " in err, err
        assert needle in err, err
    assert fake_gpu.MK.built == []


def test_version_and_queue_bound(capsys):
    """--version prints the version pyproject.toml declares; serve --max-queued N (default the engine's MAX_QUEUE)
    takes an integer >= 0."""
    import os
    import tomllib

    import jitengine_flint
    from jitengine_flint.engine.engine import MAX_QUEUE

    with pytest.raises(SystemExit) as stop:
        cli._parser().parse_args(["--version"])
    assert stop.value.code == 0
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"), "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert declared == jitengine_flint.__version__
    assert capsys.readouterr().out.strip() == f"jitengine-flint {declared}"
    a = cli._parser().parse_args(["serve", "--model", "/m"])
    assert a.max_queued == cli.DEFAULT_MAX_QUEUED == MAX_QUEUE
    assert cli._parser().parse_args(["serve", "--model", "/m", "--max-queued", "0"]).max_queued == 0
    for bad in ("-1", "two"):
        with pytest.raises(SystemExit):
            cli._parser().parse_args(["serve", "--model", "/m", "--max-queued", bad])


def test_serve_hands_its_options_to_the_engine(fake_gpu, monkeypatch, capsys):
    """serve builds the engine with --max-queued and --mode, as /health then reports them, and logs at --log-level:
    info prints the startup lines, warning none of them on a Mac where nothing goes wrong."""
    from jitengine_flint.engine.engine import Engine

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


def test_serve_fetches_a_model_id_into_the_hugging_face_cache(fake_gpu, monkeypatch, capsys):
    """A model id for --model-path is fetched at the revision the build pins, serve runs on the snapshot folder
    it returns, and the log says so at info and not at warning; with no --model-path the pinned checkpoint is it."""
    from jitengine_flint.engine.engine import Engine

    fetched, served = [], []
    monkeypatch.setattr(
        hub,
        "_snapshot_download",
        lambda repo, rev, allow: fetched.append((repo, rev)) or fake_gpu.model,
    )
    monkeypatch.setattr(cli, "_watchdog", lambda engine: None)
    monkeypatch.setattr(
        Engine, "serve", lambda self, host, port, allowed_hosts=(): served.append(self.model_path)
    )
    assert cli.main(["serve", "--context-limit", "4096"]) == 0
    assert (
        cli.main(["serve", "--model-path", "RecursiveSI/flint-int4@dev", "--context-limit", "4096"])
        == 0
    )
    assert fetched == [(pinned.CHECKPOINT, pinned.REVISION), (pinned.CHECKPOINT, "dev")]
    assert served == [fake_gpu.model] * 2
    err = capsys.readouterr().err
    assert (
        f"[hub] fetching {pinned.CHECKPOINT} at revision {pinned.REVISION[:12]} into the Hugging Face cache"
        in err
    )
    assert f"[hub] using {fake_gpu.model}." in err
    assert cli.main(["serve", "--context-limit", "4096", "--log-level", "warning"]) == 0
    assert capsys.readouterr().err == ""


def test_a_directory_needs_no_network_and_logs_nothing_more(fake_gpu, monkeypatch, capsys):
    from jitengine_flint.engine.engine import Engine

    def boom(*a):
        raise AssertionError("a directory must not reach the Hub")

    monkeypatch.setattr(hub, "_snapshot_download", boom)
    monkeypatch.setattr(cli, "_watchdog", lambda engine: None)
    monkeypatch.setattr(Engine, "serve", lambda self, host, port, allowed_hosts=(): None)
    assert cli.main(["serve", "--model-path", fake_gpu.model, "--context-limit", "4096"]) == 0
    assert "[hub]" not in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["serve", "doctor", "inspect"])
def test_a_model_that_cannot_be_fetched_is_refused_with_one_line(cmd, monkeypatch, capsys):
    """Not a directory and not a model id, or a repository that cannot be fetched: exit 2, one line, nothing loaded."""
    assert cli.main([cmd, "--model-path", "/no/such/dir"]) == 2
    err = capsys.readouterr().err
    assert err == (
        'jitengine-flint: refusing to start: "/no/such/dir" is not a directory and not a Hugging Face model id '
        "(org/name or org/name@revision).\n"
    )

    def gone(*a, **kw):
        raise hub.ModelPathError(
            '"org/nme" is not a directory, and Hugging Face has no model "org/nme" or it is private.'
        )

    monkeypatch.setattr(hub, "resolve_model_path", gone)
    assert cli.main([cmd, "--model-path", "org/nme"]) == 2
    assert capsys.readouterr().err.count("\n") == 1
