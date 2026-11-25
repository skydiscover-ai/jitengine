"""`--model-path` takes a Hugging Face model id as well as a directory: serve with none fetches the pinned checkpoint,
`--print-config` fetches nothing, and a directory is used as before. The download is replaced; nothing here needs the
network."""

from __future__ import annotations

import json
import logging

import pytest

from jitengine_qwen3_6_35b_a3b_m4pro import cli, pinned
from jitengine_runtime import hub

PROG = "jitengine-qwen3.6-35b-a3b-m4pro"


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """`serve` configures the package's logger; put it back so other tests see the default."""
    pkg = logging.getLogger("jitengine_qwen3_6_35b_a3b_m4pro")
    state = (list(pkg.handlers), pkg.level, pkg.propagate)
    yield
    pkg.handlers[:] = state[0]
    pkg.setLevel(state[1])
    pkg.propagate = state[2]


@pytest.fixture
def fetches(monkeypatch, tmp_path):
    """Replace the download: each call is recorded as (repo, revision) and returns a snapshot folder with no config.json."""
    calls = []
    snapshot = tmp_path / "hub" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)

    def fake(repo, rev, allow_patterns):
        calls.append((repo, rev))
        return str(snapshot)

    monkeypatch.setattr(hub, "_snapshot_download", fake)
    return calls


def _run(argv, capsys):
    rc = cli.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_print_config_names_the_pinned_checkpoint_and_fetches_nothing(fetches, capsys):
    rc, out, _ = _run(["serve", "--print-config"], capsys)
    doc = json.loads(out)
    assert rc == 0 and fetches == []
    assert (
        doc["resolved"]["model_path"] == pinned.CHECKPOINT
        and doc["origins"]["model_path"] == "default"
    )


def test_a_model_id_stays_an_id_in_the_configuration(fetches, tmp_path, capsys):
    """From a flag or from a config file, `org/name` is not made an absolute path; a directory is, as before."""
    _, out, _ = _run(["serve", "--print-config", "--model-path", "org/name@v2"], capsys)
    doc = json.loads(out)
    assert doc["resolved"]["model_path"] == "org/name@v2" and doc["origins"]["model_path"] == "flag"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"schema_version": 1, "model_path": "other/model"}))
    _, out, _ = _run(["serve", "--print-config", "--config", str(cfg)], capsys)
    assert json.loads(out)["resolved"]["model_path"] == "other/model"
    (tmp_path / "local").mkdir()
    cfg.write_text(json.dumps({"schema_version": 1, "model_path": "local"}))
    _, out, _ = _run(["serve", "--print-config", "--config", str(cfg)], capsys)
    assert json.loads(out)["resolved"]["model_path"] == str(tmp_path / "local")
    _, out, _ = _run(
        ["serve", "--print-config", "--config", str(cfg), "--model-path", "x/y"], capsys
    )
    assert json.loads(out)["resolved"]["model_path"] == "x/y"  # a flag beats the file
    assert fetches == []


def test_serve_with_no_model_path_fetches_the_pinned_checkpoint_and_logs_it(fetches, capsys):
    rc, _, err = _run(["serve"], capsys)
    assert fetches == [(pinned.CHECKPOINT, pinned.REVISION)]
    lines = err.splitlines()
    assert lines[0].startswith(
        f"[hub] fetching {pinned.CHECKPOINT} at revision {pinned.REVISION[:12]} into the Hugging Face cache"
    )
    assert lines[1].startswith("[hub] using ") and lines[1].endswith("/snapshots/abc.")
    assert rc == 2 and lines[2].endswith(
        "/snapshots/abc: not a checkpoint directory (no config.json)"
    )


def test_serve_takes_an_id_with_a_revision_and_logs_nothing_below_the_level(fetches, capsys):
    _run(["serve", "--model-path", f"{pinned.CHECKPOINT}@dev", "--log-level", "warning"], capsys)
    assert fetches == [(pinned.CHECKPOINT, "dev")]
    rc, _, err = _run(["serve", "--model-path", "other/model", "--log-level", "warning"], capsys)
    assert fetches[-1] == ("other/model", "main") and "[hub]" not in err and rc == 2


def test_a_directory_is_used_as_it_always_was(fetches, tmp_path, capsys):
    rc, _, err = _run(["serve", "--model-path", str(tmp_path)], capsys)
    assert rc == 2 and fetches == [] and "[hub]" not in err
    assert err == f"{PROG}: error: {tmp_path}: not a checkpoint directory (no config.json)\n"


@pytest.mark.parametrize("cmd", ["serve", "inspect", "pack"])
def test_a_model_that_cannot_be_fetched_is_refused_with_one_line(cmd, monkeypatch, capsys):
    argv = [cmd, "--model-path", "/no/such/dir"]
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert capsys.readouterr().err == (
        f'{PROG}: error: "/no/such/dir" is not a directory and not a Hugging Face model id '
        "(org/name or org/name@revision).\n"
    )

    def gone(*a, **kw):
        raise hub.ModelPathError('Hugging Face has no model "org/nme".')

    monkeypatch.setattr(hub, "resolve_model_path", gone)
    with pytest.raises(SystemExit) as e:
        cli.main([cmd, "--model-path", "org/nme"])
    assert e.value.code == 2
    assert capsys.readouterr().err == f'{PROG}: error: Hugging Face has no model "org/nme".\n'
