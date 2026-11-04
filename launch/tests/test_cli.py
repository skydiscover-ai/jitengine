"""The command line against a stand-in engine: discovery, the plan --dry-run prints, and a launch with
stand-in agents on PATH that record what they were given, and leave nothing behind."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jit_launch.cli import FALLBACK_CONTEXT, discover, main, origin


def test_the_server_may_be_an_origin_or_its_v1_url():
    assert origin("127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert origin("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert origin("https://mac.local") == "https://mac.local"
    for bad in ("http://h:1/v2", "ftp://h", "http://"):
        with pytest.raises(ValueError):
            origin(bad)


def test_discovery_reads_the_model_and_its_context(engine):
    served = discover(engine.url + "/v1")
    assert (served.origin, served.model, served.context) == (engine.url, engine.model, 32768)
    engine.context = None  # a server that reports no limit
    assert discover(engine.url).context == FALLBACK_CONTEXT


def test_no_engine_is_a_clear_error(capsys):
    assert main(["pi", "--server", "http://127.0.0.1:9"]) == 1
    err = capsys.readouterr().err
    assert "no engine answers at http://127.0.0.1:9/v1/models" in err


def test_dry_run_prints_the_plan_and_changes_nothing(engine, capsys):
    assert main(["pi", "--server", engine.url, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"model: {engine.model}" in out
    assert "context: 32768 tokens per request, 8192 of them output" in out
    assert "for the session, jitengine.js:" in out and "pi.registerProvider(" in out
    want = f"run: pi -e '<a temporary directory>/jitengine.js' --provider jitengine --model {engine.model}"
    assert want in out
    assert main(["claude", "--server", engine.url, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"set ANTHROPIC_BASE_URL={engine.url}\n" in out and "set ANTHROPIC_API_KEY=\n" in out


def test_a_small_context_is_warned_about(engine, capsys):
    engine.context = 8192
    assert main(["opencode", "--server", engine.url, "--dry-run"]) == 0
    assert "warning: the engine admits 8192 tokens per request" in capsys.readouterr().err


def _agent(bin_dir, name: str, body: str) -> None:
    """A stand-in agent on PATH: a Python script that records what it was given."""
    path = bin_dir / name
    path.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}\n")
    path.chmod(0o755)


@pytest.fixture
def bin_dir(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")
    return d


def test_pi_runs_with_its_extension_for_the_session_only(engine, bin_dir, tmp_path):
    record = tmp_path / "pi.json"
    body = f"""
extension = sys.argv[2]
json.dump({{"argv": sys.argv[1:], "extension": open(extension).read()}}, open({str(record)!r}, "w"))
sys.exit(3)
"""
    _agent(bin_dir, "pi", body)
    assert main(["pi", "--server", engine.url, "--", "-p", "hi"]) == 3  # the agent's own status
    got = json.loads(record.read_text())
    extension = got["argv"][1]
    assert got["argv"] == [
        "-e",
        extension,
        "--provider",
        "jitengine",
        "--model",
        engine.model,
        "-p",
        "hi",
    ]
    assert f'"baseUrl": "{engine.url}/v1"' in got["extension"]
    assert not os.path.exists(extension)  # gone with the session


def test_claude_code_runs_against_the_engine(engine, bin_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-key")
    record = tmp_path / "claude.json"
    keys = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_AUTH_TOKEN")
    body = f"""
env = {{k: os.environ.get(k) for k in {keys!r}}}
json.dump({{"argv": sys.argv[1:], "env": env}}, open({str(record)!r}, "w"))
"""
    _agent(bin_dir, "claude", body)
    assert main(["claude", "--server", engine.url]) == 0
    got = json.loads(record.read_text())
    assert got["argv"] == ["--model", engine.model]
    assert got["env"] == {
        "ANTHROPIC_BASE_URL": engine.url,
        "ANTHROPIC_API_KEY": "",  # the shell's key would win over the token
        "ANTHROPIC_MODEL": engine.model,
        "ANTHROPIC_AUTH_TOKEN": "jitengine",
    }


def test_codex_runs_with_its_catalog_for_the_session_only(engine, bin_dir, tmp_path):
    record = tmp_path / "codex.json"
    body = f"""
catalog = next(a for a in sys.argv if a.startswith("model_catalog_json="))
catalog = json.loads(catalog.split("=", 1)[1])
env = os.environ.get("JITENGINE_API_KEY")
got = {{"argv": sys.argv[1:], "catalog": catalog, "models": json.load(open(catalog)), "env": env}}
json.dump(got, open({str(record)!r}, "w"))
"""
    _agent(bin_dir, "codex", body)
    assert main(["codex", "--server", engine.url, "--", "exec", "hi"]) == 0
    got = json.loads(record.read_text())
    assert got["argv"][-4:] == ["-m", engine.model, "exec", "hi"] and got["env"] == "jitengine"
    assert f'base_url = "{engine.url}/v1"' in got["argv"][3]
    assert got["models"]["models"][0]["slug"] == engine.model
    assert not os.path.exists(got["catalog"])  # gone with the session


def test_a_missing_agent_says_how_to_install_it(engine, bin_dir, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(bin_dir))
    assert main(["opencode", "--server", engine.url]) == 1
    err = capsys.readouterr().err
    assert "opencode is not installed: curl -fsSL https://opencode.ai/install | bash" in err


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_to_the_launcher_ends_the_agent_and_removes_the_session_files(
    sig, engine, bin_dir, tmp_path
):
    record = tmp_path / "pi.json"
    body = f"""
import time
json.dump({{"pid": os.getpid(), "extension": sys.argv[2]}}, open({str(record)!r}, "w"))
time.sleep(60)
"""
    _agent(bin_dir, "pi", body)
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    launcher = subprocess.Popen(
        [sys.executable, "-m", "jit_launch", "pi", "--server", engine.url],
        env=env,
        stdin=subprocess.DEVNULL,  # no terminal: a SIGINT here is not a Ctrl-C the agent saw
    )
    try:
        for _ in range(200):
            if record.exists() and record.read_text():
                break
            time.sleep(0.05)
        got = json.loads(record.read_text())
        assert os.path.exists(got["extension"])
        launcher.send_signal(sig)
        assert launcher.wait(timeout=10) == 128 + sig
    finally:
        if launcher.poll() is None:
            launcher.kill()
    assert not os.path.exists(got["extension"])
    assert not os.path.exists(os.path.dirname(got["extension"]))
    with pytest.raises(ProcessLookupError):
        os.kill(got["pid"], 0)  # no orphaned agent


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(path, seconds=10.0) -> str:
    for _ in range(int(seconds / 0.05)):
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


def test_a_403_from_the_engine_says_it_refused_the_host_name_and_how_to_allow_it(capsys):
    class Refusing(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        assert main(["pi", "--server", f"http://localhost:{srv.server_address[1]}"]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    err = capsys.readouterr().err
    assert "no engine answers at" in err
    assert "refused the host name 'localhost'" in err
    assert "serve --allowed-host localhost" in err


def test_a_signal_while_the_agent_is_starting_still_cleans_up(tmp_path, monkeypatch):
    """The handlers exist before the agent does: a SIGTERM that lands right after Popen is handled,
    not the default action that would kill the launcher with its temporary directory and agent."""
    from jit_launch import cli
    from jit_launch.agents import Plan

    pidfile = tmp_path / "agent.pid"
    real_popen = subprocess.Popen

    class Popen(real_popen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            pidfile.write_text(str(self.pid))
            os.kill(os.getpid(), signal.SIGTERM)  # arrives as soon as the agent exists

    monkeypatch.setattr(subprocess, "Popen", Popen)
    made = []
    real_mkdtemp = cli.tempfile.mkdtemp
    monkeypatch.setattr(
        cli.tempfile, "mkdtemp", lambda *a, **k: made.append(real_mkdtemp(*a, **k)) or made[-1]
    )
    status = cli._run(Plan(argv=["sleep", "30"], files={"f.txt": "x"}))
    assert status == 128 + signal.SIGTERM
    assert made and not os.path.exists(made[0])
    pid = int(pidfile.read_text())
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.05)
    assert not _alive(pid)


def test_an_agent_killed_by_a_signal_exits_128_plus_the_signal():
    from jit_launch import cli
    from jit_launch.agents import Plan

    assert cli._run(Plan(argv=["sh", "-c", "kill -9 $$"])) == 128 + signal.SIGKILL
    assert cli._run(Plan(argv=["sh", "-c", "exit 3"])) == 3


def test_a_stop_ends_the_agents_own_children_when_it_runs_in_its_own_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = f"""
import sys
from jit_launch import cli
from jit_launch.agents import Plan
sys.exit(cli._run(Plan(argv=['sh', '-c', 'sleep 120 & echo $! > {pidfile}; wait'])))
"""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    launcher = subprocess.Popen([sys.executable, "-c", code], env=env, stdin=subprocess.DEVNULL)
    grandchild = None
    try:
        grandchild = int(_wait_for(pidfile))
        launcher.send_signal(signal.SIGTERM)
        assert launcher.wait(timeout=10) == 128 + signal.SIGTERM
        for _ in range(100):
            if not _alive(grandchild):
                break
            time.sleep(0.05)
        assert not _alive(grandchild), "the agent's child outlived the agent"
    finally:
        if launcher.poll() is None:
            launcher.kill()
        if grandchild and _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


def test_at_a_terminal_only_the_agent_itself_is_signalled(tmp_path, monkeypatch):
    """The agent shares the terminal's foreground group there, so a stop cannot signal the group."""
    from jit_launch import cli
    from jit_launch.agents import Plan

    pidfile = tmp_path / "child.pid"
    monkeypatch.setattr(cli, "_at_terminal", lambda: True)
    seen = {}
    real_popen = subprocess.Popen

    class Popen(real_popen):
        def __init__(self, *a, **k):
            seen["new_session"] = k.get("start_new_session")
            super().__init__(*a, **k)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    threading.Timer(0.5, lambda: (_wait_for(pidfile), os.kill(os.getpid(), signal.SIGTERM))).start()
    status = cli._run(Plan(argv=["sh", "-c", f"sleep 120 & echo $! > {pidfile}; wait"]))
    grandchild = int(pidfile.read_text())
    try:
        assert status == 128 + signal.SIGTERM
        assert not seen["new_session"]
        assert _alive(grandchild)  # documented: a process the agent started is not signalled
    finally:
        os.kill(grandchild, signal.SIGKILL)
