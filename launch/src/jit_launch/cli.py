"""jit-launch: start a coding agent against a running JITEngine engine.

    jit-launch {pi,claude,opencode,codex} [--server URL] [--dry-run] [-- AGENT ARGS ...]

It reads the served model and the tokens one request may hold from the engine's /v1/models (the
`max_model_len` each engine reports), points the agent at the engine for this session (agents.py)
and runs the agent in the foreground; a file the agent reads lives in a temporary directory until the
agent exits or the launcher is sent SIGTERM, SIGHUP or (outside a terminal) SIGINT, which also ends
the agent (and, outside a terminal, the processes it started: see `_run`). Arguments after `--` go to the agent; `--dry-run` prints the plan and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

from jit_launch import __version__
from jit_launch.agents import INSTALL, PLANS, Plan, Served, plan

DEFAULT_SERVER = "http://127.0.0.1:8080"
FALLBACK_CONTEXT = 8192  # for a server that reports no limit: guessing low only compacts sooner
AGENT_CONTEXT = 32768  # below this an agent's own prompt and a few tool results fill a request


def origin(server: str) -> str:
    """The scheme, host and port of `server`, given as an origin or as its /v1 base URL."""
    raw = server.strip()
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    path = parts.path.rstrip("/")
    if parts.scheme not in ("http", "https") or not parts.hostname or path not in ("", "/v1"):
        raise ValueError(f"--server takes an engine's origin or /v1 URL, not {server!r}")
    return f"{parts.scheme}://{parts.netloc}"


def _refused_hint(exc: Exception, base: str) -> str:
    """What to add to the error when the engine answered 403: it refuses a host name it was not
    told to serve (DNS-rebinding protection), which `serve --allowed-host` allows."""
    if not (isinstance(exc, HTTPError) and exc.code == 403):
        return ""
    name = urlsplit(base).hostname
    return (
        f". The engine answered 403: it refused the host name {name!r}, which it serves only when "
        f"told to; start it with `serve --allowed-host {name}`"
    )


def discover(server: str) -> Served:
    """The model the engine at `server` serves and the tokens one request may hold."""
    base = origin(server)
    try:
        with urlopen(f"{base}/v1/models", timeout=5) as r:
            listing = json.loads(r.read())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"no engine answers at {base}/v1/models ({exc}); start one with its `serve` "
            "command, or pass --server" + _refused_hint(exc, base)
        ) from exc
    models = listing.get("data") if isinstance(listing, dict) else None
    first = models[0] if isinstance(models, list) and models else None
    if not isinstance(first, dict) or not isinstance(first.get("id"), str):
        raise RuntimeError(f"{base}/v1/models lists no model")
    context = first.get("max_model_len")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        context = FALLBACK_CONTEXT
    return Served(base, first["id"], context)


def _print_plan(agent: str, served: Served, p: Plan) -> None:
    print(f"engine: {served.base_url}")
    print(f"model: {served.model}")
    print(f"context: {served.context} tokens per request, {served.max_output} of them output")
    for name, text in p.files.items():
        print(f"for the session, {name}:\n{text.rstrip()}")
    for key in sorted(p.env):
        print(f"set {key}={p.env[key]}")
    argv = [a.replace("{dir}", "<a temporary directory>") for a in p.argv]
    print(f"run: {shlex.join(argv)}")
    if shutil.which(p.argv[0]) is None:
        print(f"{p.argv[0]} is not installed: {INSTALL[agent]}")


GRACE_S = 5.0  # how long an agent gets to exit after SIGTERM before it is killed


def _at_terminal() -> bool:
    """Whether this process is in its terminal's foreground group, where Ctrl-C reaches the agent
    itself."""
    try:
        return os.tcgetpgrp(0) == os.getpgrp()
    except OSError:
        return False


def _signal_agent(child: subprocess.Popen, own_group: bool, sig: int) -> None:
    """Send `sig` to the agent, or to its whole process group when it leads one."""
    try:
        if own_group:
            os.killpg(child.pid, sig)
        else:
            child.send_signal(sig)
    except ProcessLookupError, PermissionError:
        pass  # already gone


def _wait(child: subprocess.Popen, own_group: bool, stopped: list[int]) -> int:
    """Wait for the agent to exit; once a stop was requested, kill it after GRACE_S."""
    deadline = None
    while True:
        try:
            return child.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            if stopped and deadline is None:
                deadline = time.monotonic() + GRACE_S
            if deadline is not None and time.monotonic() > deadline:
                _signal_agent(child, own_group, signal.SIGKILL)


def _run(p: Plan) -> int:
    """Run the agent until it exits, with its files in a temporary directory removed afterwards;
    its exit status is returned (128 + n for an agent that a signal n killed, as a shell reports it).

    SIGTERM and SIGHUP end the agent (SIGTERM, then SIGKILL after GRACE_S) and the launcher with
    status 128 + the signal, the temporary directory removed. So does SIGINT, unless a terminal
    sent it: Ctrl-C reaches the agent with the launcher, which the agent reads itself, and ending
    it there would cancel every Ctrl-C the agent means to handle. The handlers are installed before
    the agent starts, so a signal that arrives while it starts is not lost.

    Without a terminal (stdin is not the terminal's foreground group) the agent runs in a session
    of its own and a stop ends its whole process group, the processes it started included. At a
    terminal the agent must stay in the foreground group to read the keyboard, so a stop signals the
    agent itself only: a process the agent started and detached from it outlives it."""
    own_group = not _at_terminal()
    stopped: list[int] = []
    agent: list[subprocess.Popen] = []  # empty until the agent exists

    def stop(signum, _frame) -> None:
        if signum == signal.SIGINT and _at_terminal():
            return
        if not stopped:
            stopped.append(signum)
            if agent:
                _signal_agent(agent[0], own_group, signal.SIGTERM)

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with tempfile.TemporaryDirectory(prefix="jit-launch-") as tmp:
            for name, text in p.files.items():
                Path(tmp, name).write_text(text)
            if stopped:  # stopped before the agent started: start nothing
                return 128 + stopped[0]
            child = subprocess.Popen(
                [a.replace("{dir}", tmp) for a in p.argv],
                env=os.environ | p.env,
                start_new_session=own_group,
            )
            agent.append(child)
            try:
                if stopped:  # the signal came while the agent was starting
                    _signal_agent(child, own_group, signal.SIGTERM)
                status = _wait(child, own_group, stopped)
                if stopped:
                    return 128 + stopped[0]
                return 128 - status if status < 0 else status
            finally:
                if child.poll() is None:  # the launcher itself is going down: leave no agent behind
                    _signal_agent(child, own_group, signal.SIGKILL)
                    child.wait()
                if own_group and stopped:  # what the agent started and left running
                    _signal_agent(child, own_group, signal.SIGKILL)
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jit-launch",
        description="Start a coding agent against a running JITEngine engine.",
        epilog="Arguments after -- go to the agent.",
    )
    ap.add_argument("agent", choices=sorted(PLANS))
    ap.add_argument(
        "--server", default=DEFAULT_SERVER, help=f"the engine's URL (default {DEFAULT_SERVER})"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write and run nothing")
    ap.add_argument("--version", action="version", version=f"jit-launch {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    agent_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, agent_args = argv[:cut], argv[cut + 1 :]
    a = _parser().parse_args(argv)
    try:
        served = discover(a.server)
        p = plan(a.agent, served, agent_args)
    except (RuntimeError, ValueError) as exc:
        print(f"jit-launch: {exc}", file=sys.stderr)
        return 1
    if served.context < AGENT_CONTEXT:
        print(
            f"jit-launch: warning: the engine admits {served.context} tokens per request; "
            f"an agent needs about {AGENT_CONTEXT} (see the engine's README for its context limit)",
            file=sys.stderr,
        )
    if a.dry_run:
        _print_plan(a.agent, served, p)
        return 0
    if shutil.which(p.argv[0]) is None:
        print(f"jit-launch: {p.argv[0]} is not installed: {INSTALL[a.agent]}", file=sys.stderr)
        return 1
    return _run(p)
