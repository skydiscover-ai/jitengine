"""The build passes its own command to what it shares with the other Qwen3.6 build: the server names it as the owner
in /v1/models and in its startup line, and the memory monitor's thread carries it."""

import ast
import inspect

import pytest

from jitengine_qwen3_6_35b_a3b.server import api_server
from jitengine_qwen3_6_35b_a3b_m2.memory import control

COMMAND = "jitengine-qwen3.6-35b-a3b-m2"


def test_engine_serve_passes_the_build_command(monkeypatch):
    pytest.importorskip("mlx.core")
    from jitengine_qwen3_6_35b_a3b_m2.engine.engine import Engine

    seen = {}
    monkeypatch.setattr(
        api_server, "serve", lambda engine, host, port, **kw: seen.update(kw, host=host, port=port)
    )
    Engine.serve(object(), "127.0.0.1", 1234)
    assert seen["command"] == COMMAND and seen["port"] == 1234


def test_the_memory_monitor_thread_is_named_after_the_build():
    calls = [
        n
        for n in ast.walk(ast.parse(inspect.getsource(control)))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "MemoryMonitor"
    ]
    assert calls, "control.py no longer starts a MemoryMonitor"
    for call in calls:
        names = {k.arg: k.value for k in call.keywords}
        assert isinstance(names.get("thread_name"), ast.Constant)
        assert names["thread_name"].value == f"{COMMAND}-memory-monitor"
