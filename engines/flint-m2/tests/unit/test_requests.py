"""Request parsing, off a Mac: every OpenAI field the server cannot serve is refused with a 400 that names it,
never ignored, and the values that mean the default pass."""

from __future__ import annotations

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from jitengine_flint_m2.server import BadRequest, Handler, Server, parse_request

CHAT = {"messages": [{"role": "user", "content": "hi"}]}
PROMPT = {"prompt": "hi"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


class _StubEngine:
    """What parse_request reads of an Engine: generation defaults, the tokenizer's entry points, the model name."""

    cfg = SimpleNamespace(gen_defaults={})
    model_name = "flint-int4"

    def __init__(self):
        self.tools = "unrendered"

    def render_chat(self, messages, tools):
        self.tools = tools
        return [1, 2, 3]

    def encode(self, text):
        return [1, 2]


@pytest.mark.parametrize(
    ("chat", "extra", "param"),
    [
        (True, {"n": 2}, "n"),
        (False, {"n": 3}, "n"),
        (True, {"logit_bias": {"5": 1}}, "logit_bias"),
        (False, {"logit_bias": {"5": -100}}, "logit_bias"),
        (False, {"best_of": 2}, "best_of"),
        (False, {"suffix": "x"}, "suffix"),
        (True, {"functions": [{"name": "f"}]}, "functions"),
        (True, {"tools": TOOLS, "tool_choice": "required"}, "tool_choice"),
        (True, {"tool_choice": {"type": "function", "function": {"name": "f"}}}, "tool_choice"),
        (True, {"function_call": {"name": "f"}}, "function_call"),
        (True, {"response_format": {"type": "json_object"}}, "response_format"),
        (True, {"response_format": "text"}, "response_format"),
    ],
)
def test_unsupported_fields_are_refused(chat, extra, param):
    eng = _StubEngine()
    with pytest.raises(BadRequest) as e:
        parse_request(eng, {**(CHAT if chat else PROMPT), **extra}, chat)
    assert e.value.param == param and str(e.value).startswith("Unsupported "), str(e.value)
    assert eng.tools == "unrendered"  # refused before the template runs


def test_default_values_pass():
    """n and best_of 1, an empty logit_bias or suffix, text responses and tool_choice auto pass; tool_choice none
    renders the request without its tools."""
    eng = _StubEngine()
    body = {**CHAT, "n": 1, "logit_bias": {}, "response_format": {"type": "text"}}
    parse_request(eng, {**body, "tools": TOOLS, "tool_choice": "auto"}, True)
    assert eng.tools == TOOLS
    parse_request(
        eng, {**CHAT, "tools": TOOLS, "tool_choice": "none", "function_call": "none"}, True
    )
    assert eng.tools is None
    req, _ = parse_request(eng, {**PROMPT, "n": 1, "best_of": 1, "suffix": ""}, False)
    assert req.prompt_ids == [1, 2]


def test_refusal_over_http_names_the_field():
    """The 400 carries the error object with `param`, so a client can tell which field it must drop."""
    srv = Server(("127.0.0.1", 0), Handler)
    srv.engine, srv.stopping = _StubEngine(), threading.Event()
    srv.inflight, srv.lock = [0], threading.Lock()
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps({**CHAT, "n": 2}),
            {"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        err = json.loads(r.read())["error"]
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.status == 400 and err["param"] == "n" and err["type"] == "invalid_request_error", err
