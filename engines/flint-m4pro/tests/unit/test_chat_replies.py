"""Chat replies over HTTP, off a Mac, with a stand-in engine: with tools, the reply's reasoning, content
and tool calls, whole and streamed, and as the official SDK reads them; without tools, the text Flint
wrote; /v1/models' context limit; and the same reply on the Anthropic Messages API and OpenAI's Responses
API."""

from __future__ import annotations

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from jitengine_flint_m4pro.engine.io_struct import GenResult, StepOutput
from jitengine_flint_m4pro.server import Handler, Server

REPLY = (
    "<think>Read it first.</think>\nReading it.\n"
    "<tool_call>read<arg_key>path</arg_key><arg_value>a.py</arg_value>"
    "<arg_key>limit</arg_key><arg_value>20</arg_value></tool_call>"
)
READ = {
    "type": "function",
    "function": {
        "name": "read",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        },
    },
}
CONTEXT_LIMIT = 65536


class _Tokenizer:
    """Byte-level: token id = byte + 3."""

    def decode(self, ids):
        return bytes(t - 3 for t in ids if t >= 3).decode("utf-8", errors="replace")

    def piece(self, token_id):
        return chr(token_id - 3)


class _Engine:
    """What the server reads of an Engine; every request is answered with REPLY, one byte per step."""

    cfg = SimpleNamespace(gen_defaults={})
    model_name = "flint-int4"
    profile, kv_bits, context_limit = "main", 16, CONTEXT_LIMIT

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.tools = None  # what the latest chat request rendered

    def render_chat(self, messages, tools=None):
        self.tools = tools
        return self.encode(json.dumps(messages))

    def encode(self, text):
        return [b + 3 for b in text.encode()]

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def stream(self, req, cancel=None):
        ids, text = self.encode(REPLY), None
        hits = [(REPLY.find(s), s) for s in req.stop if s in REPLY]
        if hits:  # as the engine: the ids end with the stop string and the text is cut before it
            at, stop = min(hits)
            ids, text = self.encode(REPLY[: at + len(stop)]), REPLY[:at]
        for i, t in enumerate(ids):
            yield StepOutput(i, t)
        yield GenResult(ids, "stop", len(req.prompt_ids), cached_prompt_tokens=2, text=text)

    def health(self):
        return {"status": "ready"}

    def log(self, msg):
        pass


@pytest.fixture(scope="module")
def server():
    srv = Server(("127.0.0.1", 0), Handler)
    srv.engine, srv.stopping = _Engine(), threading.Event()
    srv.inflight, srv.lock = [0], threading.Lock()
    srv.stream_cancel, srv.score_cancel = True, False
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(srv, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    headers = {"Content-Type": "application/json"}
    conn.request(method, path, None if body is None else json.dumps(body), headers)
    r = conn.getresponse()
    data = r.read().decode()
    conn.close()
    return r.status, data


def _chat(srv, **body):
    msgs = [{"role": "user", "content": "read a.py"}]
    return _request(srv, "POST", "/v1/chat/completions", {"messages": msgs, **body})


def _deltas(data: str) -> list[dict]:
    frames = [b[len("data: ") :] for b in data.split("\n\n") if b.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    return [json.loads(f) for f in frames[:-1]]


def test_a_reply_with_tools_has_reasoning_content_and_tool_calls(server):
    status, data = _chat(server, tools=[READ])
    ch = json.loads(data)["choices"][0]
    assert status == 200 and server.engine.tools == [READ]
    assert ch["finish_reason"] == "tool_calls"
    msg = ch["message"]
    assert msg["reasoning_content"] == "Read it first." and msg["content"] == "Reading it."
    (call,) = msg["tool_calls"]
    assert call["id"].startswith("call_") and call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py", "limit": 20}


def test_a_stream_with_tools_sends_reasoning_then_content_then_the_calls(server):
    status, data = _chat(server, tools=[READ], stream=True)
    chunks = _deltas(data)
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert status == 200 and deltas[0]["role"] == "assistant"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "Read it first."
    assert "".join(d.get("content", "") for d in deltas) == "Reading it."
    calls = [tc for d in deltas for tc in d.get("tool_calls", ())]
    assert [(tc["index"], tc["function"]["name"]) for tc in calls] == [(0, "read")]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_without_tools_the_reply_is_the_text_flint_wrote(server):
    for body in ({}, {"tools": [READ], "tool_choice": "none"}):
        status, data = _chat(server, **body)
        ch = json.loads(data)["choices"][0]
        assert status == 200 and server.engine.tools is None
        assert ch["message"] == {"role": "assistant", "content": REPLY}
        assert ch["finish_reason"] == "stop"
        status, data = _chat(server, stream=True, **body)
        deltas = [c["choices"][0]["delta"] for c in _deltas(data)]
        assert "".join(d["content"] for d in deltas) == REPLY
        assert not any(set(d) - {"role", "content"} for d in deltas)


def _events(data: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in data.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_without_tools_the_messages_and_responses_apis_still_split_the_reasoning(server):
    """The two APIs give reasoning its own block or item, so a tool-less request must not leak
    `<think>...</think>` into the visible text (the OpenAI chat path keeps the text Flint wrote)."""
    visible = REPLY.split("</think>\n", 1)[1]  # no tools offered: the call text stays text
    msgs = [{"role": "user", "content": "read a.py"}]
    ask = {"model": "flint", "max_tokens": 400, "messages": msgs}

    status, data = _request(server, "POST", "/v1/messages", ask)
    blocks = json.loads(data)["content"]
    assert status == 200 and server.engine.tools is None  # none rendered, reasoning still read
    assert [b["type"] for b in blocks] == ["thinking", "text"]
    assert blocks[0]["thinking"] == "Read it first." and blocks[1]["text"] == visible
    status, data = _request(server, "POST", "/v1/messages", {**ask, "stream": True})
    deltas = [e["delta"] for e in _events(data) if e["type"] == "content_block_delta"]
    assert "".join(d.get("thinking", "") for d in deltas) == "Read it first."
    assert "".join(d.get("text", "") for d in deltas) == visible

    ask = {"model": "flint", "input": "read a.py"}
    status, data = _request(server, "POST", "/v1/responses", ask)
    output = json.loads(data)["output"]
    assert status == 200 and [i["type"] for i in output] == ["reasoning", "message"]
    assert output[1]["content"][0]["text"] == visible
    status, data = _request(server, "POST", "/v1/responses", {**ask, "stream": True})
    events = _events(data)
    reasoning = "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta")
    text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert reasoning == "Read it first." and text == visible


def test_the_messages_api_reports_the_stop_sequence_that_ended_the_reply(server):
    """Anthropic's reply names the client's stop sequence: stop_reason "stop_sequence" and the one
    that matched (the earliest in the text), where an end of text is "end_turn" with none."""
    ask = {"model": "flint", "max_tokens": 400, "messages": [{"role": "user", "content": "read"}]}

    def whole(stops, **extra):
        status, data = _request(
            server, "POST", "/v1/messages", {**ask, "stop_sequences": stops, **extra}
        )
        assert status == 200
        return json.loads(data)

    for extra in (
        {},
        {"tools": [{"name": "read", "input_schema": READ["function"]["parameters"]}]},
    ):
        msg = whole(["zzz", "it.", "Reading"], **extra)
        assert (msg["stop_reason"], msg["stop_sequence"]) == ("stop_sequence", "Reading")
        assert [b["type"] for b in msg["content"]] == ["thinking"]  # cut before the text
    assert whole(["zzz"])["stop_sequence"] is None
    assert whole(["zzz"])["stop_reason"] == "end_turn"

    status, data = _request(
        server, "POST", "/v1/messages", {**ask, "stop_sequences": ["it."], "stream": True}
    )
    (end,) = [e for e in _events(data) if e["type"] == "message_delta"]
    assert status == 200
    assert end["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "it."}
    # the OpenAI chat API keeps its own shape
    status, data = _chat(server, stop=["it."])
    assert "x_stop_sequence" not in json.loads(data)["choices"][0]


def test_models_report_the_context_limit(server):
    status, data = _request(server, "GET", "/v1/models")
    (model,) = json.loads(data)["data"]
    assert status == 200 and model["max_model_len"] == CONTEXT_LIMIT


def test_sdk_reads_the_reply_whole_and_streamed(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", api_key="x", max_retries=0
    )
    msgs = [{"role": "user", "content": "read a.py"}]
    whole = client.chat.completions.create(model="flint", messages=msgs, tools=[READ]).choices[0]
    with client.chat.completions.stream(model="flint", messages=msgs, tools=[READ]) as s:
        streamed = s.get_final_completion().choices[0]
    for ch in (whole, streamed):
        assert ch.finish_reason == "tool_calls" and ch.message.content == "Reading it."
        (call,) = ch.message.tool_calls
        assert json.loads(call.function.arguments) == {"path": "a.py", "limit": 20}


def test_the_messages_api_serves_the_same_reply(server):
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(
        base_url=f"http://127.0.0.1:{server.server_address[1]}", api_key="x", max_retries=0
    )
    tool = {"name": "read", "input_schema": READ["function"]["parameters"]}
    ask = {
        "model": "flint",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "read a.py"}],
    }
    whole = client.messages.create(**ask, tools=[tool])
    with client.messages.stream(**ask, tools=[tool]) as s:
        streamed = s.get_final_message()
    for m in (whole, streamed):
        assert [b.type for b in m.content] == ["thinking", "text", "tool_use"]
        assert m.content[0].thinking == "Read it first." and m.content[1].text == "Reading it."
        assert m.content[2].input == {"path": "a.py", "limit": 20} and m.stop_reason == "tool_use"
    counted = client.messages.count_tokens(model="flint", messages=ask["messages"])
    assert counted.input_tokens == len(server.engine.encode(json.dumps(ask["messages"])))
    with pytest.raises(anthropic.BadRequestError):
        client.messages.create(**ask, extra_body={"temperature": "hot"})


def test_the_responses_api_serves_the_same_reply(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", api_key="x", max_retries=0
    )
    ask = {
        "model": "flint",
        "input": "read a.py",
        "tools": [{"type": "function", **READ["function"]}],
    }
    whole = client.responses.create(**ask)
    with client.responses.stream(**ask) as s:
        streamed = s.get_final_response()
    for r in (whole, streamed):
        assert [i.type for i in r.output] == ["reasoning", "message", "function_call"]
        assert r.output[0].content[0].text == "Read it first." and r.output_text == "Reading it."
        assert json.loads(r.output[2].arguments) == {"path": "a.py", "limit": 20}
    with pytest.raises(openai.BadRequestError):
        client.responses.create(**ask, extra_body={"temperature": "hot"})
