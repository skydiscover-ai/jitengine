"""The HTTP server against OpenAI's API, over a real socket, with a stand-in engine (no MLX).

The contract is the `openai` package's types and OpenAI's API reference, not this server's own
code: the error object, and the status, type, code and Retry-After of each refusal and failure,
before the 200 and after it; the context-length error, max_completion_tokens and the max_tokens
defaults, a 400 naming each parameter the server cannot serve, and /v1/models; stop sequences, and
the stream's framing (the role chunk first, usage only on a final `choices: []` chunk when asked
for, `data: [DONE]`); tools, tool-call history and the tool calls a reply makes, whole and
streamed; and the Anthropic Messages API and OpenAI's Responses API on the same port. The tests
that drive the official SDKs skip when `openai` or `anthropic` is not installed.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from jitengine_qwen3_6_35b_a3b_m2.engine.errors import (
    AdmissionPausedError,
    ContextTooLargeError,
    MemoryPressureError,
    QueueFullError,
)
from jitengine_qwen3_6_35b_a3b_m2.engine.io_struct import GenResult
from jitengine_qwen3_6_35b_a3b_m2.server import api_server as srvmod
from jitengine_qwen3_6_35b_a3b_m2.server.openai_api import ChunkWriter
from jitengine_qwen3_6_35b_a3b_m2.server.protocol import (
    COMPLETIONS_DEFAULT_MAX_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_STOP_SEQUENCES,
)

EOS = 1
REPLY = "Hello there. Observation: tool output\nFinal answer: 42."
CALL_REPLY = (
    "Reading it.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
    "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>"
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
KV_POSITIONS = 4096


class StandInTokenizer:
    """Byte-level: token id = byte + 3, ids 0..2 are control tokens, chat is "role: content"."""

    def encode(self, text: str) -> list[int]:
        return [b + 3 for b in text.encode()]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out, raw = [], bytearray()
        for t in ids:
            if t >= 3:
                raw.append(t - 3)
            elif not skip_special_tokens:
                out.append(raw.decode("utf-8", errors="replace") + f"<|c{t}|>")
                raw = bytearray()
        return "".join(out) + raw.decode("utf-8", errors="replace")

    def render_chat(self, messages, tools=None, **kw) -> list[int]:
        text = f"tools: {','.join(t['function']['name'] for t in tools)}\n" if tools else ""
        for m in messages:
            if m["role"] not in ("system", "user", "assistant", "tool"):
                raise RuntimeError("Unexpected message role.")  # as the checkpoint's template
            c = m.get("content")
            c = "".join(p["text"] for p in c) if isinstance(c, list) else (c or "")
            calls = "".join(f" [{tc['function']['name']}]" for tc in m.get("tool_calls") or ())
            text += f"{m['role']}: {c}{calls}\n"
        return self.encode(text + "assistant: ")


class StandInEngine:
    """The engine contract the server calls. Markers in the prompt pick a behavior."""

    model_name = "stand-in-model"
    vocab = 259
    max_kv_positions = KV_POSITIONS

    def __init__(self):
        self.tokenizer = StandInTokenizer()
        self.last = None  # (prompt text, max_tokens) of the latest request
        self.cancelled = 0  # rows closed before their done event

    def events(self, req):
        """As Engine.events: a generator that raises the engine's refusal at its first step."""
        text = self.tokenizer.decode(req.prompt_ids)
        self.last = (text, int(req.max_tokens))
        if "#queuefull" in text:
            raise QueueFullError("9 requests outstanding (limit 8); retry later")
        if "#paused" in text:
            raise AdmissionPausedError("memory_pressure")
        if "#refuse" in text:
            raise ValueError("token id 300 out of range")
        if "#toolarge" in text:
            raise ContextTooLargeError(
                "prompt + max_tokens = 5000 tokens exceeds the context limit"
            )
        if "#crash" in text:
            raise RuntimeError("Metal command buffer failed")
        reply = self.tokenizer.encode(CALL_REPLY if "#call" in text else REPLY)
        reply = [] if "#empty" in text else reply
        out, finish, done = [], "length", False
        try:
            for tok in [*reply, EOS]:
                if len(out) >= req.max_tokens:
                    break
                if "#midfail" in text and len(out) == 2:
                    raise MemoryPressureError("the process would exceed its memory ceiling")
                if tok == EOS:
                    finish = "stop"
                    break
                out.append(tok)
                yield "token", tok
            done = True
            yield "done", GenResult(out, finish, len(req.prompt_ids), cached_prompt_tokens=2)
        finally:
            if not done:
                self.cancelled += 1

    def health(self) -> dict:
        return {"status": "ok"}


@pytest.fixture(scope="module")
def server():
    engine = StandInEngine()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    threading.Thread(target=srvmod.serve, args=(engine, "127.0.0.1", port), daemon=True).start()
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.02)
    return engine, port


def post(server, path: str, body: dict):
    c = http.client.HTTPConnection("127.0.0.1", server[1], timeout=30)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def chat(server, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    status, headers, data = post(server, "/v1/chat/completions", {"model": "m", **body})
    return status, headers, json.loads(data)


def frames(data: bytes) -> list:
    """The SSE data frames of a stream, `[DONE]` kept as a string."""
    out = []
    for block in data.decode().split("\n\n"):
        if block.startswith("data: "):
            payload = block[len("data: ") :]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def stream(server, path: str = "/v1/chat/completions", **body):
    if path.endswith("/chat/completions"):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
    status, headers, data = post(server, path, {"model": "m", "stream": True, **body})
    return status, headers, frames(data)


def assert_error(body, etype: str, param, code):
    err = body["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["type"] == etype and err["param"] == param and err["code"] == code
    assert isinstance(err["message"], str) and err["message"]


# ------------------------------------------------------------------ errors
def test_error_object_has_openai_shape(server):
    status, _, body = chat(server, messages=[])
    assert status == 400
    assert_error(body, "invalid_request_error", "messages", None)
    status, _, body = chat(server, temperature="hot")
    assert status == 400
    assert_error(body, "invalid_request_error", "temperature", None)
    c = http.client.HTTPConnection("127.0.0.1", server[1], timeout=30)
    c.request("GET", "/v1/nope")
    r = c.getresponse()
    assert r.status == 404
    assert_error(json.loads(r.read()), "invalid_request_error", None, None)


def test_an_oversized_body_gets_the_error_object(server):
    c = http.client.HTTPConnection("127.0.0.1", server[1], timeout=30)
    c.putrequest("POST", "/v1/completions")
    c.putheader("Content-Type", "application/json")
    c.putheader("Content-Length", str(srvmod.MAX_BODY_BYTES + 1))
    c.endheaders()
    r = c.getresponse()
    assert r.status == 413 and r.getheader("Connection") == "close"
    assert_error(json.loads(r.read()), "invalid_request_error", None, None)
    c.close()


def test_engine_refusals_map_to_status_type_and_retry_after(server):
    for marker, status, etype, code, retry in (
        ("#queuefull", 429, "rate_limit_error", "rate_limit_exceeded", "1"),
        ("#paused", 503, "service_unavailable_error", "server_is_overloaded", "5"),
        ("#midfail", 503, "service_unavailable_error", "server_is_overloaded", "5"),
        ("#refuse", 400, "invalid_request_error", None, None),
        ("#crash", 500, "server_error", None, None),
    ):
        got, headers, body = chat(server, messages=[{"role": "user", "content": marker}])
        assert got == status, marker
        assert_error(body, etype, None, code)
        assert headers.get("Retry-After") == retry, marker


def test_a_failure_after_the_200_is_an_error_frame(server):
    for include_usage in (False, True):
        status, _, fs = stream(
            server,
            messages=[{"role": "user", "content": "#midfail"}],
            stream_options={"include_usage": include_usage},
        )
        assert status == 200 and fs[-1] == "[DONE]" and set(fs[-2]) == {"error"}
        assert_error(fs[-2], "service_unavailable_error", None, "server_is_overloaded")
        # the choice still ends with "error", which a client that ignores error frames reads
        ends = [c["finish_reason"] for f in fs[:-2] for c in f["choices"]]
        assert ends[-1] == "error" and ends.count("error") == 1
        assert [f["choices"] == [] for f in fs[:-2]].count(True) == int(include_usage)
    body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "#queuefull"}]}
    status, _, data = post(server, "/v1/chat/completions", body)
    assert status == 429 and json.loads(data)["error"]["type"] == "rate_limit_error"


# ------------------------------------------------------------------ request fields
def test_context_overflow_is_context_length_exceeded(server):
    status, _, body = chat(server, max_tokens=KV_POSITIONS)
    assert status == 400
    assert_error(body, "invalid_request_error", "messages", "context_length_exceeded")
    assert body["error"]["message"].startswith("This model's maximum context length is 4096")
    long_prompt = "x" * (MAX_PROMPT_TOKENS + 1)
    status, _, data = post(server, "/v1/completions", {"model": "m", "prompt": long_prompt})
    assert status == 400
    assert_error(json.loads(data), "invalid_request_error", "prompt", "context_length_exceeded")
    # the engine's own refusal carries the same code
    status, _, body = chat(server, messages=[{"role": "user", "content": "#toolarge"}])
    assert status == 400
    assert_error(body, "invalid_request_error", None, "context_length_exceeded")


def test_max_completion_tokens_and_the_defaults(server):
    engine = server[0]
    assert chat(server, max_completion_tokens=5)[2]["usage"]["completion_tokens"] == 5
    assert chat(server, max_tokens=9, max_completion_tokens=4)[2]["usage"]["completion_tokens"] == 4
    status, _, body = chat(server)  # no limit: the rest of the request's context
    prompt_tokens = body["usage"]["prompt_tokens"]
    assert status == 200 and engine.last[1] == KV_POSITIONS - prompt_tokens
    status, _, _ = post(server, "/v1/completions", {"model": "m", "prompt": "Once"})
    assert status == 200 and engine.last[1] == COMPLETIONS_DEFAULT_MAX_TOKENS


def test_unsupported_parameters_are_refused_by_name(server):
    for body, param, code in (
        ({"n": 2}, "n", "unsupported_value"),
        ({"functions": [{"name": "f"}]}, "functions", "unsupported_parameter"),
        ({"tool_choice": "required"}, "tool_choice", "unsupported_value"),
        (
            {"tools": [READ], "tool_choice": {"type": "function"}},
            "tool_choice",
            "unsupported_value",
        ),
        ({"tools": [{"name": "read"}]}, "tools", None),
        ({"response_format": {"type": "json_object"}}, "response_format", "unsupported_value"),
        ({"logprobs": True}, "logprobs", "unsupported_parameter"),
        ({"top_logprobs": 2}, "top_logprobs", "unsupported_parameter"),
        ({"logit_bias": {"5": -100}}, "logit_bias", "unsupported_parameter"),
    ):
        status, _, got = chat(server, **body)
        assert status == 400, body
        assert_error(got, "invalid_request_error", param, code)
    for body, param in (
        ({"logprobs": 1}, "logprobs"),
        ({"echo": True}, "echo"),
        ({"suffix": "}"}, "suffix"),
        ({"best_of": 2}, "best_of"),
    ):
        status, _, data = post(server, "/v1/completions", {"model": "m", "prompt": "x", **body})
        assert status == 400 and json.loads(data)["error"]["param"] == param
    # the same parameters at their no-op values are accepted
    noop = {
        "n": 1,
        "tools": [],
        "tool_choice": "none",
        "response_format": {"type": "text"},
        "logprobs": False,
        "logit_bias": {},
        "user": "u",
        "seed": 3,
    }
    assert chat(server, **noop)[0] == 200


def test_developer_role_and_content_parts(server):
    msgs = [
        {"role": "developer", "content": "Be terse."},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    status, _, _ = chat(server, messages=msgs)
    assert status == 200 and server[0].last[0].startswith("system: Be terse.\nuser: hi\n")
    image = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    status, _, body = chat(server, messages=[{"role": "user", "content": [image]}])
    assert status == 400 and body["error"]["param"] == "messages"


def test_a_prompt_batch_of_one_is_that_prompt(server):
    for prompt in (["Once"], [[82, 113, 102, 104]]):
        status, _, data = post(server, "/v1/completions", {"model": "m", "prompt": prompt})
        assert status == 200, prompt
        assert server[0].last[0] == "Once"
    status, _, data = post(server, "/v1/completions", {"model": "m", "prompt": ["a", "b"]})
    assert status == 400 and json.loads(data)["error"]["param"] == "prompt"


def test_chat_response_shape(server):
    status, _, body = chat(server)
    assert status == 200 and body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-") and isinstance(body["created"], int)
    assert body["model"] == "m"
    ch = body["choices"][0]
    assert ch["index"] == 0 and ch["message"] == {"role": "assistant", "content": REPLY}
    assert ch["finish_reason"] == "stop"
    u = body["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["prompt_tokens_details"] == {"cached_tokens": 2}
    assert chat(server, model=7)[2]["model"] == "stand-in-model"  # not a name: not echoed
    status, _, data = post(server, "/v1/completions", {"model": "m", "prompt": "x"})
    assert json.loads(data)["id"].startswith("cmpl-")


def test_models_list_and_retrieve(server):
    c = http.client.HTTPConnection("127.0.0.1", server[1], timeout=30)
    c.request("GET", "/v1/models")
    listed = json.loads(c.getresponse().read())
    assert listed["object"] == "list" and listed["data"][0]["id"] == "stand-in-model"
    # the request's context: the engine's KV row, below the prompt bound
    assert listed["data"][0]["max_model_len"] == min(KV_POSITIONS, MAX_PROMPT_TOKENS)
    c.request("GET", "/v1/models/stand-in-model")
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read()) == listed["data"][0]
    c.request("GET", "/v1/models/other")
    r = c.getresponse()
    assert r.status == 404
    assert_error(json.loads(r.read()), "invalid_request_error", "model", "model_not_found")


# ------------------------------------------------------------------ tools
def test_a_reply_s_tool_calls_come_back_as_tool_calls(server):
    msgs = [{"role": "user", "content": "#call"}]
    status, _, body = chat(server, messages=msgs, tools=[READ])
    ch = body["choices"][0]
    assert status == 200 and ch["finish_reason"] == "tool_calls"
    assert server[0].last[0].startswith("tools: read\n")
    assert ch["message"]["content"] == "Reading it."
    (call,) = ch["message"]["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"]["name"] == "read"
    # a string parameter keeps its text, any other is read as JSON
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py", "limit": 20}
    # without tools, or with tool_choice "none", the template offers none and the reply is text
    for extra in ({}, {"tools": [READ], "tool_choice": "none"}):
        status, _, body = chat(server, messages=msgs, **extra)
        assert status == 200 and not server[0].last[0].startswith("tools:")
        assert body["choices"][0]["message"] == {"role": "assistant", "content": CALL_REPLY}
        assert body["choices"][0]["finish_reason"] == "stop"


def test_a_stream_sends_its_tool_calls_after_its_text(server):
    msgs = [{"role": "user", "content": "#call"}]
    status, _, fs = stream(server, messages=msgs, tools=[READ])
    deltas = [c["delta"] for f in fs[:-1] for c in f["choices"]]
    assert status == 200 and "".join(d.get("content") or "" for d in deltas) == "Reading it."
    calls = [tc for d in deltas for tc in d.get("tool_calls", ())]
    assert [(tc["index"], tc["function"]["name"]) for tc in calls] == [(0, "read")]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py", "limit": 20}
    assert fs[-2]["choices"][0]["finish_reason"] == "tool_calls"
    # a reply without calls streams its text whole, the whitespace held back included
    status, _, fs = stream(server, tools=[READ])
    pieces = [c["delta"].get("content", "") for f in fs[:-1] for c in f["choices"]]
    assert "".join(pieces) == REPLY and fs[-2]["choices"][0]["finish_reason"] == "stop"


def test_tool_call_history_renders(server):
    msgs = [
        {"role": "user", "content": "read a.py"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    ]
    status, _, _ = chat(server, messages=msgs, tools=[READ])
    assert status == 200
    want = "tools: read\nuser: read a.py\nassistant:  [read]\ntool: print(1)\nassistant: "
    assert server[0].last[0] == want
    no_content = [msgs[0], {k: v for k, v in msgs[1].items() if k != "content"}, msgs[2]]
    assert chat(server, messages=no_content, tools=[READ])[0] == 200


# ------------------------------------------------------------------ stop sequences and streams
def test_stop_sequences_end_the_text_before_them(server):
    engine = server[0]
    before = engine.cancelled
    for stop in ("Observation:", ["\n", "Observation:"]):
        status, _, body = chat(server, stop=stop)
        ch = body["choices"][0]
        assert status == 200 and ch["finish_reason"] == "stop"
        assert ch["message"]["content"] == "Hello there. "
        # the row ended at the stop, before the engine's done event reported its cached tokens
        assert "prompt_tokens_details" not in body["usage"]
    assert engine.cancelled == before + 2
    body = {"model": "m", "prompt": "x", "stop": "42", "max_tokens": 100}
    choice = json.loads(post(server, "/v1/completions", body)[2])["choices"][0]
    assert choice["text"] == REPLY[: REPLY.index("42")] and choice["finish_reason"] == "stop"
    status, _, body = chat(server, stop=["a"] * (MAX_STOP_SEQUENCES + 1))
    assert status == 400 and body["error"]["param"] == "stop"


def test_streamed_stop_never_leaks_a_partial_match(server):
    status, _, fs = stream(server, stop=["Observation:"], stream_options={"include_usage": True})
    pieces = [c["delta"].get("content", "") for f in fs[:-1] for c in f["choices"]]
    assert status == 200 and "".join(pieces) == "Hello there. "
    assert fs[-3]["choices"][0]["finish_reason"] == "stop"
    assert "prompt_tokens_details" not in fs[-2]["usage"]
    status, _, fs = stream(server, stop=["zzz"])  # no match: the held-back tail is released
    pieces = [c["delta"].get("content", "") for f in fs[:-1] for c in f["choices"]]
    assert "".join(pieces) == REPLY and fs[-2]["choices"][0]["finish_reason"] == "stop"


def test_the_messages_api_reports_the_stop_sequence_that_ended_the_reply(server):
    """Anthropic's reply names the client's stop sequence: stop_reason "stop_sequence" and the one
    that matched (the earliest in the text), where an end of text is "end_turn" with none."""
    ask = {"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}

    def whole(stops):
        status, _, data = post(server, "/v1/messages", {**ask, "stop_sequences": stops})
        assert status == 200
        return json.loads(data)

    msg = whole(["Final", "Observation:"])
    assert (msg["stop_reason"], msg["stop_sequence"]) == ("stop_sequence", "Observation:")
    assert msg["content"] == [{"type": "text", "text": "Hello there. "}]
    assert whole(["there", "Hello"])["stop_sequence"] == "Hello"  # the earliest in the text
    msg = whole(["zzz"])  # never met: the reply ended by itself
    assert (msg["stop_reason"], msg["stop_sequence"]) == ("end_turn", None)

    status, _, data = post(
        server, "/v1/messages", {**ask, "stop_sequences": ["Final"], "stream": True}
    )
    deltas = [
        json.loads(line[len("data: ") :])
        for line in data.decode().splitlines()
        if line.startswith("data: ") and '"message_delta"' in line
    ]
    assert status == 200 and len(deltas) == 1
    assert deltas[0]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "Final"}
    # the OpenAI APIs keep their own shapes: no extra field
    assert "x_stop_sequence" not in chat(server, stop=["Final"])[2]["choices"][0]


def test_stream_framing_role_first_and_usage_only_when_asked(server):
    status, headers, fs = stream(server)
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    assert "Content-Length" not in headers and fs[-1] == "[DONE]"
    assert fs[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(f["object"] == "chat.completion.chunk" for f in fs[:-1])
    assert not any("usage" in f for f in fs[:-1])  # not asked for: absent everywhere
    assert fs[-2]["choices"][0]["finish_reason"] == "stop"
    status, _, fs = stream(server, stream_options={"include_usage": True})
    with_usage = [f for f in fs[:-1] if f["usage"] is not None]
    assert len(with_usage) == 1 and with_usage[0] is fs[-2] and fs[-2]["choices"] == []
    assert fs[-2]["usage"]["completion_tokens"] == len(REPLY.encode())
    assert fs[-2]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    status, _, fs = stream(server, messages=[{"role": "user", "content": "#empty"}])
    assert fs[0]["choices"][0]["delta"]["role"] == "assistant"  # even with no token


def test_the_role_chunk_shares_the_first_token_s_write():
    writes = []

    class Recorder:
        def write(self, data):
            writes.append(data)

        def flush(self):
            pass

    out = ChunkWriter(Recorder(), "chatcmpl-1", "m", chat=True, include_usage=False)
    out.choice({"role": "assistant", "content": ""}, hold=True)
    out.choice({"content": "Hi"}, token_ids=[5])
    assert len(writes) == 1
    role, token = frames(writes[0])
    assert role["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert token["choices"][0]["delta"] == {"content": "Hi"}


def test_completions_stream_objects_are_text_completion(server):
    status, _, fs = stream(server, "/v1/completions", prompt="Once", max_tokens=4)
    assert status == 200 and fs[-1] == "[DONE]"
    assert {f["object"] for f in fs[:-1]} == {"text_completion"}
    assert "".join(c["text"] for f in fs[:-1] for c in f["choices"]) == "Hell"


# ------------------------------------------------------------------ the official SDK
@pytest.fixture(scope="module")
def client(server):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url=f"http://127.0.0.1:{server[1]}/v1", api_key="x", max_retries=0)


def test_sdk_raises_the_right_errors(client):
    import openai

    def ask(marker, **kw):
        msgs = [{"role": "user", "content": marker}]
        return client.chat.completions.create(model="m", messages=msgs, **kw)

    with pytest.raises(openai.RateLimitError):
        ask("#queuefull")
    with pytest.raises(openai.InternalServerError) as e:
        ask("#paused")
    assert e.value.status_code == 503 and e.value.code == "server_is_overloaded"
    with pytest.raises(openai.InternalServerError) as e:
        ask("#crash")
    assert e.value.status_code == 500 and e.value.type == "server_error"
    with pytest.raises(openai.APIError) as e:
        list(ask("#midfail", stream=True))
    assert e.value.code == "server_is_overloaded"


def test_sdk_reads_completions_models_and_the_context_length_error(client):
    import openai

    msgs = [{"role": "user", "content": "hi"}]
    r = client.chat.completions.create(model="m", messages=msgs)
    assert r.choices[0].message.content == REPLY and r.choices[0].finish_reason == "stop"
    c = client.completions.create(model="m", prompt="Once", max_tokens=3)
    assert c.object == "text_completion" and c.choices[0].text == "Hel"
    assert [m.id for m in client.models.list()] == ["stand-in-model"]
    assert client.models.retrieve("stand-in-model").id == "stand-in-model"
    with pytest.raises(openai.BadRequestError) as e:
        client.chat.completions.create(model="m", messages=msgs, max_tokens=KV_POSITIONS)
    assert e.value.code == "context_length_exceeded" and e.value.param == "messages"


def test_sdk_parses_streams(client):
    msgs = [{"role": "user", "content": "hi"}]
    chunks = list(
        client.chat.completions.create(
            model="m", messages=msgs, stream=True, stream_options={"include_usage": True}
        )
    )
    assert [c.usage is not None for c in chunks].count(True) == 1 and chunks[-1].choices == []
    assert "".join(c.choices[0].delta.content or "" for c in chunks[:-1]) == REPLY
    with client.chat.completions.stream(model="m", messages=msgs) as s:
        final = s.get_final_completion()
    assert final.choices[0].message.role == "assistant" and final.choices[0].finish_reason == "stop"
    with client.chat.completions.stream(model="m", messages=msgs, stop=["Observation:"]) as s:
        final = s.get_final_completion()
    assert final.choices[0].message.content == "Hello there. "
    texts = client.completions.create(model="m", prompt="Once", max_tokens=4, stream=True)
    assert "".join(c.choices[0].text for c in texts) == "Hell"


def test_sdk_reads_tool_calls_whole_and_streamed(client):
    msgs = [{"role": "user", "content": "#call"}]
    r = client.chat.completions.create(model="m", messages=msgs, tools=[READ])
    whole = r.choices[0]
    with client.chat.completions.stream(model="m", messages=msgs, tools=[READ]) as s:
        streamed = s.get_final_completion().choices[0]
    for ch in (whole, streamed):
        assert ch.finish_reason == "tool_calls" and ch.message.content == "Reading it."
        (call,) = ch.message.tool_calls
        assert call.function.name == "read"
        assert json.loads(call.function.arguments) == {"path": "a.py", "limit": 20}


# ------------------------------------------------------------------ the Anthropic Messages API
ANTHROPIC_READ = {"name": "read", "input_schema": READ["function"]["parameters"]}


@pytest.fixture(scope="module")
def messages(server):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(base_url=f"http://127.0.0.1:{server[1]}", api_key="x", max_retries=0)


def _ask(text: str, **kw) -> dict:
    return {"model": "m", "max_tokens": 256, "messages": [{"role": "user", "content": text}], **kw}


def test_messages_serve_the_same_chat_whole_and_streamed(messages, server):
    ask = _ask("#call", tools=[ANTHROPIC_READ])
    whole = messages.messages.create(**ask)
    with messages.messages.stream(**ask) as s:
        streamed = s.get_final_message()
    assert server[0].last[0].startswith("tools: read\n")
    for m in (whole, streamed):
        assert [b.type for b in m.content] == ["text", "tool_use"] and m.model == "m"
        assert m.content[0].text == "Reading it." and m.stop_reason == "tool_use"
        assert m.content[1].name == "read" and m.content[1].input == {"path": "a.py", "limit": 20}
        assert m.usage.cache_read_input_tokens == 2  # the stand-in's cached prompt tokens
    plain = messages.messages.create(**_ask("hi"))
    assert [b.text for b in plain.content] == [REPLY] and plain.stop_reason == "end_turn"


def test_messages_count_the_tokens_the_chat_would_render(messages, server):
    counted = messages.messages.count_tokens(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert counted.input_tokens == chat(server)[2]["usage"]["prompt_tokens"]


def test_messages_errors_in_the_api_s_shapes(messages, server):
    import anthropic

    with pytest.raises(anthropic.RateLimitError):
        messages.messages.create(**_ask("#queuefull"))
    with pytest.raises(anthropic.BadRequestError) as e:
        messages.messages.create(**_ask("hi", max_tokens=KV_POSITIONS))
    assert e.value.body["error"]["message"].startswith("prompt is too long: ")
    # after the 200, an overloaded engine is an overloaded_error event
    with pytest.raises(anthropic.APIStatusError) as e:
        with messages.messages.stream(**_ask("#midfail")) as s:
            s.get_final_message()
    assert e.value.body["error"]["type"] == "overloaded_error"


# ------------------------------------------------------------------ OpenAI's Responses API
RESPONSES_READ = {"type": "function", **READ["function"]}


def _input(text: str, **kw) -> dict:
    return {"model": "m", "max_output_tokens": 256, "input": text, **kw}


def test_responses_serve_the_same_chat_whole_and_streamed(client, server):
    ask = _input("#call", tools=[RESPONSES_READ])
    whole = client.responses.create(**ask)
    with client.responses.stream(**ask) as s:
        streamed = s.get_final_response()
    assert server[0].last[0].startswith("tools: read\n")
    for r in (whole, streamed):
        assert [i.type for i in r.output] == ["message", "function_call"] and r.model == "m"
        assert r.output_text == "Reading it." and r.status == "completed"
        assert r.output[1].name == "read"
        assert json.loads(r.output[1].arguments) == {"path": "a.py", "limit": 20}
        assert (
            r.usage.input_tokens_details.cached_tokens == 2
        )  # the stand-in's cached prompt tokens
    plain = client.responses.create(**_input("hi"))
    assert plain.output_text == REPLY and [i.type for i in plain.output] == ["message"]


def test_responses_errors_in_the_api_s_shapes(client):
    import openai

    with pytest.raises(openai.RateLimitError):
        client.responses.create(**_input("#queuefull"))
    with pytest.raises(openai.BadRequestError) as e:
        client.responses.create(**_input("hi", max_output_tokens=KV_POSITIONS))
    assert e.value.code == "context_length_exceeded"
    with pytest.raises(openai.BadRequestError, match="previous_response_id is not supported"):
        client.responses.create(**_input("hi"), previous_response_id="resp_1")
    # after the 200, an overloaded engine fails the response
    *_, last = client.responses.create(**_input("#midfail"), stream=True)
    assert last.type == "response.failed" and last.response.error.code == "server_is_overloaded"
