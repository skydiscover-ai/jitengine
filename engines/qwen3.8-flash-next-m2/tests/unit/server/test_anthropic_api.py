"""The Anthropic Messages API's reading and writing, without a server: a request as the chat body it
stands for, read as vLLM's and FreeToken's Messages endpoints read one; a reply and its stream as
that API's message and events; errors in its error object."""

import json

from jitengine_qwen3_8_flash_next_m2.server.anthropic_api import (
    Events,
    MessageStream,
    chat_request,
    count_request,
    message,
    reply,
)

CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
}
TOOL = {
    "name": "read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}


def test_a_request_becomes_chat_messages_and_function_tools():
    reminder = "<system-reminder>x</system-reminder>\n"
    use = {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.py"}}
    result = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "text", "text": "print(1)"}],
    }
    image = {"type": "image", "source": {"type": "base64", "data": "..."}}
    body = {
        "model": "m",
        "max_tokens": 512,
        "temperature": 0.2,
        "top_k": 20,
        "stop_sequences": ["END"],
        # Claude Code writes its blocks to be joined as they are
        "system": [
            {"type": "text", "text": "You are an agent."},
            {"type": "text", "text": "\nBe brief."},
        ],
        "tools": [TOOL, {"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "auto"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reminder},
                    {"type": "text", "text": "read a.py"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Look first.", "signature": ""},
                    {"type": "text", "text": "Reading."},
                    use,
                ],
            },
            {"role": "user", "content": [result, image]},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "..."}]},
        ],
    }
    chat = chat_request(body)
    call = {
        "id": "toolu_1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
    }
    assert chat["messages"] == [
        {"role": "system", "content": "You are an agent.\nBe brief."},
        {"role": "user", "content": reminder + "read a.py"},  # joined as the engines join parts
        {
            "role": "assistant",
            "content": "Reading.",
            "reasoning_content": "Look first.",
            "tool_calls": [call],
        },
        # the tool result first, as OpenAI orders them, then the rest of the turn
        {"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"},
        {"role": "user", "content": "[image omitted: this model reads text only]"},
        # an assistant turn with nothing the engine reads is dropped
    ]
    # the server tool has no input_schema: the engine cannot run it
    fn = {"name": "read", "description": "Read a file", "parameters": TOOL["input_schema"]}
    assert chat["tools"] == [{"type": "function", "function": fn}]
    assert chat["tool_choice"] == "auto" and chat["stop"] == ["END"]
    sampling = (chat["max_tokens"], chat["temperature"], chat["top_k"], chat["stream"])
    assert sampling == (512, 0.2, 20, False)
    streamed = chat_request({**body, "stream": True})
    assert streamed["stream"] and streamed["stream_options"] == {"include_usage": True}


def test_tool_choice_offers_what_the_engines_can_serve():
    other = {**TOOL, "name": "write"}
    body = {
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [TOOL, other],
    }

    def offered(choice):
        chat = chat_request({**body, "tool_choice": choice} if choice else body)
        return chat.get("tool_choice"), [t["function"]["name"] for t in chat["tools"]]

    assert offered(None) == (None, ["read", "write"])
    assert offered({"type": "none"}) == ("none", ["read", "write"])
    # the engines do not force a call: any is auto, and a named tool is that tool alone
    assert offered({"type": "any"}) == ("auto", ["read", "write"])
    assert offered({"type": "tool", "name": "write"}) == ("auto", ["write"])


def test_a_system_message_among_the_messages_is_folded_where_it_stands():
    body = {
        "model": "m",
        "max_tokens": 8,
        "system": "Base.",
        "messages": [
            {"role": "system", "content": "Before."},
            {"role": "user", "content": "hi"},
            # as Claude Code sends its environment
            {"role": "system", "content": [{"type": "text", "text": "# Environment"}]},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "Later."},
        ],
    }
    assert chat_request(body)["messages"] == [
        {"role": "system", "content": "Base.\n\nBefore."},
        {"role": "user", "content": "hi\n\n# Environment"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Later."},
    ]


def test_a_reply_becomes_blocks_with_the_cached_tokens_as_cache_reads():
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Look.",
                    "content": "Reading.",
                    "tool_calls": [CALL],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }
    m = message(resp, "m")
    assert m["type"] == "message" and m["role"] == "assistant" and m["model"] == "m"
    assert m["content"] == [
        {"type": "thinking", "thinking": "Look.", "signature": ""},
        {"type": "text", "text": "Reading."},
        {"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.py"}},
    ]
    assert m["stop_reason"] == "tool_use" and m["stop_sequence"] is None
    assert m["usage"] == {
        "input_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 100,
        "output_tokens": 9,
    }
    for finish, want in (("stop", "end_turn"), ("length", "max_tokens")):
        resp["choices"][0]["finish_reason"] = finish
        assert message(resp, "m")["stop_reason"] == want


def test_a_stream_opens_one_block_per_kind_in_order():
    s = MessageStream("m")
    events = s.start()
    for delta in (
        {"reasoning_content": "a"},
        {"reasoning_content": "b"},
        {"content": "c"},
        {"tool_calls": [{"index": 0, **CALL}]},
    ):
        events += s.chunk({"choices": [{"delta": delta, "finish_reason": None}]})
    events += s.chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    events += s.chunk({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    events += s.end()
    block = ["content_block_start", "content_block_delta", "content_block_stop"]
    assert [name for name, _ in events] == [
        "message_start",
        *block[:2],
        "content_block_delta",  # the second thinking delta joins the open block
        "content_block_delta",  # which ends with its signature
        block[2],  # and closes before the text opens
        *block,
        *block,
        "message_delta",
        "message_stop",
    ]
    kinds = [
        (e["index"], e["content_block"]["type"]) for n, e in events if n == "content_block_start"
    ]
    assert kinds == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    stops = [e["index"] for n, e in events if n == "content_block_stop"]
    assert stops == [0, 1, 2]
    deltas = [e["delta"] for n, e in events if n == "content_block_delta"]
    assert deltas == [
        {"type": "thinking_delta", "thinking": "a"},
        {"type": "thinking_delta", "thinking": "b"},
        {"type": "signature_delta", "signature": ""},
        {"type": "text_delta", "text": "c"},
        {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'},
    ]
    end = dict(events[-2][1])
    assert end["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert end["usage"]["input_tokens"] == 7 and end["usage"]["output_tokens"] == 3


def test_errors_and_counts_in_the_api_s_shapes():
    context = {
        "message": "too long",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }
    assert reply(400, {"error": context}) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "prompt is too long: too long"},
    }
    assert reply(429, {"error": {"message": "busy"}})["error"]["type"] == "rate_limit_error"
    assert reply(500, {"error": {"message": "x"}})["error"]["type"] == "api_error"
    assert reply(200, {"input_tokens": 7}) == {"input_tokens": 7}
    body = {
        "model": "m",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    counted = count_request(body)
    assert counted["max_tokens"] == 0 and not counted["stream"] and "stream_options" not in counted


def test_a_stream_s_events_from_its_chunks():
    events = Events("m")
    chunk = {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
    first = events.frame(chunk).decode()
    assert first.startswith("event: message_start\n") and "event: content_block_delta\n" in first
    events.frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    end = events.done().decode()
    names = [line.split(": ", 1)[1] for line in end.splitlines() if line.startswith("event: ")]
    assert names == ["content_block_stop", "message_delta", "message_stop"]
    # an error frame ends the stream: its event, and nothing for [DONE]
    failed = Events("m")
    out = failed.frame(
        {"error": {"message": "the server is overloaded", "code": "server_is_overloaded"}}
    )
    data = json.loads(out.decode().split("event: error\ndata: ", 1)[1])
    assert data["error"] == {"type": "overloaded_error", "message": "the server is overloaded"}
    assert failed.done() == b""
