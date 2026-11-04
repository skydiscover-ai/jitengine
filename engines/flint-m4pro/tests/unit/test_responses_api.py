"""OpenAI's Responses API, without a server: a request as the chat body it stands for, read as vLLM's
and FreeToken's Responses endpoints read Codex's; a reply and its stream as that API's response and
events; errors with the codes Codex reads."""

import json

import pytest

from jitengine_flint_m4pro.responses_api import Events, read, reply, response

READ = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a command",
    "strict": False,
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}
CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "exec_command", "arguments": '{"cmd": "ls"}'},
}


def _events(data: bytes) -> list[dict]:
    blocks = [b for b in data.decode().split("\n\n") if b]
    return [json.loads(b.split("data: ", 1)[1]) for b in blocks]


def test_codex_s_input_becomes_chat_messages():
    body = {
        "model": "m",
        "instructions": "Base.",
        "stream": True,
        "max_output_tokens": 512,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
        "tools": [
            READ,
            {"type": "namespace", "name": "multi_agent_v1", "tools": []},
            {"type": "web_search"},
        ],
        "input": [
            # Codex opens with two developer messages
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Rules."}],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Sandbox."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list files"}],
            },
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Look."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Listing."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": '{"cmd": "ls"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "a.py"},
            {"role": "developer", "content": "Later."},
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:..."}]},
            {"type": "web_search_call", "id": "ws_1"},
        ],
    }
    chat = read("/v1/responses", body)
    assert chat["messages"] == [
        {"role": "system", "content": "Base.\n\nRules.\n\nSandbox."},
        {"role": "user", "content": "list files"},
        # one assistant turn's items, merged
        {
            "role": "assistant",
            "content": "Listing.",
            "reasoning_content": "Look.",
            "tool_calls": [CALL],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
        # a later developer message stands where it was, as a user turn
        {"role": "user", "content": "Later."},
        {"role": "user", "content": "[input_image omitted: this model reads text only]"},
    ]
    # function tools only: the namespace and web search run nowhere here
    assert [t["function"]["name"] for t in chat["tools"]] == ["exec_command"]
    assert chat["tool_choice"] == "auto" and chat["max_tokens"] == 512
    assert chat["stream"] and chat["stream_options"] == {"include_usage": True}
    assert read("/v1/responses", {"model": "m", "input": "hi"})["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_a_request_that_needs_a_stored_response_is_refused():
    for key, value in (
        ("previous_response_id", "resp_1"),
        ("conversation", "conv_1"),
        ("background", True),
    ):
        with pytest.raises(ValueError, match=f"{key} is not supported"):
            read("/v1/responses", {"model": "m", "input": "hi", key: value})
    assert read("/v1/responses", {"model": "m", "input": "hi", "store": True})["messages"]


def test_tool_choice_offers_what_the_server_can_serve():
    other = {**READ, "name": "write"}
    body = {"model": "m", "input": "hi", "tools": [READ, other]}

    def offered(choice):
        chat = read("/v1/responses", {**body, "tool_choice": choice})
        return chat.get("tool_choice"), [t["function"]["name"] for t in chat["tools"]]

    assert offered("none") == ("none", ["exec_command", "write"])
    assert offered("required") == ("auto", ["exec_command", "write"])
    assert offered({"type": "function", "name": "write"}) == ("auto", ["write"])


def test_a_reply_becomes_output_items():
    resp = {
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Look.",
                    "content": "Listing.",
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
    r = reply(200, resp)
    assert r["object"] == "response" and r["status"] == "completed" and r["model"] == "m"
    assert [i["type"] for i in r["output"]] == ["reasoning", "message", "function_call"]
    assert r["output"][0]["content"] == [{"type": "reasoning_text", "text": "Look."}]
    assert r["output"][1]["content"][0] == {
        "type": "output_text",
        "text": "Listing.",
        "annotations": [],
    }
    call = r["output"][2]
    assert (call["call_id"], call["name"], call["arguments"]) == (
        "call_1",
        "exec_command",
        '{"cmd": "ls"}',
    )
    assert r["usage"]["input_tokens"] == 120 and r["usage"]["input_tokens_details"] == {
        "cached_tokens": 100
    }
    resp["choices"][0]["finish_reason"] = "length"
    cut = response(resp, "m")
    assert cut["status"] == "incomplete" and cut["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


def test_errors_keep_the_chat_api_s_object_with_codex_s_codes():
    flint = {"message": "too long", "type": "context_too_large", "code": 400}
    assert reply(400, {"error": flint})["error"]["code"] == "context_length_exceeded"
    qwen = {"message": "busy", "type": "rate_limit_error", "code": "rate_limit_exceeded"}
    assert reply(429, {"error": qwen}) == {"error": qwen}


def test_an_error_code_is_a_string_or_null():
    """Flint's chat API puts the HTTP status in `code`; a Responses error's code is a string."""
    busy = {"message": "queue full", "type": "queue_full", "code": 429}
    assert reply(429, {"error": busy})["error"] == {**busy, "code": "queue_full"}
    bare = {"message": "boom", "code": 500}
    assert reply(500, {"error": bare})["error"] == {"message": "boom", "code": None}
    assert reply(400, {"error": {"message": "x", "code": None}})["error"]["code"] is None
    failed = Events("m")
    unavailable = {"message": "stopping", "type": "unavailable", "code": 503}
    events = _events(failed.frame({"error": unavailable}))
    assert events[-1]["response"]["error"] == {"code": "unavailable", "message": "stopping"}
    failed = Events("m")
    events = _events(failed.frame({"error": bare}))
    assert events[-1]["response"]["error"] == {"code": "server_error", "message": "boom"}


def test_a_stream_s_events_from_its_chunks():
    ev = Events("m")
    out = ev.frame(
        {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    )
    for delta in (
        {"reasoning_content": "Lo"},
        {"reasoning_content": "ok."},
        {"content": "Listing."},
        {"tool_calls": [{"index": 0, **CALL}]},
    ):
        out += ev.frame({"choices": [{"delta": delta, "finish_reason": None}]})
    out += ev.frame({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    out += ev.frame({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    events = _events(out + ev.done())
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert [e["type"] for e in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",  # the reasoning item
        "response.reasoning_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
        "response.output_item.added",  # the message
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",  # the call
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    final = events[-1]["response"]
    assert [i["type"] for i in final["output"]] == ["reasoning", "message", "function_call"]
    assert (
        final["output"][0]["content"][0]["text"] == "Look." and final["usage"]["output_tokens"] == 3
    )
    # an error frame fails the response, and nothing follows for [DONE]
    failed = Events("m")
    error = {"message": "the server is overloaded", "code": "server_is_overloaded"}
    events = _events(failed.frame({"error": error}))
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"] == {
        "code": "server_is_overloaded",
        "message": "the server is overloaded",
    }
    assert failed.done() == b""
