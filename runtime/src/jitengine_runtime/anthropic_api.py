"""The Anthropic Messages API, served on the same port as the OpenAI one, for Claude Code and the
other clients that speak it: a /v1/messages body is read into the chat-completions body the server
already serves, and the reply, whole or streamed, is written back as a message.

- request, as vLLM's and FreeToken's Messages endpoints read it: the system prompt becomes a system
  message, text blocks text (joined as the template joins text parts), tool_use blocks
  `tool_calls`, tool_result blocks tool messages and thinking blocks `reasoning_content`; a message
  left with none of these is dropped, and a system message among the messages is folded into the
  user turn where it stands. Tools become function tools (server tools, which have no input_schema,
  are left out). The server cannot force a call, so tool_choice `any` is `auto` and a named tool is
  that tool alone; `none` is `none`. max_tokens, temperature, top_p, top_k and stop_sequences carry
  over. A block the server cannot read (an image, a document) becomes a note saying so, so the
  conversation can go on.
- reply: `reasoning_content` becomes a thinking block (with an empty signature), the content a text
  block and each tool call a tool_use block; finish reasons stop, length and tool_calls become
  end_turn, max_tokens and tool_use, except that a stop reason of the client's own stop sequence is
  stop_sequence, with the sequence the server names in the choice's `x_stop_sequence`; usage counts the cached prompt tokens as cache reads. A
  stream's usage, which comes last, is in message_delta.
- errors keep their status, in Anthropic's error object; a context overflow reads "prompt is too
  long", the error Claude Code compacts on, and an overloaded server is `overloaded_error`, before
  the 200 or after it.

`read` reads a request; `reply` writes a whole response or an error, and `Events` a stream, from the
chunks an OpenAI stream of the same reply would send.
"""

from __future__ import annotations

import json
import uuid

PATHS = ("/v1/messages", "/v1/messages/count_tokens")
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _stop(finish: str | None, sequence: str | None) -> tuple[str, str | None]:
    """The `stop_reason` and `stop_sequence` of a reply that ended with `finish`; the server names
    the client's stop sequence that ended it (x_stop_sequence) when one did."""
    if finish == "stop" and sequence:
        return "stop_sequence", sequence
    return STOP_REASONS.get(finish, "end_turn"), None


ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
}
CONTEXT_CODES = ("context_length_exceeded", "context_too_large")  # Qwen's code, Flint's type
OVERLOADED = ("server_is_overloaded", "service_unavailable_error", "unavailable", 503)
NOT_TEXT = "[{kind} omitted: this model reads text only]"


def _text(content) -> str:
    """A content value's text: a string as it is, else its text blocks joined as the template joins
    text parts, with a note for each block that is not text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        parts.append((block.get("text") or "") if kind == "text" else NOT_TEXT.format(kind=kind))
    return "".join(parts)


def _user(blocks: list) -> list[dict]:
    """A user turn's messages: its tool results as tool messages, in order, then the rest."""
    out, rest = [], []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            content = _text(block.get("content"))
            out.append(
                {"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": content}
            )
        else:
            rest.append(block)
    text = _text(rest)
    if text:
        out.append({"role": "user", "content": text})
    return out


def _assistant(blocks: list) -> list[dict]:
    """An assistant turn's message, or none when it holds no text, thinking or tool use."""
    text, thinking, calls = [], [], []
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            text.append(block.get("text") or "")
        elif kind == "thinking" and block.get("thinking"):
            thinking.append(block["thinking"])
        elif kind == "tool_use":
            arguments = json.dumps(block.get("input") or {}, ensure_ascii=False)
            fn = {"name": block.get("name"), "arguments": arguments}
            calls.append({"id": block.get("id"), "type": "function", "function": fn})
    msg: dict = {"role": "assistant", "content": "".join(text) or None}
    if thinking:
        msg["reasoning_content"] = "\n\n".join(thinking)
    if calls:
        msg["tool_calls"] = calls
    return [msg] if msg["content"] or thinking or calls else []


def _tools(body: dict) -> list[dict]:
    """The function tools; server tools, which have no input_schema, run nowhere here."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in body.get("tools") or ()
        if isinstance(t, dict) and "input_schema" in t
    ]


def _offered(tools: list[dict], choice) -> tuple[list[dict], str | None]:
    """The tools to offer and the OpenAI tool_choice for an Anthropic tool_choice. The server
    cannot force a call, so `any` is `auto` and a named tool is that tool alone, offered `auto`."""
    kind = choice.get("type") if isinstance(choice, dict) else None
    if kind == "none":
        return tools, "none"
    if kind == "tool":
        named = [t for t in tools if t["function"]["name"] == choice.get("name")]
        return named or tools, "auto"
    return tools, "auto" if kind in ("auto", "any") else None


def _inline_system(system: str, messages: list[dict], text: str) -> tuple[str, list[dict]]:
    """A system message among the messages: before the first it joins the system prompt; after,
    it is folded into the user turn where it stands, as vLLM and FreeToken fold one, since the
    templates take a system message only first and moving it would rewrite the cached prompt."""
    if not text:
        return system, messages
    if not messages:
        return "\n\n".join(t for t in (system, text) if t), messages
    last = messages[-1]
    if last["role"] == "user":
        messages[-1] = {**last, "content": f"{last['content']}\n\n{text}"}
        return system, messages
    return system, [*messages, {"role": "user", "content": text}]


def chat_request(body: dict) -> dict:
    """The /v1/chat/completions body for a /v1/messages body."""
    system, messages = _text(body.get("system")), []
    for m in body.get("messages") or ():
        content = m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        if m.get("role") == "system":
            system, messages = _inline_system(system, messages, _text(blocks))
        elif m.get("role") == "assistant":
            messages += _assistant(blocks)
        else:
            messages += _user(blocks)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    chat: dict = {
        "model": body.get("model"),
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]
    tools, choice = _offered(_tools(body), body.get("tool_choice"))
    if tools:
        chat["tools"] = tools
        if choice is not None:
            chat["tool_choice"] = choice
    return chat


def count_request(body: dict) -> dict:
    """The chat body whose prompt /v1/messages/count_tokens counts: the request without output."""
    return {**chat_request({**body, "stream": False}), "max_tokens": 0}


def read(path: str, body: dict) -> dict:
    """The chat-completions body for a body sent to `path`, one of PATHS."""
    return count_request(body) if path.endswith("/count_tokens") else chat_request(body)


def _error_object(status: int, err: object) -> dict:
    """Anthropic's error object for an OpenAI-shaped one."""
    err = err if isinstance(err, dict) else {}
    msg = err.get("message") or f"the server answered {status}"
    if err.get("code") in CONTEXT_CODES or err.get("type") in CONTEXT_CODES:
        msg = f"prompt is too long: {msg}"
    return {"type": ERROR_TYPES.get(status, "api_error"), "message": msg}


def reply(status: int, body: dict) -> dict:
    """What a Messages client gets for the server's response `body`: an error object for an error,
    a message for a chat completion, anything else (a token count) as it is."""
    if "error" in body:
        return {"type": "error", "error": _error_object(status, body["error"])}
    if "choices" in body:
        return message(body, body.get("model") or "")
    return body


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {
        "input_tokens": max(0, int(usage.get("prompt_tokens") or 0) - cached),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def _arguments(raw) -> dict:
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def message(resp: dict, model: str) -> dict:
    """The Messages API reply for a whole chat completion."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or ():
        fn = call.get("function") or {}
        tool_input = _arguments(fn.get("arguments"))
        content.append(
            {"type": "tool_use", "id": call.get("id"), "name": fn.get("name"), "input": tool_input}
        )
    stop_reason, stop_sequence = _stop(choice.get("finish_reason"), choice.get("x_stop_sequence"))
    return {
        "id": _message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": _usage(resp.get("usage")),
    }


class MessageStream:
    """A chat completion stream as Messages API events: `start()`, `chunk(obj)` for each chunk
    and `end()` each return a list of (event name, data)."""

    def __init__(self, model: str):
        self.model = model
        self.index = -1  # the latest content block
        self.open: str | None = None  # the type of the block still open
        self.calls: dict[int, int] = {}  # tool-call index -> its block
        self.finish: str | None = None
        self.stop_sequence: str | None = None
        self.usage: dict | None = None

    def start(self) -> list[tuple[str, dict]]:
        msg = {
            "id": _message_id(),
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return [("message_start", {"type": "message_start", "message": msg})]

    def _close(self) -> list[tuple[str, dict]]:
        """The events that close the open block; a thinking block ends with its signature,
        which is empty: nothing here signs or checks one."""
        if self.open is None:
            return []
        events = []
        if self.open == "thinking":
            events.append(self._delta(self.index, {"type": "signature_delta", "signature": ""}))
        self.open = None
        return [
            *events,
            ("content_block_stop", {"type": "content_block_stop", "index": self.index}),
        ]

    def _block(self, kind: str, block: dict) -> list[tuple[str, dict]]:
        events = self._close()
        self.index += 1
        self.open = kind
        start = {"type": "content_block_start", "index": self.index, "content_block": block}
        return [*events, ("content_block_start", start)]

    def _delta(self, index: int, delta: dict) -> tuple[str, dict]:
        return "content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        }

    def chunk(self, obj: dict) -> list[tuple[str, dict]]:
        if obj.get("usage"):
            self.usage = obj["usage"]
        events: list[tuple[str, dict]] = []
        for choice in obj.get("choices") or ():
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                if self.open != "thinking":
                    events += self._block(
                        "thinking", {"type": "thinking", "thinking": "", "signature": ""}
                    )
                d = {"type": "thinking_delta", "thinking": delta["reasoning_content"]}
                events.append(self._delta(self.index, d))
            if delta.get("content"):
                if self.open != "text":
                    events += self._block("text", {"type": "text", "text": ""})
                events.append(
                    self._delta(self.index, {"type": "text_delta", "text": delta["content"]})
                )
            for call in delta.get("tool_calls") or ():
                events += self._tool_call(call)
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
                self.stop_sequence = choice.get("x_stop_sequence")
        return events

    def _tool_call(self, call: dict) -> list[tuple[str, dict]]:
        fn = call.get("function") or {}
        n = int(call.get("index") or 0)
        events = []
        if n not in self.calls:
            block = {"type": "tool_use", "id": call.get("id"), "name": fn.get("name"), "input": {}}
            events += self._block("tool_use", block)
            self.calls[n] = self.index
        if fn.get("arguments"):
            d = {"type": "input_json_delta", "partial_json": fn["arguments"]}
            events.append(self._delta(self.calls[n], d))
        return events

    def end(self) -> list[tuple[str, dict]]:
        reason, sequence = _stop(self.finish, self.stop_sequence)
        delta = {"stop_reason": reason, "stop_sequence": sequence}
        return [
            *self._close(),
            (
                "message_delta",
                {"type": "message_delta", "delta": delta, "usage": _usage(self.usage)},
            ),
            ("message_stop", {"type": "message_stop"}),
        ]


def _stream_error(err) -> dict:
    """Anthropic's error object for a failure after the 200: overloaded (which Claude Code
    retries) when the server says so, else an API error."""
    err = err if isinstance(err, dict) else {}
    busy = err.get("code") in OVERLOADED or err.get("type") in OVERLOADED
    msg = err.get("message") or "the reply failed"
    return {"type": "overloaded_error" if busy else "api_error", "message": msg}


def _encode(events: list[tuple[str, dict]]) -> bytes:
    return b"".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    )


class Events:
    """A stream in the Messages API's events. `frame(chunk)` takes each chunk the OpenAI stream of
    the same reply would send, an error frame included, and `done()` stands for its `[DONE]`; both
    return the bytes to write. message_start goes out with the first frame."""

    def __init__(self, model: str):
        self.stream = MessageStream(model)
        self.started = False
        self.failed = False

    def _start(self) -> list[tuple[str, dict]]:
        if self.started:
            return []
        self.started = True
        return self.stream.start()

    def frame(self, chunk: dict) -> bytes:
        events = self._start()
        if "error" in chunk and not chunk.get("choices"):
            self.failed = True
            events.append(("error", {"type": "error", "error": _stream_error(chunk["error"])}))
        elif not self.failed:
            events += self.stream.chunk(chunk)
        return _encode(events)

    def done(self) -> bytes:
        """The end of the message; nothing after an error, which ends the stream itself."""
        if self.failed:
            return b""
        return _encode(self._start() + self.stream.end())
