"""OpenAI's Responses API, served on the same port as chat completions, for Codex and the other
clients that speak it: a /v1/responses body is read into the chat-completions body the server
already serves, and the reply, whole or streamed, is written back as a response.

Stateless, as FreeToken's Responses endpoint serves it to Codex, which sends the whole conversation
as `input` when it does not store responses: `store` is ignored, and `previous_response_id`,
`conversation` and `background`, which need a stored response, are refused.

- request: `instructions` and the system and developer messages that open the input become the
  system message, and a later one is folded into the user turn where it stands; user and assistant
  messages, `function_call` items (as `tool_calls`), `function_call_output` items (as tool
  messages) and `reasoning` items (as the `reasoning_content` of their assistant turn) carry over,
  one assistant turn's items merged into one message. Function tools carry over; built-in and
  namespaced tools, which the server cannot run, are left out. The server cannot force a call, so
  tool_choice `required` is `auto` and a named function is that tool alone. max_output_tokens,
  temperature, top_p and top_k carry over. Content the server cannot read (an image, a file)
  becomes a note saying so.
- reply: `reasoning_content` becomes a reasoning item, the content a message and each tool call a
  function_call item; a reply cut short is `incomplete` (max_output_tokens), and usage counts the
  cached prompt tokens. A stream sends the API's events, ending with response.completed,
  response.incomplete or, after a failure, response.failed.
- errors keep the chat API's error object, with a context overflow's code
  `context_length_exceeded`, which Codex reads to know its context is full.

`read` reads a request; `reply` writes a whole response or an error, and `Events` a stream, from
the chunks an OpenAI chat stream of the same reply would send.
"""

from __future__ import annotations

import json
import time
import uuid

PATHS = ("/v1/responses",)
STATEFUL = ("previous_response_id", "conversation", "background")
CONTEXT_CODES = ("context_length_exceeded", "context_too_large")  # Qwen's code, Flint's type
NOT_TEXT = "[{kind} omitted: this model reads text only]"
TEXT_PARTS = ("input_text", "output_text", "text")


def _text(content) -> str:
    """A content value's text: a string as it is, else its text parts joined as the template joins
    them, with a note for each part that is not text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content)
    parts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in TEXT_PARTS:
            parts.append(part.get("text") or "")
        else:
            parts.append(NOT_TEXT.format(kind=kind))
    return "".join(parts)


def _inline_system(system: str, messages: list[dict], text: str) -> tuple[str, list[dict]]:
    """A system or developer message: before the first other message it joins the system prompt;
    after, it is folded into the user turn where it stands, since the templates take a system
    message only first and moving it would rewrite the cached prompt."""
    if not text:
        return system, messages
    if not messages:
        return "\n\n".join(t for t in (system, text) if t), messages
    last = messages[-1]
    if last["role"] == "user":
        messages[-1] = {**last, "content": f"{last['content']}\n\n{text}"}
        return system, messages
    return system, [*messages, {"role": "user", "content": text}]


def _assistant(messages: list[dict]) -> dict | None:
    return messages[-1] if messages and messages[-1]["role"] == "assistant" else None


def _add_reasoning(messages: list[dict], text: str) -> None:
    """A reasoning item opens its assistant turn."""
    messages.append({"role": "assistant", "content": None, "reasoning_content": text})


def _add_text(messages: list[dict], text: str) -> None:
    """Assistant text fills the open turn if it has no text or calls yet, else opens a turn."""
    last = _assistant(messages)
    if last is not None and not last["content"] and not last.get("tool_calls"):
        last["content"] = text
    else:
        messages.append({"role": "assistant", "content": text})


def _add_call(messages: list[dict], item: dict) -> None:
    """A function call joins the open assistant turn, or opens one."""
    call = {
        "id": item.get("call_id") or item.get("id"),
        "type": "function",
        "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
    }
    last = _assistant(messages)
    if last is None:
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
    else:
        last["tool_calls"] = [*last.get("tool_calls", []), call]


def _reasoning_text(item: dict) -> str:
    content = item.get("content") if isinstance(item.get("content"), list) else []
    return "".join(c.get("text") or "" for c in content if isinstance(c, dict))


def _messages(body: dict) -> list[dict]:
    """The chat messages for a request's `instructions` and `input`."""
    system, messages = body.get("instructions") or "", []
    items = body.get("input")
    items = [{"role": "user", "content": items}] if isinstance(items, str) else items or []
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role, text = item.get("role", "user"), _text(item.get("content"))
            if role in ("system", "developer"):
                system, messages = _inline_system(system, messages, text)
            elif role == "assistant":
                if text:
                    _add_text(messages, text)
            elif text:
                messages.append({"role": "user", "content": text})
        elif kind == "function_call":
            _add_call(messages, item)
        elif kind == "function_call_output":
            output = _text(item.get("output"))
            messages.append(
                {"role": "tool", "tool_call_id": item.get("call_id"), "content": output}
            )
        elif kind == "reasoning" and _reasoning_text(item):
            _add_reasoning(messages, _reasoning_text(item))
        # other items (built-in tool calls) are the client's, run nowhere here
    return [{"role": "system", "content": system}, *messages] if system else messages


def _tools(body: dict) -> list[dict]:
    """The function tools; built-in and namespaced tools run nowhere here."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description") or "",
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for t in body.get("tools") or ()
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("name"), str)
    ]


def _offered(tools: list[dict], choice) -> tuple[list[dict], str | None]:
    """The tools to offer and the chat tool_choice. The server cannot force a call, so `required`
    is `auto` and a named function is that tool alone, offered `auto`."""
    if choice == "none":
        return tools, "none"
    if isinstance(choice, dict) and choice.get("type") == "function":
        named = [t for t in tools if t["function"]["name"] == choice.get("name")]
        return named or tools, "auto"
    return tools, "auto" if choice is not None else None


def read(path: str, body: dict) -> dict:
    """The chat-completions body for a /v1/responses body."""
    for key in STATEFUL:
        if body.get(key):
            raise ValueError(
                f"{key} is not supported: this server keeps no responses, so send the whole "
                "conversation as input"
            )
    chat: dict = {"model": body.get("model"), "messages": _messages(body)}
    chat["stream"] = bool(body.get("stream"))
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            chat[key] = body[key]
    tools, choice = _offered(_tools(body), body.get("tool_choice"))
    if tools:
        chat["tools"] = tools
        if choice is not None:
            chat["tool_choice"] = choice
    return chat


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + output,
    }


def _reasoning_item(item_id: str, text: str) -> dict:
    content = [{"type": "reasoning_text", "text": text}]
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "content": content,
        "status": "completed",
    }


def _part(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": []}


def _message_item(item_id: str, text: str, status: str) -> dict:
    content = [_part(text)]
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": content,
    }


def _call_item(item_id: str, call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _response(resp_id: str, created: int, model: str, output: list, status: str, **extra) -> dict:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": status,
        "output": output,
        "error": extra.get("error"),
        "incomplete_details": extra.get("incomplete_details"),
        "usage": extra.get("usage"),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _incomplete(finish: str | None) -> dict | None:
    return {"reason": "max_output_tokens"} if finish == "length" else None


def response(resp: dict, model: str) -> dict:
    """The response for a whole chat completion."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    output = []
    if msg.get("reasoning_content"):
        output.append(_reasoning_item(_id("rs"), msg["reasoning_content"]))
    if msg.get("content"):
        status = "incomplete" if finish == "length" else "completed"
        output.append(_message_item(_id("msg"), msg["content"], status))
    for call in msg.get("tool_calls") or ():
        fn = call.get("function") or {}
        output.append(_call_item(_id("fc"), call.get("id"), fn.get("name"), fn.get("arguments")))
    return _response(
        _id("resp"),
        int(time.time()),
        model,
        output,
        "incomplete" if finish == "length" else "completed",
        usage=_usage(resp.get("usage")),
        incomplete_details=_incomplete(finish),
    )


def _error(err: object) -> dict:
    """The chat API's error object, a context overflow's code `context_length_exceeded` and any other
    code a string."""
    err = dict(err) if isinstance(err, dict) else {"message": str(err)}
    if err.get("code") in CONTEXT_CODES or err.get("type") in CONTEXT_CODES:
        err["code"] = "context_length_exceeded"
    elif not isinstance(err.get("code"), str):
        # Flint's chat error carries the HTTP status as its code; a Responses error's code is a
        # string (or null), so the error's own type stands in for it
        err["code"] = err["type"] if isinstance(err.get("type"), str) else None
    return err


def reply(status: int, body: dict) -> dict:
    """What a Responses client gets for the server's response `body`: an error as it is (its
    context-overflow code named as OpenAI names it), a response for a chat completion."""
    if "error" in body:
        return {"error": _error(body["error"])}
    if "choices" in body:
        return response(body, body.get("model") or "")
    return body


def _encode(events: list[tuple[str, dict]]) -> bytes:
    return b"".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    )


class Events:
    """A stream in the Responses API's events. `frame(chunk)` takes each chunk the chat stream of the
    same reply would send, an error frame included, and `done()` stands for its `[DONE]`; both
    return the bytes to write. Each output item opens when its first delta arrives and closes when
    the next one opens or the stream ends."""

    def __init__(self, model: str):
        self.model = model
        self.id = _id("resp")
        self.created = int(time.time())
        self.seq = 0
        self.items: list[dict] = []  # the output items closed so far
        self.open: dict | None = None  # the item still open: kind, id, and its text or call
        self.calls: set[int] = set()  # the chat stream's tool-call indexes seen
        self.finish: str | None = None
        self.usage: dict | None = None
        self.started = False
        self.failed = False

    def _event(self, kind: str, **fields) -> tuple[str, dict]:
        self.seq += 1
        return kind, {"type": kind, "sequence_number": self.seq - 1, **fields}

    def _snapshot(self, status: str, **extra) -> dict:
        return _response(self.id, self.created, self.model, list(self.items), status, **extra)

    def _start(self) -> list[tuple[str, dict]]:
        if self.started:
            return []
        self.started = True
        return [
            self._event("response.created", response=self._snapshot("in_progress")),
            self._event("response.in_progress", response=self._snapshot("in_progress")),
        ]

    def _close(self) -> list[tuple[str, dict]]:
        item, self.open = self.open, None
        if item is None:
            return []
        at = {"item_id": item["id"], "output_index": len(self.items)}
        if item["kind"] == "reasoning":
            done = _reasoning_item(item["id"], item["text"])
            events = [
                self._event(
                    "response.reasoning_text.done", **at, content_index=0, text=item["text"]
                )
            ]
        elif item["kind"] == "message":
            status = "incomplete" if self.finish == "length" else "completed"
            done = _message_item(item["id"], item["text"], status)
            text = item["text"]
            events = [
                self._event(
                    "response.output_text.done", **at, content_index=0, text=text, logprobs=[]
                ),
                self._event("response.content_part.done", **at, content_index=0, part=_part(text)),
            ]
        else:
            done = _call_item(item["id"], item["call_id"], item["name"], item["text"])
            name, args = item["name"], item["text"]
            events = [
                self._event(
                    "response.function_call_arguments.done", **at, name=name, arguments=args
                )
            ]
        events.append(
            self._event("response.output_item.done", output_index=len(self.items), item=done)
        )
        self.items.append(done)
        return events

    def _item(self, kind: str, added: dict, **state) -> list[tuple[str, dict]]:
        events = self._close()
        self.open = {"kind": kind, "id": added["id"], "text": "", **state}
        events.append(
            self._event("response.output_item.added", output_index=len(self.items), item=added)
        )
        return events

    def _delta(self, kind: str, text: str) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        if self.open is None or self.open["kind"] != kind:
            item_id = _id("rs" if kind == "reasoning" else "msg")
            if kind == "reasoning":
                added = {"id": item_id, "type": "reasoning", "summary": [], "status": "in_progress"}
                events += self._item(kind, added)
            else:
                added = {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
                events += self._item(kind, added)
                at = {"item_id": item_id, "output_index": len(self.items)}
                events.append(
                    self._event(
                        "response.content_part.added", **at, content_index=0, part=_part("")
                    )
                )
        self.open["text"] += text
        at = {"item_id": self.open["id"], "output_index": len(self.items)}
        if kind == "reasoning":
            events.append(
                self._event("response.reasoning_text.delta", **at, content_index=0, delta=text)
            )
        else:
            events.append(
                self._event(
                    "response.output_text.delta", **at, content_index=0, delta=text, logprobs=[]
                )
            )
        return events

    def _call(self, call: dict) -> list[tuple[str, dict]]:
        fn = call.get("function") or {}
        n = int(call.get("index") or 0)
        events: list[tuple[str, dict]] = []
        if n not in self.calls:
            self.calls.add(n)
            added = _call_item(_id("fc"), call.get("id"), fn.get("name"), "")
            events += self._item(
                "call",
                {**added, "status": "in_progress"},
                call_id=call.get("id"),
                name=fn.get("name"),
            )
        if fn.get("arguments") and self.open is not None and self.open["kind"] == "call":
            self.open["text"] += fn["arguments"]
            at = {"item_id": self.open["id"], "output_index": len(self.items)}
            events.append(
                self._event("response.function_call_arguments.delta", **at, delta=fn["arguments"])
            )
        return events

    def frame(self, chunk: dict) -> bytes:
        events = self._start()
        if "error" in chunk and not chunk.get("choices"):
            if not self.failed:
                self.failed = True
                events += self._close()
                error = _error(chunk["error"])
                error = {
                    "code": error.get("code") or "server_error",
                    "message": error.get("message") or "",
                }
                events.append(
                    self._event("response.failed", response=self._snapshot("failed", error=error))
                )
            return _encode(events)
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                events += self._delta("reasoning", delta["reasoning_content"])
            if delta.get("content"):
                events += self._delta("message", delta["content"])
            for call in delta.get("tool_calls") or ():
                events += self._call(call)
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return _encode(events)

    def done(self) -> bytes:
        """The end of the response; nothing after a failure, which ends the stream itself."""
        if self.failed:
            return b""
        events = self._start() + self._close()
        status = "incomplete" if self.finish == "length" else "completed"
        final = self._snapshot(
            status, usage=_usage(self.usage), incomplete_details=_incomplete(self.finish)
        )
        events.append(self._event(f"response.{status}", response=final))
        return _encode(events)
