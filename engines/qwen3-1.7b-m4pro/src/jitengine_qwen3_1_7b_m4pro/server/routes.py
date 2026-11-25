"""The Anthropic Messages and OpenAI Responses endpoints, served by the chat-completions endpoint
the server already has.

A request to /v1/messages or /v1/responses is read into a chat-completions body, sent to this
server's own /v1/chat/completions over the loopback, and the reply, whole or streamed, is written
back in the API's own form. Generation, batching and the prompt cache are therefore the path that
was measured. A client that disconnects closes the inner connection, which stops the generation.
"""

from __future__ import annotations

import http.client
import json

from jitengine_runtime import anthropic_api, responses_api

MAX_BODY_BYTES = 32 * 1024 * 1024
APIS = {
    "/v1/messages": anthropic_api,
    "/v1/messages/count_tokens": anthropic_api,
    "/v1/responses": responses_api,
}
_WILDCARD = ("", "0.0.0.0", "::")
_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
# What a body that is JSON but not shaped like the API's request can raise while it is read.
_BAD_REQUEST = (ValueError, UnicodeDecodeError, AttributeError, TypeError, KeyError, IndexError)


def _inner_address(handler) -> tuple[str, int]:
    host, port = handler.server.server_address[:2]
    return ("127.0.0.1" if host in _WILDCARD else host), port


def _alias_reasoning(data: dict) -> None:
    """Stock mlx-lm names the thinking text `reasoning`; the translators read `reasoning_content`."""
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            part = choice.get(key)
            if (
                isinstance(part, dict)
                and part.get("reasoning")
                and not part.get("reasoning_content")
            ):
                part["reasoning_content"] = part["reasoning"]


def _normalize(data: dict) -> dict:
    """OpenAI's error object for stock mlx-lm's `{"error": "text"}`, and the reasoning field name."""
    err = data.get("error")
    if isinstance(err, str):
        return {"error": {"message": err}}
    _alias_reasoning(data)
    return data


def _send_json(handler, status: int, body: dict) -> None:
    raw = json.dumps(body).encode()
    handler._set_completion_headers(status)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
    handler.wfile.flush()


def _read_body(handler) -> dict:
    length = handler.headers.get("Content-Length")
    if length is None:
        raise ValueError("Content-Length header is required")
    n = int(length)
    if n > MAX_BODY_BYTES:
        raise ValueError(f"request body of {n} bytes is over the {MAX_BODY_BYTES} limit")
    body = json.loads(handler.rfile.read(n).decode())
    if not isinstance(body, dict):
        raise ValueError("request should be a JSON dictionary")
    return body


def _count_tokens(handler, chat: dict) -> int:
    generator = handler.response_generator
    tokenizer = generator.model_provider.tokenizer
    kwargs = getattr(generator.cli_args, "chat_template_args", None) or {}
    text = tokenizer.apply_chat_template(
        chat["messages"],
        tools=chat.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def _post_inner(handler, chat: dict) -> http.client.HTTPConnection:
    host, port = _inner_address(handler)
    conn = http.client.HTTPConnection(host, port, timeout=None)
    conn.request(
        "POST", "/v1/chat/completions", json.dumps(chat), {"Content-Type": "application/json"}
    )
    return conn


def _whole(handler, api, chat: dict) -> None:
    conn = _post_inner(handler, chat)
    try:
        resp = conn.getresponse()
        data = _normalize(json.loads(resp.read().decode() or "{}"))
        _send_json(handler, resp.status, api.reply(resp.status, data))
    finally:
        conn.close()


def _frames(resp):
    """The JSON chunks of an inner stream, then None for its [DONE]."""
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            yield None
            return
        yield json.loads(payload)


def _stream(handler, api, chat: dict, model: str) -> None:
    conn = _post_inner(handler, chat)
    try:
        resp = conn.getresponse()
        if resp.status != 200:
            data = _normalize(json.loads(resp.read().decode() or "{}"))
            return _send_json(handler, resp.status, api.reply(resp.status, data))
        events = api.Events(model)
        handler._set_stream_headers(200)
        handler.end_headers()
        for chunk in _frames(resp):
            out = events.done() if chunk is None else events.frame(_normalize(chunk))
            if out:
                handler.wfile.write(out)
                handler.wfile.flush()
    except _DISCONNECT:
        pass  # closing the inner connection below stops the generation
    finally:
        conn.close()


def _translated(handler, path: str) -> None:
    api = APIS[path]
    try:
        body = _read_body(handler)
        chat = api.read(path, body)
        if path.endswith("/count_tokens"):
            return _send_json(handler, 200, {"input_tokens": _count_tokens(handler, chat)})
    except _BAD_REQUEST as exc:
        message = str(exc) or f"malformed request ({type(exc).__name__})"
        return _send_json(handler, 400, api.reply(400, {"error": {"message": message}}))
    model = str(body.get("model") or "")
    try:
        if chat.get("stream"):
            _stream(handler, api, chat, model)
        else:
            _whole(handler, api, chat)
    except _DISCONNECT:
        pass


def _too_large(handler) -> bool:
    """A Content-Length over the cap (the translated routes check it themselves)."""
    try:
        return int(handler.headers.get("Content-Length") or 0) > MAX_BODY_BYTES
    except ValueError:
        return False


def install(handler_class) -> None:
    """Route the translated endpoints; every other POST goes to the original handler."""
    original = handler_class.do_POST

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in APIS:
            _translated(self, path)
        elif _too_large(self):
            _send_json(self, 413, {"error": {"message": "request body is over the limit"}})
        else:
            original(self)

    handler_class.do_POST = do_POST
