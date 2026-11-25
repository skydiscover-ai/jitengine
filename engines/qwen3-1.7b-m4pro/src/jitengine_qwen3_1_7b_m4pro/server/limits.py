"""Apply the token budget to mlx-lm's completion handlers."""

from __future__ import annotations

import copy
import json
import select
import socket

from jitengine_qwen3_1_7b_m4pro.engine.admission import TokenBudget, TooLong


def client_gone(handler) -> bool:
    """True when the client has closed its end while the request waits."""
    sock = handler.connection
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


def _plain_messages(messages):
    """A copy of the messages with content parts joined to text, as the server does before it
    applies the chat template (it does so later and in place, after this count is taken)."""
    from mlx_lm.server import process_message_content

    plain = copy.deepcopy(messages)
    process_message_content(plain)
    return plain


def prompt_tokens(handler, request) -> int:
    """Tokens of the request's prompt; the character count over two if it cannot be tokenized."""
    try:
        generator = handler.response_generator
        tokenizer = generator.model_provider.tokenizer
        kwargs = dict(getattr(generator.cli_args, "chat_template_args", None) or {})
        kwargs.update(getattr(handler, "chat_template_kwargs", None) or {})
        if request.request_type == "chat":
            text = tokenizer.apply_chat_template(
                _plain_messages(request.messages),
                tools=request.tools,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        else:
            text = request.prompt
            if not isinstance(text, str):
                raise TypeError("prompt is not text")
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001 - an upper bound is better than refusing a valid request
        return len(json.dumps(request.messages or request.prompt)) // 2


def valid_max_tokens(value) -> bool:
    """A positive whole number (not a bool, a float or a string)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _refuse(handler, exc: TooLong) -> None:
    _error(handler, str(exc), "context_length_exceeded")


def _error(handler, message: str, code: str) -> None:
    raw = json.dumps(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}}
    ).encode()
    handler._set_completion_headers(400)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
    handler.wfile.flush()


def install(handler_class, budget: TokenBudget) -> None:
    """Gate `handle_completion`, which runs a request already read and validated."""
    original = handler_class.handle_completion

    def handle_completion(self, request, stop_words):
        max_tokens = getattr(self, "max_tokens", None)
        if not valid_max_tokens(max_tokens):
            return _error(
                self,
                f"max_tokens / max_completion_tokens must be a positive integer, got {max_tokens!r}",
                "invalid_max_tokens",
            )
        need = prompt_tokens(self, request) + max_tokens
        try:
            admitted = budget.acquire(need, gone=lambda: client_gone(self))
        except TooLong as exc:
            return _refuse(self, exc)
        if not admitted:
            return None
        try:
            return original(self, request, stop_words)
        finally:
            budget.release(need)

    handler_class.handle_completion = handle_completion
