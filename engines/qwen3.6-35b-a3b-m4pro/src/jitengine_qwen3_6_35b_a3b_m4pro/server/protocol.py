"""What a request means: `parse_request` reads an OpenAI completions or chat body into a
`GenRequest` and the response options, and refuses what this server cannot serve with
`InvalidRequestError`, a 400 that names the parameter at fault.
"""

from __future__ import annotations

import uuid

from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest
from jitengine_qwen3_6_35b_a3b_m4pro.tokenizer import ChatTemplateError

MAX_CONTEXT = 262144  # the model's context length


MAX_PROMPT_TOKENS = 32768


# OpenAI's default for /v1/completions; chat without max_tokens may fill the request's context
COMPLETIONS_DEFAULT_MAX_TOKENS = 16


MAX_STOP_SEQUENCES = 16  # OpenAI documents up to 4; a few more are accepted


MAX_STOP_CHARS = 1024


class InvalidRequestError(ValueError):
    """A request refused with 400; `param` and `code` fill OpenAI's error object."""

    def __init__(self, message: str, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.param = param
        self.code = code


def _unsupported_parameter(param: str, why: str) -> InvalidRequestError:
    """OpenAI's refusal of a parameter the server does not support."""
    return InvalidRequestError(f"Unsupported parameter: {why}", param, "unsupported_parameter")


def _unsupported_value(param: str, why: str) -> InvalidRequestError:
    """OpenAI's refusal of a value the server does not support for a parameter it knows."""
    return InvalidRequestError(f"Unsupported value: {why}", param, "unsupported_value")


def _context_exceeded(
    limit: int, prompt: int, completion: int | None, chat: bool
) -> InvalidRequestError:
    """OpenAI's context-length refusal: agents key their context handling on its code and text."""
    what = "messages" if chat else "prompt"
    if completion is None:
        msg = (
            f"This model's maximum context length is {limit} tokens. However, your {what} "
            f"resulted in {prompt} tokens. Please reduce the length of the {what}."
        )
    else:
        msg = (
            f"This model's maximum context length is {limit} tokens. However, you requested "
            f"{prompt + completion} tokens ({prompt} in the {what}, {completion} in the "
            f"completion). Please reduce the length of the {what} or completion."
        )
    return InvalidRequestError(msg, what, "context_length_exceeded")


def _num(body: dict, key: str, default, lo=None, hi=None, integer=False):
    v = body.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidRequestError(f"{key} must be a number", key)
    if integer:
        if int(v) != v:
            raise InvalidRequestError(f"{key} must be an integer", key)
        v = int(v)
    v = float(v) if not integer else v
    if lo is not None and v < lo:
        raise InvalidRequestError(f"{key} must be >= {lo}", key)
    if hi is not None and v > hi:
        raise InvalidRequestError(f"{key} must be <= {hi}", key)
    return v


def _refuse_unsupported(body: dict, chat: bool) -> None:
    """A 400 naming each parameter this server cannot serve, instead of ignoring it."""
    if body.get("n") not in (None, 1):
        raise _unsupported_value("n", "'n' must be 1; this server returns one choice.")
    if body.get("logit_bias"):
        raise _unsupported_parameter("logit_bias", "'logit_bias' is not supported by this server.")
    if chat:
        _refuse_unsupported_chat(body)
    else:
        _refuse_unsupported_completion(body)


def _refuse_unsupported_chat(body: dict) -> None:
    """Raise for the function, tool-choice, response-format and logprobs parameters a chat request may not carry."""
    if body.get("functions"):
        raise _unsupported_parameter("functions", "'functions' is not supported: send 'tools'.")
    for key in ("tool_choice", "function_call"):
        if body.get(key) not in (None, "none", "auto"):
            why = f"'{key}' may only be 'none' or 'auto': this server does not force a tool call."
            raise _unsupported_value(key, why)
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type") == "text"):
        why = '\'response_format\' may only be {"type": "text"} on this server.'
        raise _unsupported_value("response_format", why)
    for key, off in (("logprobs", False), ("top_logprobs", 0)):
        if body.get(key) not in (None, off):
            why = f"'{key}' is not supported on /v1/chat/completions here."
            raise _unsupported_parameter(key, why)


def _refuse_unsupported_completion(body: dict) -> None:
    """Raise for the best_of, logprobs, echo and suffix parameters a completion request may not carry."""
    if body.get("best_of") not in (None, 1):
        why = "'best_of' must be 1; this server returns one choice."
        raise _unsupported_value("best_of", why)
    for key, off in (("logprobs", None), ("echo", False), ("suffix", "")):
        if body.get(key) not in (None, off):
            why = f"'{key}' is not supported on /v1/completions here."
            raise _unsupported_parameter(key, why)


def _stop_sequences(body: dict) -> list[str]:
    """The request's `stop`, one string or a list of them, without empty ones."""
    stop = body.get("stop")
    if stop is None:
        return []
    if isinstance(stop, str):
        stop = [stop]
    if (
        not isinstance(stop, list)
        or len(stop) > MAX_STOP_SEQUENCES
        or not all(isinstance(s, str) and len(s) <= MAX_STOP_CHARS for s in stop)
    ):
        raise InvalidRequestError(
            f"stop must be a string or a list of at most {MAX_STOP_SEQUENCES} strings of at "
            f"most {MAX_STOP_CHARS} characters",
            "stop",
        )
    return [s for s in stop if s]


def _refuse_non_text_content(msgs: list) -> None:
    """Text is the only content this server takes; any other content part gets a clear 400."""
    for i, m in enumerate(msgs):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type", "text") if isinstance(part, dict) else type(part).__name__
            if kind != "text":
                raise InvalidRequestError(
                    f"messages[{i}] has a content part of type {kind!r}; this server takes "
                    f"text parts only",
                    "messages",
                )


def _chat_tools(body: dict) -> list[dict] | None:
    """The request's function tools, or None when it offers none or `tool_choice` is "none"."""
    tools = body.get("tools")
    if tools is None or tools == []:
        return None
    if not isinstance(tools, list) or not all(
        isinstance(t, dict)
        and t.get("type") == "function"
        and isinstance(t.get("function"), dict)
        and isinstance(t["function"].get("name"), str)
        for t in tools
    ):
        raise InvalidRequestError(
            'tools must be a list of {"type": "function", "function": {"name", ...}}', "tools"
        )
    return None if body.get("tool_choice") == "none" else tools


def _chat_prompt(tok, body: dict, tools: list[dict] | None) -> list[int]:
    """The token ids of a chat request's messages and tools, rendered by the checkpoint's template."""
    msgs = body.get("messages")
    if (
        not isinstance(msgs, list)
        or not msgs
        or not all(
            isinstance(m, dict) and "role" in m and ("content" in m or m.get("tool_calls"))
            for m in msgs
        )
    ):
        raise InvalidRequestError(
            "messages must be a non-empty list of {role, content}", "messages"
        )
    _refuse_non_text_content(msgs)
    # OpenAI treats `developer` as `system` for models that predate the role, and this checkpoint's
    # template knows only `system`
    msgs = [{**m, "role": "system"} if m.get("role") == "developer" else m for m in msgs]
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise InvalidRequestError("chat_template_kwargs must be an object", "chat_template_kwargs")
    try:
        return tok.render_chat(msgs, tools, **kw)
    except ChatTemplateError as exc:  # the reason is the client's to see
        raise InvalidRequestError(str(exc), "messages") from exc
    except Exception as exc:
        raise InvalidRequestError(f"chat template failed: {exc}", "messages") from exc


def _completion_prompt(tok, body: dict) -> list[int]:
    """The token ids of a completion request's prompt: its text encoded, or its token ids."""
    p = body.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
        p = p[0]  # a batch of one prompt is that prompt
    elif isinstance(p, list) and len(p) > 1 and all(isinstance(x, (str, list)) for x in p):
        why = "'prompt' must hold a single prompt; send a batch as separate requests."
        raise _unsupported_value("prompt", why)
    if isinstance(p, str):
        return tok.encode(p)
    if isinstance(p, list) and p and all(isinstance(t, int) and not isinstance(t, bool) for t in p):
        return [int(t) for t in p]
    raise InvalidRequestError("prompt must be a non-empty string or a list of token ids", "prompt")


def _prompt(engine, body: dict, chat: bool, tools: list[dict] | None) -> list[int]:
    """The request's prompt as token ids; an empty, overlong or out-of-vocabulary one is refused."""
    param = "messages" if chat else "prompt"
    tok = engine.tokenizer
    prompt = _chat_prompt(tok, body, tools) if chat else _completion_prompt(tok, body)
    if not prompt:
        raise InvalidRequestError("prompt is empty after tokenization", param)
    if len(prompt) > MAX_PROMPT_TOKENS:
        raise _context_exceeded(MAX_PROMPT_TOKENS, len(prompt), None, chat)
    if any(t < 0 or t >= engine.vocab for t in prompt):
        raise InvalidRequestError("prompt contains an out-of-range token id", param)
    return prompt


def _context(engine) -> int:
    """The prompt plus output tokens one request may hold."""
    # a row reserves KV for its whole generation at admission; refuse one larger than the budget
    limit = int(getattr(engine, "max_kv_positions", 0) or 0)
    return min(MAX_CONTEXT, limit) if limit else MAX_CONTEXT


def max_model_len(engine) -> int:
    """The longest request this server admits however it splits into prompt and output, as
    `/v1/models` reports it for clients that size their context to it."""
    return min(_context(engine), MAX_PROMPT_TOKENS)


def _max_tokens(engine, body: dict, chat: bool, n_prompt: int) -> int:
    """How many tokens the request may generate: chat's `max_completion_tokens` or `max_tokens`,
    else OpenAI's default. One that does not fit the context gets OpenAI's context-length error."""
    context = _context(engine)
    key = "max_tokens"
    if chat and body.get("max_completion_tokens") is not None:
        key = "max_completion_tokens"  # max_tokens' replacement; it wins when both are set
    if body.get(key) is not None:
        max_tokens = _num(body, key, None, lo=0, integer=True)
    elif chat:
        # OpenAI caps a chat request without a limit only by the model: generate until EOS or
        # until the request's context is full
        max_tokens = context - n_prompt
        if max_tokens < 1:
            raise _context_exceeded(context, n_prompt, None, chat)
    else:
        max_tokens = COMPLETIONS_DEFAULT_MAX_TOKENS
    if n_prompt + max_tokens > context:
        raise _context_exceeded(context, n_prompt, max_tokens, chat)
    return max_tokens


def parse_request(engine, body: dict, chat: bool) -> tuple[GenRequest, dict]:
    """Read a completions or chat `body` into a `GenRequest` and `meta` (`stream`,
    `include_usage`, `model`, `stop`, and `tools`, the tools whose calls the reply is read for);
    raises `InvalidRequestError`, a 400 naming the parameter at fault, for what this server
    cannot serve."""
    if not isinstance(body, dict):
        raise InvalidRequestError("body must be a JSON object")
    _refuse_unsupported(body, chat)
    tools = _chat_tools(body) if chat else None
    prompt = _prompt(engine, body, chat, tools)
    max_tokens = _max_tokens(engine, body, chat, len(prompt))
    temperature = _num(body, "temperature", 1.0, lo=0.0)
    top_p = _num(body, "top_p", 1.0, lo=0.0, hi=1.0)
    top_k = _num(body, "top_k", 0, lo=0, integer=True)
    seed = body.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidRequestError("seed must be an integer", "seed")
    stream = bool(body.get("stream", False))
    so = body.get("stream_options") or {}
    include_usage = bool(isinstance(so, dict) and so.get("include_usage"))
    stop = _stop_sequences(body)
    model = body.get("model")
    req = GenRequest(
        prompt_ids=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stop_at_eos=True,
        request_id=f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:16]}",
    )
    meta = {
        "stream": stream,
        "include_usage": include_usage,
        # a name is echoed back; anything else gets the served model's name
        "model": model if isinstance(model, str) and model else engine.model_name,
        "stop": stop,
        "tools": tools,
    }
    return req, meta
