"""What the server sends back, in OpenAI's shapes: the error object (`error_object`), and the
status each engine failure maps to (`error_response`); usage (`usage_object`); the logprobs of a
completion and of its drawn EOS (`completion_logprobs`, `drawn_eos`); a stream's chunks
(`ChunkWriter`, and `EventWriter` for the Messages and Responses APIs); and the text held back while it
may still become a stop sequence (`StopText`).
"""

from __future__ import annotations

import json
import time


class StopText:
    """Stop sequences over text that arrives piece by piece.

    `push` returns the text that is safe to send: once a stop sequence appears, what precedes it;
    until then all but the last len(longest) - 1 characters, which could begin one. `flush`
    releases that held-back tail when generation ends without a stop.
    """

    def __init__(self, stops: list[str]):
        self.stops = list(stops)
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.pending = ""
        self.hit = False
        self.matched: str | None = None  # the stop sequence that was hit, once `hit`

    def push(self, piece: str) -> str:
        """Take the next `piece` of text and return what is now safe to send; sets `hit` when a
        stop sequence appears, and returns "" from then on."""
        if self.hit:
            return ""
        text = self.pending + piece
        found = [(i, n) for n, s in enumerate(self.stops) if (i := text.find(s)) >= 0]
        if found:
            cut, which = min(found)  # the earliest; of two at one place, the one listed first
            self.hit, self.pending, self.matched = True, "", self.stops[which]
            return text[:cut]
        keep = min(self.hold, len(text))
        self.pending = text[len(text) - keep :]
        return text[: len(text) - keep]

    def flush(self) -> str:
        """Return, and clear, the held-back tail: what is left when generation ends without a
        stop."""
        out, self.pending = self.pending, ""
        return out


def usage_object(prompt_tokens: int, completion_tokens: int, cached: int | None) -> dict:
    """OpenAI's `usage` object; the cached-token fields are there only when `cached` is not None."""
    out = {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }
    if cached is not None:  # from the engine's done event, which a request ended early lacks
        out["cached_prompt_tokens"] = int(cached)
        out["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    return out


def drawn_eos(trace: list, n_generated: int, finish: str, eos_ids) -> tuple[int, dict] | None:
    """The EOS token a generation that stopped drew, with the trace entry of the step that drew
    it; the trace keeps that step's distribution but not its token. None when there is none."""
    if finish != "stop" or len(trace) != n_generated + 1:
        return None
    e = trace[-1]
    eos = [int(t) for t in eos_ids or ()]
    drawn = [int(t) for t, v in e["top"] if int(t) in eos and v == e["logprob"]]
    if not (drawn or eos):
        return None
    return (drawn or eos)[0], e


def completion_logprobs(tok, ids: list[int], dists: list, k: int) -> dict:
    """OpenAI's completion `logprobs` for `ids`: each token's text and offset in the echoed text,
    its logprob, and the top `k` of its distribution `dists[i]` (None for a token without one) as
    {text: logprob}, the token's own included. `top_logprob_ids` repeats the top entries as
    [id, logprob], since two tokens can share a text."""
    pieces = [tok.decode([t], skip_special_tokens=False) for t in ids]
    offsets, off = [], 0
    for piece in pieces:
        offsets.append(off)
        off += len(piece)
    token_lps, tops, top_ids = [], [], []
    for piece, e in zip(pieces, dists):
        if e is None:
            token_lps.append(None)
            tops.append(None)
            top_ids.append(None)
            continue
        pairs = [(int(t), float(v)) for t, v in e["top"][:k]]
        top: dict[str, float] = {}
        for t, v in pairs:
            top.setdefault(tok.decode([t], skip_special_tokens=False), v)
        if e["logprob"] is not None:
            top.setdefault(piece, float(e["logprob"]))
        token_lps.append(e["logprob"])
        tops.append(top)
        top_ids.append([[t, v] for t, v in pairs])
    return {
        "tokens": pieces,
        "token_ids": list(ids),
        "token_logprobs": token_lps,
        "top_logprobs": tops,
        "top_logprob_ids": top_ids,
        "text_offset": offsets,
    }


def error_object(
    message: str, etype: str, param: str | None = None, code: str | None = None
) -> dict:
    """OpenAI's error object: what an error response and a stream's error frame hold as "error"."""
    return {"message": message, "type": etype, "param": param, "code": code}


def error_response(exc: BaseException) -> tuple[int, dict, int | None]:
    """The HTTP status, OpenAI error object and Retry-After seconds for an engine exception: this
    engine refuses nothing that a retry would cure, so there are none."""
    if isinstance(exc, ValueError):  # the engine's own admission checks
        return 400, error_object(str(exc), "invalid_request_error"), None
    return 500, error_object(f"engine error: {exc}", "server_error"), None


class ChunkWriter:
    """Writes one completion as an SSE stream: `data:` frames of chunk objects, then `[DONE]`."""

    def __init__(self, wfile, request_id: str, model: str, chat: bool, include_usage: bool):
        self.wfile = wfile
        self.chat = chat
        # OpenAI streams /v1/completions as "text_completion" objects, the shape it returns whole
        obj = "chat.completion.chunk" if chat else "text_completion"
        self.base = {"id": request_id, "object": obj, "created": int(time.time()), "model": model}
        if include_usage:
            self.base["usage"] = None  # null on every chunk but the final usage chunk
        self.held = b""

    def encode(self, payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    def done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def frame(self, payload: dict, hold: bool = False) -> None:
        """Write one frame, with any held before it; a held frame waits for the next."""
        self.held += self.encode(payload)
        if not hold:
            self.wfile.write(self.held)
            self.wfile.flush()
            self.held = b""

    def choice(
        self,
        delta: dict,
        finish: str | None = None,
        token_ids: list[int] | None = None,
        hold: bool = False,
        stop_sequence: str | None = None,
    ) -> None:
        """A chunk of the one choice: chat's `delta`, or its content as a completion's `text`."""
        ch = {"index": 0, "finish_reason": finish}
        if token_ids is not None:
            ch["token_ids"] = token_ids
        ch["logprobs"] = None
        if (
            stop_sequence is not None
        ):  # the Messages API reports which stop sequence ended the reply
            ch["x_stop_sequence"] = stop_sequence
        if self.chat:
            ch["delta"] = delta
        else:
            ch["text"] = delta.get("content", "")
        self.frame({**self.base, "choices": [ch]}, hold)

    def end(
        self,
        rest: str,
        finish: str,
        usage: dict | None,
        error: dict | None,
        tool_calls: list[dict] | None = None,
        stop_sequence: str | None = None,
    ) -> None:
        """Any text held back for a stop sequence, a chunk for each tool call, the chunk with the
        finish reason, the usage chunk when asked for, an error frame after a failure, then
        `[DONE]`."""
        if rest:
            self.choice({"content": rest})
        for i, call in enumerate(tool_calls or ()):
            self.choice({"tool_calls": [{"index": i, **call}]})
        self.choice({}, finish, stop_sequence=stop_sequence)
        if usage is not None:
            self.frame({**self.base, "choices": [], "usage": usage})
        if error is not None:
            # after the 200 a failure is a data frame holding an error object, not a chunk
            self.frame({"error": error})
        self.done()


class EventWriter(ChunkWriter):
    """Writes one chat completion as the Anthropic Messages API or OpenAI's Responses API streams
    it: the chunks ChunkWriter would send, as `events` turns them into that API's events
    (server/anthropic_api.py, server/responses_api.py)."""

    def __init__(self, events, wfile, request_id: str, model: str, chat: bool, include_usage: bool):
        super().__init__(wfile, request_id, model, chat, include_usage)
        self.events = events

    def encode(self, payload: dict) -> bytes:
        return self.events.frame(payload)

    def done(self) -> None:
        self.wfile.write(self.events.done())
        self.wfile.flush()
