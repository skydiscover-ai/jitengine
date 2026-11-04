"""OpenAI-compatible HTTP server over the Engine (stdlib ThreadingHTTPServer).

POST /v1/completions        prompt (text or ids), max_tokens | max_completion_tokens, temperature, top_p,
                            top_k, seed, stop, stream, stream_options.include_usage, logprobs (0..20),
                            echo, ignore_eos, forced_tokens, logprob_positions, x_fault
POST /v1/chat/completions   messages (+ tools, tool_choice none | auto: none renders no tools), the same sampling
                            fields, logprobs (bool) + top_logprobs (control-token text in any client string, tool
                            definitions and assistant turns included, is escaped as the template escapes user text;
                            see tokenizer.py).
                            With tools the reply comes back as reasoning_content, content and tool_calls, read as
                            Recursive's flint parsers read it (reply.py); without, as the text Flint wrote
                            (on /v1/messages and /v1/responses its reasoning is split off either way).
Refused with a 400 that names the field, never ignored: n or best_of other than 1, logit_bias, suffix, functions,
tool_choice / function_call other than none or auto, response_format other than {"type": "text"}.
Host / Origin: a request naming a host that is not loopback, an IP literal, the bind address or an `--allowed-host`
(or sent by a browser page of another origin) gets 403 before any work, on every route: the API has no
authentication, and a page whose domain re-resolves to 127.0.0.1 (DNS rebinding) would otherwise read it.
POST /v1/messages, /v1/messages/count_tokens   the Anthropic Messages API on the same chat (anthropic_api.py)
POST /v1/responses          OpenAI's Responses API on the same chat (responses_api.py)
GET  /health  /v1/models  /metrics   counts, timings, statuses, configuration and versions (/v1/models: the
                            context limit as max_model_len); never another request's seed or request_id (those
                            are in that request's own x_metrics only)
Bounds checked before any model work: JSON content type (415), body size (413), parameter ranges and
token ids (400), the context limit (400). One request runs at a time; up to the engine's queue bound
wait, the rest get 429.
Disconnects: every POST has a watcher that cancels its job when the client closes the connection, whether
the job is queued, prefilling or decoding (a non-streaming response, or a stream in its prefill, writes nothing that
could fail), so an abandoned request stops using the GPU within one release group or one prefill chunk.
Limits (`serve --request-deadline-s S --max-output-tokens N`, both off by default and shown in /health as
request_deadline_s / max_output_tokens, null when off): the cap clamps max_tokens (an absent max_tokens becomes the
cap); a generation request still running S seconds after it was submitted stops at its next pass (or prefill chunk,
or before it starts if still queued), frees the engine and is answered normally with the tokens so far:
finish_reason "length" (HTTP 200, or the usual final SSE chunk then [DONE]) and x_metrics.deadline_exceeded true.
SIGTERM/SIGINT: new requests get 503; the engine's stop_all cancels the running job and every queued one, and each
client is answered (503 JSON, or an SSE error event then [DONE]); then the engine releases its memory and serve()
returns (exit 0), within SHUTDOWN_DEADLINE_S of the signal.
"""

from __future__ import annotations

import errno
import inspect
import ipaddress
import json
import re
import select
import signal
import socket
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jitengine_flint_m4pro import anthropic_api, logs, responses_api
from jitengine_flint_m4pro.engine.io_struct import (
    Cancelled,
    ContextTooLarge,
    EngineUnavailable,
    GenRequest,
    GenResult,
    QueueFull,
    StepOutput,
)
from jitengine_flint_m4pro.reply import ReplyStream, parse
from jitengine_flint_m4pro.tokenizer import IncrementalDecoder, StopMatcher

MAX_BODY_BYTES = 64 * 2**20
# the APIs served on the same chat as /v1/chat/completions, by path
APIS = {path: api for api in (anthropic_api, responses_api) for path in api.PATHS}
SHUTDOWN_GRACE_S = (
    20.0  # SIGTERM: wait this long for the in-flight handlers (each job is cancelled first) ...
)
SHUTDOWN_DEADLINE_S = (
    27.0  # ... and join the worker until this long after the signal (exit 0 within 30 s)
)
WATCH_POLL_S = 0.1  # disconnect watcher poll
JSON_TYPES = ("application/json",)
# Host names always served, besides IP literals (which DNS cannot rebind) and the bind address. The
# API has no authentication, so any other name in Host is refused: a page whose domain re-resolves
# to 127.0.0.1 would otherwise be same-origin with the server and read every response (DNS
# rebinding, as Ollama's CVE-2024-28224). `serve(..., allowed_hosts=)` adds names.
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
WILDCARD_BINDS = ("", "0.0.0.0", "::")


_ORIGIN = re.compile(r"(https?)://([^/?#\\\s@]+)/?")
_HOST_LABEL = re.compile(r"[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?")
_PORT = re.compile(r"[0-9]{1,5}")


def host_name(authority: str) -> str | None:
    """The host of a Host header value or URL authority ("name", "name:port", "[v6]:port"),
    lower-cased without its port or a trailing dot; None when it is not exactly that: userinfo
    (`@`), backslashes, spaces, a path or any other character a host cannot hold, a port that is
    not digits, a name that is not dot-separated labels or an IP literal."""
    a = authority.strip()
    if not a or not a.isascii() or not all(c.isalnum() or c in "-._:[]%" for c in a):
        return None
    if a.startswith("["):
        end = a.find("]")
        rest = a[end + 1 :] if end > 0 else "x"
        if rest and not (rest.startswith(":") and _PORT.fullmatch(rest[1:])):
            return None
        name = a[1:end] if end > 0 else ""
        if ":" not in name or _ip(name) is None:
            return None
    elif a.count(":") == 1:
        name, port = a.split(":", 1)
        if not _PORT.fullmatch(port):
            return None
    elif a.count(":") > 1:
        name = a  # an unbracketed IPv6 literal
        if _ip(name) is None:
            return None
    else:
        name = a
    name = name.lower().rstrip(".")
    if not name:
        return None
    if ":" not in name and _ip(name) is None:
        if not all(_HOST_LABEL.fullmatch(label) for label in name.split(".")):
            return None
    return name


def _ip(name: str):
    try:
        return ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return None


def served_names(bind_host: str, extra=()) -> frozenset:
    """The host names a request may name: the loopback names, the bind address unless it is a
    wildcard, and `extra`."""
    names = set(LOOPBACK_NAMES)
    b = host_name(bind_host or "")
    if b and b not in WILDCARD_BINDS:
        names.add(b)
    names.update(n for n in (host_name(str(x)) for x in extra) if n)
    return frozenset(names)


def refusal(headers, names: frozenset) -> str | None:
    """Why a request must be refused before any work, or None. `Host` must be a served name or an
    IP literal (a missing one is not a browser's: HTTP/1.1 browsers always send it). `Origin`,
    which browsers send on cross-origin requests and on every POST, must be a loopback or served
    host, or the request's own; non-browser clients (the OpenAI SDKs, curl) send none."""
    host = headers.get("Host")
    hname = None
    if host is not None:
        hname = host_name(host)
        if hname is None or (hname not in names and _ip(hname) is None):
            return f"host {host!r} is not served here"
    origin = headers.get("Origin")
    if origin is not None:
        m = _ORIGIN.fullmatch(origin.strip())
        oname = host_name(m.group(2)) or "" if m else ""  # "null" (a sandboxed page) has none
        oname = oname or None
        ip = _ip(oname) if oname else None
        if oname is None or not (
            oname in names or oname == hname or (ip is not None and ip.is_loopback)
        ):
            return f"origin {origin!r} may not use this server"
    return None


class BadRequest(ValueError):
    """A 400; `param` names the request field at fault, when there is one."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


def _unsupported(param: str, why: str, value: bool = False) -> BadRequest:
    """OpenAI's refusal of a parameter (or, with value, of a value) this server does not support."""
    return BadRequest(f"Unsupported {'value' if value else 'parameter'}: {why}", param)


def _refuse_unsupported(body: dict, chat: bool) -> None:
    """A 400 naming each parameter this server cannot serve, instead of ignoring it: one choice per request, no
    logit bias, no forced tool call, text responses only."""
    if body.get("n") not in (None, 1):
        raise _unsupported("n", "'n' must be 1; this server returns one choice.", value=True)
    if body.get("logit_bias"):
        raise _unsupported("logit_bias", "'logit_bias' is not supported by this server.")
    if not chat:
        if body.get("best_of") not in (None, 1):
            raise _unsupported(
                "best_of", "'best_of' must be 1; this server returns one choice.", value=True
            )
        if body.get("suffix") not in (None, ""):
            raise _unsupported("suffix", "'suffix' is not supported on /v1/completions here.")
        return
    if body.get("functions"):
        raise _unsupported("functions", "'functions' is not supported: send 'tools'.")
    for key in ("tool_choice", "function_call"):
        if body.get(key) not in (None, "none", "auto"):
            why = f"'{key}' may only be 'none' or 'auto': this server does not force a tool call."
            raise _unsupported(key, why, value=True)
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type") == "text"):
        why = '\'response_format\' may only be {"type": "text"} on this server.'
        raise _unsupported("response_format", why, value=True)


def _num(body, key, default, lo=None, hi=None, integer=False, lo_open=False):
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BadRequest(f"{key} must be a number")
    if v != v or v in (float("inf"), float("-inf")):  # NaN / infinity (json accepts NaN, Infinity)
        raise BadRequest(f"{key} must be finite")
    if integer:
        if int(v) != v:
            raise BadRequest(f"{key} must be an integer")
        v = int(v)
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise BadRequest(f"{key} must be {'>' if lo_open else '>='} {lo}")
    if hi is not None and v > hi:
        raise BadRequest(f"{key} must be <= {hi}")
    return v


def _ids(v, name):
    if not isinstance(v, list) or not all(
        isinstance(t, int) and not isinstance(t, bool) for t in v
    ):
        raise BadRequest(f"{name} must be a list of integer token ids")
    return v


def parse_request(engine, body: dict, chat: bool):
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    _refuse_unsupported(body, chat)
    gen = engine.cfg.gen_defaults
    prompt, tools = _prompt(engine, body, chat)
    if not prompt:
        raise BadRequest("prompt is empty after tokenization")
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if mt is not None:
        mt = _num({"max_tokens": mt}, "max_tokens", None, lo=0, integer=True)
    temperature = _num(body, "temperature", float(gen.get("temperature", 1.0)), lo=0.0)
    top_p = _num(body, "top_p", float(gen.get("top_p", 0.95)), lo=0.0, hi=1.0, lo_open=True)
    top_k = _num(body, "top_k", int(gen.get("top_k", 20)), lo=0, integer=True)
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise BadRequest("seed must be an integer")
    stop = _stop_strings(body)
    logprobs, echo = _logprobs(body, chat)
    forced = body.get("forced_tokens")
    if forced is not None:
        if chat:
            raise BadRequest("forced_tokens is a /v1/completions field")
        forced = _ids(forced, "forced_tokens")
    positions = body.get("logprob_positions")
    if positions is not None:
        positions = _ids(positions, "logprob_positions")
    fault = body.get("x_fault")
    if fault is not None and not isinstance(fault, str):
        raise BadRequest("x_fault must be a string")
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise BadRequest("ignore_eos must be a boolean")
    otp = body.get("one_token_per_pass")
    req = GenRequest(
        prompt_ids=prompt,
        max_tokens=mt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        ignore_eos=ignore_eos,
        stop=stop,
        logprobs=logprobs,
        forced_tokens=forced,
        one_token_per_pass=otp if isinstance(otp, bool) else None,
        fault=fault,
        request_id=("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:16],
    )
    so = body.get("stream_options") or {}
    meta = {
        "stream": bool(body.get("stream", False)),
        "include_usage": bool(isinstance(so, dict) and so.get("include_usage")),
        "model": body.get("model") or engine.model_name,
        "echo": echo,
        "positions": positions,
        "chat": chat,
        "top_lp": logprobs,
        "tools": tools or None,  # the tools whose calls the reply is read for
    }
    _check_echo(echo, forced, mt, logprobs)
    return req, meta


def _prompt(engine, body, chat) -> tuple:
    """The request's prompt ids, and the tools a chat request renders with them (None without)."""
    if chat:
        msgs, tools = _chat_messages(body)
        try:
            return engine.render_chat(msgs, tools), tools
        except Exception as exc:
            raise BadRequest(f"chat template failed: {exc}") from exc
    p = body.get("prompt")
    if isinstance(p, str):
        return engine.encode(p), None
    if isinstance(p, list) and p:
        return _ids(p, "prompt"), None
    raise BadRequest("prompt must be a non-empty string or a list of token ids")


def _chat_messages(body) -> tuple:
    """The chat request's messages and tools, checked: a non-empty list of known roles, and a list of tools if any."""
    msgs = body.get("messages")
    if (
        not isinstance(msgs, list)
        or not msgs
        or not all(isinstance(m, dict) and "role" in m for m in msgs)
    ):
        raise BadRequest("messages must be a non-empty list of {role, content}")
    for m in msgs:
        if m["role"] not in ("system", "developer", "user", "assistant", "tool"):
            raise BadRequest(f"unsupported role {m['role']!r}")
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise BadRequest("tools must be a list", "tools")
    if body.get("tool_choice") == "none":  # no tool may be called: the template offers none
        tools = None
    return msgs, tools


def _stop_strings(body) -> list:
    """The request's stop strings, as a list."""
    stop = body.get("stop")
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    elif not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise BadRequest("stop must be a string or a list of strings")
    return stop


def _logprobs(body, chat) -> tuple:
    """How many top log-probabilities each step returns (None: none), and whether a completion echoes its prompt for scoring."""
    if chat:
        want_lp = body.get("logprobs")
        if want_lp is not None and not isinstance(want_lp, bool):
            raise BadRequest("logprobs must be a boolean for chat completions")
        top_lp = _num(body, "top_logprobs", 0, lo=0, hi=20, integer=True)
        logprobs = top_lp if want_lp else None
        echo = False
    else:
        lp = body.get("logprobs")
        logprobs = None if lp is None else _num(body, "logprobs", 0, lo=0, hi=20, integer=True)
        echo = bool(body.get("echo", False))
    return logprobs, echo


def _check_echo(echo, forced, mt, logprobs) -> None:
    """Echo scoring takes no forced stream, no new tokens and a logprobs count."""
    if echo:
        if forced is not None:
            raise BadRequest("echo and forced_tokens cannot be combined")
        if (mt or 0) != 0:
            raise BadRequest("echo scoring requires max_tokens 0")
        if logprobs is None:
            raise BadRequest("echo requires logprobs (the number of top log-probabilities, 0..20)")


def _live_listener(addr) -> bool:
    """Does a LIVE listener answer `addr` = (host, port)? Called only while our socket is NOT bound to
    the port (macOS drops a SYN aimed at an address that a bound, not yet listening socket holds, so a
    probe after bind cannot tell a live listener from a free port). Refused: no listener, only
    TIME_WAIT ends hold the port. Answered, timed out (a bound socket or a full backlog swallowed the
    SYN) or any other error: treat as in use."""
    host, port = addr[0], addr[1]
    target = "127.0.0.1" if host in ("", "0.0.0.0") else host
    try:
        socket.create_connection((target, port), timeout=1.0).close()
    except ConnectionRefusedError:
        return False
    except OSError:
        return True
    return True


def _accepts(fn, name: str) -> bool:
    """Does `fn` take keyword `name` (the server also runs over duck-typed engines, e.g. a test stub)?"""
    try:
        ps = inspect.signature(fn).parameters
    except TypeError, ValueError:
        return False
    return name in ps or any(p.kind == p.VAR_KEYWORD for p in ps.values())


def _watch_disconnect(sock, cancel: threading.Event, done: threading.Event) -> None:
    """Set `cancel` when the client closes its connection before the request is answered. Peeks, never consumes:
    readable data (a pipelined request) is not a disconnect; end of stream or a reset is."""
    while not done.is_set() and not cancel.is_set():
        try:
            r, _, _ = select.select([sock], [], [], WATCH_POLL_S)
            if not r:
                continue
            if not sock.recv(1, socket.MSG_PEEK):
                cancel.set()
                return
            time.sleep(WATCH_POLL_S)
        except OSError, ValueError:
            cancel.set()
            return


def _usage(prompt_tokens, completion_tokens, cached):
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "prompt_tokens_details": {"cached_tokens": int(cached)},
    }


def _log_engine_error(engine, exc: BaseException) -> None:
    """An error no status code names: logged with its traceback before the client gets it."""
    engine.log(
        logs.error("[server] engine error: " + "".join(traceback.format_exception(exc)).rstrip())
    )


def _fingerprint(engine) -> str:
    return f"flint-engine/profile={engine.profile}/kv{engine.kv_bits}"


def _chat_message(text: str, tools: list | None) -> dict:
    """A whole reply's assistant message: the text Flint wrote when `tools` is None, else its reasoning,
    content and tool calls (a request without tools on another API: its reasoning and content)."""
    if tools is None:
        return {"role": "assistant", "content": text}
    reasoning, content, calls = parse(text, tools)
    msg = {"role": "assistant", "content": content or (None if calls else "")}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def _stop_sequence(stops, text: str) -> str | None:
    """The stop string that `text` contains first (of two at one place, the one listed first):
    the one that ended the reply, which the Messages API reports as its `stop_sequence`."""
    found = [(i, n) for n, s in enumerate(stops) if (i := text.find(s)) >= 0]
    return stops[min(found)[1]] if found else None


def _top_entries(engine, top):
    return [
        {"token": engine.tokenizer.piece(i), "token_id": int(i), "logprob": float(v)}
        for i, v in top
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # the API a request came in on besides OpenAI's (anthropic_api, responses_api), which reads it
    # into a chat request and writes the reply back in its own shapes
    api = None
    events = None  # that API's events for a stream, while one is open

    def log_message(self, *a):
        pass

    # --------------------------------------------------------------- helpers
    def _json(self, code, obj, headers=None):
        if self.api is not None:
            obj = self.api.reply(code, obj)
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection and "Connection" not in (headers or {}):
            self.send_header("Connection", "close")  # so the client does not reuse it
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, msg, etype="invalid_request_error", param=None, headers=None):
        err = {"message": msg, "type": etype, "code": code}
        if param is not None:
            err["param"] = param
        self._json(code, {"error": err}, headers)

    def _forbidden(self) -> bool:
        """Refuse (403, connection closed: no body is read) a request for another host or from
        another origin; True when it was refused."""
        why = refusal(self.headers, getattr(self.server, "served_names", None) or served_names(""))
        if why is None:
            return False
        self.close_connection = True
        self._error(403, f"forbidden: {why}", "forbidden", headers={"Connection": "close"})
        return True

    def _refusal(self, exc) -> None:
        if isinstance(exc, QueueFull):
            return self._error(429, str(exc), "queue_full")
        if isinstance(exc, EngineUnavailable):
            return self._error(503, str(exc), "unavailable")
        if isinstance(exc, ContextTooLarge):
            return self._error(400, str(exc), "context_too_large")
        if isinstance(exc, ValueError):
            return self._error(400, str(exc))
        _log_engine_error(self.server.engine, exc)
        return self._error(500, f"engine error: {exc}", "server_error")

    # --------------------------------------------------------------- GET
    def do_GET(self):
        self.api = None
        if self._forbidden():
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/v1/models":
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.engine.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "flint-engine",
                            "max_model_len": self.server.engine.context_limit,
                        }
                    ],
                },
            )
        if path in ("/health", "/"):
            h = self.server.engine.health()
            if self.server.stopping.is_set():
                h["status"] = "stopping"
            return self._json(200 if h["status"] in ("ready", "busy") else 503, h)
        if path in ("/metrics", "/stats"):
            return self._json(200, self.server.engine.stats())
        return self._error(404, "not found")

    # --------------------------------------------------------------- POST
    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        self.api = APIS.get(path)  # first, so even a refusal is in the API's own error form
        if self._forbidden():
            return
        chat = path == "/v1/chat/completions" or self.api is not None
        if not chat and path != "/v1/completions":
            # the body is not read, so the connection cannot carry another request
            self.close_connection = True
            return self._error(404, "not found", headers={"Connection": "close"})
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype not in JSON_TYPES:
            self.close_connection = True
            return self._error(415, "Content-Type must be application/json")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True  # the body's length is unknown
            return self._error(400, "bad Content-Length")
        if n < 0:  # a negative length would make rfile.read(n) read to EOF
            self.close_connection = True
            return self._error(400, "bad Content-Length")
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            return self._error(413, f"request body of {n} bytes exceeds {MAX_BODY_BYTES}")
        if self.server.stopping.is_set():
            return self._error(503, "server is stopping", "unavailable")
        try:
            req, meta = self._read_request(n, chat, path)
        except (BadRequest, ValueError, UnicodeDecodeError) as exc:
            return self._error(400, str(exc), param=getattr(exc, "param", None))
        if path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": len(req.prompt_ids)})
        cancel, done = threading.Event(), threading.Event()
        with self.server.lock:
            self.server.inflight[0] += 1
        try:
            threading.Thread(
                target=_watch_disconnect,
                args=(self.connection, cancel, done),
                daemon=True,
                name="flint-watch",
            ).start()
            if meta["echo"]:
                return self._echo(req, meta, cancel)
            try:
                events = (
                    self.server.engine.stream(req, cancel=cancel)
                    if self.server.stream_cancel
                    else self.server.engine.stream(req)
                )
            except Exception as exc:  # noqa: BLE001  (_refusal logs what it answers with a 500)
                return self._refusal(exc)
            if meta["stream"]:
                return self._stream(req, meta, events)
            return self._complete(req, meta, events)
        finally:
            done.set()
            with self.server.lock:
                self.server.inflight[0] -= 1

    def _read_request(self, n, chat, path):
        """The request in `n` body bytes and its metadata; another API's body is read as the chat
        body it stands for."""
        body = json.loads((self.rfile.read(n) if n else b"").decode("utf-8") or "{}")
        try:
            if self.api is not None:
                body = self.api.read(path, body)
        except (AttributeError, TypeError) as exc:
            raise BadRequest(f"bad request for {path}: {exc}") from exc
        req, meta = parse_request(self.server.engine, body, chat)
        if self.api is not None and meta["tools"] is None:
            # the Messages and Responses APIs give reasoning its own block or item, tools or not
            meta["tools"] = []
        return req, meta

    # --------------------------------------------------------------- echo scoring
    def _echo(self, req, meta, cancel):
        try:
            kw = {"cancel": cancel} if self.server.score_cancel else {}
            scored = self.server.engine.score(
                req.prompt_ids, meta["positions"], top_k=meta["top_lp"], **kw
            )
        except Cancelled:  # the client went away: nothing to answer
            self.close_connection = True
            return None
        except Exception as exc:  # noqa: BLE001  (_refusal logs what it answers with a 500)
            return self._refusal(exc)
        prompt = list(req.prompt_ids)
        plp = [
            {
                "position": s.position,
                "token_id": s.token_id,
                "logprob": s.logprob,
                "top": [[int(i), float(v)] for i, v in s.top],
            }
            for s in scored
        ]
        choice = {
            "index": 0,
            "text": self.server.engine.decode(prompt),
            "finish_reason": "length",
            "token_ids": [],
            "prompt_token_ids": prompt,
            "prompt_logprobs": plp,
            "logprobs": None,
        }
        self._json(
            200,
            {
                "id": req.request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": meta["model"],
                "system_fingerprint": _fingerprint(self.server.engine),
                "choices": [choice],
                "usage": _usage(len(prompt), 0, 0),
            },
        )

    # --------------------------------------------------------------- non-streaming
    def _complete(self, req, meta, events):
        steps, result = [], None
        try:
            for item in events:
                if isinstance(item, StepOutput):
                    steps.append(item)
                else:
                    result = item
        except Exception as exc:  # noqa: BLE001  (_refusal logs what it answers with a 500)
            return self._refusal(exc)
        et = (
            (getattr(result, "metrics", None) or {}).get("error_type")
            if result is not None
            else None
        )
        if et == "Cancelled":  # the client went away: nothing to answer
            self.close_connection = True
            return None
        if (
            et == "EngineUnavailable"
        ):  # stopped by the engine (SIGTERM / close) before or during its decode
            self.close_connection = True
            return self._json(
                503,
                {
                    "error": {
                        "message": result.error or "server is stopping",
                        "type": "unavailable",
                        "code": 503,
                    },
                    "x_metrics": result.metrics,
                },
            )
        if result is None or result.finish_reason == "error":
            msg = result.error if result is not None else "request failed"
            return self._json(
                500,
                {
                    "error": {"message": msg, "type": "engine_error", "code": 500},
                    "x_metrics": getattr(result, "metrics", {}),
                },
            )
        text = (
            result.text if result.text is not None else self.server.engine.decode(result.token_ids)
        )
        choice = {
            "index": 0,
            "finish_reason": result.finish_reason,
            "token_ids": list(result.token_ids),
        }
        if req.forced_tokens is not None:
            choice["forced_steps"] = [self._forced_entry(req, s) for s in steps]
        choice["logprobs"] = (
            self._logprobs(meta, steps)
            if meta["top_lp"] is not None and req.forced_tokens is None
            else None
        )
        if meta["chat"]:
            choice["message"] = _chat_message(text, meta["tools"])
            if "tool_calls" in choice["message"]:
                choice["finish_reason"] = "tool_calls"
            elif self.api is not None and choice["finish_reason"] == "stop" and req.stop:
                stopped_by = _stop_sequence(req.stop, self.server.engine.decode(result.token_ids))
                if stopped_by is not None:
                    choice["x_stop_sequence"] = stopped_by
        else:
            choice["text"] = text
        self._json(
            200,
            {
                "id": req.request_id,
                "object": "chat.completion" if meta["chat"] else "text_completion",
                "created": int(time.time()),
                "model": meta["model"],
                "system_fingerprint": _fingerprint(self.server.engine),
                "choices": [choice],
                "usage": _usage(
                    len(req.prompt_ids), len(result.token_ids), result.cached_prompt_tokens
                ),
                "x_metrics": result.metrics,
            },
        )

    def _forced_entry(self, req, s):
        return {
            "position": len(req.prompt_ids) + s.index,
            "forced_token": s.token_id,
            "forced_logprob": s.logprob,
            "top": [[int(i), float(v)] for i, v in s.top],
        }

    def _logprobs(self, meta, steps):
        if meta["chat"]:
            return {
                "content": [
                    {
                        "token": self.server.engine.tokenizer.piece(s.token_id),
                        "token_id": s.token_id,
                        "logprob": s.logprob,
                        "top_logprobs": _top_entries(self.server.engine, s.top),
                    }
                    for s in steps
                ]
            }
        return {
            "tokens": [self.server.engine.tokenizer.piece(s.token_id) for s in steps],
            "token_ids": [s.token_id for s in steps],
            "token_logprobs": [s.logprob for s in steps],
            "top_logprobs": [
                {self.server.engine.tokenizer.piece(i): float(v) for i, v in s.top} for s in steps
            ],
            "top_logprob_ids": [[[int(i), float(v)] for i, v in s.top] for s in steps],
        }

    # --------------------------------------------------------------- streaming
    def _sse(self, obj) -> None:
        if self.events is not None:
            self.wfile.write(self.events.frame(obj))
        else:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(self.events.done() if self.events is not None else b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream(self, req, meta, events):
        chunks = _Chunks(self, req, meta)
        self.events = self.api.Events(meta["model"]) if self.api is not None else None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        text = _StreamText(self.server.engine.tokenizer, req.stop, meta["tools"])
        n_out = 0
        result = None
        try:
            for item in events:  # on SIGTERM the engine itself ends the job (stop_all), so read on
                if isinstance(item, GenResult):
                    result = item
                    break
                if item.token_id is not None:
                    n_out += 1
                self._sse(chunks.make(text.step(item.token_id), item))
        except BrokenPipeError, ConnectionResetError:
            events.close()
            return
        except Exception as exc:  # noqa: BLE001  (logged, then sent as the stream's error event)
            _log_engine_error(self.server.engine, exc)
            result = GenResult(
                [], "error", len(req.prompt_ids), error=f"{type(exc).__name__}: {exc}"
            )
        if result is None:  # the iterator ended without its final item
            events.close()
            result = GenResult(
                [], "error", len(req.prompt_ids), error="the request ended without a result"
            )
        if (result.metrics or {}).get("error_type") == "Cancelled":
            return  # the client went away: nothing to send
        try:
            self._stream_end(req, meta, chunks, text, result, n_out)
        except BrokenPipeError, ConnectionResetError:
            return

    def _stream_end(self, req, meta, chunks, text, result, n_out) -> None:
        """A failed stream's error event; otherwise the rest of the text, a chunk for each tool
        call, the final chunk with the usage, and the usage chunk when asked for. Then [DONE]."""
        if result.finish_reason == "error":
            unavailable = (result.metrics or {}).get("error_type") == "EngineUnavailable"
            self._sse(
                {
                    "error": {
                        "message": result.error or "request failed",
                        "type": "unavailable" if unavailable else "engine_error",
                        "code": 503 if unavailable else 500,
                    },
                    "x_metrics": result.metrics,
                }
            )
        else:
            delta, calls = text.end()
            for i, call in enumerate(calls):
                self._sse(chunks.make({"tool_calls": [{"index": i, **call}]}))
            finish = "tool_calls" if calls else result.finish_reason
            stopped_by = text.stopped_by if self.api is not None and finish == "stop" else None
            final = chunks.make(delta, None, finish, stopped_by)
            final["usage"] = _usage(len(req.prompt_ids), n_out, result.cached_prompt_tokens)
            final["x_metrics"] = result.metrics
            self._sse(final)
            if meta["include_usage"]:
                self._sse({**chunks.base, "choices": [], "usage": final["usage"]})
        self._sse_done()


class _Chunks:
    """The chunk objects of one stream: the fields every chunk shares, the role on the first chat
    delta, and each step's token id with its logprobs or forced step."""

    def __init__(self, handler, req, meta):
        self.handler, self.req, self.meta = handler, req, meta
        self.base = {
            "id": req.request_id,
            "object": "chat.completion.chunk" if meta["chat"] else "text_completion",
            "created": int(time.time()),
            "model": meta["model"],
            "system_fingerprint": _fingerprint(handler.server.engine),
        }
        self.first = True

    def make(self, delta: dict, s=None, finish=None, stop_sequence=None) -> dict:
        ch = {"index": 0, "finish_reason": finish}
        if (
            stop_sequence is not None
        ):  # the Messages API reports which stop sequence ended the reply
            ch["x_stop_sequence"] = stop_sequence
        if s is not None:
            ch["token_ids"] = [s.token_id] if s.token_id is not None else []
            if self.req.forced_tokens is not None:
                ch["forced_step"] = self.handler._forced_entry(self.req, s)
            elif self.meta["top_lp"] is not None:
                ch["logprobs"] = self.handler._logprobs(self.meta, [s])
        if self.meta["chat"]:
            ch["delta"] = {"role": "assistant", **delta} if self.first else delta
        else:
            ch["text"] = delta.get("content", "")
        self.first = False
        return {**self.base, "choices": [ch]}


class _StreamText:
    """A stream's text: decoded token by token and cut before a stop string, holding back what may
    still begin one; with tools, read into reasoning, content and tool calls (reply.py)."""

    def __init__(self, tokenizer, stop, tools):
        self.dec = IncrementalDecoder(tokenizer)
        self.stops = StopMatcher(stop)
        self.sent = 0
        self.stopped_by: str | None = None  # the stop string that ended the text, once `end` ran
        self.reply = ReplyStream(tools) if tools is not None else None

    def step(self, token_id) -> dict:
        """The delta one engine step adds."""
        if token_id is not None:
            self.dec.push(token_id)
        text = self.dec.text
        cut = self.stops.find(text)
        limit = cut if cut is not None else self.stops.safe_prefix(text)
        piece = text[self.sent : limit] if limit > self.sent else ""
        self.sent = max(self.sent, limit)
        if self.reply is None:
            return {"content": piece}
        return self.reply.push(piece) or {"content": ""}

    def end(self) -> tuple[dict, list]:
        """The last delta and the tool calls, once the engine has finished."""
        self.dec.flush()
        text = self.dec.text
        cut = self.stops.find(text)
        if cut is not None:
            self.stopped_by = _stop_sequence(self.stops.stops, text)
        end = cut if cut is not None else len(text)
        rest = text[self.sent : end] if end > self.sent else ""
        if self.reply is None:
            return {"content": rest}, []
        delta = self.reply.push(rest)
        more, calls = self.reply.finish()
        for key, value in more.items():
            delta[key] = delta.get(key, "") + value
        return delta or {"content": ""}, calls


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # Bind WITHOUT SO_REUSEADDR first, so every holder of the port refuses the bind: a live listener on
    # the same address, and also a live wildcard (0.0.0.0) or dual-stack ([::]) one, which with
    # SO_REUSEADDR a 127.0.0.1 bind would sit beside and shadow for loopback clients (BSD semantics).
    # Only on EADDRINUSE, and only if no live listener answers the address, are the holders the
    # TIME_WAIT ends of a previous server (it closes streams, Connection: close requests and, when it
    # exits, every open connection first; ~30 s on macOS): rebind with SO_REUSEADDR, so a restart on
    # the same port is not refused for that long.
    allow_reuse_address = False
    request_queue_size = 64

    def __init__(self, server_address, handler, bind_and_activate: bool = True, allowed_hosts=()):
        # served_names first: server_bind (run by the base __init__) does not read it, handlers do
        self.served_names = served_names(str(server_address[0]), allowed_hosts)
        super().__init__(server_address, handler, bind_and_activate)

    def server_bind(self):
        try:
            super().server_bind()
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or _live_listener(self.server_address):
                raise
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()


def serve(
    engine, host: str = "127.0.0.1", port: int = 8080, ready_event=None, allowed_hosts=()
) -> None:
    """Serve `engine` until SIGTERM or SIGINT. `allowed_hosts`: host names served besides the
    loopback names, IP literals and `host`; any other Host, and any other browser Origin, gets 403."""
    stopping = threading.Event()
    t_signal = [0.0]
    inflight = [0]  # POST handlers between parse and response (the shutdown grace waits for them)
    lock = threading.Lock()
    stream_cancel = _accepts(engine.stream, "cancel")
    score_cancel = _accepts(engine.score, "cancel")

    srv = Server(
        (host, port), Handler, allowed_hosts=allowed_hosts
    )  # bind first: a port in use fails before any weight loads
    # what the handlers read, as self.server.<name>
    srv.engine = engine
    srv.stopping = stopping
    srv.inflight = inflight
    srv.lock = lock
    srv.stream_cancel = stream_cancel
    srv.score_cancel = score_cancel
    if not getattr(engine, "started", True):
        engine.start()

    def on_signal(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        t_signal[0] = time.monotonic()
        engine.log(
            f"[server] signal {signum}: stopping; the running and queued requests are cancelled and answered"
        )
        if hasattr(engine, "stop_all"):
            engine.stop_all(
                "server is stopping"
            )  # refuse new jobs; end the running one; answer the queued ones
        threading.Thread(target=srv.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, on_signal)
        except ValueError:  # not the main thread (an in-process caller); the caller owns signals
            break
    engine.log(f"[server] {engine.model_name} profile={engine.profile} on http://{host}:{port}")
    if ready_event is not None:
        ready_event.set()
    try:
        srv.serve_forever(poll_interval=0.2)
    finally:
        srv.server_close()
    t0 = t_signal[0] or time.monotonic()
    while inflight[0] and time.monotonic() < t0 + SHUTDOWN_GRACE_S:
        time.sleep(0.05)
    left = max(1.0, t0 + SHUTDOWN_DEADLINE_S - time.monotonic())
    if _accepts(engine.close, "timeout"):
        engine.close(timeout=left)
    else:
        engine.close()
