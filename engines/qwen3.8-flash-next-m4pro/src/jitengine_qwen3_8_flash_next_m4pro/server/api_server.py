"""OpenAI-compatible HTTP server over the Engine: completions, chat completions, models, stats, and on
the same chat the Anthropic Messages API's /v1/messages and /v1/messages/count_tokens
(server/anthropic_api.py) and OpenAI's Responses API's /v1/responses (server/responses_api.py).

Handlers run on their own threads and the engine serializes model work. Bad requests are refused
(415, 413 or 400) before any model work; SIGTERM and SIGINT end open streams with `data: [DONE]`.
A request naming a host other than this server's (DNS rebinding), or sent by a browser page of
another origin, is refused with 403 before anything else is read.

The wire format is OpenAI's. An error is {"error": {"message", "type", "param", "code"}} with a
string or null code. A stream is `data:` frames ending in `data: [DONE]`: a chat stream's first
chunk carries the role, /v1/completions streams "text_completion" objects, and usage comes only on
a final `choices: []` chunk when `stream_options.include_usage` is set; a failure after the 200 adds
a `data: {"error": ...}` frame before `[DONE]`. Text ends before the first stop sequence. A chat
request's `tools` go to the template, and the calls in the reply come back as `tool_calls` with
finish reason "tool_calls" (server/tool_calls.py); a stream sends them, whole, after its text. A
parameter this server cannot serve (n > 1, functions, a tool_choice that forces a call,
response_format, chat logprobs, logit_bias, suffix, best_of, and logprobs or echo outside prompt
scoring) is refused with a 400 naming it, never ignored.

This module is the HTTP server itself: the handler and its routes, body limits, the signal handling
and `serve`. server/protocol.py reads a request, and server/openai_api.py writes the response.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import select
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import ClientDisconnected
from jitengine_qwen3_8_flash_next_m4pro.server import anthropic_api, responses_api
from jitengine_qwen3_8_flash_next_m4pro.server.openai_api import (
    ChunkWriter,
    EventWriter,
    StopText,
    completion_logprobs,
    drawn_eos,
    error_object,
    error_response,
    usage_object,
)
from jitengine_qwen3_8_flash_next_m4pro.server.protocol import (
    InvalidRequestError,
    max_model_len,
    parse_request,
)
from jitengine_qwen3_8_flash_next_m4pro.server.tool_calls import ToolCallText, parse
from jitengine_qwen3_8_flash_next_m4pro.tokenizer import IncrementalDecoder

logger = logging.getLogger(__name__)
# the APIs served on the same chat as /v1/chat/completions, by path
APIS = {path: api for api in (anthropic_api, responses_api) for path in api.PATHS}

MAX_BODY_BYTES = 32 * 2**20  # request bodies above this are refused unread
DRAIN_CAP_BYTES = (
    256 * 2**20
)  # how much of a refused body is read (discarded) so the client sees the 413
SHUTDOWN_GRACE_S = 20.0  # how long a graceful stop waits for open streams to end
# JSON only: CORS "simple" content types (text/plain, form encodings) would let any web page the
# operator visits POST to the loopback API without a preflight.
JSON_CONTENT_TYPES = ("application/json",)
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


def _new_kqueue():
    """A kqueue (macOS, the BSDs), or None where there is none. Tests make it fail as a process out
    of descriptors does."""
    return select.kqueue() if hasattr(select, "kqueue") else None


class PeerWatch:
    """Whether the client of one request has closed its connection, asked many times while the
    request runs (`Handler._client_alive`).

    Read from the socket's state, whatever the client sent that nobody has read yet: a pipelined
    request or a stray CRLF must not hide the close (a MSG_PEEK returns those bytes
    and never reaches the end of file behind them). kqueue's EV_EOF on macOS and the BSDs, through one
    kqueue opened when the request starts and closed when it ends; poll's POLLRDHUP on Linux (no
    descriptor); elsewhere, or when the kqueue cannot be opened (the process is out of descriptors),
    a peek, which needs no descriptor and sees the close whenever nothing is buffered.

    `closed()` is True only for a close it has seen (end of file, a reset), False while the client
    is there, and None when the probe cannot run: the caller then keeps serving, never cancels (a
    kqueue per probe failed at the descriptor limit, and every request in flight was
    cancelled as if its client had left). A half-closed client (shutdown of its sending side) reads
    as closed.
    """

    def __init__(self, sock):
        self.sock = sock
        self.kq = None
        try:
            kq = _new_kqueue()
        except OSError:
            kq = None  # out of descriptors: the peek serves this request
        if kq is not None:
            try:
                kq.control(
                    [select.kevent(sock.fileno(), select.KQ_FILTER_READ, select.KQ_EV_ADD)], 0, 0
                )
                self.kq = kq
            except OSError, ValueError:
                kq.close()

    def closed(self) -> bool | None:
        if self.kq is not None:
            try:
                evs = self.kq.control(None, 1, 0)
            except OSError, ValueError:
                return self._peek()
            return any(e.flags & select.KQ_EV_EOF for e in evs)
        if hasattr(select, "POLLRDHUP"):
            try:
                p = select.poll()
                p.register(self.sock.fileno(), select.POLLIN | select.POLLRDHUP)
                evs = p.poll(0)
            except OSError, ValueError:
                return self._peek()
            return any(m & (select.POLLRDHUP | select.POLLHUP) for _, m in evs)
        return self._peek()

    def _peek(self) -> bool | None:
        try:
            return not self.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except BlockingIOError:
            return False  # nothing to read: still connected
        except ConnectionResetError:
            return True  # the client reset the connection
        except OSError, ValueError:
            return None  # cannot tell: keep serving

    def close(self) -> None:
        if self.kq is not None:
            self.kq.close()
            self.kq = None


def peer_closed(sock) -> bool:
    """One probe (`PeerWatch`): True only for a close it has seen."""
    w = PeerWatch(sock)
    try:
        return w.closed() is True
    finally:
        w.close()


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


def _stream_tail(stop: StopText, calls: ToolCallText | None, finish: str) -> tuple:
    """What a stream still holds at its end: the text held back, the tool calls read from the
    reply (none after a failure) and the finish reason, "tool_calls" when there are calls."""
    rest = stop.flush()
    if calls is None:
        return rest, [], finish
    rest = calls.push(rest)
    more, tool_calls = calls.finish()
    if finish == "error":
        return rest + more, [], finish
    return rest + more, tool_calls, "tool_calls" if tool_calls else finish


class _Counter:
    """How many SSE streams are open right now (so a graceful stop can wait for them)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.value = 0

    def enter(self) -> None:
        """Count one more open stream."""
        with self._lock:
            self.value += 1

    def leave(self) -> None:
        """Count one stream fewer, never going below zero."""
        with self._lock:
            self.value = max(0, self.value - 1)


def _install_signal_handlers(srv, stopping: threading.Event) -> str:
    """Make SIGTERM/SIGINT stop the listener and let open streams finish cleanly.

    `srv.shutdown()` blocks until `serve_forever` returns, so the handler runs it on its own thread.
    Returns the installed signal names, or "" off the main thread, where `signal.signal` fails.
    """

    def handle(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        logger.info(f"[server] signal {signum}: draining open streams, then stopping")
        threading.Thread(target=srv.shutdown, name="server-shutdown", daemon=True).start()

    installed = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle)
        except ValueError, OSError:  # not the main thread
            return ""
        installed.append(sig.name)
    return "+".join(installed)


class Handler(BaseHTTPRequestHandler):
    """One HTTP connection, on its own thread. It reads `engine`, `stopping` and `open_streams`
    from `self.server`, where `serve` put them."""

    protocol_version = "HTTP/1.1"
    # the API a request came in on besides OpenAI's (anthropic_api, responses_api), which reads it
    # into a chat request and writes the reply back in its own shapes
    api = None

    def log_message(self, *a):  # quiet
        """Drop the access-log line that `BaseHTTPRequestHandler` would write to stderr."""
        pass

    def _json(self, status: int, obj: dict, headers: dict | None = None) -> None:
        if self.api is not None:
            obj = self.api.reply(status, obj)
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection and "Connection" not in (headers or {}):
            self.send_header("Connection", "close")  # so the client does not reuse it
        self.end_headers()
        self.wfile.write(data)

    def _error(
        self,
        status: int,
        message: str,
        etype: str | None = None,
        *,
        param: str | None = None,
        code: str | None = None,
        headers: dict | None = None,
    ) -> None:
        """Send OpenAI's error object; its type follows the status unless given."""
        if etype is None:
            etype = "server_error" if status >= 500 else "invalid_request_error"
        self._json(status, {"error": error_object(message, etype, param, code)}, headers)

    def _engine_error(self, exc: BaseException) -> None:
        """Answer an engine exception with its status, OpenAI error object and Retry-After."""
        status, error, retry_after = error_response(exc)
        headers = None if retry_after is None else {"Retry-After": str(retry_after)}
        self._json(status, {"error": error}, headers)

    def _client_alive(self) -> bool:
        """False only once the request's `PeerWatch` has seen the client close; read from the
        socket's state without consuming anything it sent, so a pipelined next request is still
        served. The engine asks while a request waits, decodes unstreamed or is scored, so a client
        that leaves cancels its row; a probe that cannot run keeps
        it served."""
        watch = getattr(self, "_peer", None)
        if watch is None:
            return True
        try:
            return watch.closed() is not True
        except Exception:  # noqa: BLE001  a probe that fails is no evidence the client left
            return True

    def _gone(self) -> None:
        """The client left: nobody to answer, so the connection just closes."""
        self.close_connection = True
        logger.info("[server] client closed its connection; its request was cancelled")

    def _too_large(self, n: int) -> None:
        """Send 413, then drain a bounded part of the body so the client is not reset."""
        self.close_connection = True
        msg = f"request body of {n} bytes exceeds the {MAX_BODY_BYTES}-byte limit"
        self._error(413, msg, headers={"Connection": "close"})
        self.wfile.flush()
        try:
            self.connection.settimeout(5.0)
            left = min(n, DRAIN_CAP_BYTES)
            while left > 0:
                chunk = self.rfile.read(min(left, 2**20))
                if not chunk:
                    break
                left -= len(chunk)
        except OSError, ValueError:
            pass  # draining is best effort: the connection closes after the 413 either way

    def _forbidden(self) -> bool:
        """Refuse (403, connection closed: no body is read) a request for another host or from
        another origin; True when it was refused."""
        why = refusal(self.headers, getattr(self.server, "served_names", None) or served_names(""))
        if why is None:
            return False
        self.close_connection = True
        self._error(403, f"forbidden: {why}", code="forbidden", headers={"Connection": "close"})
        return True

    def do_GET(self):
        """Answer GET `/v1/models`, `/v1/models/{id}`, `/health` (also `/`) and `/stats`;
        anything else is a 404. `/health` is a 503 when the engine's `health()` status is not
        "ok"."""
        self.api = None
        if self._forbidden():
            return
        path = self.path.split("?", 1)[0]
        name = self.server.engine.model_name
        model = {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "jitengine-qwen3.8-flash-next-m4pro",
            "max_model_len": max_model_len(self.server.engine),
        }
        if path in ("/v1/models", "/v1/models/"):
            return self._json(200, {"object": "list", "data": [model]})
        if path.startswith("/v1/models/"):
            # the id is the checkpoint's path, whose slashes a client may join or encode
            want = unquote(path[len("/v1/models/") :])
            if want.strip("/") == str(name).strip("/"):
                return self._json(200, model)
            msg = f"The model '{want}' does not exist"
            return self._error(404, msg, param="model", code="model_not_found")
        if path in ("/health", "/"):
            # health is observed (model thread alive and not failed), never a constant
            if hasattr(self.server.engine, "health"):
                h = self.server.engine.health()
                return self._json(200 if h.get("status") == "ok" else 503, h)
            return self._json(
                200, {"status": "ok", "memory_bytes": self.server.engine.memory_bytes()}
            )
        if path in ("/stats", "/stats/"):
            return self._json(200, self.server.engine.stats())
        return self._error(404, "not found")

    def do_POST(self):
        """Route POST `/v1/chat/completions`, `/v1/completions` and the other APIs' paths; a
        request for another host or from another origin is refused (403), one that is not JSON,
        too large or invalid is refused (415, 413 or 400) before the engine sees it, and a valid
        one is counted, scored (a prompt echoed with `logprobs`), streamed or served whole."""
        path = self.path.split("?", 1)[0]
        self.api = APIS.get(path)  # first, so even a refusal is in the API's own error form
        if self._forbidden():
            return
        chat = path == "/v1/chat/completions" or self.api is not None
        if not (chat or path == "/v1/completions"):
            # the body is not read, so the connection cannot carry another request
            self.close_connection = True
            return self._error(404, "not found", headers={"Connection": "close"})
        ctype = self._content_type()
        if ctype not in JSON_CONTENT_TYPES:
            self.close_connection = True
            return self._error(
                415, f"Content-Type must be application/json (got {ctype or 'none'})"
            )
        try:
            n = self._content_length()
        except ValueError:
            self.close_connection = True  # the body's length is unknown
            return self._error(400, "bad Content-Length")
        if n > MAX_BODY_BYTES:
            return self._too_large(n)
        try:
            req, meta = self._read_json_request(n, chat, path)
        except InvalidRequestError as exc:
            return self._error(400, str(exc), param=exc.param, code=exc.code)
        except (ValueError, UnicodeDecodeError) as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error(400, f"bad request: {exc}")
        if path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": len(req.prompt_ids)})
        self._peer = PeerWatch(self.connection)  # one per request, closed with it
        try:
            if meta["logprobs"] is not None:  # prompt scoring: echo with logprobs
                return self._echo_logprobs(req, meta)
            if meta["stream"]:
                return self._stream(req, meta, chat)
            return self._complete(req, meta, chat)
        finally:
            self._peer.close()
            self._peer = None

    def _content_type(self) -> str:
        """The request's media type, lower-cased without parameters ('' when absent)."""
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        return ctype

    def _content_length(self) -> int:
        """Content-Length as an int (0 when absent); ValueError if it is not a number."""
        n = int(self.headers.get("Content-Length") or 0)
        return n

    def _read_json_request(self, n: int, chat: bool, path: str) -> tuple:
        """Read `n` body bytes as JSON (another API's body as the chat body it stands for) and
        validate them against the engine; returns (request, meta)."""
        raw = self.rfile.read(n) if n > 0 else b""
        body = json.loads(raw.decode("utf-8") or "{}")
        if self.api is not None:
            body = self.api.read(path, body)
        req, meta = parse_request(self.server.engine, body, chat)
        return req, meta

    # ---------------------------------------------------------- echo logprobs (scoring)
    def _echo_logprobs(self, req, meta):
        """Echo logprobs in OpenAI's shape, scored by `Engine.score` (and `generate`).

        OpenAI's entry i describes token i: its logprob given the tokens before it, and the top
        tokens of that distribution; token 0 has no context and gets nulls. Prompt entry i of
        `score` is the distribution after `prompt[:i+1]`, so it describes token i + 1. Generated
        entry j is the distribution token j was drawn from, and the entry of the step that drew
        EOS stands for that EOS token, as OpenAI lists a stop token among the logprobs.
        """
        engine = self.server.engine
        ids = [int(t) for t in req.prompt_ids]
        k = int(meta["logprobs"])
        try:
            scored = engine.score(ids, (), max(1, k), alive=self._client_alive)
        except ClientDisconnected:
            return self._gone()
        except Exception as exc:  # noqa: BLE001
            return self._engine_error(exc)
        if len(scored) != len(ids):
            return self._error(500, f"score returned {len(scored)} entries for {len(ids)} tokens")
        gen_ids: list[int] = []
        trace: list[dict] = []
        finish, cached = "length", 0
        if int(req.max_tokens) > 0:
            try:
                res = engine.generate(req, logprobs=max(1, k), alive=self._client_alive)
            except ClientDisconnected:
                return self._gone()
            except Exception as exc:  # noqa: BLE001
                return self._engine_error(exc)
            if res.finish_reason == "error":  # an engine ValueError, which generate returns
                return self._error(400, (res.extra or {}).get("error", "request failed"))
            gen_ids = [int(t) for t in res.token_ids]
            trace = list((res.extra or {}).get("logprobs") or [])
            finish = res.finish_reason
            cached = int(getattr(res, "cached_prompt_tokens", 0) or 0)
        listed_ids = ids + gen_ids
        dists = [None, *scored[:-1], *trace[: len(gen_ids)]]
        eos = drawn_eos(trace, len(gen_ids), finish, getattr(engine, "eos_ids", ()))
        if eos is not None:
            listed_ids.append(eos[0])
            dists.append(eos[1])
        tok = engine.tokenizer
        choice = {
            "index": 0,
            # echo: the prompt, then the output, decoded as `tokens` are so `text_offset` indexes it
            "text": tok.decode(ids + gen_ids, skip_special_tokens=False),
            "finish_reason": finish,
            "logprobs": completion_logprobs(tok, listed_ids, dists, k),
            "token_ids": list(gen_ids),
        }
        self._json(
            200,
            {
                "id": req.request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": meta["model"],
                "choices": [choice],
                "usage": usage_object(len(ids), len(gen_ids), cached),
            },
        )

    # ---------------------------------------------------------- non-streaming
    def _stopped_by(self, stop, finish: str) -> str | None:
        """The stop sequence that ended the reply, for the Messages API to report as its
        `stop_sequence`; None on other APIs and for any other end."""
        if self.api is None or stop is None or finish != "stop":
            return None
        return stop.matched

    def _complete(self, req, meta, chat):
        """Serve one request whole. The engine's events are read as a stream reads them, so a
        refusal or failure reaches `error_response` as the exception the engine raised, and a stop
        sequence ends the row where it appears."""
        engine = self.server.engine
        events = engine.events(req, alive=self._client_alive)
        stop, dec = self._stop_filter(engine, meta)
        ids: list[int] = []
        parts: list[str] = []
        finish, cached = "length", None
        try:
            finish, cached = self._read_events(events, ids, parts, stop, dec, finish, cached)
        except ClientDisconnected:
            return self._gone()
        except Exception as exc:  # noqa: BLE001
            return self._engine_error(exc)
        finally:
            events.close()  # after a stop sequence this cancels the row
        text = self._final_text(engine, stop, parts, ids)
        choice = {"index": 0, "finish_reason": finish, "token_ids": ids, "logprobs": None}
        self._fill_choice(choice, chat, text, meta["tools"])
        stopped_by = self._stopped_by(stop, choice["finish_reason"])
        if stopped_by is not None:
            choice["x_stop_sequence"] = stopped_by
        self._json(
            200,
            {
                "id": req.request_id,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": meta["model"],
                "choices": [choice],
                "usage": usage_object(len(req.prompt_ids), len(ids), cached),
            },
        )

    def _stop_filter(self, engine, meta: dict) -> tuple:
        """The stop matcher and its incremental decoder (both None without `stop`)."""
        stop = StopText(meta["stop"]) if meta["stop"] else None
        dec = IncrementalDecoder(engine.tokenizer) if stop is not None else None
        return stop, dec

    def _read_events(
        self, events, ids: list[int], parts: list[str], stop, dec, finish: str, cached
    ) -> tuple:
        """Read the events: collect the tokens, stop at a stop sequence; returns (finish reason, cached count)."""
        for kind, payload in events:
            if kind == "token":
                ids.append(int(payload))
                if stop is not None:
                    parts.append(stop.push(dec.push(payload)))
                    if stop.hit:
                        finish = "stop"
                        break
            else:
                finish = payload.finish_reason
                cached = int(getattr(payload, "cached_prompt_tokens", 0) or 0)
        return finish, cached

    def _final_text(self, engine, stop, parts: list[str], ids: list[int]) -> str:
        """The completion text: the stop matcher's output with its tail flushed, else the decoded ids."""
        if stop is not None:
            text = "".join(parts) + stop.flush()
        else:
            text = engine.tokenizer.decode(ids)
        return text

    def _fill_choice(self, choice: dict, chat: bool, text: str, tools: list[dict] | None) -> None:
        """Put `text` in the choice as a chat message, with the tool calls read from it when the
        request offered tools, or as completion text."""
        if not chat:
            choice["text"] = text
            return
        content, calls = parse(text, tools) if tools else (text, [])
        choice["message"] = {"role": "assistant", "content": content}
        if calls:
            choice["message"] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": calls,
            }
            choice["finish_reason"] = "tool_calls"

    # ---------------------------------------------------------- streaming
    def _stream(self, req, meta, chat):
        """One SSE stream, counted as open through its final `data: [DONE]`."""
        events = self.server.engine.events(req, alive=self._client_alive)
        self.server.open_streams.enter()
        try:
            self._stream_body(req, meta, chat, events)
        finally:
            events.close()  # leaving early (a stop sequence, a gone client) cancels the row
            self.server.open_streams.leave()

    def _stream_body(self, req, meta, chat, events):
        args = (self.wfile, req.request_id, meta["model"], chat, meta["include_usage"])
        if self.api is None:
            out = ChunkWriter(*args)
        else:
            out = EventWriter(self.api.Events(meta["model"]), *args)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        dec = IncrementalDecoder(self.server.engine.tokenizer)
        stop = StopText(meta["stop"])
        calls = ToolCallText(meta["tools"]) if meta["tools"] else None
        n_out = 0
        finish = "length"
        cached = None
        error = None
        try:
            if chat:
                # the first chunk carries the role, as OpenAI's does, tokens or not. It goes out
                # in one write with the next chunk, so no write precedes the first token's and a
                # client timing its first chunk still times the first token
                out.choice({"role": "assistant", "content": ""}, hold=True)
            for kind, payload in events:
                if self.server.stopping.is_set():
                    # a graceful stop ends the stream like a client cancellation: final chunk
                    # and [DONE]; leaving the loop cancels the row
                    break
                if kind == "token":
                    n_out += 1
                    text = stop.push(dec.push(payload))
                    if calls is not None:  # text that may belong to a tool call is held back
                        text = calls.push(text)
                    out.choice({"content": text}, token_ids=[int(payload)])
                    if stop.hit:
                        finish = "stop"
                        break
                else:
                    finish = payload.finish_reason
                    cached = int(getattr(payload, "cached_prompt_tokens", 0) or 0)
        except BrokenPipeError, ConnectionResetError, ClientDisconnected:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[server] stream error: {exc}")
            # the final chunk still says "error": clients that ignore error frames, such as
            # benchmarks/benchmark.py, count the stream as failed by its finish reason
            finish, error = "error", error_response(exc)[1]
        usage = usage_object(len(req.prompt_ids), n_out, cached) if meta["include_usage"] else None
        rest, tool_calls, finish = _stream_tail(stop, calls, finish)
        try:
            out.end(rest, finish, usage, error, tool_calls, self._stopped_by(stop, finish))
        except BrokenPipeError, ConnectionResetError:
            return


class Server(ThreadingHTTPServer):
    """The HTTP server `serve` runs: daemon handler threads, an address that may be rebound at
    once, a backlog of 64 pending connections, and the host names it answers (`served_names`)."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, server_address, handler, bind_and_activate: bool = True, allowed_hosts=()):
        super().__init__(server_address, handler, bind_and_activate)
        self.served_names = served_names(str(server_address[0]), allowed_hosts)


def serve(engine, host: str = "127.0.0.1", port: int = 8080, allowed_hosts=()) -> None:
    """Serve `engine` on `host:port` until SIGTERM or SIGINT (the handlers are installed on the
    main thread only), then wait up to `SHUTDOWN_GRACE_S` for open streams to end. Blocks.
    `allowed_hosts`: host names served besides the loopback names, IP literals and `host`."""
    stopping = threading.Event()  # set by SIGTERM/SIGINT; open streams end at the next token
    open_streams = _Counter()

    srv = Server((host, port), Handler, allowed_hosts=allowed_hosts)
    # what the handlers read, as self.server.<name>
    srv.engine = engine
    srv.stopping = stopping
    srv.open_streams = open_streams
    signals = _install_signal_handlers(srv, stopping)
    ip = _ip(host_name(host) or "")
    if host_name(host) not in LOOPBACK_NAMES and not (ip is not None and ip.is_loopback):
        logger.warning(
            f"[server] listening on {host}, beyond this machine: the API has no authentication, so "
            f"anyone who can reach port {port} can use it"
        )
    logger.info(
        f"[server] jitengine-qwen3.8-flash-next-m4pro serving {engine.model_name} on http://{host}:{port}"
        f" (graceful shutdown on {signals or 'no signals: not the main thread'})"
    )
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    if stopping.is_set():
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        while open_streams.value and time.monotonic() < deadline:
            time.sleep(0.05)
        logger.info(
            f"[server] stopped; {open_streams.value} stream(s) still open after the grace period"
        )
