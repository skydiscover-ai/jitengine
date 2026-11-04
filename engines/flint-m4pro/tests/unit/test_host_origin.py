"""The server answers only its own host names and refuses browser pages of other origins: the API has
no authentication, so a page whose domain re-resolves to 127.0.0.1 (DNS rebinding) would otherwise be
same-origin with it and read completions and /metrics. The real handler over a socket, with a
stand-in engine that records every call; no MLX."""

from __future__ import annotations

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from jitengine_flint_m4pro import server as srvmod
from jitengine_flint_m4pro.engine.io_struct import GenResult, StepOutput


class _Engine:
    cfg = SimpleNamespace(gen_defaults={})
    model_name = "m"
    profile, kv_bits, context_limit = "main", 16, 4096

    def __init__(self):
        self.calls: list[str] = []

    def encode(self, text):
        self.calls.append("encode")
        return [5, 6, 7]

    def render_chat(self, messages, tools=None):
        self.calls.append("render_chat")
        return [5, 6, 7]

    def decode(self, ids):
        return "x" * len(list(ids))

    def stream(self, req, cancel=None):
        self.calls.append("stream")
        yield StepOutput(0, 9)
        yield GenResult([9], "length", len(req.prompt_ids))

    def health(self):
        self.calls.append("health")
        return {"status": "ready"}

    def stats(self):
        self.calls.append("stats")
        return {}

    def log(self, msg):
        pass


def _start(**kw):
    engine = _Engine()
    srv = srvmod.Server(("127.0.0.1", 0), srvmod.Handler, **kw)
    srv.engine, srv.stopping = engine, threading.Event()
    srv.inflight, srv.lock = [0], threading.Lock()
    srv.stream_cancel, srv.score_cancel = True, False
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return engine, srv


@pytest.fixture
def server():
    engine, srv = _start()
    yield engine, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _ask(port, method, path, host=None, origin=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    if host is not None:
        c.putheader("Host", host)
    if origin is not None:
        c.putheader("Origin", origin)
    data = json.dumps(body).encode() if body is not None else b""
    if body is not None:
        c.putheader("Content-Type", "application/json")
        c.putheader("Content-Length", str(len(data)))
    c.endheaders(data if body is not None else None)
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


COMPLETION = {"prompt": "hi", "max_tokens": 1}
CHAT = {"messages": [{"role": "user", "content": "hi"}]}
ROUTES = (
    ("GET", "/health", None),
    ("GET", "/v1/models", None),
    ("GET", "/metrics", None),
    ("POST", "/v1/completions", COMPLETION),
    ("POST", "/v1/chat/completions", CHAT),
    ("POST", "/v1/messages", {"max_tokens": 1, **CHAT}),
    ("POST", "/v1/messages/count_tokens", CHAT),
    ("POST", "/v1/responses", {"input": "hi"}),
)


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example",
        "attacker.example:{port}",
        "127.0.0.1.nip.io:{port}",
        "evil.localhost:{port}",
    ],
)
def test_a_foreign_host_is_refused_before_any_engine_work(server, host):
    engine, port = server
    host = host.format(port=port)
    for method, path, body in ROUTES:
        status, raw = _ask(port, method, path, host=host, body=body)
        assert status == 403, (method, path, status)
        err = json.loads(raw)
        assert err["type"] == "error" if path.startswith("/v1/messages") else "error" in err
    assert engine.calls == []


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:{port}",
        "localhost:{port}",
        "LOCALHOST",
        "[::1]:{port}",
        "localhost.:{port}",
        "10.1.2.3:{port}",
        None,
    ],
)
def test_loopback_names_ip_literals_and_hostless_clients_are_served(server, host):
    _engine, port = server
    host = host.format(port=port) if host else None
    assert _ask(port, "GET", "/v1/models", host=host)[0] == 200
    status, raw = _ask(port, "POST", "/v1/completions", host=host, body=COMPLETION)
    assert status == 200 and json.loads(raw)["choices"][0]["text"] == "x"


@pytest.mark.parametrize(
    "origin",
    [
        "http://attacker.example",
        "https://attacker.example:8443",
        "null",
        "file://",
        "chrome-extension://abc",
    ],
)
def test_a_cross_origin_browser_request_is_refused(server, origin):
    engine, port = server
    for method, path, body in ROUTES:
        status, _ = _ask(port, method, path, host=f"127.0.0.1:{port}", origin=origin, body=body)
        assert status == 403, (method, path)
    assert engine.calls == []


@pytest.mark.parametrize(
    "origin", ["http://localhost:3000", "http://127.0.0.1:{port}", "http://[::1]:5173"]
)
def test_a_loopback_origin_is_served(server, origin):
    _engine, port = server
    status, _ = _ask(
        port,
        "POST",
        "/v1/completions",
        host=f"127.0.0.1:{port}",
        origin=origin.format(port=port),
        body=COMPLETION,
    )
    assert status == 200


def test_the_bind_address_and_named_hosts_are_served():
    names = srvmod.served_names("my-mac.local", ["api.internal"])
    assert {"localhost", "127.0.0.1", "::1", "my-mac.local", "api.internal"} <= names
    assert srvmod.served_names("0.0.0.0") == srvmod.served_names("")
    assert srvmod.refusal({"Host": "api.internal:80"}, names) is None
    assert srvmod.refusal({"Host": "other.internal"}, names) is not None


def test_an_allowed_host_is_served_and_others_still_are_not():
    _engine, srv = _start(allowed_hosts=("api.internal",))
    try:
        port = srv.server_address[1]
        assert _ask(port, "GET", "/v1/models", host=f"api.internal:{port}")[0] == 200
        assert _ask(port, "GET", "/v1/models", host="other.internal")[0] == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_post_to_an_unknown_path_does_not_break_the_next_request_on_the_connection(server):
    _engine, port = server
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps({"x": "y" * 500}).encode()
    c.request("POST", "/v1/nope", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read()
    assert r.status == 404
    c.request("GET", "/v1/models")  # same connection object: reopened if the server closed it
    r = c.getresponse()
    r.read()
    assert r.status == 200
    c.close()


def test_serve_options_name_allowed_host():
    from jitengine_flint_m4pro import cli

    a = cli._parser().parse_args(
        ["serve", "--model-path", "/m", "--allowed-host", "a", "--allowed-host", "b"]
    )
    assert a.allowed_host == ["a", "b"]
    assert cli._parser().parse_args(["serve", "--model-path", "/m"]).allowed_host is None


@pytest.mark.parametrize(
    "host",
    [
        "localhost:80@evil.example",
        "evil.example\\@localhost",
        "localhost@evil.example",
        "localhost evil.example",
        "localhost:abc",
        "localhost:80:80",
        "localhost/",
        "localhost:",
        "[::1",
        "[::1]x",
        "a..b",
        "-x.example",
        "loc\talhost",
    ],
)
def test_a_malformed_host_is_refused(server, host):
    engine, port = server
    assert _ask(port, "GET", "/v1/models", host=host)[0] == 403
    assert engine.calls == []


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example\\@localhost",
        "http://localhost@evil.example",
        "http://localhost:80@evil.example",
        "http://localhost/x",
        "http://local host",
        "http://localhost:abc",
        "http://",
        "localhost",
    ],
)
def test_a_malformed_origin_is_refused(server, origin):
    engine, port = server
    host = f"127.0.0.1:{port}"
    assert _ask(port, "GET", "/v1/models", host=host, origin=origin)[0] == 403
    assert engine.calls == []


def test_the_forms_the_check_accepts_are_unchanged(server):
    _engine, port = server
    for host in ("localhost:80", "LocalHost", "[::1]:8080", "127.0.0.1", "[fe80::1%25en0]:80"):
        assert _ask(port, "GET", "/v1/models", host=host)[0] == 200, host
    for origin in ("http://localhost:3000/", "https://[::1]:5173", "http://127.0.0.1"):
        assert _ask(port, "GET", "/v1/models", host=f"127.0.0.1:{port}", origin=origin)[0] == 200


def _headers_of(port, method, path, extra, body=b""):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_host="Host" in extra)
    for k, v in extra.items():
        c.putheader(k, v)
    c.endheaders(body)
    r = c.getresponse()
    r.read()
    out = r.status, r.getheader("Connection")
    c.close()
    return out


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ({"Content-Type": "text/plain", "Content-Length": "2"}, 415),
        ({"Content-Type": "application/json", "Content-Length": "abc"}, 400),
    ],
)
def test_an_early_error_that_leaves_the_body_unread_says_connection_close(server, extra, status):
    _engine, port = server
    got = _headers_of(port, "POST", "/v1/completions", extra, b"{}")
    assert got == (status, "close")
    assert _headers_of(port, "POST", "/v1/nope", {"Content-Length": "2"}, b"{}") == (404, "close")
    assert _headers_of(port, "GET", "/v1/models", {"Host": "evil.example"})[1] == "close"
