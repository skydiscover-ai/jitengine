"""The live server refuses other host names and browser pages of other origins."""

import http.client
import json

from ._http import MODEL, chat_body


def _status(server, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    r.read()
    return r.status


def test_rebinding_hosts_and_foreign_origins_get_403(shared):
    for path in ("/health", "/stats", "/v1/models"):
        assert _status(shared, "GET", path) == 200
        assert _status(shared, "GET", path, {"Host": "rebind.example"}) == 403
        assert _status(shared, "GET", path, {"Origin": "https://evil.example"}) == 403
    body = json.dumps(chat_body("hi", max_tokens=2)).encode()
    post = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    assert (
        _status(shared, "POST", "/v1/chat/completions", {**post, "Host": "rebind.example"}, body)
        == 403
    )
    assert (
        _status(shared, "POST", "/v1/messages", {**post, "Origin": "http://evil.example"}, body)
        == 403
    )
    assert _status(shared, "POST", "/v1/chat/completions", post, body) == 200


def test_a_name_given_to_allowed_host_is_served(checkpoint):
    from .conftest import Server

    srv = Server(checkpoint, "--allowed-host", "mac-mini.local")
    try:
        assert _status(srv, "GET", "/health", {"Host": "mac-mini.local:8080"}) == 200
        assert _status(srv, "GET", "/health", {"Host": "rebind.example"}) == 403
    finally:
        srv.stop()
    assert MODEL
