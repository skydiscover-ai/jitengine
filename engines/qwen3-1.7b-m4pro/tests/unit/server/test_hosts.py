import http.client
import http.server
import json
import threading

import pytest

from jitengine_qwen3_1_7b_m4pro.server import hosts

NAMES = hosts.served_names("127.0.0.1", ["mac-mini.local"])


@pytest.mark.parametrize(
    "authority,expected",
    [
        ("localhost", "localhost"),
        ("LocalHost:8080", "localhost"),
        ("[::1]:8080", "::1"),
        ("example.com.", "example.com"),
        ("a@evil.com", None),
        ("evil.com\\x", None),
        ("evil.com:notaport", None),
        ("", None),
    ],
)
def test_host_name(authority, expected):
    assert hosts.host_name(authority) == expected


def test_served_names_include_the_bind_address_unless_it_is_a_wildcard():
    assert "192.168.1.5" in hosts.served_names("192.168.1.5")
    assert hosts.served_names("0.0.0.0") == frozenset(hosts.LOOPBACK_NAMES)
    assert "mac-mini.local" in NAMES


@pytest.mark.parametrize(
    "headers,refused",
    [
        ({"Host": "localhost:8080"}, False),
        ({"Host": "127.0.0.1:8080"}, False),
        ({"Host": "mac-mini.local:8080"}, False),
        ({"Host": "192.168.9.9:8080"}, False),  # an IP literal cannot be re-pointed by DNS
        ({"Host": "evil.example:8080"}, True),  # a rebinding name
        ({"Host": "localhost", "Origin": "http://localhost:3000"}, False),
        ({"Host": "localhost", "Origin": "https://evil.example"}, True),
        ({"Host": "localhost", "Origin": "null"}, True),
        ({"Host": "mac-mini.local", "Origin": "http://mac-mini.local"}, False),
        ({}, False),  # not a browser: the OpenAI SDKs and curl send no Origin
    ],
)
def test_refusal(headers, refused):
    assert (hosts.refusal(headers, NAMES) is not None) is refused


def test_install_refuses_before_the_handler_runs():
    ran = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _set_completion_headers(self, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")

        def do_GET(self):
            ran.append("GET")
            self._set_completion_headers()
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_OPTIONS = do_GET

    hosts.install(H, NAMES)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        for method in ("GET", "POST", "OPTIONS"):
            conn = http.client.HTTPConnection("127.0.0.1", port)
            conn.request(method, "/x", headers={"Host": "evil.example"})
            r = conn.getresponse()
            assert r.status == 403 and json.loads(r.read())["error"]["code"] == "forbidden"
        assert ran == []
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/x")
        assert conn.getresponse().status == 200 and ran == ["GET"]
    finally:
        srv.shutdown()
