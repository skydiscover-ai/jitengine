"""A stand-in engine over HTTP: its /v1/models, which reports the served model and its context."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

MODEL = "/Users/demo/models/Qwen3.6-35B-A3B-3bit"
CONTEXT = 32768


class _Engine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        status, body = 404, {"error": {"message": "not found"}}
        if self.path == "/v1/models":
            model = {"id": MODEL, "object": "model", "max_model_len": self.server.context}
            status, body = 200, {"object": "list", "data": [model]}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def engine():
    """The stand-in engine's server: `.url`, and `.model` and `.context`, what /v1/models reports."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    srv.daemon_threads = True
    srv.model, srv.context = MODEL, CONTEXT
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()
