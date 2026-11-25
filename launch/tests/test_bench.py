"""jit-launch bench against a stand-in engine that streams tokens at a known pace, and a stand-in registry."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jit_launch import bench, cli

GAP_S = 0.01  # the stand-in streams one token every 10 ms: about 100 tok/s


class _Streamer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "demo-model", "max_model_len": 32768}]})
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/report":
            self.server.reports.append(body)
            return self._json(self.server.report_status, self.server.report_answer)
        if self.path != "/v1/chat/completions":  # an engine serves chat on this path only
            return self._json(404, {"error": {"message": "not found"}})
        self.server.prompts.append(body["messages"][0]["content"])
        n = body["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(0.05)  # the wait for the first token
        for i in range(n):
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(GAP_S)
        usage = {"choices": [], "usage": {"prompt_tokens": 1040, "completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.close_connection = True


@pytest.fixture
def streamer():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Streamer)
    srv.daemon_threads = True
    srv.prompts, srv.reports = [], []
    srv.report_status, srv.report_answer = 200, {"message": "Thanks: report r1 stored."}
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def a_mac(monkeypatch, tmp_path):
    facts = {"machdep.cpu.brand_string": "Apple M4 Pro", "hw.memsize": str(48 * 2**30)}
    monkeypatch.setattr(bench, "_sysctl", facts.get)
    monkeypatch.setattr(bench.platform, "mac_ver", lambda: ("26.6.2", ("", "", ""), "arm64"))
    (tmp_path / "engine.env").write_text(
        'ENGINE_CMD=jitengine-qwen3.6-35b-a3b-m4pro\nMODEL_PATH="/m"\n'
    )
    (tmp_path / "serve.args").write_text("--max-context 32768  --build-expert-pack \n")
    monkeypatch.setenv("JITENGINE_HOME", str(tmp_path))


def test_prompt_is_the_sweeps_with_a_fresh_nonce():
    a, b = bench.prompt(11), bench.prompt(11)
    assert a != b and a.split("\n", 1)[1] == b.split("\n", 1)[1]
    assert a.count("def pick_") == 11 and "Rewrite it with type hints" in a


def test_measures_decode_and_first_token(streamer):
    out = bench.measure(streamer.url, "demo-model")  # the engine's origin, as the CLI passes it
    assert len(streamer.prompts) == 1 + bench.RUNS  # a warmup, then the timed runs
    assert out["completion_tokens"] == bench.MAX_TOKENS and out["prompt_tokens"] == 1040
    assert 40 < out["decode_tok_s"] < 1 / GAP_S * 1.2, (
        out
    )  # about 100 tok/s, slower on a busy machine
    assert 0.04 < out["first_token_s"] < 2, out
    assert len(out["runs_decode_tok_s"]) == bench.RUNS


def test_sends_nothing_without_report(streamer, a_mac, capsys):
    assert (
        cli.main(["bench", "--server", streamer.url, "--report-url", streamer.url + "/report"]) == 0
    )
    assert streamer.reports == []
    err = capsys.readouterr().err
    assert "Add --report" in err


def test_report_prints_then_sends_what_it_sends(streamer, a_mac, capsys):
    rc = cli.main(
        ["bench", "--server", streamer.url, "--report", "--report-url", streamer.url + "/report"]
    )
    assert rc == 0 and len(streamer.reports) == 1
    sent = streamer.reports[0]
    assert {k: sent[k] for k in ("chip", "mem_gb", "macos", "engine", "serve_args", "model")} == {
        "chip": "M4 Pro",
        "mem_gb": 48,
        "macos": "26.6.2",
        "engine": "qwen3.6-35b-a3b-m4pro",
        "serve_args": "--max-context 32768 --build-expert-pack",
        "model": "demo-model",
    }
    assert sent["decode_tok_s"] > 0 and sent["client"].startswith("jit-launch ")
    err = capsys.readouterr().err
    assert json.dumps(sent, indent=2) in err and "Thanks: report r1 stored." in err


def test_report_shows_the_registrys_refusal(streamer, a_mac, capsys):
    streamer.report_status, streamer.report_answer = (
        400,
        {"error": "Too many reports from your network."},
    )
    rc = cli.main(
        ["bench", "--server", streamer.url, "--report", "--report-url", streamer.url + "/report"]
    )
    assert (
        rc == 1
        and "did not go through: Too many reports from your network." in capsys.readouterr().err
    )


def test_report_needs_a_mac(streamer, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bench, "_sysctl", lambda name: None)
    monkeypatch.setenv("JITENGINE_HOME", str(tmp_path))
    rc = cli.main(
        ["bench", "--server", streamer.url, "--report", "--report-url", streamer.url + "/report"]
    )
    assert (
        rc == 1
        and streamer.reports == []
        and "needs an Apple silicon Mac" in capsys.readouterr().err
    )


def test_no_engine(capsys):
    assert cli.main(["bench", "--server", "http://127.0.0.1:9"]) == 1
    assert "no engine answers" in capsys.readouterr().err
