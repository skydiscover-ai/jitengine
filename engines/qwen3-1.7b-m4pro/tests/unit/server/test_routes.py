"""The translated endpoints against a stand-in for the chat-completions endpoint."""

import http.client
import http.server
import json
import socket
import threading
import time
import types

import pytest

from jitengine_qwen3_1_7b_m4pro.server import routes

SEEN = {"bodies": [], "write_failed": threading.Event()}


def _chunk(text=None, finish=None, usage=None):
    delta = {} if text is None else {"content": text}
    obj = {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n".encode()


class Upstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _set_completion_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")

    def _set_stream_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN["bodies"].append(body)
        if body.get("model") == "missing":
            raw = json.dumps({"error": "no such model"}).encode()
            self._set_completion_headers(404)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif body.get("stream"):
            self._set_stream_headers()
            self.end_headers()
            n = 200 if body.get("model") == "slow" else 2
            try:
                if body.get("model") == "thinker":  # stock mlx-lm's field name is `reasoning`
                    think = {"choices": [{"index": 0, "delta": {"reasoning": "pondering"}}]}
                    self.wfile.write(f"data: {json.dumps(think)}\n\n".encode())
                for i in range(n):
                    self.wfile.write(_chunk(f"w{i} "))
                    self.wfile.flush()
                    time.sleep(0.02 if n > 2 else 0)
                self.wfile.write(
                    _chunk(finish="stop", usage={"prompt_tokens": 5, "completion_tokens": n})
                )
                self.wfile.write(b"data: [DONE]\n\n")
            except BrokenPipeError, ConnectionResetError:
                SEEN["write_failed"].set()
        else:
            message = {"role": "assistant", "content": "hello"}
            if body.get("model") == "thinker":
                message["reasoning"] = "pondering"
            raw = json.dumps(
                {
                    "model": body.get("model"),
                    "choices": [
                        {
                            "message": message,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                }
            ).encode()
            self._set_completion_headers()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


class Tok:
    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=True, **kw
    ):
        return " ".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture(scope="module")
def server():
    routes.install(Upstream)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    httpd.RequestHandlerClass.response_generator = types.SimpleNamespace(
        model_provider=types.SimpleNamespace(tokenizer=Tok()),
        cli_args=types.SimpleNamespace(chat_template_args={}),
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _post(port, path, body, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else json.dumps(body)
    conn.request("POST", path, data, {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, r.read().decode()


MSG = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


def test_messages_whole_reply(server):
    status, text = _post(server, "/v1/messages", MSG)
    out = json.loads(text)
    assert status == 200 and out["type"] == "message"
    assert (
        out["content"] == [{"type": "text", "text": "hello"}] and out["stop_reason"] == "end_turn"
    )
    assert (
        SEEN["bodies"][-1]["max_tokens"] == 8
        and SEEN["bodies"][-1]["messages"][0]["content"] == "hi"
    )


def test_messages_stream_events_in_order(server):
    status, text = _post(server, "/v1/messages", {**MSG, "stream": True})
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert status == 200
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert "content_block_delta" in names and "message_delta" in names


def test_responses_whole_reply(server):
    status, text = _post(server, "/v1/responses", {"model": "m", "input": "hi"})
    out = json.loads(text)
    assert status == 200 and out["object"] == "response" and out["status"] == "completed"
    assert out["output"][0]["content"][0]["text"] == "hello"


def test_responses_stream_ends_with_completed(server):
    status, text = _post(server, "/v1/responses", {"model": "m", "input": "hi", "stream": True})
    assert status == 200 and "response.completed" in text


def test_an_upstream_error_keeps_status_and_message(server):
    status, text = _post(server, "/v1/messages", {**MSG, "model": "missing"})
    out = json.loads(text)
    assert status == 404 and out["type"] == "error" and "no such model" in out["error"]["message"]
    status, text = _post(server, "/v1/messages", {**MSG, "model": "missing", "stream": True})
    assert status == 404 and "no such model" in text


def test_bad_bodies_are_400_in_the_api_error_form(server):
    status, text = _post(server, "/v1/messages", None, raw="not json")
    assert status == 400 and json.loads(text)["type"] == "error"
    status, text = _post(server, "/v1/responses", None, raw="[1, 2]")
    assert status == 400 and "error" in json.loads(text)
    status, _ = _post(
        server, "/v1/responses", {"model": "m", "input": "x", "previous_response_id": "r1"}
    )
    assert status == 400


def test_count_tokens(server):
    status, text = _post(
        server,
        "/v1/messages/count_tokens",
        {"model": "m", "messages": [{"role": "user", "content": "a b c"}]},
    )
    assert status == 200 and json.loads(text) == {"input_tokens": 3}


def test_other_paths_go_to_the_original_handler(server):
    status, text = _post(server, "/v1/chat/completions", {"model": "m", "messages": []})
    assert status == 200 and json.loads(text)["choices"][0]["message"]["content"] == "hello"


def test_a_disconnecting_client_closes_the_inner_stream(server):
    SEEN["write_failed"].clear()
    body = json.dumps({**MSG, "model": "slow", "stream": True}).encode()
    request = (
        b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    with socket.create_connection(("127.0.0.1", server), timeout=10) as sock:
        sock.sendall(request)
        assert sock.recv(200)  # the stream has started
    assert SEEN["write_failed"].wait(10), (
        "the stand-in generation kept running after the client left"
    )


def test_reasoning_under_stock_name_reaches_a_thinking_block(server):
    _, text = _post(server, "/v1/messages", {**MSG, "model": "thinker"})
    blocks = json.loads(text)["content"]
    assert blocks[0]["type"] == "thinking" and blocks[0]["thinking"] == "pondering"
    _, text = _post(server, "/v1/responses", {"model": "thinker", "input": "x"})
    assert any(item["type"] == "reasoning" for item in json.loads(text)["output"])


def test_streamed_reasoning_under_stock_name_is_kept(server):
    _, text = _post(server, "/v1/messages", {**MSG, "model": "thinker", "stream": True})
    assert "thinking_delta" in text and "pondering" in text


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "messages": ["hi"]},
        {"model": "m", "messages": [{"role": "user", "content": 5}]},
        {"model": "m", "messages": "text"},
    ],
)
def test_a_malformed_body_is_a_400_not_a_dropped_connection(server, body):
    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        status, text = _post(server, path, body)
        assert status == 400 and "error" in json.loads(text), path


def test_an_oversized_body_on_the_original_route_is_refused(server):
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Length", str(routes.MAX_BODY_BYTES + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413
