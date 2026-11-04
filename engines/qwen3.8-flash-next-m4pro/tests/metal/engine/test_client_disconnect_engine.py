"""A client that closes its connection mid-request cancels it, on the tiny model through the HTTP
server: an unstreamed completion's row is retired within a poll interval of the
close and hands back its slot and KV, its /stats record says it was cancelled (finish_reason stays
"length", which callers read), and the next request does not queue behind the abandoned
decode; an echo score stops at its next chunk. Neither runs to the end."""

import http.client
import json
import socket
import threading
import time

import pytest

M = 8000  # max_tokens of the abandoned request: many seconds of tiny-model decode
PROMPT = list(range(1000, 1030))


@pytest.fixture
def served(flashnext_engine, requires_mlx):
    from jitengine_qwen3_8_flash_next_m4pro.server import api_server as srvmod

    e = flashnext_engine(prefix_grid_tokens=64, prefix_min_divergence_tokens=16)
    srv = srvmod.Server(("127.0.0.1", 0), srvmod.Handler)
    srv.engine, srv.stopping, srv.open_streams = e, threading.Event(), srvmod._Counter()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield e, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _send(port, body: dict) -> socket.socket:
    data = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(data)}\r\n\r\n".encode()
        + data
    )
    return s


def _record(e, rid):
    recs = [r for r in e.stats().get("recent") or [] if r.get("id") == rid]
    return recs[-1] if recs else None


def test_an_abandoned_unstreamed_completion_is_cancelled_and_released(served):
    e, port = served
    s = _send(port, {"prompt": PROMPT, "max_tokens": M, "temperature": 0, "request_id": "u-1"})
    t0 = time.monotonic()
    while e.stats()["batcher"]["live_rows"] == 0 and time.monotonic() - t0 < 30:
        time.sleep(0.01)
    time.sleep(0.5)  # decoding
    s.close()
    t_close = time.monotonic()
    while e.stats()["batcher"]["live_rows"] and time.monotonic() - t_close < 10:
        time.sleep(0.01)
    released = time.monotonic() - t_close
    b = e.stats()["batcher"]
    assert b["live_rows"] == 0 and b["kv_live_bytes"] == 0 and released < 5.0, (released, b)
    rec = _record(e, "u-1")
    assert rec is not None and rec.get("cancelled") is True and rec["finish_reason"] == "length"
    assert 0 < rec["completion_tokens"] < M
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    body = {"prompt": PROMPT, "max_tokens": 8, "temperature": 0, "request_id": "q-1"}
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())["usage"]["completion_tokens"] == 8
    c.close()
    rq = _record(e, "q-1")
    assert rq is not None and rq["queue_ms"] < 5000 and not rq.get("cancelled")


@pytest.mark.parametrize(
    "extra",
    [b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n", b"\r\n"],
    ids=["pipelined", "crlf"],
)
def test_a_client_that_sends_more_bytes_then_closes_is_cancelled(served, extra):
    """Bytes sent after the request
    sit unread in front of the close, which a peek never reached; the row ran all its tokens."""
    e, port = served
    rid = "x-" + str(len(extra))
    s = _send(port, {"prompt": PROMPT, "max_tokens": 3000, "temperature": 0, "request_id": rid})
    t0 = time.monotonic()
    while e.stats()["batcher"]["live_rows"] == 0 and time.monotonic() - t0 < 30:
        time.sleep(0.01)
    time.sleep(0.5)  # decoding
    s.sendall(extra)
    time.sleep(0.2)
    s.close()
    t_close = time.monotonic()
    while e.stats()["batcher"]["live_rows"] and time.monotonic() - t_close < 10:
        time.sleep(0.01)
    released = time.monotonic() - t_close
    assert e.stats()["batcher"]["live_rows"] == 0 and released < 5.0, released
    rec = _record(e, rid)
    assert rec is not None and rec.get("cancelled") is True and rec["completion_tokens"] < 3000, rec


@pytest.mark.parametrize("stream", [False, True], ids=["unstreamed", "stream"])
def test_a_client_that_stays_is_served_whole_at_the_descriptor_limit(served, stream):  # noqa: PLR0915
    """With the process out of descriptors for 1.5 s while a client that never leaves is
    decoding, a per-probe kqueue failed and the failure
    read as a close: the row was cancelled at ~140 of 1500 tokens, the unstreamed client got no byte
    and the stream ended with no [DONE] and no error frame. The probe must keep it served."""
    import errno
    import os
    import resource

    e, port = served
    rid = "emfile-" + ("s" if stream else "u")
    body = {
        "prompt": PROMPT,
        "max_tokens": 1500,
        "temperature": 0,
        "request_id": rid,
        "stream": stream,
    }
    s = _send(port, body)
    s.settimeout(90)
    got, done = bytearray(), threading.Event()

    def complete() -> bool:
        if stream:
            return b"data: [DONE]" in got
        head, sep, rest = bytes(got).partition(b"\r\n\r\n")
        n = [
            int(ln.split(b":")[1])
            for ln in head.split(b"\r\n")
            if ln.lower().startswith(b"content-length")
        ]
        return bool(sep and n and len(rest) >= n[0])

    def reader():
        try:
            while not complete():
                x = s.recv(65536)
                if not x:
                    break
                got.extend(x)
        except OSError:
            pass
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.monotonic()
    while e.stats()["batcher"]["live_rows"] == 0 and time.monotonic() - t0 < 30:
        time.sleep(0.01)
    time.sleep(0.3)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (len(os.listdir("/dev/fd")) + 16, hard))
    held, err = [], None
    try:
        try:
            while True:
                held.append(os.open("/dev/null", os.O_RDONLY))
        except OSError as exc:
            err = exc.errno
        time.sleep(1.5)
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert err == errno.EMFILE
    assert done.wait(90)
    s.close()
    text = bytes(got)
    rec = _record(e, rid)
    assert rec is not None and not rec.get("cancelled") and rec["completion_tokens"] == 1500, rec
    assert text.startswith(b"HTTP/1.1 200"), text[:80]
    if stream:
        assert b"data: [DONE]" in text and b'"error"' not in text
    else:
        assert json.loads(text.partition(b"\r\n\r\n")[2])["usage"]["completion_tokens"] == 1500


def test_an_abandoned_echo_score_stops_at_its_next_chunk(served):
    try:
        from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import ClientDisconnected
    except ImportError:  # a tree without it: the check below then fails on what the score did
        ClientDisconnected = ()

    e, port = served
    seen = {}
    real = e._run_score

    def spy(*args, **kwargs):
        seen["started"] = time.monotonic()
        try:
            out = real(*args, **kwargs)
        except BaseException as exc:
            seen["ended"] = (time.monotonic(), exc)
            raise
        seen["ended"] = (time.monotonic(), None)
        return out

    e._run_score = spy
    prompt = [1000 + (i % 5000) for i in range(20_000)]  # 313 chunks of 64
    s = _send(port, {"prompt": prompt, "max_tokens": 0, "echo": True, "logprobs": 1})
    t0 = time.monotonic()
    while "started" not in seen and time.monotonic() - t0 < 30:
        time.sleep(0.01)
    time.sleep(0.2)
    s.close()
    t_close = time.monotonic()
    while "ended" not in seen and time.monotonic() - t_close < 60:
        time.sleep(0.01)
    t_end, exc = seen["ended"]
    assert isinstance(exc, ClientDisconnected), exc
    assert t_end - t_close < 5.0
    assert "ClientDisconnected" not in e.stats().get("error_types", {})
    assert e.health()["status"] == "ok"
