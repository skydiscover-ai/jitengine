"""A request whose client has gone does no further work: the server hands the
engine an `alive` check for unstreamed completions, echo scoring and streams alike, and the engine
asks it while the request waits or decodes. An unstreamed or scored request must not read every
event before writing a byte, or a closed socket goes unnoticed and the row decodes to its
max_tokens holding its whole KV reservation. Model-free: the real handler over a socket, and a
stand-in engine that, like the engine, stops once `alive()` says the client left."""

import contextlib
import errno
import itertools
import os
import queue
import resource
import select
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import GenResult
from jitengine_qwen3_8_flash_next_m4pro.scheduler.batch import Job
from jitengine_qwen3_8_flash_next_m4pro.server import api_server as srvmod

try:
    from jitengine_qwen3_8_flash_next_m4pro.engine.io_struct import ClientDisconnected
except ImportError:  # a tree without it
    ClientDisconnected = ConnectionAbortedError

STEP_S = 0.005  # one stand-in decode step
LONG = 4000  # max_tokens: 20 s of steps, far past any test's wait


class Tok:
    def encode(self, text):
        return [5] * max(1, len(text))

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(list(ids))


class SlowEngine:
    """Decodes one token per STEP_S until max_tokens, or until `alive()` (asked every step, as the
    engine's watch asks it) says the client has gone. Records how each request ended."""

    model_name = "m"
    vocab = 100
    max_kv_positions = 100_000
    eos_ids = (1,)

    def __init__(self):
        self.tokenizer = Tok()
        self.ended: list[tuple[str, int]] = []  # (how, tokens produced)
        self.ended_event = threading.Event()

    def _end(self, how, n):
        self.ended.append((how, n))
        self.ended_event.set()

    def events(self, req, logprobs=0, alive=None):
        n = 0
        try:
            while n < req.max_tokens:
                if alive is not None and not alive():
                    self._end("cancelled", n)
                    raise ClientDisconnected("client gone")
                time.sleep(STEP_S)
                n += 1
                yield "token", 7
            self._end("completed", n)
            yield "done", GenResult([7] * n, "length", len(req.prompt_ids))
        finally:
            if not self.ended_event.is_set():
                self._end("closed", n)  # the consumer closed the events (a stream's failed write)

    def generate(self, req, logprobs=0, alive=None):
        for kind, payload in self.events(req, logprobs=logprobs, alive=alive):
            if kind == "done":
                return payload

    def score(self, prompt_ids, continuation_ids=(), top_k=5, alive=None):
        for chunk in range(int(LONG * STEP_S / 0.01)):  # 10 ms chunks
            if alive is not None and not alive():
                self._end("score cancelled", chunk)
                raise ClientDisconnected("client gone")
            time.sleep(0.01)
        self._end("score completed", 0)
        return [{"logprob": None, "top": [[7, -0.1]]} for _ in prompt_ids]

    def health(self):
        return {"status": "ok"}


@pytest.fixture
def server():
    engine = SlowEngine()
    srv = srvmod.Server(("127.0.0.1", 0), srvmod.Handler)
    srv.engine, srv.stopping, srv.open_streams = engine, threading.Event(), srvmod._Counter()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield engine, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _post_and_close(port, body: bytes, after_s: float):
    s = socket.create_connection(("127.0.0.1", port))
    head = (
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    s.sendall(head + body)
    time.sleep(after_s)
    t_close = time.monotonic()
    s.close()
    return t_close


@pytest.mark.parametrize(
    "body",
    [
        b'{"prompt": "hi", "max_tokens": %d}' % LONG,  # unstreamed
        b'{"prompt": "hi", "max_tokens": %d, "stream": true}' % LONG,  # streamed
        b'{"prompt": "hi", "max_tokens": %d, "stop": ["zz"]}' % LONG,  # unstreamed, stop sequences
    ],
)
def test_a_closed_client_stops_its_decode(server, body):
    engine, port = server
    t_close = _post_and_close(port, body, after_s=0.3)
    assert engine.ended_event.wait(5.0), "the request ran on after its client left"
    ((how, n),) = engine.ended
    assert how in ("cancelled", "closed") and n < LONG
    assert time.monotonic() - t_close < 2.0


def test_a_closed_client_stops_its_echo_score(server):
    engine, port = server
    body = b'{"prompt": "hello", "max_tokens": 0, "echo": true, "logprobs": 1}'
    _post_and_close(port, body, after_s=0.3)
    assert engine.ended_event.wait(5.0), "the score ran on after its client left"
    assert engine.ended[0][0] == "score cancelled"


PIPELINED = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"


def _post_extra_and_close(port, body: bytes, extra: bytes, after_s: float):
    """The request, then `extra` once it is being served (a pipelined next request, which HTTP/1.1
    allows, or a stray CRLF), then the close: those bytes sit unread in front of the end of file."""
    s = socket.create_connection(("127.0.0.1", port))
    head = (
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    s.sendall(head + body)
    time.sleep(after_s)
    s.sendall(extra)
    time.sleep(0.2)
    t_close = time.monotonic()
    s.close()
    return t_close


@pytest.mark.parametrize("extra", [PIPELINED, b"\r\n"], ids=["pipelined", "crlf"])
@pytest.mark.parametrize(
    "body",
    [
        b'{"prompt": "hi", "max_tokens": %d}' % LONG,  # unstreamed
        b'{"prompt": "hello", "max_tokens": 0, "echo": true, "logprobs": 1}',  # echo scoring
    ],
    ids=["unstreamed", "echo"],
)
def test_a_client_that_sent_more_bytes_and_closed_is_still_seen_to_leave(server, body, extra):
    engine, port = server
    t_close = _post_extra_and_close(port, body, extra, after_s=0.3)
    assert engine.ended_event.wait(5.0), "the request ran on after its client left"
    assert engine.ended[0][0] in ("cancelled", "score cancelled"), engine.ended
    assert time.monotonic() - t_close < 2.0


def _response(f) -> tuple[int, bytes]:
    """One HTTP/1.1 response with a Content-Length body, read from file object `f`."""
    status = int(f.readline().split()[1])
    length = 0
    while True:
        line = f.readline().strip()
        if not line:
            break
        k, _, v = line.partition(b":")
        if k.strip().lower() == b"content-length":
            length = int(v)
    return status, f.read(length)


def test_a_pipelined_next_request_from_a_client_that_stays_is_served(server):
    """The liveness probe consumes nothing: the next request, sent while the first decodes, is
    answered after it on the same connection."""
    import json

    engine, port = server
    body = json.dumps({"prompt": "hi", "max_tokens": 40}).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(30)
    s.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    time.sleep(0.1)  # decoding
    s.sendall(PIPELINED)
    f = s.makefile("rb")
    st1, b1 = _response(f)
    st2, b2 = _response(f)
    s.close()
    assert st1 == 200 and json.loads(b1)["usage"]["completion_tokens"] == 40
    assert st2 == 200 and json.loads(b2)["status"] == "ok"
    assert engine.ended == [("completed", 40)]


def test_peer_closed_sees_the_close_behind_buffered_bytes():
    from jitengine_qwen3_8_flash_next_m4pro.server.api_server import peer_closed

    def pair():
        lst = socket.socket()
        lst.bind(("127.0.0.1", 0))
        lst.listen(1)
        c = socket.create_connection(lst.getsockname())
        s, _ = lst.accept()
        lst.close()
        return c, s

    c, s = pair()
    assert not peer_closed(s)
    c.sendall(PIPELINED)
    time.sleep(0.05)
    assert not peer_closed(s)  # bytes waiting, client still there
    c.close()
    time.sleep(0.05)
    assert peer_closed(s)  # the close behind them
    assert s.recv(64).startswith(b"GET /health")  # and nothing was consumed
    s.close()


@contextlib.contextmanager
def out_of_descriptors(hold_s: float):
    """The process at its descriptor limit for `hold_s` seconds: the soft
    RLIMIT_NOFILE lowered to what is open + 16, then /dev/null opened until EMFILE; then released."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (len(os.listdir("/dev/fd")) + 16, hard))
    held = []
    try:
        try:
            while True:
                held.append(os.open("/dev/null", os.O_RDONLY))
        except OSError as exc:
            assert exc.errno == errno.EMFILE
        time.sleep(hold_s)
        yield
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.parametrize("stream", [False, True], ids=["unstreamed", "stream"])
def test_a_client_that_stays_is_served_whole_while_the_process_is_out_of_descriptors(
    server, stream
):
    """A liveness probe that cannot run (EMFILE) is no evidence the client left.
    An earlier server opened a kqueue per probe and read the failure as a close: the request was cancelled, the
    unstreamed client got no byte and the stream stopped with no [DONE] and no error frame."""
    import json

    engine, port = server
    n = 400  # 2 s of stand-in decode
    body = json.dumps({"prompt": "hi", "max_tokens": n, "stream": stream}).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(30)
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
    s.sendall(
        b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    time.sleep(0.3)  # decoding
    with out_of_descriptors(1.0):
        pass
    assert done.wait(30)
    s.close()
    assert engine.ended == [("completed", n)], engine.ended
    text = bytes(got)
    assert text.startswith(b"HTTP/1.1 200")
    if stream:
        assert b"data: [DONE]" in text and b'"error"' not in text
    else:
        assert json.loads(text.partition(b"\r\n\r\n")[2])["usage"]["completion_tokens"] == n


def test_a_real_close_is_still_seen_when_no_kqueue_can_be_opened(server, monkeypatch):
    """The fallback probe (a peek, no descriptor) still cancels a client that closes."""

    def no_kqueue():
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(srvmod, "_new_kqueue", no_kqueue)
    engine, port = server
    t_close = _post_and_close(port, b'{"prompt": "hi", "max_tokens": %d}' % LONG, after_s=0.3)
    assert engine.ended_event.wait(5.0), "the request ran on after its client left"
    ((how, n),) = engine.ended
    assert how == "cancelled" and n < LONG and time.monotonic() - t_close < 2.0


def test_the_peer_watch_says_closed_only_for_a_close_it_saw():
    def pair():
        lst = socket.socket()
        lst.bind(("127.0.0.1", 0))
        lst.listen(1)
        c = socket.create_connection(lst.getsockname())
        s, _ = lst.accept()
        lst.close()
        return c, s

    c, s = pair()
    w = srvmod.PeerWatch(s)
    assert w.closed() is False
    with out_of_descriptors(0.0):
        assert w.closed() is False  # its kqueue was opened with the request: it still works
        late = srvmod.PeerWatch(s)  # opened at the limit: the peek
        assert late.kq is None and late.closed() is False
    c.sendall(PIPELINED)
    c.close()
    time.sleep(0.05)
    assert w.closed() is True  # the close behind the buffered request
    # without a kqueue, Linux's POLLRDHUP still sees that close; the peek (macOS at the descriptor
    # limit) cannot see past the buffered bytes, so it keeps serving and never cancels
    assert late.closed() is hasattr(select, "POLLRDHUP")
    w.close()
    late.close()
    s.close()
    gone = srvmod.PeerWatch.__new__(srvmod.PeerWatch)
    gone.sock, gone.kq = s, None
    assert gone.closed() is None  # a socket the probe cannot read: unknown, not a close


def test_a_client_that_stays_is_served_to_the_end(server):
    import http.client
    import json

    engine, port = server
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(
        "POST",
        "/v1/completions",
        json.dumps({"prompt": "hi", "max_tokens": 40}),
        {"Content-Type": "application/json"},
    )
    r = c.getresponse()
    out = json.loads(r.read())
    assert r.status == 200 and out["usage"]["completion_tokens"] == 40
    assert engine.ended == [("completed", 40)]


# ---------------------------------------------------------------- the engine's side of the contract
@pytest.mark.parametrize("flowing", [True, False])
def test_the_watch_cancels_the_job_whether_events_flow_or_not(flowing, monkeypatch):
    """`alive()` is asked at most CLIENT_POLL_S apart whether events flow (a decode) or not (a
    prefill, the queue), and a gone client cancels the job within one poll interval of leaving. On
    a fake clock, so a loaded host cannot fail it: each read of the watch's clock advances it a
    millisecond while events flow, and a poll interval while the queue is empty (the wait that timed
    out)."""
    from jitengine_qwen3_8_flash_next_m4pro.engine import engine as E

    step = 0.001 if flowing else E.CLIENT_POLL_S
    clock = SimpleNamespace(t=0.0)

    def monotonic():
        clock.t += step
        return clock.t

    monkeypatch.setattr(E, "time", SimpleNamespace(monotonic=monotonic))
    job = Job("generate")
    for _ in range(1000 if flowing else 0):  # flowing: far more events than 0.2 s of reads takes
        job.out.put(("token", 7))
    gone_at, asked = 0.2, []

    def alive():
        asked.append(clock.t)
        assert len(asked) < 100, "asked 100 times and the client never left: the clock stood still"
        return clock.t < gone_at

    watch = E._ClientWatch(job, alive)
    start = clock.t  # the watch's first reading
    with pytest.raises(ClientDisconnected):
        while True:
            watch.get()
    assert job.cancelled and asked and asked[-1] - gone_at <= E.CLIENT_POLL_S + step, asked
    gaps = [b - a for a, b in itertools.pairwise([start, *asked])]
    assert max(gaps) <= E.CLIENT_POLL_S + step + 1e-9, gaps


def test_without_alive_the_watch_is_a_plain_wait():
    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import _ClientWatch

    job = Job("generate")
    job.out.put(("done", None))
    assert _ClientWatch(job, None).get() == ("done", None) and not job.cancelled
    with pytest.raises(queue.Empty):
        job.out.get_nowait()
