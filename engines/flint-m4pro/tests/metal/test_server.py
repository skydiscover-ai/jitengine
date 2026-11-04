"""The request deadline and a restart on the same port, through the real Engine and HTTP server on a stub model
(no weights load), on the M4 Pro with a real checkpoint."""

import os
import sys
import time

from jitengine_flint_m4pro.engine.io_struct import GenRequest
from jitengine_flint_m4pro.model.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))


class _FakeHist:
    """The History surface the graph backend's worker, prefill and decode loop touch (no arrays)."""

    rk = None

    def __init__(self, *a):
        self.length, self.valid, self.reserved = 0, False, 0

    def reserve(self, n):
        self.reserved = max(self.reserved, n)

    def invalidate(self):
        self.valid = False

    def state_arrays(self):
        return []

    def ring_mask(self, p):
        return None

    def release(self):
        pass


def _fake_model(model):
    """Stub the model of the graph backend on the CPU, so a real Engine (worker thread, queue, cancel / deadline
    plumbing, chunked prefill, graph decode loop, echo scoring) runs without weights: the logits at position p select
    token 100 + 37 p mod 5000; a decode pass takes T["pass_s"] and a prefill chunk T["chunk_s"] of host time.
    Returns (T, restore)."""
    import mlx.core as mx

    from jitengine_flint_m4pro import kv_cache as history
    from jitengine_flint_m4pro import loader
    from jitengine_flint_m4pro.model import model as graph

    saved = (loader.load_weights, history.History, graph.decode_step, graph.prefill_chunk)
    T = {"pass_s": 0.0, "chunk_s": 0.0}
    ids = mx.arange(load_config(model).vocab, stream=mx.cpu)
    mx.eval(ids)  # concrete here: a lazy array cannot be evaluated by the worker

    def logits(p):
        return mx.where(ids == 100 + (37 * p) % 5000, 8.0, 0.0).astype(mx.bfloat16)

    def decode_step(W, hist, token, p, ring, trace=None):
        time.sleep(T["pass_s"])
        return logits(p)

    def prefill_chunk(W, hist, toks, start, rows=None):
        time.sleep(T["chunk_s"])
        return mx.stack([logits(start + r) for r in rows])

    loader.load_weights = lambda *a, **k: None
    history.History = _FakeHist
    graph.decode_step, graph.prefill_chunk = decode_step, prefill_chunk

    def restore():
        loader.load_weights, history.History, graph.decode_step, graph.prefill_chunk = saved
        mx.set_default_device(mx.gpu)

    return T, restore


def _fake_engine(model, **opts):
    import mlx.core as mx

    from jitengine_flint_m4pro.engine.engine import Engine

    eng = Engine(
        model,
        **{
            "context_limit": 4096,
            "decode_backend": "graph",
            "prefix_cache": False,
            "log": lambda m: None,
            **opts,
        },
    )
    mx.set_default_device(mx.cpu)  # every array of the stubbed model and the decode loop: CPU
    return eng


def _stream_of(n_prompt, k):
    return [100 + (37 * (n_prompt - 1 + j)) % 5000 for j in range(k)]


_SERVE_CHILD = """
import sys
tests, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, tests)
import test_server as T
T._fake_model(model)
from jitengine_flint_m4pro.server import serve
serve(T._fake_engine(model), "127.0.0.1", port)
"""


def test_request_deadline(m4pro_checkpoint):  # noqa: PLR0915
    """With request_deadline_s set, a request that runs out stops at its next released
    step (between prefill chunks, or before it starts if still queued), frees the engine and is answered cleanly:
    finish_reason "length", the tokens released so far (a prefix of the full stream), deadline_exceeded; the next
    request is served normally; echo scoring is not cut. Off (the default), a long request runs to its length.
    Through the engine API and the HTTP server (streamed and not)."""
    import http.client
    import json
    import socket
    import threading

    DL = 0.3
    prompt = list(range(1, 33))
    T, restore = _fake_model(m4pro_checkpoint)
    engines = []
    try:
        off = _fake_engine(m4pro_checkpoint)
        engines.append(off)
        r = off.generate(
            GenRequest(prompt_ids=prompt, max_tokens=300, temperature=0.0, ignore_eos=True)
        )
        assert (
            r.finish_reason == "length"
            and r.token_ids == _stream_of(32, 300)
            and "deadline_exceeded" not in r.metrics
        )
        T.update(pass_s=0.01, chunk_s=0.02)
        r = off.generate(
            GenRequest(prompt_ids=prompt, max_tokens=50, temperature=0.0, ignore_eos=True)
        )  # 0.5 s > DL
        assert (
            r.finish_reason == "length"
            and len(r.token_ids) == 50
            and "deadline_exceeded" not in r.metrics
        )
        capped = _fake_engine(m4pro_checkpoint, max_output_tokens=7)
        engines.append(capped)
        for mt, want in ((None, 7), (3, 3), (500, 7)):
            r = capped.generate(
                GenRequest(prompt_ids=prompt, max_tokens=mt, temperature=0.0, ignore_eos=True)
            )
            assert r.finish_reason == "length" and r.token_ids == _stream_of(32, want), (
                mt,
                r.token_ids,
            )
        eng = _fake_engine(m4pro_checkpoint, request_deadline_s=DL, prefill_chunk=8)
        engines.append(eng)
        # decode cut: within one released step of the deadline
        fired = []
        expire = eng._expire
        eng._expire = lambda job: (fired.append(time.perf_counter()), expire(job))
        t0 = time.perf_counter()
        r = eng.generate(
            GenRequest(
                prompt_ids=list(range(1, 9)), max_tokens=3000, temperature=0.0, ignore_eos=True
            )
        )
        took = time.perf_counter() - t0
        assert (
            r.finish_reason == "length"
            and r.error is None
            and r.metrics.get("deadline_exceeded") is True
        ), r
        assert 0 < len(r.token_ids) < 3000 and r.token_ids == _stream_of(8, len(r.token_ids))
        assert len(fired) == 1 and sum(s.t_release > fired[0] for s in r.steps) <= 1, (
            fired,
            r.steps[-3:],
        )
        assert DL <= took < DL + 0.5, took
        assert r.metrics["finish_reason"] == "length" and r.metrics["completion_tokens"] == len(
            r.token_ids
        )
        eng._expire = expire
        # the engine serves the next request normally (a short request is not cut)
        r2 = eng.generate(
            GenRequest(prompt_ids=prompt, max_tokens=5, temperature=0.0, ignore_eos=True)
        )
        assert (
            r2.finish_reason == "length"
            and r2.token_ids == _stream_of(32, 5)
            and "deadline_exceeded" not in r2.metrics
        )
        assert eng.health()["status"] == "ready" and eng.errors == 0
        # prefill cut: 64 chunks of 8 at 0.02 s do not fit -> zero tokens, clean; then a cold request is exact
        r3 = eng.generate(
            GenRequest(prompt_ids=list(range(1, 513)), max_tokens=10, temperature=0.0)
        )
        assert (
            r3.finish_reason == "length"
            and r3.token_ids == []
            and r3.metrics["deadline_exceeded"] is True
        )
        assert eng.generate(
            GenRequest(prompt_ids=prompt, max_tokens=3, temperature=0.0)
        ).token_ids == _stream_of(32, 3)
        # queued cut: an echo score (not subject to the deadline, 1.28 s of prefill) holds the engine; the generation
        # queued behind it runs out while it waits and is answered without starting
        out = {}
        ta = threading.Thread(
            target=lambda: out.__setitem__("score", eng.score(list(range(1, 513)), top_k=0))
        )
        ta.start()
        time.sleep(0.05)
        rb = eng.generate(
            GenRequest(prompt_ids=prompt, max_tokens=3000, temperature=0.0, ignore_eos=True)
        )
        ta.join()
        assert len(out["score"]) == 511, len(out["score"])
        assert (
            rb.finish_reason == "length"
            and rb.token_ids == []
            and rb.metrics["deadline_exceeded"] is True
        )
        assert eng.errors == 0 and eng.health()["status"] == "ready"
        # HTTP: streamed and non-streamed answers are clean; /health and /metrics carry no seed / request_id
        from jitengine_flint_m4pro.server import serve

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        ready = threading.Event()
        threading.Thread(target=serve, args=(eng, "127.0.0.1", port, ready), daemon=True).start()
        assert ready.wait(10)

        def post(body):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.request(
                "POST",
                "/v1/completions",
                body=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            rr = c.getresponse()
            return rr.status, rr.read().decode()

        long = {
            "prompt": list(range(1, 9)),
            "max_tokens": 3000,
            "temperature": 1.0,
            "seed": 987654321,
            "ignore_eos": True,
        }
        st, txt = post(long)
        j = json.loads(txt)
        ch = j["choices"][0]
        assert st == 200 and ch["finish_reason"] == "length" and 0 < len(ch["token_ids"]) < 3000
        assert (
            j["x_metrics"]["deadline_exceeded"] is True and j["x_metrics"]["seed"] == 987654321
        )  # its own response
        st, txt = post({**long, "stream": True, "stream_options": {"include_usage": True}})
        ev = [json.loads(x[6:]) for x in txt.split("\n\n") if x.startswith("data: {")]
        fin = [e for e in ev if e.get("choices") and e["choices"][0]["finish_reason"]]
        assert (
            st == 200
            and txt.rstrip().endswith("data: [DONE]")
            and len(fin) == 1
            and '"error"' not in txt
        )
        assert (
            fin[0]["choices"][0]["finish_reason"] == "length"
            and fin[0]["x_metrics"]["deadline_exceeded"] is True
        )
        assert 0 < fin[0]["usage"]["completion_tokens"] < 3000
        st, txt = post({"prompt": prompt, "max_tokens": 4, "temperature": 0})
        assert st == 200 and json.loads(txt)["choices"][0]["token_ids"] == _stream_of(32, 4)
        for path in ("/health", "/metrics"):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", path)
            rr = c.getresponse()
            body = rr.read().decode()
            d = json.loads(body)
            assert (
                rr.status == 200
                and d["request_deadline_s"] == DL
                and d["max_output_tokens"] is None
            )
            assert (
                '"seed"' not in body
                and '"request_id"' not in body
                and "987654321" not in body
                and "cmpl-" not in body
            )
            assert d["last_request"]["completion_tokens"] == 4 and d["requests"] >= 9
        assert len(d["recent"]) >= 3
    finally:
        for e in engines:
            e.close(timeout=10)
        restore()


def test_server_restart_same_port(m4pro_checkpoint):  # noqa: PLR0915
    """A restart on the same port right after a clean SIGTERM exit binds although the first server's
    connection ends sit in TIME_WAIT (a stream, or a request with Connection: close, is closed server-side): the
    server rebinds with SO_REUSEADDR when no live listener answers; a LIVE listener on the port still refuses the
    bind (tests/unit/test_server.py: also a wildcard one). Two real server processes (stub model), the second started
    at once after the first exits."""
    import json
    import signal
    import socket
    import subprocess

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    def start():
        return subprocess.Popen(
            [sys.executable, "-B", "-c", _SERVE_CHILD, HERE, str(port), m4pro_checkpoint],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def request(proc):
        """One completion over a raw socket with Connection: close, read to EOF before closing: the server closes
        first, so its end of the connection goes to TIME_WAIT (as after every stream, which always closes)."""
        body = json.dumps(
            {"prompt": list(range(1, 33)), "max_tokens": 4, "temperature": 0}
        ).encode()
        req = (
            f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            f"Connection: close\r\nContent-Length: {len(body)}\r\n\r\n"
        ).encode() + body
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            if proc.poll() is not None:
                raise AssertionError(
                    f"server exited rc {proc.returncode}: {proc.stdout.read().decode()[-400:]}"
                )
            try:
                c = socket.create_connection(("127.0.0.1", port), timeout=30)
            except ConnectionRefusedError:
                time.sleep(0.1)
                continue
            c.sendall(req)
            data = b""
            while True:
                b = c.recv(65536)
                if not b:
                    break
                data += b
            c.close()
            head, _, payload = data.partition(b"\r\n\r\n")
            j = json.loads(payload)
            assert head.startswith(b"HTTP/1.") and b" 200 " in head.split(b"\r\n")[0], head[:80]
            assert j["choices"][0]["token_ids"] == _stream_of(32, 4), j
            return
        raise AssertionError("server not ready within 60 s")

    def stop(proc):
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0, proc.stdout.read().decode()[-400:]

    p1 = start()
    try:
        request(p1)
        live = socket.socket()
        live.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            live.bind(("127.0.0.1", port))
            raise AssertionError("bound a port with a live listener")
        except OSError:
            pass
        finally:
            live.close()
    finally:
        if p1.poll() is None:
            stop(p1)
    t_exit = time.monotonic()
    tw = subprocess.run(["netstat", "-an", "-p", "tcp"], capture_output=True, text=True).stdout
    assert any(
        ln.split()[3:4] == [f"127.0.0.1.{port}"] and ln.split()[-1] == "TIME_WAIT"
        for ln in tw.splitlines()
        if len(ln.split()) >= 6
    ), "the server's end is not in TIME_WAIT: the restart check would be vacuous"
    p2 = start()
    try:
        assert time.monotonic() - t_exit < 2.0
        request(p2)
    finally:
        if p2.poll() is None:
            stop(p2)
