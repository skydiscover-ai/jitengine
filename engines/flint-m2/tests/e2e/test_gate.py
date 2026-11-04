"""The Flint M2 end-to-end test: greedy-512 through the HTTP server, repeat for repeat, and decode.

It starts `python -m jitengine_flint_m2 serve --model-path M` (the default: on a 24 GB Mac the whole
model does not fit, so the experts stream from the SSD within an 18 GiB budget, on stock MLX ops,
with the GPU keep-alive), waits for /health to say ready, sends one warmup and then three greedy
chat requests of 512 tokens with end-of-sequence ignored, and hashes each reply's text as
`hashlib.sha256(text.encode()).hexdigest()[:16]`. Every repeat must give the same hash, mlx must be
0.32.2, and SIGTERM must end the server with exit 0.

On the 24 GB M2 this build gave 46cc7a346d0ee2fb on every repeat, so every repeat must give that hash unless
JITENGINE_FLINT_M2_E2E_SHA names another, and the server must report the measured configuration:
offload, the graph backend, profile main. The decode rate is reported.

On any other Mac, run it with JITENGINE_FLINT_M2_E2E_SHA=report: the hash and the configuration
the server chose are printed and recorded, not required; the repeats must still agree.

Environment: JITENGINE_FLINT_M2_E2E_CHECKPOINT (required: the flint-int4 checkpoint directory);
JITENGINE_FLINT_M2_E2E_SHA (default 46cc7a346d0ee2fb, the M2's; `report` records instead);
JITENGINE_FLINT_M2_E2E_MIN_TOKS (a decode tok/s floor; unset, the rate is only reported). Run
with `-s` to see the hash and the rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

import pytest

PREFIX = "JITENGINE_FLINT_M2_E2E_"
M2_SHA = "46cc7a346d0ee2fb"  # the greedy-512 hash on the 24 GB M2
PROMPT = (
    "Write a Python function that parses an ISO 8601 duration such as P3DT4H5M into a number of "
    "seconds, with a docstring and three doctests, then explain each branch."
)
READY_S = 900  # load and the startup checks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read())


def _chat(port: int, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _report_mode() -> bool:
    return os.environ.get(PREFIX + "SHA") == "report"


@pytest.fixture
def e2e_checkpoint(requires_metal):
    """The checkpoint, on any Mac with Metal (off the 24 GB M2, JITENGINE_FLINT_M2_E2E_SHA=report records
    what the server chose instead of requiring the M2's hash and configuration)."""
    path = os.environ.get(PREFIX + "CHECKPOINT")
    if not path:
        pytest.skip(f"set {PREFIX}CHECKPOINT to the flint-int4 checkpoint directory")
    return os.path.expanduser(path)


def _wait_ready(server: subprocess.Popen, port: int) -> dict:
    deadline = time.time() + READY_S
    health: dict = {}
    while time.time() < deadline and server.poll() is None:
        try:
            health = _get(port, "/health")
        except OSError:
            health = {}
        if health.get("status") == "ready":
            break
        time.sleep(2)
    return health


def _stop(server: subprocess.Popen) -> int | None:
    server.send_signal(signal.SIGTERM)
    try:
        return server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        return None


def _configuration(health: dict) -> str:
    """What the server chose on this Mac, in one line."""
    d, mem = health.get("device") or {}, health.get("memory") or {}
    return (
        f"{d.get('device_name')} {d.get('architecture')}, {d.get('gpu_cores')} GPU cores; memory "
        f"{mem.get('mode')}, budget {mem.get('budget_bytes')} ({mem.get('budget_source')}), experts "
        f"{mem.get('pool_slots')}/{mem.get('total_records')}; backend {health.get('decode_backend')}, "
        f"context {health.get('context_limit')} ({health.get('context_limit_source')})"
    )


def _problems(shas, replies, health, rate, code) -> list:
    problems = []
    want = os.environ.get(PREFIX + "SHA") or M2_SHA
    if len(set(shas)) != 1:
        problems.append(f"the repeats differ: text_sha {shas}")
    if not _report_mode() and set(shas) != {want}:
        problems.append(f"text_sha {shas}, want {want} on every repeat")
    if [r["usage"]["completion_tokens"] for r in replies] != [512] * 3:
        problems.append(f"usage {[r['usage'] for r in replies]}, want 512 completion tokens each")
    measured = (
        health.get("profile") == "main"
        and health.get("decode_backend") == "graph"
        and (health.get("memory") or {}).get("mode") == "offload"
    )
    if not _report_mode() and not measured:
        problems.append(f"not the measured configuration: {_configuration(health)}")
    mlx = health.get("versions", {}).get("mlx")
    if mlx != "0.32.2":
        problems.append(f"mlx {mlx}, the build is certified on 0.32.2")
    floor = os.environ.get(PREFIX + "MIN_TOKS")
    if floor and rate < float(floor):
        problems.append(f"decode {rate:.2f} tok/s is below {PREFIX}MIN_TOKS={floor}")
    if code != 0:
        problems.append(f"SIGTERM ended the server with {code}, want exit 0 within 30 s")
    return problems


def test_greedy_512_repeats_and_decode(e2e_checkpoint, record_testsuite_property):
    port = _free_port()
    cmd = [sys.executable, "-m", "jitengine_flint_m2", "serve", "--model-path"]
    server = subprocess.Popen([*cmd, e2e_checkpoint, "--port", str(port)])
    try:
        health = _wait_ready(server, port)
        assert health.get("status") == "ready", f"not ready within {READY_S} s: {health}"
        _chat(port, 512)  # warmup
        replies = [_chat(port, 512) for _ in range(3)]
        health = _get(port, "/health")
    finally:
        code = _stop(server)
    texts = [r["choices"][0]["message"]["content"] for r in replies]
    shas = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]
    rate = statistics.median(float(r["x_metrics"]["decode_tok_s"]) for r in replies)
    record_testsuite_property("flint_m2_decode_tok_per_sec", rate)
    record_testsuite_property("flint_m2_text_sha", ",".join(shas))
    record_testsuite_property("flint_m2_configuration", _configuration(health))
    print(
        f"\n[e2e] Flint M2 greedy-512: text_sha {shas}, decode {rate:.2f} tok/s (median of 3)"
        f"\n[e2e] {_configuration(health)}; mlx {health.get('versions', {}).get('mlx')}"
    )
    problems = _problems(shas, replies, health, rate, code)
    assert not problems, "\n".join(problems)
