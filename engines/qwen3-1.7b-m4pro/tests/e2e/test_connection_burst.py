"""A burst of simultaneous connections must not be reset: stock's listen backlog of 5 resets them."""

import json
import subprocess
import threading
import urllib.request

import pytest

MODEL = "mlx-community/Qwen3-1.7B-4bit"


def _listen_queue_size(port: int) -> int | None:
    out = subprocess.run(["netstat", "-Lan"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f".{port}" in line:
            return int(line.split()[0].split("/")[2])
    return None


def test_listen_backlog_is_deeper_than_five(server):
    size = _listen_queue_size(server.port)
    if size is None:
        pytest.skip("netstat -Lan does not list the socket here")
    assert size > 5


@pytest.mark.parametrize("round_", range(3))
def test_sixteen_simultaneous_clients_all_connect(server, round_):
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 4,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Say hi."}],
        }
    ).encode()
    results = [None] * 16
    gate = threading.Barrier(16)

    def go(i):
        req = urllib.request.Request(
            f"{server.url}/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        gate.wait()
        try:
            results[i] = urllib.request.urlopen(req, timeout=120).status
        except Exception as e:  # noqa: BLE001
            results[i] = repr(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200] * 16, results
