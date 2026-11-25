"""Small HTTP helpers for the end-to-end tests."""

import json
import socket
import time
import urllib.error
import urllib.request

MODEL = "mlx-community/Qwen3-1.7B-4bit"


def post(server, path, body, timeout=300):
    """(status, parsed JSON body) of a POST; an error status is returned, not raised."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(f"{server.url}{path}", data, {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def get(server, path, timeout=10):
    return json.load(urllib.request.urlopen(f"{server.url}{path}", timeout=timeout))


def chat_body(text, **extra):
    return {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def stream_chunks(server, body):
    """Every JSON chunk of a streamed chat completion, and whether [DONE] ended the stream."""
    req = urllib.request.Request(
        f"{server.url}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    chunks, done = [], False
    for line in urllib.request.urlopen(req, timeout=300):
        line = line.decode().strip()
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: "):
            chunks.append(json.loads(line[6:]))
    return chunks, done


def short_latency(server):
    t = time.time()
    status, _ = post(server, "/v1/chat/completions", chat_body("Say hi.", max_tokens=8))
    assert status == 200
    return time.time() - t


def start_stream_then_leave(server, path, body, read_bytes=3000, timeout=30):
    """Open a streamed request, read some of it, then close the socket without finishing."""
    raw = json.dumps(body).encode()
    request = (
        f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode()
        + raw
    )
    got = 0
    with socket.create_connection(("127.0.0.1", server.port), timeout=timeout) as sock:
        sock.sendall(request)
        deadline = time.time() + timeout
        while got < read_bytes and time.time() < deadline:
            data = sock.recv(4096)
            if not data:
                break
            got += len(data)
    return got
