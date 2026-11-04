"""HTTP surface of the memory controls, on a tiny real engine over a real socket."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _req(url, body=None, ctype="application/json", method=None):
    data = (
        None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    )
    r = urllib.request.Request(
        url, data=data, method=method or ("POST" if data is not None else "GET")
    )
    if data is not None:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}"), dict(exc.headers)


class _Server:
    def __init__(self, engine, host="127.0.0.1"):
        from jitengine_qwen3_6_35b_a3b_m4pro.server.api_server import serve

        self.port = _free_port()
        self.url = f"http://{host}:{self.port}"
        self.th = threading.Thread(target=serve, args=(engine, host, self.port), daemon=True)
        self.th.start()
        for _ in range(100):
            try:
                _req(self.url + "/health")
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.05)


def test_memory_endpoint_contract(tiny_engine, requires_mlx):
    e = tiny_engine(mode="offload", experts=64)
    try:
        srv = _Server(e)
        u = srv.url + "/v1/memory"
        for bad in (
            {},
            {"budget_bytes": "1"},
            {"budget_bytes": 1.5},
            {"budget_bytes": -1},
            {"budget_bytes": True},
            {"budget_bytes": 10, "extra": 1},
        ):
            code, body, _ = _req(u, bad)
            assert code == 400, (bad, code, body)
        code, _, _ = _req(u, b"budget_bytes=5", ctype="application/x-www-form-urlencoded")
        assert code == 415
        code, body, _ = _req(u, {"budget_bytes": e.ceiling_bytes + 1})
        assert code == 422 and body["status"] == "rejected"
        small = (
            e.capacity.fixed_mlx_bytes
            + e.capacity.host_side_bytes
            + 40 * e.capacity.expert_slot_bytes
        )
        code, body, _ = _req(u, {"budget_bytes": int(small)})
        assert code == 200 and body["status"] == "applied" and body["after"]["pool_slots"] == 40
        code, st, _ = _req(srv.url + "/stats")
        assert st["memory"]["last_resize"]["status"] == "applied" and st["expert_pool_slots"] == 40
        assert st["memory"]["plan"]["pool_slots"] == 40 and st["memory_mode"] == "offload"
        code, body, _ = _req(u, {"budget_bytes": 1024})
        assert code == 422
        code, h, _ = _req(srv.url + "/health")
        assert code == 200 and h["admission"] == "open"
    finally:
        e.close()


def test_generation_refusals_map_to_http_status(tiny_engine, requires_mlx):
    e = tiny_engine(max_context_tokens=64)
    try:
        srv = _Server(e)
        u = srv.url + "/v1/completions"
        code, body, _ = _req(u, {"prompt": list(range(1, 60)), "max_tokens": 16})
        assert code == 400 and body["error"]["code"] == "context_length_exceeded"
        assert "maximum context length is 64 tokens" in body["error"]["message"]
        e.set_admission("memory_pressure")
        code, body, hdr = _req(u, {"prompt": [1, 2, 3], "max_tokens": 4})
        assert code == 503 and body["error"]["type"] == "service_unavailable_error"
        assert body["error"]["code"] == "server_is_overloaded" and "Retry-After" in hdr
        assert "memory_pressure" in body["error"]["message"]
        code, body, hdr = _req(u, {"prompt": [1, 2, 3], "max_tokens": 4, "stream": True})
        assert code == 503 and body["error"]["type"] == "service_unavailable_error"
        e.set_admission(None)
        code, body, _ = _req(u, {"prompt": [1, 2, 3], "max_tokens": 4, "temperature": 0})
        assert code == 200 and len(body["choices"][0]["token_ids"]) == 4
    finally:
        e.close()


def test_memory_endpoint_disabled_off_loopback(tiny_engine, requires_mlx):
    from jitengine_qwen3_6_35b_a3b_m4pro.server import api_server as srvmod

    e = tiny_engine()
    try:
        port = _free_port()
        th = threading.Thread(
            target=srvmod.serve,
            args=(e, "127.0.0.1", port),
            kwargs={"memory_control": False},
            daemon=True,
        )
        th.start()
        time.sleep(0.3)
        code, _body, _ = _req(f"http://127.0.0.1:{port}/v1/memory", {"budget_bytes": 10})
        assert code == 403
    finally:
        e.close()


def test_adaptive_engine_runs_monitor_and_reports_it(tiny_engine, requires_mlx):
    e = tiny_engine(mode="offload", memory_policy="adaptive", monitor_interval_s=0.1)
    try:
        time.sleep(0.5)
        m = e.stats()["memory"]
        assert m["observed"]["samples"] >= 2 and m["observed"]["running"]
        assert m["pressure"]["state"] in ("normal", "warning", "critical")
        assert m["observed"]["process"]["footprint_bytes"] > 0
    finally:
        e.close()
        assert not e.monitor.running
