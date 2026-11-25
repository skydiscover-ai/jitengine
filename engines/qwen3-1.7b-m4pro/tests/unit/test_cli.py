import json

from jitengine_qwen3_1_7b_m4pro import __version__, cli
from jitengine_qwen3_1_7b_m4pro.cli import build_parser, build_report, main
from jitengine_qwen3_1_7b_m4pro.provenance import MLX_LM_COMMIT, MODEL_REPO, MODEL_REVISION


def test_version_flag(capsys):
    try:
        build_parser().parse_args(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "serve" in capsys.readouterr().out


def test_pins_are_full_hashes():
    assert len(MODEL_REVISION) == 40 and len(MLX_LM_COMMIT) == 40


def test_serve_defaults():
    a = build_parser().parse_args(["serve", "--model-path", "/m"])
    assert (a.host, a.port, a.adapter_path) == ("127.0.0.1", 8080, None)
    assert json.loads(a.chat_template_args) == {"enable_thinking": False}


def test_serve_with_no_model_path_serves_the_pinned_checkpoint():
    assert build_parser().parse_args(["serve"]).model_path == MODEL_REPO


def test_a_model_path_resolves_through_the_hub_at_the_pinned_revision(monkeypatch, tmp_path):
    seen = []

    def resolve(value, *, pinned, log):
        seen.append((value, pinned))
        return str(tmp_path) if value == MODEL_REPO else value

    monkeypatch.setattr(cli.hub, "resolve_model_path", resolve)
    assert cli._checkpoint(MODEL_REPO) == str(tmp_path)
    assert seen == [(MODEL_REPO, {MODEL_REPO: MODEL_REVISION})]


def test_a_model_path_that_is_neither_a_directory_nor_an_id_exits_2(capsys):
    try:
        cli._checkpoint("not a model")
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("no exit")
    assert "not a directory and not a Hugging Face model id" in capsys.readouterr().err


def test_serve_refuses_a_bad_model_directory(tmp_path, capsys):
    assert main(["serve", "--model-path", str(tmp_path)]) == 2


def test_doctor_report_is_json_serializable():
    rep = build_report()
    json.dumps(rep, default=str)
    assert rep["engine"]["model_revision"] == MODEL_REVISION and "supported" in rep


def test_serve_bounds_the_prompt_cache_in_bytes(monkeypatch, tmp_path):
    import json
    import sys

    from jitengine_qwen3_1_7b_m4pro.engine import config

    seen = {}
    monkeypatch.setattr(config, "validate", lambda p: {"ok": True, "problems": [], "warnings": []})
    monkeypatch.setattr(sys, "platform", "darwin")
    fake_patches = type(
        "P", (), {"install": staticmethod(lambda budget=None: type("R", (), {"skipped": {}})())}
    )
    monkeypatch.setitem(sys.modules, "jitengine_qwen3_1_7b_m4pro.engine.patches", fake_patches)
    import jitengine_qwen3_1_7b_m4pro.engine as engine_pkg

    monkeypatch.setattr(engine_pkg, "patches", fake_patches, raising=False)
    server = type("S", (), {"main": staticmethod(lambda: seen.setdefault("argv", list(sys.argv)))})
    monkeypatch.setitem(sys.modules, "mlx_lm.server", server)
    import importlib

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: server if name == "mlx_lm.server" else sys.modules[name],
    )
    assert main(["serve", "--model-path", str(tmp_path), "--prompt-cache-gib", "2"]) == 0
    argv = seen["argv"]
    assert argv[argv.index("--prompt-cache-bytes") + 1] == str(2 * 2**30)
    assert json.loads(argv[argv.index("--chat-template-args") + 1]) == {"enable_thinking": False}


def test_status_reports_an_unreachable_server_without_raising(capsys):
    assert main(["status", "--url", "http://127.0.0.1:9", "--timeout", "1"]) == 1
    assert "127.0.0.1:9" in capsys.readouterr().out


def test_status_reads_health_and_stats_from_a_running_server(capsys):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = (
                {"status": "ok"}
                if self.path == "/health"
                else {
                    "admission": {
                        "padded_tokens": 5,
                        "max_inflight_tokens": 49152,
                        "waiting": 0,
                        "peak_padded_tokens": 9,
                    },
                    "prompt_cache": {"sequences": 2, "bytes": 2**30},
                    "selfcheck": {"exact_shapes_ok": True, "fusion_ok": True, "disabled": []},
                    "power": {"source": "AC Power"},
                }
            )
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert main(["status", "--url", f"http://127.0.0.1:{srv.server_address[1]}"]) == 0
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "ok" in out and "5 of 49152" in out and "1.00 GiB" in out


def test_serve_names_every_package_that_differs_from_the_measured_stack():
    packages = {
        "mlx": {"installed": "0.32.2", "measured": "0.32.4"},
        "transformers": {"installed": "5.20.2", "measured": "5.20.2"},
        "mlx-lm": {"installed": "0.31.0", "measured": "commit c39139d70", "matches": False},
    }
    text = " ".join(cli._version_warnings(packages))
    assert "mlx 0.32.2" in text and "mlx-lm 0.31.0" in text and "transformers" not in text
    packages["mlx"]["installed"] = "0.32.4"
    packages["mlx-lm"]["matches"] = True
    assert cli._version_warnings(packages) == []
