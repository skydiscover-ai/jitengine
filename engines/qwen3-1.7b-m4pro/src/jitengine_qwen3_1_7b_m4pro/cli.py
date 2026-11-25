"""Command line: `serve` runs the engine, `doctor` checks this Mac."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import sys

from jitengine_qwen3_1_7b_m4pro import __version__, hardware
from jitengine_qwen3_1_7b_m4pro.engine import admission
from jitengine_qwen3_1_7b_m4pro.provenance import (
    MEASURED,
    MLX_LM_COMMIT,
    MODEL_REPO,
    MODEL_REVISION,
    PINNED,
)
from jitengine_runtime import hub

DEFAULT_CHAT_TEMPLATE_ARGS = '{"enable_thinking": false}'  # as measured: Qwen3's thinking mode off
MODEL_PATH_HELP = (
    "checkpoint directory, or a Hugging Face model id (org/name or org/name@revision) fetched into the "
    "Hugging Face cache"
)


def _die(msg: str, code: int = 2) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def _checkpoint(value: str) -> str:
    """The checkpoint directory for a --model-path value: the directory itself, or the Hugging Face cache's
    snapshot of that model id (fetched now, at the revision this build pins for its own checkpoint)."""
    try:
        return hub.resolve_model_path(value, pinned=PINNED, log=hub.log_to_stderr)
    except hub.ModelPathError as exc:
        raise SystemExit(_die(str(exc))) from None


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _packages() -> dict:
    mlx_lm = _version("mlx-lm")
    return {
        "mlx": {"installed": _version("mlx"), "measured": MEASURED["mlx"]},
        "transformers": {
            "installed": _version("transformers"),
            "measured": MEASURED["transformers"],
        },
        "mlx-lm": {
            "installed": mlx_lm,
            "measured": f"commit {MLX_LM_COMMIT[:9]}",
            "matches": bool(mlx_lm and MLX_LM_COMMIT[:9] in mlx_lm),
        },
    }


def _version_warnings(packages: dict) -> list[str]:
    """What differs from the measured stack. The hash guards and the self-check keep results
    correct; this says why the engine may run at stock speed."""
    out = []
    for name in ("mlx", "transformers"):
        p = packages[name]
        if p["installed"] != p["measured"]:
            out.append(
                f"{name} {p['installed']} is installed; the engine was measured with {p['measured']}"
            )
    lm = packages["mlx-lm"]
    if not lm["matches"]:
        out.append(
            f"mlx-lm {lm['installed']} is installed; the engine needs {lm['measured']} "
            "(unmatched edits are left as stock)"
        )
    return out


def _patch_sources() -> dict | str:
    """Whether the two mlx-lm functions the engine edits are the pinned source."""
    try:
        from jitengine_qwen3_1_7b_m4pro.engine import fusion, patches, source_patch

        generate = importlib.import_module("mlx_lm.generate")
        server = importlib.import_module("mlx_lm.server")
    except ImportError as e:
        return f"unavailable: {e}"
    return {
        "GenerationBatch._step": source_patch.source_sha256(generate.GenerationBatch._step)
        == patches.STEP_SHA256,
        "ResponseGenerator._generate": source_patch.source_sha256(
            server.ResponseGenerator._generate
        )
        == patches.GENERATE_SHA256,
        "qwen3.Attention/MLP.__call__": fusion.guard() is None,
    }


def build_report(model_path: str | None = None) -> dict:
    rep = hardware.base_report()
    rep["engine"] = {"version": __version__, "model": MODEL_REPO, "model_revision": MODEL_REVISION}
    rep["packages"] = _packages()
    rep["power"] = hardware.power() if sys.platform == "darwin" else {}
    metal = None
    if sys.platform == "darwin" and rep["machine"] == "arm64":
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
                metal = {
                    "device": info.get("device_name"),
                    "memory_gib": round(info.get("memory_size", 0) / 2**30, 1),
                }
        except Exception as e:  # noqa: BLE001
            metal = {"error": str(e)}
    rep["metal"] = metal
    chip = metal.get("device") if isinstance(metal, dict) else None
    mem = metal.get("memory_gib") if isinstance(metal, dict) else None
    rep["mac"] = hardware.tested_status(chip, mem)
    rep["patch_sources"] = _patch_sources() if metal and "device" in metal else "not checked"
    if model_path:
        from jitengine_qwen3_1_7b_m4pro.engine import config

        rep["model"] = config.validate(model_path)
    rep["supported"] = bool(isinstance(metal, dict) and "device" in metal)
    return rep


def _selfcheck(rep: dict) -> None:
    if not rep["supported"]:
        return
    try:
        from jitengine_qwen3_1_7b_m4pro.kernels import selfcheck

        res = selfcheck.run()
        rep["kernel_selfcheck"] = {
            k: res[k] for k in ("exact_shapes_ok", "fusion_ok", "disabled", "seconds")
        }
    except Exception as e:  # noqa: BLE001
        rep["kernel_selfcheck"] = {"error": f"{type(e).__name__}: {e}"}


def _print_report(rep: dict) -> None:
    print(f"jitengine-qwen3-1.7b-m4pro {rep['engine']['version']}")
    print(
        f"  platform   {rep['platform']} {rep['machine']}, Python {rep['python']}, macOS {rep['macos']}"
    )
    print(f"  metal      {rep['metal']}")
    print(
        f"  this Mac   {rep['mac']} (measured on {hardware.TESTED['chip']}, {hardware.TESTED['memory_gib']} GB)"
    )
    power = rep.get("power") or {}
    print(
        f"  power      {power.get('source')} (mode {power.get('power_mode')}); battery or low-power mode lowers speed"
    )
    for name, p in rep["packages"].items():
        flag = (
            ""
            if p.get("installed") in (p.get("measured"),) or p.get("matches")
            else "  (differs from measured)"
        )
        print(f"  {name:<12} {p['installed']}  measured {p['measured']}{flag}")
    print(f"  patches    {rep['patch_sources']}")
    if "kernel_selfcheck" in rep:
        print(f"  selfcheck  {rep['kernel_selfcheck']}")
    if "model" in rep:
        m = rep["model"]
        print(f"  model      {'ok' if m['ok'] else 'NOT OK'}  {m['problems'] + m['warnings']}")
    print(f"  supported  {rep['supported']}")


def cmd_doctor(a) -> int:
    rep = build_report(_checkpoint(a.model_path) if a.model_path else None)
    _selfcheck(rep)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        _print_report(rep)
    ok = rep["supported"] and rep.get("model", {"ok": True})["ok"]
    return 0 if ok else 1


def cmd_serve(a) -> int:
    if sys.platform != "darwin":
        return _die("the engine runs on macOS with Apple silicon")
    from jitengine_qwen3_1_7b_m4pro.engine import config

    model_path = _checkpoint(a.model_path)
    check = config.validate(model_path)
    for w in check["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if not check["ok"]:
        for p in check["problems"]:
            print(f"error: {p}", file=sys.stderr)
        return 2
    from jitengine_qwen3_1_7b_m4pro.engine import patches
    from jitengine_qwen3_1_7b_m4pro.engine.state import STATE

    STATE["model_path"] = model_path
    STATE["served_names"] = {MODEL_REPO, a.served_model_name}
    STATE["served_primary"] = a.served_model_name
    STATE["host"] = a.host
    STATE["allowed_hosts"] = list(a.allowed_host)
    try:
        budget = admission.TokenBudget(a.max_context, a.max_inflight_tokens)
    except ValueError as e:
        return _die(str(e))
    for text in _version_warnings(_packages()):
        print(f"warning: {text}", file=sys.stderr)
    report = patches.install(budget=budget)
    for name, why in report.skipped.items():
        print(f"warning: {name} left as stock: {why}", file=sys.stderr)
    argv = [
        "jitengine-qwen3-1.7b-m4pro",
        "--model",
        model_path,
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--chat-template-args",
        a.chat_template_args,
        "--prompt-cache-bytes",
        str(int(a.prompt_cache_gib * 2**30)),
    ]
    if a.adapter_path:
        argv += ["--adapter-path", a.adapter_path]
        print("note: with an adapter the projections stay unfused", file=sys.stderr)
    import mlx.core as mx

    mx.set_cache_limit(int(a.mlx_cache_gib * 2**30))
    sys.argv = argv
    importlib.import_module("mlx_lm.server").main()
    return 0


def _fetch(url: str, timeout: float) -> dict:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "body": exc.read().decode(errors="replace")[:500]}
    except Exception as exc:  # noqa: BLE001 - a status command reports what it could not reach
        return {"error": repr(exc)}


def cmd_status(a) -> int:
    base = a.url.rstrip("/")
    out = {
        "url": a.url,
        "health": _fetch(base + "/health", a.timeout),
        "stats": _fetch(base + "/stats", a.timeout),
    }
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        h, st = out["health"], out["stats"]
        adm = st.get("admission") or {}
        cache = st.get("prompt_cache") or {}
        check = st.get("selfcheck") or {}
        print(f"{a.url}: {h.get('status', h)}")
        print(
            f"  padded {adm.get('padded_tokens')} of {adm.get('max_inflight_tokens')} tokens, "
            f"{adm.get('waiting')} waiting, peak {adm.get('peak_padded_tokens')}"
        )
        print(
            f"  prompt cache {cache.get('sequences')} sequences, {(cache.get('bytes') or 0) / 2**30:.2f} GiB"
        )
        print(
            f"  kernel check ok {check.get('exact_shapes_ok')}, fusion ok {check.get('fusion_ok')}, disabled {check.get('disabled')}"
        )
        print(f"  power {(st.get('power') or {}).get('source')}")
    return 0 if out["health"].get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jitengine-qwen3-1.7b-m4pro",
        description="Qwen3-1.7B 4-bit on a 24 GB Apple M4 Pro, tuned for 16 concurrent requests.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("doctor", help="check this Mac, the installed packages and the kernels")
    d.add_argument("--model-path", help=f"also check this checkpoint: a {MODEL_PATH_HELP}")
    d.add_argument("--json", action="store_true")
    s = sub.add_parser("serve", help="serve the model over an OpenAI-compatible API")
    s.add_argument(
        "--model-path",
        default=MODEL_REPO,
        help=f"{MODEL_PATH_HELP} (default: {MODEL_REPO} at {MODEL_REVISION[:9]})",
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--chat-template-args", default=DEFAULT_CHAT_TEMPLATE_ARGS)
    s.add_argument(
        "--max-context",
        type=int,
        default=admission.DEFAULT_MAX_CONTEXT,
        help="tokens one request may hold, prompt and output",
    )
    s.add_argument(
        "--max-inflight-tokens",
        type=int,
        default=admission.DEFAULT_MAX_INFLIGHT,
        help="padded KV tokens of the running requests (their number x the longest); others wait in order",
    )
    s.add_argument(
        "--prompt-cache-gib",
        type=float,
        default=admission.DEFAULT_PROMPT_CACHE_GIB,
        help="most memory the KV caches of finished (cached) requests may hold; running requests are limited by --max-inflight-tokens",
    )
    s.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help="a host name clients may use besides localhost, IP literals and --host (repeatable); "
        "a request naming any other host, or from a browser page of another origin, gets 403",
    )
    s.add_argument(
        "--mlx-cache-gib",
        type=float,
        default=admission.DEFAULT_MLX_CACHE_GIB,
        help="most memory MLX keeps for reuse after arrays are freed",
    )
    s.add_argument("--served-model-name", default=MODEL_REPO, help="the model name clients send")
    s.add_argument("--adapter-path", default=None)
    st = sub.add_parser("status", help="health and limits of a running server (read-only)")
    st.add_argument("--url", default="http://127.0.0.1:8080")
    st.add_argument("--json", action="store_true")
    st.add_argument("--timeout", type=float, default=10.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0
    return {"doctor": cmd_doctor, "serve": cmd_serve, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
