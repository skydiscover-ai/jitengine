"""`jitengine-qwen3.6-35b-a3b-m2` command line: doctor, inspect, pack, serve and status.

MLX is imported only inside the commands that need the GPU, so `--help`, `--version`,
`--print-config` and config validation work on any platform.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from jitengine_qwen3_6_35b_a3b_m2 import __version__, logs

logger = logging.getLogger(__name__)

GIB = 1 << 30
MIB = 1 << 20


def _die(msg: str, code: int = 2) -> int:
    print(f"jitengine-qwen3.6-35b-a3b-m2: error: {msg}", file=sys.stderr)
    return code


def _gib(n) -> str:
    return "n/a" if n is None else f"{n / GIB:.2f} GiB"


def _bytes_flag(unit: int):
    def conv(text: str) -> int:
        from decimal import Decimal, InvalidOperation

        try:
            d = Decimal(text)
        except InvalidOperation:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not d.is_finite() or d < 0:
            raise argparse.ArgumentTypeError(f"must be a finite non-negative number: {text!r}")
        return int(d * unit)

    return conv


# ---------------------------------------------------------------- doctor
def _tuned_on(info: dict) -> dict:
    """The Mac this build was tuned on, and a note in words on what that means for this one."""
    from jitengine_qwen3_6_35b_a3b_m2.kernels.selfcheck import TUNED_GPU, TUNED_MAC, tuned_note

    note = tuned_note(str(info.get("device_name", "")), str(info.get("architecture")))
    same = f"This build was tuned on {TUNED_MAC} ({TUNED_GPU}), this Mac's GPU family."
    return {"mac": TUNED_MAC, "gpu": TUNED_GPU, "note": note or same}


def cmd_doctor(a) -> int:
    rep = _base_report()
    rep["physical_bytes"] = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    from jitengine_qwen3_6_35b_a3b_m2.memory.budget import device_info_or_none

    info, err = device_info_or_none()
    rep["metal"] = info if info is not None else {"error": err}
    if info:
        rep["tuned_on"] = _tuned_on(info)
    _note_packages(rep)
    if info is not None:
        rep["three_bit_ops"] = _probe_three_bit()
    from jitengine_qwen3_6_35b_a3b_m2.memory.monitor import capability_report

    rep["memory_telemetry"] = capability_report()
    _note_wired_limit(rep)
    rep["supported"] = bool(
        sys.platform == "darwin" and rep["machine"] == "arm64" and info is not None
    )
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        _print_report(rep, info, err)
    return 0 if rep["supported"] else 1


def _base_report() -> dict:
    """The doctor report's start: versions, platform and the chip name."""
    import platform

    rep: dict = {
        "jitengine-qwen3.6-35b-a3b-m2": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import subprocess

        rep["chip"] = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception as exc:  # noqa: BLE001
        rep["chip"] = None
        rep["chip_error"] = repr(exc)
    return rep


def _note_packages(rep: dict) -> None:
    """Add the installed versions of the engine's packages to `rep`."""
    try:
        import importlib.metadata as md

        rep["packages"] = {
            p: _version(md, p) for p in ("mlx", "mlx-metal", "mlx-lm", "numpy", "transformers")
        }
    except Exception as exc:  # noqa: BLE001
        rep["packages"] = {"error": repr(exc)}


def _note_wired_limit(rep: dict) -> None:
    """Add the GPU wired-memory limit (MB) to `rep`, or None when sysctl cannot say."""
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=5
        )
        rep["iogpu_wired_limit_mb"] = (
            int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
        )
    except Exception:  # noqa: BLE001
        rep["iogpu_wired_limit_mb"] = None


def _print_report(rep: dict, info: dict | None, err: str | None) -> None:
    """Print the doctor report as text."""
    print(f"jitengine-qwen3.6-35b-a3b-m2 {__version__} on {rep['platform']} ({rep.get('chip')})")
    print(f"  physical memory {_gib(rep['physical_bytes'])}")
    if info:
        print(
            f"  Metal: {info.get('device_name')} {info.get('architecture')}, recommended working set "
            f"{_gib(info.get('max_recommended_working_set_size'))}, max buffer {_gib(info.get('max_buffer_length'))}"
        )
    else:
        print(f"  Metal: unavailable ({err})")
    if "tuned_on" in rep:
        print(f"  {rep['tuned_on']['note']}")
    print(f"  packages: {rep['packages']}")
    if "three_bit_ops" in rep:
        print(f"  3-bit ops: {rep['three_bit_ops']}")
    mt = rep["memory_telemetry"]
    print(
        f"  footprint telemetry: {'ok' if mt['process_footprint']['ok'] else mt['process_footprint']['error']}; "
        f"system telemetry: {'ok' if mt['system_memory']['ok'] else mt['system_memory']['error']}"
    )
    print(f"  supported for serving: {rep['supported']}")


def _version(md, p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def _probe_three_bit() -> dict:
    try:
        import mlx.core as mx

        w = mx.random.normal((64, 256), key=mx.random.key(1)).astype(mx.bfloat16)
        x = mx.random.normal((2, 256), key=mx.random.key(2)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=3)
        y = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=3)
        g = mx.gather_qmm(
            x[:, None, None, :],
            q[None],
            s[None],
            b[None],
            rhs_indices=mx.zeros((2, 1), dtype=mx.uint32),
            transpose=True,
            group_size=64,
            bits=3,
        )
        d = mx.dequantize(q, s, b, group_size=64, bits=3)
        mx.eval(y, g, d)
        return {
            "quantized_matmul": True,
            "gather_qmm": True,
            "dequantize": True,
            "gather_equals_matmul": bool(mx.array_equal(g[:, 0, 0], y).item()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


# ---------------------------------------------------------------- config plumbing
def _flags_from(a) -> dict:
    if (
        getattr(a, "memory_budget_gib", None) is not None
        and getattr(a, "memory_budget_gb", None) is not None
    ):
        raise SystemExit(_die("pass --memory-budget-gib or --memory-budget-gb, not both"))
    budget = a.memory_budget_gib if a.memory_budget_gib is not None else a.memory_budget_gb
    return {
        "model_path": getattr(a, "model_path", None),
        "memory_budget_bytes": budget,
        "system_reserve_bytes": a.reserve_system_gib,
        "mode": a.mode,
        "memory_policy": a.memory_policy,
        "max_slots": a.max_slots,
        "max_queued_requests": a.max_queued,
        "max_context_tokens": a.max_context,
        "prefill_chunk_tokens": a.prefill_chunk,
        "prefix_cache_bytes": a.prefix_cache_mib,
        "expert_pack_path": a.expert_pack_path,
        "build_expert_pack": True if a.build_expert_pack else None,
        # Only an explicitly passed host/port overrides a profile or config file (see _Explicit).
        "host": a.host if "host" in getattr(a, "_explicit", set()) else None,
        "port": a.port if "port" in getattr(a, "_explicit", set()) else None,
    }


class _Explicit(argparse.Action):
    """Store the value and remember that the flag was given (so a default never overrides a file)."""

    def __call__(self, parser, namespace, values, option_string=None):
        if self.type is not None and isinstance(values, str):
            values = self.type(values)
        setattr(namespace, self.dest, values)
        explicit = set(getattr(namespace, "_explicit", set()))
        explicit.add(self.dest)
        namespace._explicit = explicit


def _add_config_flags(p) -> None:
    g = p.add_argument_group("configuration (precedence: defaults < --profile < --config < flags)")
    g.add_argument("--profile", help="packaged profile (24gb-static, 24gb-adaptive)")
    g.add_argument("--config", help="JSON configuration file (schema_version 1)")
    g.add_argument(
        "--print-config", action="store_true", help="print the resolved configuration and exit"
    )
    g.add_argument(
        "--memory-budget-gib",
        type=_bytes_flag(GIB),
        default=None,
        help="process footprint ceiling in GiB (binary; default 18)",
    )
    g.add_argument(
        "--memory-budget-gb",
        type=_bytes_flag(10**9),
        default=None,
        help="the same ceiling in decimal GB (10^9 bytes); not with --memory-budget-gib",
    )
    g.add_argument(
        "--reserve-system-gib",
        type=_bytes_flag(GIB),
        default=None,
        help="physical memory left to the OS and other apps, GiB (default 6)",
    )
    g.add_argument("--mode", choices=("auto", "resident", "offload"), default=None)
    g.add_argument("--memory-policy", choices=("static", "adaptive"), default=None)
    g.add_argument("--max-slots", type=int, default=None, help="concurrent generations (default 8)")
    g.add_argument(
        "--max-queued", type=int, default=None, help="waiting requests before 429 (default 16)"
    )
    g.add_argument(
        "--max-context", type=int, default=None, help="prompt + output tokens (default 8192)"
    )
    g.add_argument(
        "--prefill-chunk", type=int, default=None, help="prefill chunk tokens (default 256)"
    )
    g.add_argument(
        "--prefix-cache-mib",
        type=_bytes_flag(MIB),
        default=None,
        help="prefix cache MiB (default 128)",
    )
    g.add_argument("--expert-pack-path", default=None, help="expert pack file (offload mode)")
    g.add_argument(
        "--build-expert-pack",
        action="store_true",
        help="build a missing/stale expert pack at startup instead of refusing",
    )


def _resolve(a) -> tuple[dict, dict]:
    from jitengine_qwen3_6_35b_a3b_m2.engine.args import ConfigError, resolve

    try:
        return resolve(a.profile, a.config, _flags_from(a))
    except ConfigError as exc:
        raise SystemExit(_die(str(exc))) from None


def _print_config(cfg: dict, origin: dict) -> None:
    out = {"schema_version": 1, "resolved": cfg, "origins": origin}
    print(json.dumps(out, indent=1, sort_keys=True))


# ---------------------------------------------------------------- inspect
def cmd_inspect(a) -> int:
    from jitengine_qwen3_6_35b_a3b_m2.checkpoint.inventory import CheckpointError
    from jitengine_qwen3_6_35b_a3b_m2.engine.args import resolve_engine_options
    from jitengine_qwen3_6_35b_a3b_m2.memory.budget import (
        BudgetError,
        capacity_for,
        device_info_or_none,
        limits_for,
        make_plan,
    )

    cfg, _origin = _resolve(a)
    if not cfg.get("model_path"):
        return _die("--model-path is required")
    try:
        opt = resolve_engine_options(cfg)
        cap, _mcfg, inv, scan, _kv = capacity_for(cfg["model_path"], opt)
    except (CheckpointError, OSError, ValueError) as exc:
        return _die(f"unsupported or unreadable checkpoint: {exc}", 1)
    info, err = device_info_or_none() if not a.no_metal else (None, "skipped (--no-metal)")
    rep = {
        "model": inv.summary(),
        "expert_record_bytes": scan.record_bytes,
        "expert_record_stride": scan.record_stride,
        "num_expert_records": scan.num_records,
        "metal": info or {"error": err},
    }
    rep["model"].pop("overrides", None)
    try:
        plan = make_plan(cap, limits_for(opt, info), opt["mode"])
        rep["plan"] = plan.to_dict()
        rep["plan_text"] = plan.describe()
    except BudgetError as exc:
        rep["plan_error"] = str(exc)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        m = rep["model"]
        q = m["quantization"]
        print(
            f"{m['model_path']}: {m['model_type']}, affine {q['bits']}-bit g{q['group_size']} "
            f"(overrides at {m['override_bits']} bits)"
        )
        print(
            f"  dense {_gib(m['dense_bytes'])}, experts {_gib(m['expert_bytes'])} "
            f"({scan.num_records} records x {scan.record_bytes} B), text {_gib(m['text_bytes'])}, "
            f"not loaded {_gib(m['dropped_bytes'])} {m['family_bytes']}"
        )
        print(f"  plan: {rep.get('plan_text') or rep.get('plan_error')}")
        if info is None:
            print(
                f"  (Metal limits unknown: {err}; the plan used physical memory as the Metal cap)"
            )
    return 0 if "plan" in rep else 1


# ---------------------------------------------------------------- pack
def cmd_pack(a) -> int:
    from jitengine_qwen3_6_35b_a3b_m2.checkpoint.pack import (
        CheckpointScan,
        PackError,
        build_pack,
        default_pack_path,
        validate,
    )

    try:
        scan = CheckpointScan(a.model_path)
    except Exception as exc:  # noqa: BLE001
        return _die(f"cannot scan {a.model_path}: {exc}", 1)
    out = os.path.abspath(os.path.expanduser(a.output)) if a.output else default_pack_path(scan)
    if os.path.exists(out):
        ok, why = validate(out, scan)
        if ok and not a.force:
            print(f"{out}: valid for this checkpoint; nothing to do (--force rebuilds)")
            return 0
        print(f"{out}: {'rebuilding' if ok else 'rejecting: ' + why}")
    need = scan.num_records * scan.record_stride + (1 << 20)
    free = _free_bytes(os.path.dirname(out) or ".")
    slack = 2 * GIB
    if free is not None and free < need + slack:
        return _die(f"{out}: needs {_gib(need)} + {_gib(slack)} slack, {_gib(free)} free", 1)
    try:
        build_pack(scan, out, log=lambda m: print(m, flush=True))
    except PackError as exc:
        return _die(str(exc), 1)
    ok, why = validate(out, scan)
    print(f"{out}: {'valid' if ok else 'INVALID: ' + why}")
    return 0 if ok else 1


def _free_bytes(path: str) -> int | None:
    try:
        os.makedirs(path, exist_ok=True)
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


# ---------------------------------------------------------------- serve
def _preflight(planned_process_bytes: int, max_wait_s: float) -> tuple[bool, str]:
    """Wait up to `max_wait_s` for the available-memory estimate to cover the planned footprint.

    Passes if telemetry is unavailable; never changes a sysctl, purges caches or signals processes.
    """
    from jitengine_qwen3_6_35b_a3b_m2.memory.monitor import system_memory

    t0 = time.monotonic()
    last = ""
    while True:
        try:
            s = system_memory()
        except Exception as exc:  # noqa: BLE001
            return True, f"preflight skipped: system telemetry unavailable ({exc})"
        avail = s.get("available_bytes", s["available_bytes_estimate"])
        short = planned_process_bytes - avail
        last = (
            f"available (estimate) {_gib(avail)}, planned process {_gib(planned_process_bytes)}, "
            + (f"short by {_gib(short)}, " if short > 0 else "")
            + f"swap used {_gib(s['swap_used_bytes'])}"
        )
        if avail >= planned_process_bytes:
            return True, last
        if time.monotonic() - t0 >= max_wait_s:
            return False, last
        time.sleep(2.0)


def cmd_serve(a) -> int:
    logs.configure(a.log_level)
    cfg, origin = _resolve(a)
    if a.print_config:
        _print_config(cfg, origin)
        return 0
    if not cfg.get("model_path"):
        return _die("--model-path is required (downloads are never automatic)")
    if not os.path.isfile(os.path.join(cfg["model_path"], "config.json")):
        return _die(f"{cfg['model_path']}: not a checkpoint directory (no config.json)")
    from jitengine_qwen3_6_35b_a3b_m2.checkpoint.inventory import CheckpointError
    from jitengine_qwen3_6_35b_a3b_m2.engine.args import resolve_engine_options
    from jitengine_qwen3_6_35b_a3b_m2.memory.budget import (
        BudgetError,
        capacity_for,
        device_info_or_none,
        limits_for,
        make_plan,
    )

    opt = resolve_engine_options(cfg)
    try:
        cap, *_ = capacity_for(cfg["model_path"], opt)
        info, err = device_info_or_none()
        if info is None:
            return _die(
                f"Metal is unavailable on this machine ({err}); serving needs Apple Silicon", 1
            )
        plan = make_plan(cap, limits_for(opt, info), opt["mode"])
    except (CheckpointError, BudgetError) as exc:
        return _die(str(exc), 1)
    if plan.mode == "offload" and not opt.get("build_expert_pack"):
        from jitengine_qwen3_6_35b_a3b_m2.checkpoint.pack import (
            CheckpointScan,
            default_pack_path,
            validate,
        )

        scan = CheckpointScan(cfg["model_path"])
        pack = opt.get("expert_pack_path") or default_pack_path(scan)
        ok, why = validate(pack, scan) if os.path.exists(pack) else (False, "does not exist")
        if not ok:
            return _die(
                f"offload mode needs an expert pack; {pack}: {why}. Prepare it with\n"
                f"  jitengine-qwen3.6-35b-a3b-m2 pack --model-path {cfg['model_path']} --output {pack}\n"
                f"or pass --build-expert-pack.",
                1,
            )
        opt["expert_pack_path"] = pack
    ok, why = _preflight(plan.planned_process_bytes, a.preflight_wait)
    logger.info(f"[jitengine-qwen3.6-35b-a3b-m2] preflight: {why}")
    if not ok:
        return _die(
            f"not enough available memory for the planned footprint after "
            f"{a.preflight_wait:.0f}s ({why}); close applications or lower --memory-budget-gib",
            1,
        )
    from jitengine_qwen3_6_35b_a3b_m2.engine.engine import create_engine

    engine = create_engine(cfg["model_path"], **opt)
    try:
        engine.serve(cfg["host"], int(cfg["port"]), allowed_hosts=tuple(a.allowed_host or ()))
    finally:
        engine.close()
    return 0


# ---------------------------------------------------------------- status
def cmd_status(a) -> int:
    import urllib.error
    import urllib.request

    out = {"url": a.url}
    for path in ("/health", "/stats"):
        try:
            with urllib.request.urlopen(a.url.rstrip("/") + path, timeout=a.timeout) as r:
                out[path.strip("/")] = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            out[path.strip("/")] = {
                "http_status": exc.code,
                "body": exc.read().decode(errors="replace")[:2000],
            }
        except Exception as exc:  # noqa: BLE001
            out[path.strip("/")] = {"error": repr(exc)}
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        h, s = out.get("health", {}), out.get("stats", {})
        print(
            f"{a.url}: {h.get('status', h)}; mode {s.get('memory_mode')}; admission {h.get('admission')}"
        )
        mem = s.get("memory") or {}
        obs = (mem.get("observed") or {}).get("process") or {}
        print(f"  plan: {s.get('memory_plan')}")
        print(
            f"  MLX active {_gib(mem.get('mlx_active_bytes'))}, footprint {_gib(obs.get('footprint_bytes'))}, "
            f"last resize {(mem.get('last_resize') or {}).get('status')}"
        )
    return 0 if out.get("health", {}).get("status") == "ok" else 1


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jitengine-qwen3.6-35b-a3b-m2",
        description="Serve Qwen3.6-35B-A3B on an Apple silicon Mac.",
    )
    ap.add_argument(
        "--version", action="version", version=f"jitengine-qwen3.6-35b-a3b-m2 {__version__}"
    )
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser(
        "doctor", help="report platform, MLX/Metal and telemetry capabilities (read-only)"
    )
    d.add_argument("--json", action="store_true")
    i = sub.add_parser(
        "inspect", help="header-only checkpoint inventory and candidate plan (read-only)"
    )
    i.add_argument("--model-path", required=True)
    i.add_argument("--json", action="store_true")
    i.add_argument("--no-metal", action="store_true", help="do not import MLX for Metal limits")
    _add_config_flags(i)
    p = sub.add_parser(
        "pack", help="build the expert pack (writes only the pack and its temp file)"
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--force", action="store_true")
    s = sub.add_parser(
        "serve", help="validate, plan, preflight and serve the OpenAI-compatible API"
    )
    s.add_argument("--model-path", default=None)
    s.add_argument(
        "--host",
        default="127.0.0.1",
        action=_Explicit,
        help="bind address (default 127.0.0.1, loopback only)",
    )
    s.add_argument("--port", type=int, default=8080, action=_Explicit, help="port (default 8080)")
    s.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        metavar="NAME",
        help="a host name requests may carry in Host besides localhost, IP literals and --host "
        "(repeatable); any other is refused with 403, which stops DNS rebinding",
    )
    s.add_argument(
        "--preflight-wait",
        type=float,
        default=60.0,
        help="seconds to wait for available memory before refusing (default 60)",
    )
    s.add_argument(
        "--log-level",
        choices=logs.LEVELS,
        default="info",
        help="the least severe log records to print (default info)",
    )
    _add_config_flags(s)
    st = sub.add_parser("status", help="health and memory state of a running server (read-only)")
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
    return {
        "doctor": cmd_doctor,
        "inspect": cmd_inspect,
        "pack": cmd_pack,
        "serve": cmd_serve,
        "status": cmd_status,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
