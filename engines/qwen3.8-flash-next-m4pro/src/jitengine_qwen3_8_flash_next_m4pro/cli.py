"""`jitengine-qwen3.8-flash-next-m4pro` command line: doctor, inspect, pack, serve and status.

MLX is imported only by the commands that need it, so `--help` and `--version` work anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from jitengine_qwen3_8_flash_next_m4pro import __version__, logs

logger = logging.getLogger(__name__)

GIB = 1 << 30
# a budget over this fraction of unified memory is refused without --force: macOS needs the rest
MAX_BUDGET_FRACTION = 0.80
DEFAULT_BUDGET_FRACTION = 0.62
# below this the expert pool holds too few records for a good hit rate
LOW_BUDGET_BYTES = 14 * 10**9


def _die(msg: str, code: int = 2) -> int:
    print(f"jitengine-qwen3.8-flash-next-m4pro: error: {msg}", file=sys.stderr)
    return code


def _unified_memory_bytes() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _gib(n) -> str:
    return "?" if n is None else f"{n / GIB:.2f} GiB"


def _bytes_flag(unit: int):
    """An argparse type: a non-negative number of `unit`s, converted to bytes exactly."""

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


def _requested_budget(a) -> int | None:
    """The budget the flags request, in bytes: --memory-budget-gib, or --memory-budget-gb in the
    decimal GB the published budgets were measured in."""
    gib, gb = a.memory_budget_gib, a.memory_budget_gb
    if gib is not None and gb is not None:
        raise SystemExit(
            _die("pass the budget once: --memory-budget-gib or --memory-budget-gb, not both")
        )
    return gib if gib is not None else gb


def _default_budget_bytes() -> int:
    """62% of unified memory, at a hundredth of a GiB."""
    return int(round(_unified_memory_bytes() * DEFAULT_BUDGET_FRACTION / GIB, 2) * GIB)


def _budget_bytes(requested: int | None, force: bool) -> int:
    """The engine's memory budget in bytes: the request, or 62% of unified memory; refuses a budget
    over 80% of it without --force."""
    ram = _unified_memory_bytes()
    budget = requested if requested is not None else _default_budget_bytes()
    if budget > MAX_BUDGET_FRACTION * ram:
        if not force:
            raise SystemExit(
                f"jitengine-qwen3.8-flash-next-m4pro: refusing: a budget of {_gib(budget)} is more "
                f"than 80% of {_gib(ram)} of unified memory; macOS needs the rest. Pass --force to "
                "run it anyway."
            )
        logger.warning(
            f"[jitengine-qwen3.8-flash-next-m4pro] WARNING: a budget of {_gib(budget)} is more than "
            f"80% of {_gib(ram)} (--force)"
        )
    if budget < LOW_BUDGET_BYTES:
        logger.warning(
            f"[jitengine-qwen3.8-flash-next-m4pro] WARNING: a budget of {_gib(budget)} leaves about "
            f"{_gib(max(budget - 11 * 10**9, 0))} for the expert pool; expect a low hit rate and "
            "slow decode"
        )
    logger.info(
        f"[jitengine-qwen3.8-flash-next-m4pro] unified memory {_gib(ram)}, engine budget {_gib(budget)}"
    )
    return budget


def _tuned_on(info: dict) -> dict:
    """The Mac this build was tuned on, and a note in words on what that means for this one."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels.selfcheck import (
        TUNED_GPU,
        TUNED_MAC,
        tuned_note,
    )

    note = tuned_note(str(info.get("device_name", "")), str(info.get("architecture")))
    same = f"This build was tuned on {TUNED_MAC} ({TUNED_GPU}), this Mac's GPU family."
    return {"mac": TUNED_MAC, "gpu": TUNED_GPU, "note": note or same}


def cmd_doctor(a) -> int:
    """The platform, MLX and Metal this engine would run on (read-only)."""
    import importlib.metadata as md
    import platform
    import subprocess

    rep: dict = {
        "jitengine-qwen3.8-flash-next-m4pro": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "physical_bytes": _unified_memory_bytes(),
    }
    _probe_chip(rep, subprocess)
    info, err = _probe_metal()
    rep["metal"] = info if info is not None else {"error": err}
    if info:
        rep["tuned_on"] = _tuned_on(info)

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    rep["packages"] = {p: version(p) for p in ("mlx", "mlx-metal", "numpy", "transformers")}
    rep["supported"] = bool(sys.platform == "darwin" and rep["machine"] == "arm64" and info)
    _print_doctor(a, rep, info, err)
    return 0 if rep["supported"] else 1


def _probe_chip(rep: dict, subprocess) -> None:
    """Put the CPU brand string in rep, or None and the error when sysctl fails."""
    try:
        rep["chip"] = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        rep["chip"], rep["chip_error"] = None, repr(exc)


def _probe_metal() -> tuple:
    """The Metal device info, or None and why MLX or Metal is unavailable."""
    try:
        import mlx.core as mx

        info, err = dict(mx.device_info()), None
        if not mx.metal.is_available():
            info, err = None, "Metal is not available"
    except Exception as exc:  # noqa: BLE001  any failure means: this Mac cannot serve, and why
        info, err = None, repr(exc)
    return info, err


def _print_doctor(a, rep: dict, info: dict | None, err: str | None) -> None:
    """Print the doctor report as JSON or as the human summary."""
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(
            f"jitengine-qwen3.8-flash-next-m4pro {__version__} on {rep['platform']} ({rep.get('chip')})"
        )
        print(f"  unified memory {_gib(rep['physical_bytes'])}")
        if info:
            print(
                f"  Metal: {info.get('device_name')} {info.get('architecture')}, recommended working "
                f"set {_gib(info.get('max_recommended_working_set_size'))}"
            )
        else:
            print(f"  Metal: unavailable ({err})")
        if "tuned_on" in rep:
            print(f"  {rep['tuned_on']['note']}")
        print(f"  packages: {rep['packages']}")
        print(f"  supported for serving: {rep['supported']}")


def cmd_inspect(a) -> int:
    """The memory plan a budget gives on this checkpoint, from its headers alone (read-only)."""
    from jitengine_qwen3_8_flash_next_m4pro.budget import (
        WAVE_TOKENS_SERVED,
        BudgetError,
        plan_budget,
    )
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.expert_reader import CheckpointScan
    from jitengine_qwen3_8_flash_next_m4pro.model.config import load_config
    from jitengine_qwen3_8_flash_next_m4pro.model.moe import MOE_KERNEL_DEFAULT

    model_path = os.path.expanduser(a.model_path)
    cfg, scan = load_config(model_path), CheckpointScan(model_path)
    budget = _requested_budget(a)
    if budget is None:
        budget = _default_budget_bytes()
    opts = {
        "compute_dtype": "float32",
        "moe_kernel": MOE_KERNEL_DEFAULT,
        "wave_tokens": WAVE_TOKENS_SERVED,
    }
    try:
        plan = plan_budget(budget, scan, cfg, opts)
    except BudgetError as exc:
        print(
            f"jitengine-qwen3.8-flash-next-m4pro: a budget of {_gib(budget)} does not fit: {exc}",
            file=sys.stderr,
        )
        return 1
    if a.json:
        print(json.dumps({"budget_bytes": budget, "plan": plan.as_dict()}, indent=1, default=str))
    else:
        print(f"{model_path}: {scan.num_layers} layers, {scan.num_records} expert records")
        print(plan.describe())
    return 0


def cmd_status(a) -> int:
    """Health and memory of a running server (read-only)."""
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
        except Exception as exc:  # noqa: BLE001  report any failure to reach the server
            out[path.strip("/")] = {"error": repr(exc)}
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        h, st = out.get("health", {}), out.get("stats", {})
        print(f"{a.url}: {h.get('status', h)}")
        print(f"  plan: {st.get('memory_plan')}")
    return 0 if out.get("health", {}).get("status") == "ok" else 1


def _install_startup_term_handler() -> None:
    """Exit 0 on SIGTERM during the model load; `serve()` replaces it."""
    import signal

    def handle(signum, _frame):
        logger.info(f"[engine] signal {signum} during startup: exiting")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, handle)
    except ValueError, OSError:
        pass  # not the main thread, or no signal support: the default SIGTERM handling stays


def cmd_pack(a) -> int:
    """Build the expert and PLE packs the engine reads, once, each verified against the checkpoint;
    with `--mtp`, then the multi-token-prediction head from the official release; with `--pack16`
    (or `--pack16-out`), then the compact-record pack16 file (every record verified)."""
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_experts, build_ple

    for build in (build_experts, build_ple):
        rc = build.main(["--model", a.model_path, "--verify", str(a.verify)])
        if rc:
            return rc
    if a.mtp:
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_mtp

        rc = build_mtp.main(["--model", a.model_path])
        if rc:
            return rc
    out = a.pack16_out
    if a.pack16 and not out:
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint.p16_format import pack16_path_for
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
            ExpertScan,
            default_pack_path,
        )

        scan = ExpertScan(a.model_path)
        out = pack16_path_for(default_pack_path(scan), scan)
        os.makedirs(os.path.dirname(out), exist_ok=True)
    if out:
        from jitengine_qwen3_8_flash_next_m4pro.checkpoint import build_p16

        return build_p16.main(["--model", a.model_path, "--out", out, "--all"])
    return 0


# the serve flags that become engine options of the same name when given
SERVE_OPTIONS = (
    "max_slots",
    "kv_budget_mb",
    "prefix_grid_tokens",
    "prefix_min_divergence_tokens",
    "compute_dtype",
    "residual_dtype",
    "moe_kernel",
    "moe_kernel_variant",
    "dense_kernel",
    "wave_width",
    "admit_warm_quiet_s",
    "prefetch_lookahead",
    "prefetch_reserve",
    "mtp_min_slots",
    "draft_vocab",
    "expert_pool_evict",
    "embed_stream",
    "embed_cache_rows",
    "store_spill",
    "ple_prefetch",
    "ple_host_context",
    "pack16_disk",
)
SERVE_SWITCHES = ("transient_prefill", "moe_sync_each_layer")  # on when given


def cmd_serve(a) -> int:
    """Check the checkpoint and the budget, build the engine and serve the OpenAI-compatible API."""
    logs.configure(a.log_level)
    model_path = os.path.expanduser(a.model_path)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        return _die(
            f"no checkpoint at {model_path} (need config.json, the tokenizer and the shards)"
        )
    opts = {"memory_budget_bytes": _budget_bytes(_requested_budget(a), a.force)}
    _install_startup_term_handler()
    for name in ("expert_pack_path", "ple_pack_path"):
        if getattr(a, name) is not None:
            opts[name] = os.path.expanduser(getattr(a, name))
    opts.update({k: getattr(a, k) for k in SERVE_OPTIONS if getattr(a, k) is not None})
    opts.update({k: True for k in SERVE_SWITCHES if getattr(a, k)})
    from jitengine_qwen3_8_flash_next_m4pro.engine.engine import create_engine

    engine = create_engine(model_path, **opts)
    engine.serve(a.host, a.port, allowed_hosts=tuple(a.allowed_host or ()))
    return 0


def _budget_flags(p) -> None:
    p.add_argument(
        "--memory-budget-gib",
        type=_bytes_flag(GIB),
        default=None,
        help="memory the engine may use, in GiB (binary; default 62%% of unified memory)",
    )
    p.add_argument(
        "--memory-budget-gb",
        type=_bytes_flag(10**9),
        default=None,
        help="the same budget in decimal GB (10^9 bytes), as the published budgets were measured",
    )


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    ap = argparse.ArgumentParser(
        prog="jitengine-qwen3.8-flash-next-m4pro",
        description="Serve Qwen3.8-Flash-Next on an Apple silicon Mac.",
    )
    ap.add_argument(
        "--version", action="version", version=f"jitengine-qwen3.8-flash-next-m4pro {__version__}"
    )
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("doctor", help="report the platform, MLX and Metal this would run on")
    d.add_argument("--json", action="store_true")
    i = sub.add_parser(
        "inspect", help="the memory plan a budget gives, from the checkpoint headers"
    )
    i.add_argument("--model-path", required=True, help="checkpoint directory")
    _budget_flags(i)
    i.add_argument("--json", action="store_true")
    p = sub.add_parser("pack", help="build the expert and PLE packs the engine reads (once)")
    p.add_argument("--model-path", required=True, help="checkpoint directory")
    p.add_argument(
        "--verify",
        type=int,
        default=64,
        help="records of each pack checked against the checkpoint, byte for byte (default 64)",
    )
    p.add_argument(
        "--mtp",
        action="store_true",
        help="also build the multi-token-prediction head, <model path>/mtp.safetensors (1.5 GB, from "
        "4.9 GB of the official bf16 release); one user's decode drafts with it",
    )
    p.add_argument(
        "--pack16",
        action="store_true",
        help="also write the compact-record pack16 file (68 GB) where the engine reads it, "
        "<pack dir>/../pack16/<checkpoint>-<sha12>.p16x",
    )
    p.add_argument(
        "--pack16-out",
        default=None,
        help="also write the pack16 file, at this path (point the engine at it by placing it as above)",
    )
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--model-path", required=True, help="checkpoint directory")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        metavar="NAME",
        help="a host name requests may carry in Host besides localhost, IP literals and --host "
        "(repeatable); any other is refused with 403, which stops DNS rebinding",
    )
    _budget_flags(s)
    s.add_argument(
        "--force", action="store_true", help="run a budget over 80%% of unified memory anyway"
    )
    s.add_argument(
        "--log-level",
        choices=logs.LEVELS,
        default="info",
        help="the least severe log records to print (default info)",
    )
    s.add_argument("--expert-pack-path", default=None, help="expert pack file")
    s.add_argument(
        "--pack16-disk",
        choices=("on", "off"),
        default=None,
        help="read compact pack16 records from <pack dir>/../pack16/<checkpoint>-<sha12>.p16x when it "
        "validates (on, the default), or read the expert pack and encode at publish (off)",
    )
    s.add_argument("--ple-pack-path", default=None, help="PLE pack file")
    s.add_argument("--max-slots", type=int, default=None, help="batcher state slots (default 8)")
    s.add_argument(
        "--kv-budget-mb",
        type=float,
        default=None,
        help="live attention KV budget in MB (default: planned from the memory budget)",
    )
    s.add_argument(
        "--prefix-grid-tokens",
        dest="prefix_grid_tokens",
        type=int,
        default=None,
        help="absolute snapshot grid for a prompt, also the partition score() walks (default 2048)",
    )
    s.add_argument(
        "--wave-chunk", dest="prefix_grid_tokens", type=int, default=None, help=argparse.SUPPRESS
    )
    s.add_argument(
        "--prefix-min-divergence-tokens",
        type=int,
        default=None,
        help="minimum divergence before the prefix store snapshots (default 256)",
    )
    s.add_argument(
        "--compute-dtype",
        default=None,
        choices=["bfloat16", "float32"],
        help="activation dtype: float32 (default) or bfloat16",
    )
    s.add_argument(
        "--residual-dtype",
        default=None,
        choices=["bfloat16", "float32"],
        help="hyper-connection residual dtype, with bf16 matmul inputs (default: the "
        "compute dtype)",
    )
    s.add_argument(
        "--moe-kernel",
        default=None,
        choices=["grouped", "stock"],
        help="routed-expert kernel: grouped (default; reads each distinct expert once) "
        "or stock gather_qmm",
    )
    s.add_argument(
        "--dense-kernel",
        default=None,
        choices=["rb", "stock"],
        help="dense linear kernel: rb (default; row-blocked, on the shapes its table "
        "covers) or stock",
    )
    s.add_argument(
        "--moe-kernel-variant",
        default=None,
        choices=["fused", "split"],
        help=argparse.SUPPRESS,  # grouped kernel variant: split (default) or fused gate+up
    )
    s.add_argument(
        "--wave-width",
        type=int,
        default=None,
        help="max rows in one layer-synchronous prefill wave; 1 prefills one row at a time",
    )
    s.add_argument(
        "--admit-warm-quiet-s",
        type=float,
        default=None,
        help="how long an idle engine keeps the arrival window open when every pending "
        "prompt is already in the prefix store (default 0.1 s; 0 disables)",
    )
    s.add_argument(
        "--mtp-min-slots",
        type=int,
        default=None,
        help="expert slots a lone greedy row must have before it decodes two positions per pass "
        "with the MTP head (default 10500; 0 whenever a head is found)",
    )
    s.add_argument(
        "--draft-vocab",
        choices=("subset", "full"),
        default=None,
        help="the MTP draft's readout: the shipped top-K token ids plus the request's prompt ids "
        "(subset, the default) or every token (full); the draft decides how many positions a pass "
        "verifies, never a token",
    )
    s.add_argument(
        "--expert-pool-evict",
        "--evict",
        choices=["cyc64", "decay32"],
        default=None,
        help="single-row decode's expert-slot victim rule: cyc64, the expected next use in the "
        "decode's fixed layer cycle (default), or decay32, the decayed use count",
    )
    s.add_argument(
        "--embed-stream",
        choices=("on", "off"),
        default=None,
        help="read token-embedding rows from the checkpoint (on, the default: the 0.37 GiB table "
        "goes to the expert pool) or hold the table resident (off)",
    )
    s.add_argument(
        "--embed-cache-rows",
        type=int,
        default=None,
        help="rows the streamed embedding's LRU keeps in memory (default 8192; charged in the plan)",
    )
    s.add_argument(
        "--store-spill",
        choices=("on", "off"),
        default=None,
        help="while one row decodes, spill the prefix store to a scratch file and lend its memory "
        "to the expert cache (on, the default), or keep it resident (off)",
    )
    s.add_argument(
        "--ple-prefetch",
        default=None,
        choices=["on", "off"],
        help="the MTP verify pass issues both positions' PLE row reads at round start (default on, "
        "or $JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PREFETCH); off reads them in the PLE layer",
    )
    s.add_argument(
        "--ple-host-context",
        default=None,
        choices=["on", "off"],
        help="keep the PLE n-gram context as a host-made array, so reading it never waits for the "
        "GPU (default on, or $JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_HOST_CONTEXT)",
    )
    s.add_argument(
        "--transient-prefill",
        action="store_true",
        help=argparse.SUPPRESS,  # expert records fetched during prefill are evicted first
    )
    s.add_argument(
        "--prefetch-lookahead",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # router-lookahead prefetch distance in layers (default 0, off)
    )
    s.add_argument(
        "--prefetch-reserve",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # slots reserved for speculative reads (default 32, about 98 MB)
    )
    s.add_argument(
        "--moe-sync-each-layer",
        action="store_true",
        help=argparse.SUPPRESS,  # finish each layer's expert gather before the next may evict a slot
    )
    st = sub.add_parser("status", help="health and memory of a running server (read-only)")
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
