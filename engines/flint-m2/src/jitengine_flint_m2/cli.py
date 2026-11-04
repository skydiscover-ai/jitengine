"""Command line: `jitengine-flint-m2 serve --model-path DIR ...` (or `python -m jitengine_flint_m2 serve ...`),
`jitengine-flint-m2 doctor --model-path DIR [--json [FILE]]` (doctor.py: what this Mac runs, exit 0 / 1 / 2) and
`jitengine-flint-m2 inspect --model-path DIR [--json [FILE]]` (what serve would plan, without loading a weight).

Every refusal (unknown or unbuilt profile, context limit outside 1..262,144, a GPU that is not Apple
silicon, checkpoint digest/header mismatch, a context limit the megakernel's wired limit cannot hold,
a memory budget too small for one layer of experts, a port in use) happens before any weight is
allocated and exits non-zero with a one-line reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from jitengine_flint_m2 import __version__, logs
from jitengine_flint_m2.budget import MEMORY_MODES

PROG = "jitengine-flint-m2"
DEFAULT_MAX_QUEUED = (
    8  # engine.MAX_QUEUE (the engine module is not imported here: it loads the tokenizer)
)


def context_limit_arg(v: str):
    """`auto` or an integer (range and fit are checked by the engine, before any weight)."""
    if v == "auto":
        return v
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto' or an integer, got {v!r}") from None


def nonneg_int(v: str) -> int:
    """An argparse type: an integer >= 0."""
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {v!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0: {v!r}")
    return n


def bytes_arg(unit: int):
    """An argparse type: a positive number of `unit`s, converted to bytes exactly."""

    def conv(v: str) -> int:
        from decimal import Decimal, InvalidOperation

        try:
            d = Decimal(v)
        except InvalidOperation:
            raise argparse.ArgumentTypeError(f"not a number: {v!r}") from None
        if not d.is_finite() or d <= 0:
            raise argparse.ArgumentTypeError(f"must be a positive number: {v!r}")
        return int(d * unit)

    return conv


def simulate_arg(v: str) -> dict:
    """A JSON object, or @FILE holding one (device.Sim)."""
    try:
        if v.startswith("@"):
            with open(v[1:]) as f:
                return json.load(f)
        return json.loads(v)
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"--simulate: {exc}") from None


def add_engine_args(s: argparse.ArgumentParser) -> None:
    """The engine options `serve` and `doctor` share."""
    s.add_argument(
        "--model-path",
        "--model",
        dest="model",
        required=True,
        help="the flint-int4 checkpoint directory (RecursiveSI/flint-int4)",
    )
    s.add_argument(
        "--context-limit",
        type=context_limit_arg,
        default="auto",
        metavar="auto|N",
        help="history positions: auto (default) = what the GPU wired limit holds, at most 262,144; an "
        "explicit N that does not fit is refused, never truncated",
    )
    s.add_argument(
        "--memory-budget-gib",
        type=bytes_arg(2**30),
        default=None,
        metavar="N",
        help="the memory the engine may use, in GiB. Default: the whole model resident when the GPU wired limit "
        "holds it, else the wired limit at most physical memory less 6 GiB. A budget that cannot hold the 22.6 GB "
        "of weights keeps the non-expert weights and as many experts as fit, reads the rest from the checkpoint "
        "and decodes with stock MLX ops",
    )
    s.add_argument(
        "--memory-budget-gb",
        type=bytes_arg(10**9),
        default=None,
        metavar="N",
        help="the same budget in decimal GB (10^9 bytes)",
    )
    s.add_argument(
        "--mode",
        default="auto",
        choices=MEMORY_MODES,
        help="auto (default): stream the experts from the checkpoint only when the whole model does not fit; "
        "resident: refuse instead; offload: stream them even when it fits (profile main, stock MLX ops)",
    )
    s.add_argument(
        "--profile",
        default="main",
        help="main (default), or kv8: the 14 full-attention layers' history at 8 bits, which halves the "
        "history read per token at long context and changes the outputs slightly",
    )
    s.add_argument(
        "--decode-backend",
        default="auto",
        choices=("auto", "mk", "graph"),
        help="auto (default): the persistent megakernel if its load check passes on this GPU, else stock "
        "MLX ops (profile main; kv8 refuses); mk: the megakernel or refuse; graph: the MLX-graph "
        "reference decode",
    )
    s.add_argument(
        "--mk-grid",
        type=int,
        default=None,
        metavar="G",
        help="pin the megakernel grid (threadgroups; a multiple of 4 in 16..64): probed at G only and "
        "refused if not co-resident (default: tuned or sized from the GPU core count, searched down)",
    )
    s.add_argument(
        "--simulate",
        type=simulate_arg,
        default=None,
        metavar="JSON|@FILE",
        help=argparse.SUPPRESS,  # test hook (serve: needs --allow-fault-injection): injected device facts
    )


def add_json_arg(s: argparse.ArgumentParser) -> None:
    s.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="write the report as JSON to FILE (no FILE or '-': stdout)",
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG)
    ap.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    add_engine_args(s)
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    s.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        metavar="NAME",
        help="a host name requests may carry in Host besides localhost, IP literals and --host "
        "(repeatable); any other is refused with 403, which stops DNS rebinding",
    )
    s.add_argument(
        "--max-queued",
        type=nonneg_int,
        default=DEFAULT_MAX_QUEUED,
        metavar="N",
        help=f"requests that may wait while one runs; more get 429 (default {DEFAULT_MAX_QUEUED})",
    )
    s.add_argument(
        "--one-token-per-pass",
        action="store_true",
        help="commit exactly one token per model pass: no speculation (slower; the reference the "
        "speculative output is checked against)",
    )
    s.add_argument(
        "--allow-fault-injection",
        action="store_true",
        help=argparse.SUPPRESS,  # honour the x_fault request field (tests only; see faults.py)
    )
    s.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="do not keep the last prompt's prefill state; a repeated prompt is prefilled again",
    )
    s.add_argument(
        "--log-level",
        default="info",
        choices=tuple(logs.LEVELS),
        help="the least severe log lines to print (default info; the engine logs nothing below info)",
    )
    s.add_argument(
        "--verify-weights",
        action="store_true",
        help="sha256 every safetensors file at start (reads all 22.6 GB; the default checks the small "
        "files and the safetensors headers)",
    )
    s.add_argument(
        "--request-deadline-s",
        type=float,
        default=None,
        metavar="S",
        help="wall-clock budget per generation request, queue wait included (default: off); a request "
        "that runs out stops at its next pass and is answered with the tokens so far, "
        'finish_reason "length", x_metrics.deadline_exceeded true',
    )
    s.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        metavar="N",
        help="cap on max_tokens (default: off, i.e. the room left under the context limit)",
    )
    d = sub.add_parser(
        "doctor",
        help="what this Mac runs: load the engine, report device, grid, backend and context as JSON; "
        "exit 0 (megakernel), 1 (stock MLX ops, or speculation off), 2 (refused)",
    )
    add_engine_args(d)
    add_json_arg(d)
    d.add_argument(
        "--smoke-tokens",
        type=int,
        default=16,
        metavar="N",
        help="greedy tokens decoded from a fixed prompt once loaded (0: none)",
    )
    d.add_argument(
        "--tune",
        action="store_true",
        help="opt-in: before loading, time every co-resident megakernel grid on this Mac and cache the "
        "fastest (FLINT_GRID_CACHE, default ~/.cache/flint-engine/grid-v1.json); later starts on a "
        "device outside the tuned table try it first (the tuned M4 Pro keeps G 40)",
    )
    i = sub.add_parser(
        "inspect",
        help="what serve would plan on this Mac, from the checkpoint headers and without loading a weight: "
        "admission, the memory plan, the context limit, the decode backend; exit 0 (would start), 2 (refused)",
    )
    add_engine_args(i)
    add_json_arg(i)
    return ap


def _watchdog(engine) -> None:
    """An engine that became unhealthy (worker dead, reset failed) exits the process non-zero."""
    while True:
        time.sleep(0.5)
        if engine.health()["status"] == "unhealthy":
            print(
                f"[server] engine unhealthy: {engine.health().get('fatal')}; exiting",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(1.0)
            os._exit(3)


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "doctor":
        from jitengine_flint_m2.doctor import run

        return run(a)
    if a.cmd == "inspect":
        from jitengine_flint_m2.doctor import run_inspect

        return run_inspect(a)
    from jitengine_flint_m2.budget import flag_budget
    from jitengine_flint_m2.engine.engine import Engine
    from jitengine_flint_m2.model.config import StartupError

    log = logs.leveled(a.log_level)
    try:
        budget = flag_budget(a.memory_budget_gib, a.memory_budget_gb)
        eng = Engine(
            a.model,
            profile=a.profile,
            context_limit=a.context_limit,
            one_token_per_pass=a.one_token_per_pass,
            allow_fault_injection=a.allow_fault_injection,
            prefix_cache=not a.no_prefix_cache,
            verify_weights=a.verify_weights,
            start=False,
            decode_backend=a.decode_backend,
            mk_grid=a.mk_grid,
            sim=a.simulate,
            request_deadline_s=a.request_deadline_s,
            max_output_tokens=a.max_output_tokens,
            memory_budget_bytes=budget,
            memory_mode=a.mode,
            max_queue=a.max_queued,
            log=log,
        )
    except (StartupError, OSError, ValueError) as exc:
        print(f"{PROG}: refusing to start: {exc}", file=sys.stderr, flush=True)
        return 2
    log(
        f"[server] profile={eng.profile} kv_bits={eng.kv_bits} context_limit={eng.context_limit} "
        f"({eng.context_limit_source}) decode_backend={eng.decode_backend_requested} "
        f"one_token_per_pass={eng.one_token_per_pass} memory={eng.memory.mode}"
    )
    threading.Thread(target=_watchdog, args=(eng,), daemon=True).start()
    try:
        eng.serve(a.host, a.port, allowed_hosts=tuple(a.allowed_host or ()))
    except OSError as exc:
        print(f"{PROG}: cannot bind {a.host}:{a.port}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0
