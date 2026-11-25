"""Cold-start throughput measurement of the engine against stock mlx-lm on this Mac.

Each run starts a fresh server (no warmup), waits for /health, drives it with benchmark.py, and
stops it. With two arms the runs alternate order, so a slow minute hits both. Every run is kept,
with the machine's power state and any busy process seen before it; nothing is dropped or rerun.

    python benchmarks/run.py --model-path ~/models/Qwen3-1.7B-4bit --arms engine stock \
        --concurrency 16 --repeats 5 --out results/c16.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import _servers

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {"scored": "prompts_scored.txt", "heldout": "prompts_heldout.txt"}
TOTAL_REQUESTS = 64
MAX_TOKENS = 128
TOKEN_BOUNDS = (122, 128)  # per request, as the benchmark's workload produces


def one_run(arm: str, a, index: int, log_dir: str) -> dict:
    busy = _servers.busy_processes()
    server = _servers.Server(arm, a.model_path, os.path.join(log_dir, f"{arm}_{index}.server.log"))
    out = os.path.join(log_dir, f"{arm}_{index}.json")
    try:
        proc = subprocess.run(
            [
                sys.executable, os.path.join(HERE, "benchmark.py"), "--url", server.chat_url,
                "--model", "default_model", "--concurrency", str(a.concurrency),
                "--total-requests", str(TOTAL_REQUESTS), "--max-tokens", str(MAX_TOKENS),
                "--prompt-file", os.path.join(HERE, PROMPTS[a.prompts]), "--output", out,
            ],
            capture_output=True, text=True,
        )  # fmt: skip
    finally:
        server.stop()
    if proc.returncode != 0 or not os.path.exists(out):
        return {
            "arm": arm,
            "index": index,
            "ok": False,
            "error": proc.stderr[-400:],
            "busy_before": busy,
        }
    d = json.load(open(out))
    per = d["per_request"]
    tokens = [p["tokens_received"] for p in per]
    ttft = d["aggregate_metrics"]["time_to_first_token"]
    return {
        "arm": arm,
        "index": index,
        "ok": True,
        "tokens_per_sec": d["total_tokens"] / d["total_time"],  # total tokens over total time
        "rolling_tokens_per_sec": d["aggregate_tokens_per_sec"],  # the client's rolling aggregate
        "total_tokens": d["total_tokens"],
        "total_time": d["total_time"],
        "successful": d["successful_requests"],
        "tokens_min": min(tokens),
        "tokens_max": max(tokens),
        "ttft_avg": ttft["avg"],
        "ttft_p95": ttft["p95"],
        "latency_mean": statistics.mean(p["total_time"] for p in per),
        "latency_max": max(p["total_time"] for p in per),
        "within_bounds": d["successful_requests"] == TOTAL_REQUESTS
        and TOKEN_BOUNDS[0] <= min(tokens)
        and max(tokens) <= TOKEN_BOUNDS[1],
        "ready_seconds": server.ready_seconds,
        "busy_before": busy,
        "power_before": _servers.machine()["power_source"],
    }


def summarize(runs: list[dict], arms: list[str]) -> dict:
    table = {}
    for arm in arms:
        ok = [r for r in runs if r["arm"] == arm and r["ok"]]
        if not ok:
            continue
        speeds = [r["tokens_per_sec"] for r in ok]
        totals = [r["total_tokens"] for r in ok]
        table[arm] = {
            "runs": len(ok),
            "failed_runs": sum(r["arm"] == arm and not r["ok"] for r in runs),
            "tokens_per_sec_median": statistics.median(speeds),
            "tokens_per_sec_min": min(speeds),
            "tokens_per_sec_max": max(speeds),
            "total_tokens_min": min(totals),
            "total_tokens_max": max(totals),
            "all_within_bounds": all(r["within_bounds"] for r in ok),
            "ttft_avg_median": statistics.median(r["ttft_avg"] for r in ok),
            "latency_mean_median": statistics.median(r["latency_mean"] for r in ok),
        }
    if len(table) == 2:
        first, second = arms
        table["ratio_of_medians"] = {
            f"{first}/{second}": table[first]["tokens_per_sec_median"]
            / table[second]["tokens_per_sec_median"]
        }
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--arms", nargs="+", choices=["engine", "stock"], default=["engine", "stock"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompts", choices=sorted(PROMPTS), default="scored")
    ap.add_argument("--settle", type=float, default=5, help="seconds to wait between runs")
    ap.add_argument("--out", required=True, help="JSON file for every run and the summary")
    a = ap.parse_args(argv)

    log_dir = tempfile.mkdtemp(prefix="jit-bench-")
    runs = []
    for repeat in range(a.repeats):
        order = a.arms if repeat % 2 == 0 else a.arms[::-1]  # alternate who goes first
        for arm in order:
            r = one_run(arm, a, len(runs), log_dir)
            runs.append(r)
            shown = (
                f"{r['tokens_per_sec']:.1f} tok/s, {r['total_tokens']} tokens"
                if r["ok"]
                else r["error"]
            )
            print(f"[{len(runs):>2}] {arm:<6} c={a.concurrency} {a.prompts}: {shown}", flush=True)
            time.sleep(a.settle)
    result = {
        "machine": _servers.machine(),
        "settings": {k: v for k, v in vars(a).items() if k != "out"},
        "benchmark_sha256": __import__("hashlib")
        .sha256(open(os.path.join(HERE, "benchmark.py"), "rb").read())
        .hexdigest(),
        "runs": runs,
        "summary": summarize(runs, a.arms),
        "logs": log_dir,
    }
    _servers.dump(result, a.out)
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
