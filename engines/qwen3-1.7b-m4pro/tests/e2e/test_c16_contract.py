"""The scored workload against the engine: 64 of 64 requests, 122 to 128 tokens each.

The total tokens vary from run to run with the batch sizes a run passes through (stock MLX uses
different arithmetic below 13 rows), so the test reports the total and does not floor it."""

import json
import os
import subprocess
import sys

BENCH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks"
)


def run_client(server, concurrency, prompts, tmp_path):
    out = tmp_path / f"c{concurrency}.json"
    proc = subprocess.run(
        [
            sys.executable, os.path.join(BENCH, "benchmark.py"), "--url", f"{server.url}/v1/chat/completions",
            "--model", "default_model", "--concurrency", str(concurrency), "--total-requests", "64",
            "--max-tokens", "128", "--prompt-file", os.path.join(BENCH, prompts), "--output", str(out),
        ],
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-500:]
    return json.load(open(out))


def check(d, lowest):
    tokens = [p["tokens_received"] for p in d["per_request"]]
    assert d["successful_requests"] == 64, d["successful_requests"]
    assert min(tokens) >= lowest and max(tokens) <= 128, (min(tokens), max(tokens))
    print(f"total tokens {d['total_tokens']}, {d['total_tokens'] / d['total_time']:.1f} tok/s")


def test_scored_workload_at_16(server, tmp_path):
    check(run_client(server, 16, "prompts_scored.txt", tmp_path), lowest=122)


def test_heldout_workload_at_16(server, tmp_path):
    # some held-out prompts end early (a short poem), so only the 128-token cap applies
    check(run_client(server, 16, "prompts_heldout.txt", tmp_path), lowest=1)
