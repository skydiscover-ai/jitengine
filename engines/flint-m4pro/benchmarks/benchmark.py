"""Decode-throughput benchmark for the Flint engine, standard library only.

    python3 benchmark.py --impl <engine dir> [--contexts 1024,...] [--modes code_greedy,code_sampled,raw]
                         [--set code|heldout] [--repeats 5] [--steps 1024] [--out-dir DIR] [--profile main|kv8]

--impl is a directory holding engine.py (normally this benchmarks directory). Prompts are read from
$FLINT_BENCH_DIR/bench/prompts (default ~/flint-bench; build them with make_prompts.py).

--profile (default main) selects an engine profile through the public create_engine option; the timing
protocol is identical, and every record and result.json carry the profile so a variant's numbers are
reported separately and never mixed with the main profile's.

What it does, in order:
  1. digests the engine tree (sha256 over its source files) and refuses a directory that has no engine.py;
  2. takes an exclusive lock ($FLINT_BENCH_DIR/box.lock) and waits on a memory predicate (available memory
     and unwound wired memory, never a clock), recording co-tenants above 2 GB;
  3. starts driver.py in the MLX venv (it loads the engine only through the public interface) and
     samples the footprint of the driver's whole process tree from outside every 0.25 s, plus swap;
  4. writes <out-dir>/result.json: per (mode, context) the median of the repeats' decode tok/s (each =
     1,024 decode passes / the window from the first post-prefill step to the last), all repeats, the
     spread, whether repeats produced identical tokens, the prefill restore check, external peak
     footprint per context, swap-out delta, co-tenants, and the engine digest.

Timing protocol (every mode): exactly 1,024 decode passes per run after a prefill that is excluded;
EOS ignored (code modes) or impossible (raw: forced stream); 5 repeats, median. A run whose co-tenant
list is non-empty, or whose swap-out counter grew, is flagged invalid in result.json (exclusivity is
binary: rerun it alone).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import footprint  # noqa: E402

CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 258048]
SOURCE_EXT = (".py", ".metal", ".h", ".c", ".cc", ".cpp", ".json", ".toml", ".txt")


def bench_dir() -> str:
    """$FLINT_BENCH_DIR, default ~/flint-bench: holds bench/prompts, bench/runs and box.lock."""
    return os.path.abspath(os.path.expanduser(os.environ.get("FLINT_BENCH_DIR") or "~/flint-bench"))


def tree_digest(root: str) -> str:
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__" and not d.startswith("."))
        for fn in sorted(fns):
            if fn.endswith(SOURCE_EXT):
                p = os.path.join(dp, fn)
                h.update(os.path.relpath(p, root).encode() + b"\0")
                with open(p, "rb") as f:
                    h.update(f.read())
                h.update(b"\0")
    return h.hexdigest()


def need_bytes(contexts) -> int:
    """Memory to have free before loading: weights + 16-bit history at the largest context + slack."""
    hist = (max(contexts) + 1025) * 28672
    return int(22.6e9 + hist + 4e9)


def summarize(runs: list, sampler: dict, cotenants_before: list) -> dict:
    by = {}
    for r in runs:
        by.setdefault((r["mode"], r["context"]), []).append(r)
    out = {}
    for (mode, ctx), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        v = [r["decode_tok_s"] for r in rs]
        out.setdefault(mode, {})[str(ctx)] = {
            "median_tok_s": round(statistics.median(v), 3), "repeats": [round(x, 3) for x in v],
            "spread_pct": round((max(v) - min(v)) / statistics.median(v) * 100, 2),
            "release_median_tok_s": round(statistics.median([r["release_decode_tok_s"] for r in rs]), 3),
            "decode_steps": rs[0]["decode_steps"], "prompt_tokens": rs[0]["prompt_tokens"],
            "tokens_identical_across_repeats": len({r["tokens_sha256"] for r in rs}) == 1,
            "cached_after_first": all(r["cached_prompt_tokens"] == r["prompt_tokens"] for r in rs[1:]),
            "prefill_s": rs[0]["engine_prefill_s"], "eos_in_output": rs[0].get("eos_in_output"),
            "peak_tree_footprint_gb": max((g for k, g in sampler["per_phase_peak_gb"].items()
                                           if k.startswith(f"{ctx}/")), default=None)}
    valid = not cotenants_before and not sampler["cotenants_ge_2gb"] and sampler["swapouts_delta_pages"] == 0
    return {"modes": out, "valid": valid,
            "invalid_reasons": [r for r, bad in (("co-tenant before start", bool(cotenants_before)),
                                                 ("co-tenant during run", bool(sampler["cotenants_ge_2gb"])),
                                                 ("swap-outs grew", sampler["swapouts_delta_pages"] != 0)) if bad]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", required=True, help="directory holding engine.py (normally this benchmarks directory)")
    ap.add_argument("--model", default=os.environ.get("FLINT_MODEL", os.path.expanduser("~/models/flint-int4")))
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    ap.add_argument("--modes", default="code_greedy,code_sampled,raw")
    ap.add_argument("--set", default="code", choices=("code", "heldout"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--python", default=os.environ.get("FLINT_PYTHON", sys.executable))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-lock", action="store_true", help="caller already holds the lock")
    ap.add_argument("--profile", default="main", help="engine profile (main, or kv8 for the 8-bit-history variant)")
    ap.add_argument("--need-gb", type=float, default=None,
                    help="available-memory predicate before start (default: weights + history + 4 GB)")
    a = ap.parse_args()
    root = bench_dir()
    impl = os.path.abspath(a.impl)
    if not os.path.isfile(os.path.join(impl, "engine.py")):
        raise SystemExit(f"{impl}: no engine.py (not an engine directory)")
    digest = tree_digest(impl)
    contexts = [int(c) for c in a.contexts.split(",")]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(a.out_dir or os.path.join(root, "bench", "runs", digest[:16], f"{a.set}_{stamp}"))
    os.makedirs(out, exist_ok=True)
    cmd = [a.python, "-u", os.path.join(HERE, "driver.py"), "--impl", impl, "--model", a.model,
           "--prompts", os.path.join(root, "bench", "prompts"), "--set", a.set, "--contexts", a.contexts,
           "--modes", a.modes, "--repeats", str(a.repeats), "--steps", str(a.steps), "--out-dir", out]
    if a.profile != "main":
        cmd += ["--profile", a.profile]
    lock = footprint.BoxLock(os.path.join(root, "box.lock"))
    t_lock = time.time()
    if not a.no_lock:
        lock.__enter__()
    try:
        wait = footprint.wait_memory(int(a.need_gb * 1e9) if a.need_gb is not None else need_bytes(contexts),
                                     wired_max_bytes=(2**62 if a.need_gb == 0 else 4 * 2**30))
        before = footprint.other_big_processes(set())
        print(f"[bench] lock after {time.time() - t_lock:.0f}s, memory {wait}, co-tenants {before}", flush=True)
        t0 = time.time()
        with open(os.path.join(out, "driver.log"), "a") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            smp = footprint.TreeSampler(p.pid, os.path.join(out, "phase.txt")).start()
            rc = p.wait()
            mem = smp.stop()
    finally:
        if not a.no_lock:
            lock.__exit__()
    runs = []
    rp = os.path.join(out, "runs.jsonl")
    if os.path.exists(rp):
        runs = [json.loads(l) for l in open(rp) if l.strip()]
    res = {"benchmark": "flint-m4pro decode throughput", "impl": impl, "impl_digest": digest,
           "set": a.set, "profile": a.profile, "contexts": contexts, "modes": a.modes.split(","), "repeats": a.repeats,
           "decode_steps": a.steps, "driver_rc": rc, "wall_s": round(time.time() - t0, 1),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "memory_wait": wait,
           "cotenants_before": before, "external_memory": mem, "out_dir": out,
           "host": {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                    "wired_limit_mb": subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True).stdout.strip()}}
    if runs:
        res.update(summarize(runs, mem, before))
    footprint.write_json(os.path.join(out, "result.json"), res)
    print(json.dumps({k: res[k] for k in ("impl_digest", "driver_rc", "wall_s", "valid") if k in res}), flush=True)
    for mode, d in (res.get("modes") or {}).items():
        print(mode, {c: v["median_tok_s"] for c, v in d.items()}, flush=True)
    print(f"[bench] result: {os.path.join(out, 'result.json')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
