"""Benchmark driver (runs in the MLX venv, inside benchmark.py's external sampler).

Loads the engine only through its public interface: `<impl>/engine.py` imported as `impl_under_test`
with $FLINT_IMPL_DIR = <impl>, then `create_engine(model)` with default options, or, with
--profile <variant>, `create_engine(model, profile=<variant>)` (reported separately; every record carries it).

Per context and mode, R repeats (default 5) of one request each, all through `engine.stream()`:
  code_greedy   chat prompt bench/prompts/<set>/<ctx>.json (its ids must equal the engine's own render
                of its messages), temperature 0, max_tokens 1,025, ignore_eos -> 1 prefill + 1,024 passes;
  code_sampled  same prompt, temperature 1.0 / top_p 0.95 / top_k 20, seed 20261023;
  raw           bench/prompts/raw/<ctx>.json ids, forced_tokens = its 1,024-id decode stream -> the same
                1 prefill + 1,024 passes, every pass returning top-20 + the forced token's logprob.
decode tok/s = 1,024 / (receipt time of step 1,024 - receipt time of step 0), consumer-side
time.perf_counter() as each released step arrives (prefill and step 0 excluded; the same definition as
rec's benchmark). Repeats of one prompt are served from the engine's post-prefill snapshot; the record
carries `cached_prompt_tokens` so the harness can verify it.

Outputs (in --out-dir): runs.jsonl (one record per run), tokens.jsonl (all token ids per run),
steps.jsonl.gz (raw mode: every step's top-20 and forced logprob, from the timed run itself),
meta.json, and phase.txt (the current phase, read by the external sampler).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time

SEED = 20261023


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_impl(impl_dir: str):
    impl_dir = os.path.abspath(impl_dir)
    os.environ["FLINT_IMPL_DIR"] = impl_dir
    spec = importlib.util.spec_from_file_location("impl_under_test", os.path.join(impl_dir, "engine.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["impl_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_request(mod, mode: str, prompt: dict, steps: int):
    GenRequest = mod.GenRequest
    ids = list(prompt["ids"])
    if mode == "raw":
        return GenRequest(prompt_ids=ids, forced_tokens=list(prompt["decode_stream"][:steps]),
                          request_id=f"raw-{len(ids)}")
    if mode == "code_greedy":
        return GenRequest(prompt_ids=ids, max_tokens=steps + 1, temperature=0.0, top_p=1.0, top_k=0,
                          seed=SEED, ignore_eos=True, request_id=f"greedy-{len(ids)}")
    if mode == "code_sampled":
        return GenRequest(prompt_ids=ids, max_tokens=steps + 1, temperature=1.0, top_p=0.95, top_k=20,
                          seed=SEED, ignore_eos=True, request_id=f"sampled-{len(ids)}")
    raise ValueError(mode)


def run_one(engine, mod, mode, prompt, steps, eos):
    req = build_request(mod, mode, prompt, steps)
    t_req = time.perf_counter()
    recv, rel, toks, fsteps = [], [], [], []
    result = None
    for item in engine.stream(req):
        now = time.perf_counter()
        if isinstance(item, mod.GenResult):
            result = item
            break
        recv.append(now)
        rel.append(item.t_release)
        if item.token_id is not None:
            toks.append(int(item.token_id))
        if mode == "raw":
            fsteps.append([item.logprob, [[int(i), float(v)] for i, v in item.top]])
    if result is None or result.finish_reason == "error":
        raise RuntimeError(f"{mode}: request failed: {getattr(result, 'error', None)}")
    n_steps = len(recv)
    if n_steps != steps + 1:
        raise RuntimeError(f"{mode}: released {n_steps} steps, expected {steps + 1} (1 prefill + {steps} passes)")
    win = recv[-1] - recv[0]
    win_rel = rel[-1] - rel[0]
    m = result.metrics or {}
    rec = {"mode": mode, "context": prompt["context"], "prompt_tokens": len(prompt["ids"]),
           "decode_steps": steps, "decode_tok_s": steps / win, "window_s": win,
           "release_decode_tok_s": steps / win_rel if win_rel > 0 else None,
           "first_step_latency_s": recv[0] - t_req, "cached_prompt_tokens": result.cached_prompt_tokens,
           "engine_prefill_s": m.get("prefill_s"), "engine_decode_tok_s": m.get("decode_tok_s"),
           "engine_peak_memory_gb": (m.get("peak_memory_bytes") or 0) / 1e9, "finish_reason": result.finish_reason,
           "profile": m.get("profile"), "kv_bits": m.get("kv_bits"), "completion_tokens": len(result.token_ids),
           "eos_in_output": sum(1 for t in toks if t in eos) if mode != "raw" else None,
           "tokens_sha256": hashlib.sha256(json.dumps(toks).encode()).hexdigest(),
           "first_tokens": toks[:16], "max_gap_ms": max((b - a) for a, b in zip(recv, recv[1:])) * 1e3,
           "wall": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if mode == "raw":
        lps = [s[0] for s in fsteps[:-1]]
        rec["forced_logprob_mean"] = statistics.fmean(lps) if lps else None
        rec["steps_with_top20"] = sum(1 for s in fsteps if len(s[1]) == 20)
    return rec, toks, fsteps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="the prompt directory make_prompts.py wrote ($FLINT_BENCH_DIR/bench/prompts)")
    ap.add_argument("--set", default="code", help="code | heldout (prompt set for the code modes)")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--modes", default="code_greedy,code_sampled,raw")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profile", default="main", help="main | a variant profile name such as kv8 (reported separately)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    phase = os.path.join(a.out_dir, "phase.txt")

    def set_phase(s):
        with open(phase, "w") as f:
            f.write(s)

    set_phase("load")
    mod = load_impl(a.impl)
    t0 = time.perf_counter()
    engine = mod.create_engine(a.model) if a.profile == "main" else mod.create_engine(a.model, profile=a.profile)
    load_s = time.perf_counter() - t0
    health = engine.health()
    eos = set(health.get("stop_token_ids") or [])
    meta = {"impl": os.path.abspath(a.impl), "impl_module_file": mod.__file__, "load_s": load_s,
            "health": health, "python": sys.version.split()[0], "argv": sys.argv}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    log(f"engine loaded in {load_s:.1f}s profile={health.get('profile')} kv_bits={health.get('kv_bits')}")
    if a.profile == "main" and (health.get("profile") != "main" or health.get("kv_bits") != 16):
        raise SystemExit("the main profile must use a 16-bit history")
    if a.profile != "main" and health.get("profile") != a.profile:
        raise SystemExit(f"variant run: the engine declares profile {health.get('profile')!r}, not {a.profile!r}")
    modes = [m for m in a.modes.split(",") if m]
    code_modes = [m for m in modes if m.startswith("code")]
    runs_f = open(os.path.join(a.out_dir, "runs.jsonl"), "a")
    toks_f = open(os.path.join(a.out_dir, "tokens.jsonl"), "a")
    steps_f = gzip.open(os.path.join(a.out_dir, "steps.jsonl.gz"), "at")
    for ctx in [int(c) for c in a.contexts.split(",")]:
        plan = []
        if code_modes:
            p = json.load(open(os.path.join(a.prompts, a.set, f"{ctx}.json")))
            rendered = engine.render_chat(p["messages"])
            if rendered != p["ids"]:
                raise SystemExit(f"{a.set}/{ctx}: the engine's chat render ({len(rendered)} ids) differs from the saved prompt ({len(p['ids'])})")
            plan += [(m, p) for m in code_modes]
        if "raw" in modes:
            plan.append(("raw", json.load(open(os.path.join(a.prompts, "raw", f"{ctx}.json")))))
        for mode, p in plan:
            vals = []
            for rep in range(a.repeats):
                set_phase(f"{ctx}/{mode}/{rep}")
                rec, toks, fsteps = run_one(engine, mod, mode, p, a.steps, eos)
                rec.update(repeat=rep, set=a.set if mode != "raw" else "raw")
                vals.append(rec["decode_tok_s"])
                runs_f.write(json.dumps(rec) + "\n"); runs_f.flush()
                toks_f.write(json.dumps({"mode": mode, "context": ctx, "repeat": rep, "tokens": toks}) + "\n"); toks_f.flush()
                if fsteps:
                    steps_f.write(json.dumps({"context": ctx, "repeat": rep, "steps": fsteps}) + "\n"); steps_f.flush()
                log(f"{ctx} {mode} rep {rep}: {rec['decode_tok_s']:.2f} tok/s (release {rec['release_decode_tok_s']:.2f}) "
                    f"prefill {rec['engine_prefill_s']:.1f}s cached {rec['cached_prompt_tokens']} "
                    f"first {rec['first_tokens'][:6]} sha {rec['tokens_sha256'][:10]}")
            log(f"{ctx} {mode}: median {statistics.median(vals):.2f} tok/s over {len(vals)}")
    set_phase("done")
    runs_f.close(); toks_f.close(); steps_f.close()
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
