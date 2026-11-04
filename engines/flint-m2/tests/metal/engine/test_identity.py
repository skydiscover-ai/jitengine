"""Multi-token identity on long streams, through the engine's public API (run alone on the box; ~25 GB).

    JITENGINE_FLINT_M2_TEST_CHECKPOINT=~/models/flint-int4 FLINT_BENCH_DIR=~/flint-bench \
        pytest tests/metal/engine/test_identity.py -s
    python tests/metal/engine/test_identity.py --out <json> --model DIR --bench-dir DIR [--contexts 1024,8192] \
        [--steps 1024] [--profile kv8]

<bench dir>/bench/prompts holds the code prompts benchmarks/make_prompts.py builds.

For every (prompt, mode) the same request runs twice on one engine: multi-token (the default: drafts + verify
passes) and GenRequest.one_token_per_pass=True (the reference: one M = 1 pass per token). Oracle: identical token
ids, finish_reason, step count, and every step's log-probability and top-20 (ids and values) BIT-identical.
Prompts: the benchmark's code prompts (copy-friendly) and a copy-hostile random-id prompt; modes: greedy and seeded
sampling at 1.0 / 0.95 / 20. Also records the speculation statistics and both decode rates.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import pytest

MODES = {
    "greedy": dict(temperature=0.0, top_k=0, top_p=1.0, seed=0),
    "sampled_s20261023": dict(temperature=1.0, top_k=20, top_p=0.95, seed=20261023),
    "sampled_s1": dict(temperature=1.0, top_k=20, top_p=0.95, seed=1),
}


def bits(x) -> str:
    return "none" if x is None else struct.pack(">f", float(x)).hex()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--bench-dir",
        default=os.environ.get("FLINT_BENCH_DIR", ""),
        help="benchmark directory (holds bench/prompts)",
    )
    ap.add_argument("--contexts", default="1024,8192")
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument(
        "--random", type=int, default=600, help="copy-hostile random-id prompt length (0 = skip)"
    )
    ap.add_argument(
        "--profile", default="main", help="engine profile (main | kv8): both runs of a pair use it"
    )
    a = ap.parse_args(argv)
    import random

    from jitengine_flint_m2.engine.engine import Engine
    from jitengine_flint_m2.engine.io_struct import GenRequest

    eng = Engine(a.model, profile=a.profile)
    R = {"argv": sys.argv, "health": eng.health(), "cases": [], "ok": True}
    prompts = {}
    for c in a.contexts.split(","):
        if c:
            prompts[f"code/{c}"] = (
                json.load(
                    open(
                        os.path.join(
                            os.path.expanduser(a.bench_dir), "bench", "prompts", "code", f"{c}.json"
                        )
                    )
                )["ids"],
                True,
            )
    if a.random:
        rng = random.Random(600)
        prompts[f"random/{a.random}"] = (
            [rng.randrange(1000, 150000) for _ in range(a.random)],
            False,
        )
    for pname, (ids, ignore_eos) in prompts.items():
        for mname in a.modes.split(","):
            kw = MODES[mname]
            runs = {}
            for otp in (None, True):
                req = GenRequest(
                    prompt_ids=ids,
                    max_tokens=a.steps + 1,
                    ignore_eos=ignore_eos,
                    logprobs=20,
                    one_token_per_pass=otp,
                    **kw,
                )
                r = eng.generate(req)
                runs[otp] = r
                m = r.metrics or {}
                print(
                    f"[identity] {pname} {mname} otp={otp}: {len(r.token_ids)} tokens, finish {r.finish_reason}, "
                    f"{m.get('decode_tok_s')} tok/s, passes {m.get('decode_passes')}, tpp {m.get('tokens_per_pass')}",
                    flush=True,
                )
            a_, b_ = runs[None], runs[True]
            sa = [
                [s.token_id, bits(s.logprob), [[int(i), bits(v)] for i, v in s.top]]
                for s in a_.steps
            ]
            sb = [
                [s.token_id, bits(s.logprob), [[int(i), bits(v)] for i, v in s.top]]
                for s in b_.steps
            ]
            first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
            ok = (
                a_.token_ids == b_.token_ids
                and a_.finish_reason == b_.finish_reason
                and len(sa) == len(sb)
                and first is None
                and a_.error is None
                and b_.error is None
            )
            R["ok"] = R["ok"] and ok
            case = {
                "prompt": pname,
                "mode": mname,
                "ok": ok,
                "tokens": len(a_.token_ids),
                "finish": [a_.finish_reason, b_.finish_reason],
                "steps": [len(sa), len(sb)],
                "first_step_diff": first,
                "errors": [a_.error, b_.error],
                "multi": {
                    k: (a_.metrics or {}).get(k)
                    for k in (
                        "decode_tok_s",
                        "decode_passes",
                        "verify_passes",
                        "tokens_per_pass",
                        "passes_by_rows",
                        "accepted_hist",
                        "drafted_tokens",
                        "host_ms_per_pass",
                    )
                },
                "one": {k: (b_.metrics or {}).get(k) for k in ("decode_tok_s", "decode_passes")},
            }
            if first is not None:
                case["diff"] = {"multi": sa[first][:2], "one": sb[first][:2]}
            R["cases"].append(case)
            print(
                f"[identity] {pname} {mname}: {'IDENTICAL' if ok else 'DIFFERENT'} {json.dumps(case)[:600]}",
                flush=True,
            )
            with open(a.out, "w") as f:
                json.dump(R, f, indent=1)
    eng.close()
    print(f"[identity] ok={R['ok']} cases={len(R['cases'])}", flush=True)
    return 0 if R["ok"] else 1


@pytest.mark.parametrize("profile", ["main", "kv8"])
def test_multi_token_identity(m4pro_checkpoint, tmp_path, profile):
    run = os.path.expanduser(os.environ.get("FLINT_BENCH_DIR", ""))
    contexts = ("1024", "8192")
    if not run or not all(
        os.path.isfile(os.path.join(run, "bench", "prompts", "code", f"{c}.json")) for c in contexts
    ):
        pytest.skip("no code prompts: build them with benchmarks/make_prompts.py (FLINT_BENCH_DIR)")
    out = tmp_path / f"{profile}.json"
    argv = [
        "--model",
        m4pro_checkpoint,
        "--out",
        str(out),
        "--bench-dir",
        run,
        "--profile",
        profile,
    ]
    import gc

    import mlx.core as mx

    try:
        status = main(argv)
    finally:  # the next test in this process loads its own 23 GB of weights
        gc.collect()
        mx.clear_cache()
    assert status == 0, json.loads(out.read_text())["cases"]


if __name__ == "__main__":
    raise SystemExit(main())
