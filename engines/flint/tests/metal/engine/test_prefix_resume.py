"""A prompt resumed from the last prompt's head against the same prompt cold, on the real checkpoint (run alone on
the box; ~25 GB).

    python tests/metal/engine/test_prefix_resume.py --model DIR --out JSON [--mode resident|offload] \
        [--budget-gib N] [--profile main|kv8] [--steps 64]

One engine with the prefix cache. Turn 1 is a prompt of 4,662 ids (the size of Claude Code's first request through
jit-launch); turn 2 is turn 1 plus 300 ids, as an agent's next turn repeats the conversation so far, so it restores
turn 1's head (its rings at 4,096) and prefills from there. An unrelated prompt then replaces the cache, and turn 2
runs again from the start. Oracle, for greedy and seeded sampling: the resumed and the cold turn 2 give identical
token ids and finish reasons, and every step's log-probability and top-20 (ids and values) BIT-identical; the resumed
run reports 4,096 cached prompt tokens and the cold one 0.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys

import pytest

MODES = {
    "greedy": dict(temperature=0.0, top_k=0, top_p=1.0, seed=0),
    "sampled_s20261104": dict(temperature=1.0, top_k=20, top_p=0.95, seed=20261104),
}
TURN1, MORE, HEAD = 4662, 300, 4096


def bits(x) -> str:
    return "none" if x is None else struct.pack(">f", float(x)).hex()


def steps(r) -> list:
    return [[s.token_id, bits(s.logprob), [[int(i), bits(v)] for i, v in s.top]] for s in r.steps]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="resident", choices=("resident", "offload"))
    ap.add_argument(
        "--budget-gib", type=float, default=None, help="memory budget (offload streams the experts)"
    )
    ap.add_argument("--profile", default="main")
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args(argv)
    from jitengine_flint.engine.engine import Engine
    from jitengine_flint.engine.io_struct import GenRequest

    kw = {"profile": a.profile, "memory_mode": a.mode}
    if a.budget_gib:
        kw["memory_budget_bytes"] = int(a.budget_gib * 2**30)
    eng = Engine(a.model, **kw)
    rng = random.Random(4662)
    # code-like repetition, so the decode drafts from the prompt as an agent's does, then copy-hostile ids
    block = [rng.randrange(1000, 150000) for _ in range(331)]
    turn1 = (block * 15)[: TURN1 - 600] + [rng.randrange(1000, 150000) for _ in range(600)]
    turn2 = turn1 + [rng.randrange(1000, 150000) for _ in range(MORE)]
    other = [rng.randrange(1000, 150000) for _ in range(2500)]
    R = {"argv": sys.argv, "health": eng.health(), "cases": [], "ok": True}
    for mname, mode in MODES.items():

        def run(ids, _mode=mode):
            req = GenRequest(
                prompt_ids=ids, max_tokens=a.steps, ignore_eos=True, logprobs=20, **_mode
            )
            return eng.generate(req)

        run(turn1)
        warm = run(turn2)
        run(other)
        cold = run(turn2)
        sw, sc = steps(warm), steps(cold)
        first = next((i for i, (x, y) in enumerate(zip(sw, sc)) if x != y), None)
        ok = (
            warm.error is None
            and cold.error is None
            and warm.token_ids == cold.token_ids
            and warm.finish_reason == cold.finish_reason
            and len(sw) == len(sc)
            and first is None
            and warm.cached_prompt_tokens == HEAD
            and cold.cached_prompt_tokens == 0
        )
        R["ok"] = R["ok"] and ok
        case = {
            "mode": mname,
            "ok": ok,
            "tokens": [len(warm.token_ids), len(cold.token_ids)],
            "cached": [warm.cached_prompt_tokens, cold.cached_prompt_tokens],
            "first_step_diff": first,
            "errors": [warm.error, cold.error],
            "prefill_s": [(r.metrics or {}).get("prefill_s") for r in (warm, cold)],
        }
        R["cases"].append(case)
        print(
            f"[prefix-resume] {mname}: {'IDENTICAL' if ok else 'DIFFERENT'} {json.dumps(case)}",
            flush=True,
        )
        with open(a.out, "w") as f:
            json.dump(R, f, indent=1)
    eng.close()
    print(f"[prefix-resume] ok={R['ok']} cases={len(R['cases'])}", flush=True)
    return 0 if R["ok"] else 1


@pytest.mark.parametrize("mode", ["resident", "offload"])
def test_a_resumed_prompt_is_bit_identical_to_a_cold_one(m4pro_checkpoint, tmp_path, mode):
    out = tmp_path / f"{mode}.json"
    argv = ["--model", m4pro_checkpoint, "--out", str(out), "--mode", mode]
    if mode == "offload":
        argv += ["--budget-gib", "18"]
    assert main(argv) == 0, out.read_text() if out.exists() else "no output"


if __name__ == "__main__":
    sys.exit(main())
