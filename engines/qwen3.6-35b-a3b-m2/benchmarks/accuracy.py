"""Greedy token-exact accuracy check against mlx-lm on the same checkpoint.

The reference is mlx-lm (`mlx_lm.generate`, greedy, one prompt at a time) on the same
checkpoint, so "correct" is well defined: same weights, same quantization, same tokenizer. The
reference is generated live once per checkpoint and cached under .data/ (keyed by a hash of the
checkpoint's config and weight index), because the reference and the engine cannot both be resident
in 24 GB at the same time.

    python accuracy.py reference --model-path ~/models/Qwen3.6-35B-A3B-3bit
    python accuracy.py check --model-path ~/models/Qwen3.6-35B-A3B-3bit --url http://127.0.0.1:8080

Prompt sets (accuracy_prompts.json):
  single_prompts       run one at a time: checks the model itself (weights, DeltaNet, attention, MoE)
  shared_prefix_sets   prefix + branch; each branch run alone for the reference. Checked twice
                       against the engine: solo, and with all branches of a set fired concurrently,
                       which must not change any branch's output (batch invariance)
  long_prefix          the benchmark's real ~2k-token system prompt + several tasks (the
                       benchmark's shape), greedy: proves the full prefix is attended on the benchmarked path

Policy. bf16 numerics differ between batch sizes, prefill chunkings and kernels, so near-tie tokens
can legitimately flip (mlx-lm's own batched server flips them against mlx-lm's single-stream
output). Therefore:
  - the `single` group must match token for token (confident prompts, no excuse), with one
    exception: an EXACT tie, where the reference's own top-1 and the engine's token carry bitwise
    equal logprobs and the reference therefore prefers neither. No --delta slack applies in this
    group, so "close enough" still fails; only a position the reference itself cannot decide is
    tolerated (`single/0` is such a tie).
  - elsewhere a divergence passes only if the engine's token is a near-tie in the reference's own
    stored top-k logprobs (gap to the reference's top-1 <= --delta nats); any other divergence fails;
  - over the whole suite, >= --min-matched of the reference tokens must be matched before the first
    divergence, so an engine that is "slightly wrong everywhere" cannot hide behind near-ties.
Exit 0 on pass, 1 on failure, 2 if the check could not run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "accuracy_prompts.json"
DATA = HERE / ".data"
WORKLOAD = HERE / "workload.json"


def model_hash(model_path: str, max_new: int, long_new: int) -> str:
    """Keyed by the checkpoint AND by everything that defines the cases (prompt file, workload,
    generation lengths), so editing any of them invalidates the cached reference."""
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        p = Path(model_path) / name
        if p.exists():
            h.update(p.read_bytes())
    for p in (PROMPTS, WORKLOAD):
        if p.exists():
            h.update(p.read_bytes())
    h.update(f"{max_new}/{long_new}".encode())
    return h.hexdigest()[:16]


def chat_text(tok, msgs, **kw) -> str:
    """The rendered chat-template text (what a server tokenizes)."""
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


def text_ids(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def build_cases(model_path: str, max_new: int, long_new: int) -> list[dict]:
    """Every case is {id, prompt_ids, max_tokens, group}. Prompt ids are what both sides see."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    spec = json.loads(PROMPTS.read_text())
    cases: list[dict] = []
    for i, p in enumerate(spec["single_prompts"]):
        cases.append(
            {
                "id": f"single/{i}",
                "group": "single",
                "prompt_text": p,
                "prompt_ids": text_ids(tok, p),
                "max_tokens": max_new,
            }
        )
    for s in spec["shared_prefix_sets"]:
        for j, b in enumerate(s["branches"]):
            cases.append(
                {
                    "id": f"shared/{s['name']}/{j}",
                    "group": f"shared/{s['name']}",
                    "prompt_text": s["prefix"] + b,
                    "prompt_ids": text_ids(tok, s["prefix"] + b),
                    "max_tokens": max_new,
                }
            )
    if WORKLOAD.exists():
        wl = json.loads(WORKLOAD.read_text())
        for t in wl["tasks"][: spec.get("long_prefix_tasks", 4)]:
            msgs = [
                {"role": "system", "content": wl["system_prompt"]},
                {"role": "user", "content": t["text"]},
            ]
            text = chat_text(tok, msgs, **wl.get("chat_template_args", {}))
            cases.append(
                {
                    "id": f"long/{t['id']}",
                    "group": "long",
                    "prompt_text": text,
                    "prompt_ids": text_ids(tok, text),
                    "max_tokens": long_new,
                }
            )
    return cases


# ---------------------------------------------------------------------------
# reference: mlx-lm greedy, one prompt at a time
# ---------------------------------------------------------------------------
def cmd_reference(args) -> int:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    cases = build_cases(args.model_path, args.max_new, args.long_new)
    model, tok = load(args.model_path)
    sampler = make_sampler(temp=0.0)
    out = {
        "model_hash": model_hash(args.model_path, args.max_new, args.long_new),
        "model_path": args.model_path,
        "max_new": args.max_new,
        "long_new": args.long_new,
        "cases": [],
    }
    for c in cases:
        ids: list[int] = []
        finish = "length"
        tops: list[list[list[float]]] = []  # per position: [[token_id, logprob], ...] top-K
        for r in stream_generate(
            model, tok, mx.array(c["prompt_ids"]), max_tokens=c["max_tokens"], sampler=sampler
        ):
            lp = r.logprobs
            k = min(args.topk, lp.shape[-1])
            idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[..., :k]
            vals = mx.take_along_axis(lp, idx, axis=-1)
            pairs = sorted(zip(idx.tolist(), vals.tolist()), key=lambda p: -p[1])
            tops.append([[int(t), float(v)] for t, v in pairs])
            if r.token in tok.eos_token_ids:
                finish = "stop"
                break
            ids.append(r.token)
        text = tok.decode(ids, skip_special_tokens=True)
        out["cases"].append(
            {
                **c,
                "ref_ids": ids,
                "ref_text": text,
                "ref_finish": finish,
                "ref_top": tops[: len(ids)],
            }
        )
        print(f"[ref] {c['id']:28s} {len(ids):3d} tok  {text[:60]!r}", file=sys.stderr)
    DATA.mkdir(exist_ok=True)
    p = DATA / f"reference-{out['model_hash']}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"reference written: {p} ({len(cases)} cases)")
    return 0


# ---------------------------------------------------------------------------
# check: an OpenAI-compatible server, greedy, solo and concurrent
# ---------------------------------------------------------------------------
async def complete(
    client, url: str, prompt, max_tokens: int, model: str, stream: bool = False
) -> tuple[str, list[int] | None]:
    """`prompt` is the text (default) or the token-id list (--prompt-format ids). Greedy, no seed
    (a seed makes mlx-lm serve the request unbatched, which would void the concurrent check)."""
    import httpx

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "stream": stream,
    }
    timeout = httpx.Timeout(connect=30, read=1800, write=60, pool=30)
    if not stream:
        r = await client.post(f"{url}/v1/completions", json=body, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"http {r.status_code}: {r.text[:300]}")
        ch = r.json()["choices"][0]
        return ch.get("text", ""), ch.get("token_ids") or ch.get("tokens")
    parts: list[str] = []
    async with client.stream("POST", f"{url}/v1/completions", json=body, timeout=timeout) as resp:
        if resp.status_code != 200:
            raw = await resp.aread()
            raise RuntimeError(f"http {resp.status_code}: {raw.decode('utf-8', 'replace')[:300]}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    parts.append(ch["text"])
    return "".join(parts), None


def classify_divergence(
    ref_top: list[list[list[float]]], ref_ids: list[int], n: int, got_text: str, tok, delta: float
) -> tuple[str, float | None]:
    """At the first divergent position n, is the engine's next token a near-tie in the reference's
    own top-k (logprob gap to the reference's top-1 <= delta)? Returns (kind, gap):
    kind = "near-tie" | "real" | "unknown" (no logprobs stored)."""
    if not ref_top or n >= len(ref_top):
        return "unknown", None
    matched_text = tok.decode(ref_ids[:n], skip_special_tokens=True).lstrip()
    rest = got_text[len(matched_text) :]
    top = ref_top[n]
    best = top[0][1]
    for tid, lp in top:
        if tid == ref_ids[n]:
            continue
        piece = tok.decode([tid], skip_special_tokens=True)
        if piece and rest.startswith(piece) and best - lp <= delta:
            return "near-tie", best - lp
    return "real", None


def compare(ref_text: str, ref_ids: list[int], got_text: str, tok) -> tuple[bool, int, int]:
    # A server stops at EOS and never renders it; the reference ids may end with an EOS id, so the
    # reference text is re-rendered without special tokens for the comparison.
    ref_text = tok.decode(ref_ids, skip_special_tokens=True)
    """(match, first_divergent_token_index, matched_tokens). Compared on decoded text: the engine's
    text (leading whitespace normalized) must start with the reference text rendered without
    special tokens. Shorter output is a mismatch; the reference already stops at EOS."""
    # Servers commonly strip the leading whitespace of the first streamed token when rendering text
    # (mlx-lm does); that is presentation, not a decode difference, so leading whitespace is
    # normalized on both sides before comparing.
    got_text = got_text.lstrip()
    ref_text = ref_text.lstrip()
    if got_text == ref_text:
        return True, -1, len(ref_ids)
    # locate divergence in token space of the reference
    n = 0
    for k in range(1, len(ref_ids) + 1):
        pre = tok.decode(ref_ids[:k], skip_special_tokens=True).lstrip()
        if got_text.startswith(pre):
            n = k
        else:
            break
    return False, n, n


async def cmd_check_async(args) -> int:
    import httpx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    p = DATA / f"reference-{model_hash(args.model_path, args.max_new, args.long_new)}.json"
    if not p.exists():
        print(
            f"no reference for this checkpoint at {p}; run `accuracy.py reference` first",
            file=sys.stderr,
        )
        return 2
    ref = json.loads(p.read_text())
    cases = ref["cases"]
    # The prompt text is rebuilt from the spec (tokenizer only), and must tokenize to the cached ids.
    fresh = {c["id"]: c for c in build_cases(args.model_path, args.max_new, args.long_new)}
    for c in cases:
        f = fresh.get(c["id"])
        if f is None or f["prompt_ids"] != c["prompt_ids"]:
            print(
                f"cached reference no longer matches the prompt spec at {c['id']}; regenerate it",
                file=sys.stderr,
            )
            return 2
        c["prompt_text"] = f["prompt_text"]
    fails: list[str] = []
    rows: list[dict] = []
    key = "prompt_ids" if args.prompt_format == "ids" else "prompt_text"
    total_ref = 0
    total_matched = 0

    def judge(c: dict, mode: str, got: str) -> tuple[bool, str]:
        """Exact match passes. Otherwise a divergence passes only outside the `single` group AND only
        when it is a near-tie (gap <= --delta) in the reference's stored top-k. Matched-token
        counts feed the suite-level coverage requirement (--min-matched)."""
        nonlocal total_ref, total_matched
        ok, div, matched = compare(c["ref_text"], c["ref_ids"], got, tok)
        total_ref += len(c["ref_ids"])
        total_matched += matched
        if ok:
            rows.append(
                {"id": c["id"], "mode": mode, "ok": True, "matched": matched, "kind": "exact"}
            )
            return True, f"[{mode[:4]}] PASS {c['id']:28s} exact {matched}/{len(c['ref_ids'])}"
        kind, gap = classify_divergence(
            c.get("ref_top") or [], c["ref_ids"], div, got, tok, args.delta
        )
        # An EXACT tie (gap 0.0: the reference's own top-1 and the engine's token carry bitwise
        # equal logprobs) is the one divergence the `single` group tolerates. No --delta slack
        # applies there, so the suite still fails any engine that is merely close.
        if kind == "near-tie" and gap == 0.0:
            kind = "exact-tie"
        allowed = (kind == "near-tie" and c["group"] != "single") or kind == "exact-tie"
        rows.append(
            {
                "id": c["id"],
                "mode": mode,
                "ok": allowed,
                "diverge_at": div,
                "matched": matched,
                "kind": kind,
                "gap": gap,
                "got": got[:200],
            }
        )
        tag = "PASS" if allowed else "FAIL"
        return allowed, (
            f"[{mode[:4]}] {tag} {c['id']:28s} diverged at {div}/{len(c['ref_ids'])}: {kind}"
            + (f" (gap {gap:.2f} nats)" if gap is not None else "")
        )

    async with httpx.AsyncClient() as client:
        # 1) every case solo
        for c in cases:
            try:
                got, _ = await complete(client, args.url, c[key], c["max_tokens"], args.model)
            except Exception as exc:  # noqa: BLE001
                fails.append(f"{c['id']} solo: {exc}")
                rows.append({"id": c["id"], "mode": "solo", "ok": False, "error": str(exc)})
                continue
            ok, line = judge(c, "solo", got)
            print(line, file=sys.stderr)
            if not ok:
                fails.append(
                    f"{c['id']} solo: {line.split(': ', 1)[-1]}: got {got[:80]!r} want {c['ref_text'][:80]!r}"
                )
        # 2) each shared-prefix set and the long set, all branches concurrently
        groups: dict[str, list[dict]] = {}
        for c in cases:
            if c["group"] != "single":
                groups.setdefault(c["group"], []).append(c)
        for g, cs in groups.items():
            try:
                outs = await asyncio.gather(
                    *[complete(client, args.url, c[key], c["max_tokens"], args.model) for c in cs]
                )
            except Exception as exc:  # noqa: BLE001
                fails.append(f"group {g} concurrent: {exc}")
                rows.extend(
                    {"id": c["id"], "mode": "concurrent", "ok": False, "error": str(exc)}
                    for c in cs
                )
                continue
            for c, (got, _) in zip(cs, outs):
                ok, line = judge(c, "concurrent", got)
                print(line, file=sys.stderr)
                if not ok:
                    fails.append(f"{c['id']} concurrent: {line.split(': ', 1)[-1]}")
        # 3) the benchmark's shape, streamed: the long-prefix set concurrently with stream=True. The path
        #    that is timed (streaming, batched, prefix-cached) must be the path that is exact.
        cs = groups.get("long", [])
        if cs:
            try:
                outs = await asyncio.gather(
                    *[
                        complete(client, args.url, c[key], c["max_tokens"], args.model, stream=True)
                        for c in cs
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                fails.append(f"group long streaming: {exc}")
                outs = []
            for c, (got, _) in zip(cs, outs):
                ok, line = judge(c, "stream", got)
                print(line, file=sys.stderr)
                if not ok:
                    fails.append(f"{c['id']} streaming: {line.split(': ', 1)[-1]}")
    coverage = total_matched / total_ref if total_ref else 0.0
    if coverage < args.min_matched:
        fails.append(
            f"suite coverage {coverage:.3f} below --min-matched {args.min_matched}: too many early divergences"
        )
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(
                {
                    "reference": str(p),
                    "delta": args.delta,
                    "min_matched": args.min_matched,
                    "coverage": coverage,
                    "rows": rows,
                    "fails": fails,
                },
                indent=1,
            )
        )
    print()
    passed = sum(1 for r in rows if r.get("ok"))
    exact = sum(1 for r in rows if r.get("kind") == "exact")
    print(
        f"accuracy: {passed}/{len(rows)} checks passed ({exact} exact, {passed - exact} near-tie), "
        f"{len(fails)} failures, matched-token coverage {coverage:.3f}"
    )
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--model-path", required=True)
    r.add_argument("--max-new", type=int, default=32)
    r.add_argument("--long-new", type=int, default=48)
    r.add_argument(
        "--topk",
        type=int,
        default=5,
        help="top-k logprobs stored per position for near-tie classification",
    )
    c = sub.add_parser("check")
    c.add_argument("--model-path", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8080")
    c.add_argument(
        "--model",
        default=None,
        help="model name in the request body (default: --model-path, which mlx-lm requires)",
    )
    c.add_argument("--output-json", default=None)
    c.add_argument("--max-new", type=int, default=32)
    c.add_argument("--long-new", type=int, default=48)
    c.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="max logprob gap (nats) for a divergence to count as a near-tie outside the single group",
    )
    c.add_argument(
        "--min-matched",
        type=float,
        default=0.75,
        help="required fraction of reference tokens matched before first divergence, over the whole suite",
    )
    c.add_argument(
        "--prompt-format",
        choices=["text", "ids"],
        default="text",
        help="send the prompt as text (what mlx-lm's server accepts) or as a token-id list",
    )
    args = ap.parse_args()
    if args.cmd == "reference":
        sys.exit(cmd_reference(args))
    if args.model is None:
        args.model = args.model_path
    sys.exit(asyncio.run(cmd_check_async(args)))


if __name__ == "__main__":
    main()
