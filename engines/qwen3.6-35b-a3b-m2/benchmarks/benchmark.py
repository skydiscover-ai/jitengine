"""Local coding-agent serving benchmark for an OpenAI-compatible server on a Mac.

Workload (workload.json): one shared ~2.5k-token system prompt (agent rules + an open file) and many
distinct user turns, each with its own ~380-token snippet. Two modes:

  agent        C concurrent streams (default 8), distinct tasks sharing the prefix, sampled,
               max_tokens generated each. Headline = aggregate generated tokens per second of
               wall clock from first dispatch to last completion (prefix warmed first; per-request
               suffix prefill is inside the window, as it is in real agent traffic).
  interactive  one stream, one task, repeated. Headline = decode tokens per second measured from
               the first streamed token to the last (prefill excluded), plus TTFT.

Each mode runs R repeats (default 3) and reports the median. Requests go to /v1/chat/completions by
default (messages + chat_template_kwargs, the shape a real agent sends and the one that lets servers
reuse the shared system prompt); --endpoint completions sends the same prompt as rendered text or
token ids instead. Both render to identical token ids with the checkpoint's template.

Checks built into the client:
  - generated tokens are counted independently: min(server usage, re-tokenized text, max_tokens);
    a usage figure more than 5% (or 8 tokens) away from the re-tokenized text fails the stream;
  - a stream that was not streamed per decode step (chunks < 50% of tokens) fails the stream;
  - identical outputs for different streams of one burst at temperature > 0 fail the run
    (identical output across repeats of the same prompt is reported as a warning);
  - memory is sampled externally (macOS footprint of the whole process tree, which includes Metal
    allocations that ps RSS misses) and the kernel's lifetime peak is recorded;
  - a held-out draw (--heldout) uses only tasks absent from the default draw.

    python benchmark.py --url http://localhost:8080 --model-path ~/models/Qwen3.6-35B-A3B-4bit \
        --mode agent --concurrency 8 --server-pid <pid> --output-json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_WORKLOAD = HERE / "workload.json"


@dataclass
class StreamResult:
    task: str
    ttft_s: float | None
    total_s: float
    chunks: int
    usage_tokens: int | None
    retok_tokens: int
    output_tokens: int  # the count used for scoring
    decode_tps: float
    finish_reason: str | None
    text_sha: str
    max_gap_s: float
    error: str | None


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def messages_for(workload: dict, task: dict) -> list[dict]:
    return [
        {"role": "system", "content": workload["system_prompt"]},
        {"role": "user", "content": task["text"]},
    ]


def render(tok, workload: dict, task: dict) -> tuple[str, list[int]]:
    """(rendered chat text, its token ids) with the checkpoint's template."""
    kw = workload.get("chat_template_args", {})
    text = tok.apply_chat_template(messages_for(workload, task), add_generation_prompt=True, tokenize=False, **kw)
    return text, list(tok(text, add_special_tokens=False)["input_ids"])


def pick_tasks(workload: dict, n: int, task_seed: int, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(task_seed)
    tasks = [t for t in workload["tasks"] if not exclude or t["id"] not in exclude]
    rng.shuffle(tasks)
    if n > len(tasks):
        raise SystemExit(f"only {len(tasks)} eligible tasks, asked for {n}")
    return tasks[:n]


def common_prefix_len(id_lists: list[list[int]]) -> int:
    if not id_lists:
        return 0
    n = 0
    for toks in zip(*id_lists):
        if all(t == toks[0] for t in toks):
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# External memory sampler (never trusts the server's own numbers)
# ---------------------------------------------------------------------------
def footprint_bytes(pid: int) -> tuple[int | None, int | None]:
    """(phys_footprint, phys_footprint_peak) from macOS `footprint -f bytes`. Metal/unified-memory
    allocations are NOT in ps RSS; footprint's IOAccelerator rows are."""
    try:
        out = subprocess.run(["footprint", "-p", str(pid), "-f", "bytes"], capture_output=True, text=True, timeout=20)
    except Exception:
        return None, None
    cur = peak = None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("phys_footprint_peak:"):
            peak = int(s.split(":")[1].strip().split()[0])
        elif s.startswith("phys_footprint:"):
            cur = int(s.split(":")[1].strip().split()[0])
    return cur, peak


def descendants(root: int) -> list[int]:
    """root plus every descendant (any depth), from one ps listing."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True)
    except Exception:
        return [root]
    kids: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [root]
    while stack:
        p = stack.pop()
        seen.append(p)
        stack.extend(kids.get(p, []))
    return seen


class MemorySampler:
    def __init__(self, pid: int | None, interval: float = 1.0):
        self.pid = pid
        self.interval = interval
        self.peak_bytes = 0          # max sampled footprint of the tree during the window
        self.kernel_peak_bytes = 0   # max phys_footprint_peak reported by the kernel (lifetime)
        self.samples = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def sample(self) -> tuple[int, int]:
        cur_total = peak_total = 0
        for p in descendants(self.pid):
            cur, peak = footprint_bytes(p)
            cur_total += cur or 0
            peak_total += peak or 0
        return cur_total, peak_total

    def _record(self):
        cur, peak = self.sample()
        self.peak_bytes = max(self.peak_bytes, cur)
        self.kernel_peak_bytes = max(self.kernel_peak_bytes, peak)
        self.samples += 1

    def _loop(self):
        while not self._stop.is_set():
            self._record()
            self._stop.wait(self.interval)

    def start(self):
        if self.pid is None:
            return
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        if self._t is None:
            return
        self._stop.set()
        self._t.join(timeout=5)
        self._record()


def swap_used_mb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        used = out.split("used =")[1].split()[0]
        return float(used[:-1]) if used.endswith("M") else float(used[:-1]) * 1024
    except Exception:
        return None


# ---------------------------------------------------------------------------
# One streaming request
# ---------------------------------------------------------------------------
async def run_stream(client: httpx.AsyncClient, args, tok, workload: dict, task: dict, text: str,
                     ids: list[int], seed: int | None, max_tokens: int) -> StreamResult:
    if args.endpoint == "chat":
        path = "/v1/chat/completions"
        body = {"model": args.model, "messages": messages_for(workload, task),
                "chat_template_kwargs": workload.get("chat_template_args", {})}
    else:
        path = "/v1/completions"
        body = {"model": args.model, "prompt": ids if args.prompt_format == "ids" else text}
    body.update({
        "max_tokens": max_tokens, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "stream": True, "stream_options": {"include_usage": True},
    })
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    t_first: float | None = None
    t_last: float = t0
    max_gap = 0.0
    chunks = 0
    usage_tokens: int | None = None
    finish: str | None = None
    text_parts: list[str] = []
    error: str | None = None
    try:
        async with client.stream("POST", f"{args.url}{path}", json=body, headers={"content-type": "application/json"},
                                 timeout=httpx.Timeout(connect=30, read=1800, write=120, pool=30)) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                raise RuntimeError(f"http {resp.status_code}: {raw.decode('utf-8', 'replace')[:400]}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in ev.get("choices") or []:
                    piece = ch.get("text")
                    if piece is None and isinstance(ch.get("delta"), dict):
                        piece = ch["delta"].get("content")
                    if piece:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        else:
                            max_gap = max(max_gap, now - t_last)
                        t_last = now
                        chunks += 1
                        text_parts.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                usage = ev.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    out_text = "".join(text_parts)
    retok = len(tok(out_text, add_special_tokens=False)["input_ids"]) if out_text else 0
    # Engine-independent count: never more than the re-tokenized text (+5%/8 for tokenization
    # ambiguity) and never more than max_tokens. A usage figure far from the re-tokenization fails.
    cap = min(int(retok * 1.05) + 8, max_tokens)
    out_tokens = min(usage_tokens, cap) if usage_tokens is not None else min(max(chunks, retok), cap)
    if error is None:
        if usage_tokens is not None and abs(usage_tokens - retok) > max(8, 0.05 * retok):
            error = f"usage.completion_tokens={usage_tokens} disagrees with re-tokenized text={retok}"
        elif out_tokens == 0:
            error = "no tokens generated"
        elif chunks < 0.5 * out_tokens:
            error = f"not streamed per decode step: {chunks} chunks for {out_tokens} tokens"
        elif finish not in (None, "stop", "length"):
            error = f"unexpected finish_reason {finish!r}"
    ttft = (t_first - t0) if t_first is not None else None
    decode_window = max(t_end - (t_first or t0), 1e-6)
    return StreamResult(
        task=task["id"], ttft_s=ttft, total_s=t_end - t0, chunks=chunks, usage_tokens=usage_tokens,
        retok_tokens=retok, output_tokens=out_tokens,
        decode_tps=(out_tokens / decode_window) if out_tokens else 0.0, finish_reason=finish,
        text_sha=hashlib.sha256(out_text.encode()).hexdigest()[:16], max_gap_s=max_gap, error=error,
    )


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


# ---------------------------------------------------------------------------
# One burst
# ---------------------------------------------------------------------------
async def burst(args, client, tok, workload, prompts: list[tuple[dict, str, list[int]]], seed_base: int | None) -> dict:
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        run_stream(client, args, tok, workload, task, text, ids,
                   None if seed_base is None else seed_base + i, args.max_tokens)
        for i, (task, text, ids) in enumerate(prompts)
    ])
    wall = time.perf_counter() - t0
    ok = [r for r in results if r.error is None]
    bad = [r for r in results if r.error is not None]
    total_out = sum(r.output_tokens for r in ok)
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    decs = [r.decode_tps for r in ok if r.decode_tps > 0]
    return {
        "wall_s": wall,
        "total_output_tokens": total_out,
        "aggregate_tps": total_out / wall if wall > 0 else 0.0,
        "completed": len(ok),
        "failed": len(bad),
        "ttft_p50": pct(ttfts, 50) if ttfts else None,
        "ttft_p95": pct(ttfts, 95) if ttfts else None,
        "decode_tps_mean": statistics.mean(decs) if decs else None,
        "decode_tps_p50": pct(decs, 50) if decs else None,
        "mean_output_len": (total_out / len(ok)) if ok else 0,
        "distinct_outputs": len({r.text_sha for r in ok}),
        "streams": [asdict(r) for r in results],
    }


async def run(args) -> dict:
    workload = json.loads(Path(args.workload).read_text())
    tok = load_tokenizer(args.model_path)
    n = args.concurrency if args.mode == "agent" else 1
    exclude = None
    if args.heldout:
        exclude = {t["id"] for t in pick_tasks(workload, n, args.task_seed)}
        tasks = pick_tasks(workload, n, args.task_seed + 1, exclude)
    else:
        tasks = pick_tasks(workload, n, args.task_seed)
    prompts = [(t, *render(tok, workload, t)) for t in tasks]
    prompt_lens = [len(ids) for _, _, ids in prompts]
    all_ids = [render(tok, workload, t)[1] for t in workload["tasks"]]
    prefix_len = common_prefix_len(all_ids)
    print(
        f"[bench] mode={args.mode} endpoint={args.endpoint} streams={n} shared prefix={prefix_len} tok, "
        f"prompt lens={prompt_lens}, out={args.max_tokens}, "
        f"sampling(t={args.temperature},p={args.top_p},k={args.top_k}, seeds={'per-stream' if args.seed is not None else 'none'}) "
        f"repeats={args.repeats} tasks={[t['id'] for t in tasks]}",
        file=sys.stderr,
    )
    sampler = MemorySampler(args.server_pid)
    swap0 = swap_used_mb()
    cap = max(n * 2, 64)
    limits = httpx.Limits(max_connections=cap, max_keepalive_connections=cap)
    reps: list[dict] = []
    async with httpx.AsyncClient(limits=limits) as client:
        if args.warmup:
            # Warm the shared prefix once (and page the weights in) before anything is timed.
            w0 = time.perf_counter()
            task, text, ids = prompts[0]
            w = await run_stream(client, args, tok, workload, task, text, ids, None, min(8, args.max_tokens))
            print(f"[warmup] {time.perf_counter()-w0:.2f}s ttft={w.ttft_s} err={w.error}", file=sys.stderr)
            if w.error:
                raise SystemExit(f"warmup request failed: {w.error}")
        sampler.start()
        for r in range(args.repeats):
            res = await burst(args, client, tok, workload, prompts,
                              None if args.seed is None else args.seed + r * 1000)
            reps.append(res)
            print(
                f"[rep {r}] wall={res['wall_s']:.2f}s out={res['total_output_tokens']} "
                f"agg={res['aggregate_tps']:.2f} tok/s decode/stream p50={res['decode_tps_p50']} "
                f"ttft p50={res['ttft_p50']} failed={res['failed']} distinct={res['distinct_outputs']}/{res['completed']}",
                file=sys.stderr,
            )
        sampler.stop()
    swap1 = swap_used_mb()

    problems: list[str] = []
    failed = sum(r["failed"] for r in reps)
    if failed:
        problems.append(f"{failed} streams failed")
        for r in reps:
            for s in r["streams"]:
                if s["error"]:
                    problems.append(f"  {s['task']}: {s['error'][:160]}")
    warnings: list[str] = []
    if args.temperature > 0:
        # Two streams of one burst are distinct tasks: identical text means a broken sampler or a
        # replay, so that fails. Identical text across repeats of the same prompt is only a warning:
        # a legitimate engine may derive its RNG from the request (mlx-lm's batched path does),
        # so memoization is for the reader to judge (timing, a one-token prompt edit), not this client.
        for i, r in enumerate(reps):
            shas = [s["text_sha"] for s in r["streams"] if not s["error"]]
            if len(set(shas)) < len(shas):
                problems.append(f"repeat {i}: identical outputs for different streams at temperature > 0")
        per_task: dict[str, set[str]] = {}
        for r in reps:
            for s in r["streams"]:
                if not s["error"]:
                    per_task.setdefault(s["task"], set()).add(s["text_sha"])
        same = [t for t, shas in per_task.items() if len(reps) > 1 and len(shas) < len(reps)]
        if same:
            warnings.append(f"identical output across repeats for {len(same)}/{len(per_task)} tasks (request-derived RNG or replay; check before trusting the rate)")
    if swap0 is not None and swap1 is not None and swap1 - swap0 > 256:
        problems.append(f"swap grew by {swap1 - swap0:.0f} MB during the run")

    aggs = [r["aggregate_tps"] for r in reps]
    decs = [r["decode_tps_p50"] or 0.0 for r in reps]
    ttfts = [r["ttft_p50"] or 0.0 for r in reps]
    if args.mode == "agent":
        headline, headline_name = statistics.median(aggs), "aggregate_tok_per_sec"
    else:
        headline, headline_name = statistics.median(decs), "decode_tok_per_sec"
    valid = not problems
    peak_gb = sampler.peak_bytes / 1e9 if sampler.pid else None
    kpeak_gb = sampler.kernel_peak_bytes / 1e9 if sampler.pid else None

    print()
    print("=" * 64)
    print("  MoE-on-Mac coding-agent serving benchmark")
    print("=" * 64)
    print(f"Backend:           {args.url} ({args.endpoint})  model={args.model}")
    print(f"Mode:              {args.mode}  streams={n}  max_tokens={args.max_tokens}  repeats={args.repeats}"
          f"  {'held-out draw' if args.heldout else 'default draw'} task_seed={args.task_seed}")
    print(f"Completed/failed:  {sum(r['completed'] for r in reps)} / {failed}")
    print(f"Aggregate tok/s:   {' '.join(f'{a:.2f}' for a in aggs)}  (median {statistics.median(aggs):.2f})")
    print(f"Decode tok/s p50:  {' '.join(f'{d:.2f}' for d in decs)}  (median {statistics.median(decs):.2f})")
    print(f"TTFT p50 (s):      {' '.join(f'{t:.3f}' for t in ttfts)}")
    print(f"Mean output len:   {statistics.mean(r['mean_output_len'] for r in reps):.1f}")
    if peak_gb is not None:
        print(f"Server footprint:  peak in window {peak_gb:.2f} GB ({sampler.samples} samples); kernel lifetime peak {kpeak_gb:.2f} GB")
    if swap0 is not None and swap1 is not None:
        print(f"Swap used (MB):    {swap0:.0f} -> {swap1:.0f}")
    print()
    for w in warnings:
        print(f"WARNING: {w}")
    if valid:
        print(f"{headline_name} = {headline:.2f}")
    else:
        print(f"{headline_name} = INVALID ({headline:.2f} measured, not a score)")
        for p in problems:
            print("  " + p)
    print()
    result = {
        "config": {
            "url": args.url, "endpoint": args.endpoint, "prompt_format": args.prompt_format, "model": args.model,
            "model_path": args.model_path, "mode": args.mode, "concurrency": n, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
            "task_seed": args.task_seed, "heldout": bool(args.heldout), "repeats": args.repeats,
            "warmup": bool(args.warmup), "tasks": [t["id"] for t in tasks], "prefix_tokens": prefix_len,
            "prompt_tokens": prompt_lens,
        },
        "headline_name": headline_name,
        "headline": headline if valid else None,
        "measured": headline,
        "valid": valid,
        "problems": problems,
        "warnings": warnings,
        "aggregate_tok_per_sec_median": statistics.median(aggs),
        "decode_tok_per_sec_median": statistics.median(decs),
        "ttft_s_median": statistics.median(ttfts),
        "num_failed": failed,
        "server_peak_footprint_gb": peak_gb,
        "server_lifetime_peak_footprint_gb": kpeak_gb,
        "swap_used_mb": [swap0, swap1],
        "repeats_detail": reps,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"Results written to {args.output_json}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None, help="model name in the request body (default: --model-path, which mlx-lm requires)")
    ap.add_argument("--model-path", required=True, help="checkpoint dir (tokenizer + chat template)")
    ap.add_argument("--workload", default=str(DEFAULT_WORKLOAD))
    ap.add_argument("--mode", choices=["agent", "interactive"], default="agent")
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--prompt-format", choices=["text", "ids"], default="text",
                    help="for --endpoint completions: rendered chat text or token-id list")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None,
                    help="per-stream sampling seed base (seed+i, +1000 per repeat). Off by default: a seed makes "
                         "mlx-lm serve the request unbatched, so the baseline never sends one")
    ap.add_argument("--task-seed", type=int, default=0, help="which tasks are drawn")
    ap.add_argument("--heldout", action="store_true", help="draw only tasks absent from the default draw of --task-seed")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--server-pid", type=int, default=None, help="server pid for external memory sampling")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = args.model_path
    if args.server_pid is None and os.environ.get("BENCH_SERVER_PID"):
        args.server_pid = int(os.environ["BENCH_SERVER_PID"])
    res = asyncio.run(run(args))
    sys.exit(0 if res["valid"] else 1)


if __name__ == "__main__":
    main()
