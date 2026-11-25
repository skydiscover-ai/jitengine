"""jit-launch bench: one user's decode speed and wait for the first token on the running engine, measured the way the
budget sweep of 2026-11-03 measured every engine on our Macs, so a Mac's numbers compare with ours.

    jit-launch bench [--server URL] [--report] [--json]      (or `jitengine bench`, which starts the engine first)

- decode: a code prompt of about 1K tokens (a Python module to rewrite with type hints), 256 tokens at temperature 0,
  streamed through /v1/chat/completions; one warmup, then three timed runs. Decode is the median of
  (tokens - 1) / (time of the last token - time of the first), and the first token the median wait for it. Each prompt
  starts with a fresh nonce, so no prefix cache helps.
- --report sends the numbers, with this Mac's chip, memory and macOS version and the engine's build and serve
  arguments, to jitengine.ai's registry, after printing exactly what it sends. Nothing is sent without it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jit_launch import __version__

REPORT_URL = "https://jitengine.ai/registry/api/bench"
RUNS, MAX_TOKENS, FUNCS = 3, 256, 11  # FUNCS functions make about 1K prompt tokens, as in the sweep
TIMEOUT_S = 900.0  # a streaming engine on a small budget can take minutes for the first token
CODE = '''def {name}(items, key=None, limit={n}):
    """Return the first `limit` items sorted by `key`, skipping None values."""
    out = []
    for i, item in enumerate(items):
        if item is None:
            continue
        out.append((key(item) if key else item, i, item))
    out.sort()
    return [x[2] for x in out[:limit]]
'''


def prompt(n_funcs: int) -> str:
    """The sweep's code prompt: `n_funcs` small functions to rewrite, after a fresh nonce."""
    body = "\n".join(CODE.format(name=f"pick_{i}", n=i % 9 + 1) for i in range(n_funcs))
    return (
        f"# request {uuid.uuid4().hex}\nHere is a Python module. Rewrite it with type hints and a short docstring "
        f"for every function, starting from the first one.\n\n```python\n{body}\n```"
    )


def stream_chat(
    base: str, model: str, text: str, max_tokens: int, timeout: float = TIMEOUT_S
) -> dict:
    """One streamed chat completion: the wait for the first token, the tokens and the decode rate after it."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = last = None
    chunks, usage = 0, None
    with urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if (
                    d.get("content")
                    or d.get("reasoning_content")
                    or d.get("reasoning")
                    or d.get("tool_calls")
                ):
                    now = time.monotonic()
                    chunks += 1
                    first = first or now
                    last = now
    n = (usage or {}).get("completion_tokens") or chunks
    out = {
        "first_token_s": round(first - t0, 3) if first else None,
        "completion_tokens": n,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
    }
    if first and last and last > first and n > 1:
        out["decode_tok_s"] = round((n - 1) / (last - first), 3)
    return out


def measure(base: str, model: str, runs: int = RUNS) -> dict:
    """One warmup, then `runs` timed decodes; the medians and every run."""
    stream_chat(base, model, prompt(4), 32)
    timed = [stream_chat(base, model, prompt(FUNCS), MAX_TOKENS) for _ in range(runs)]
    decode = [r["decode_tok_s"] for r in timed if r.get("decode_tok_s")]
    first = [r["first_token_s"] for r in timed if r.get("first_token_s") is not None]
    if not decode or not first:
        raise RuntimeError("the engine streamed no tokens to time")
    return {
        "decode_tok_s": round(statistics.median(decode), 2),
        "first_token_s": round(statistics.median(first), 2),
        "runs_decode_tok_s": decode,
        "prompt_tokens": timed[0].get("prompt_tokens"),
        "completion_tokens": timed[0].get("completion_tokens"),
    }


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return None


def mac() -> dict:
    """This Mac's chip ("M4 Pro"), memory in GB as macOS counts it, and macOS version; None where unknown."""
    brand = _sysctl("machdep.cpu.brand_string") or ""
    mem = _sysctl("hw.memsize")
    return {
        "chip": brand.removeprefix("Apple ") if brand.startswith("Apple M") else None,
        "mem_gb": int(mem) // 2**30 if mem and mem.isdigit() else None,
        "macos": platform.mac_ver()[0] or None,
    }


def engine() -> dict:
    """The installed build and the arguments `jitengine start` served it with, from its JITENGINE_HOME."""
    home = Path(os.environ.get("JITENGINE_HOME") or Path.home() / ".jitengine")
    build = args = None
    try:
        for line in (home / "engine.env").read_text().splitlines():
            if line.startswith("ENGINE_CMD="):
                build = line.split("=", 1)[1].strip().strip('"').removeprefix("jitengine-") or None
    except OSError:
        pass
    try:
        args = " ".join((home / "serve.args").read_text().split()) or None
    except OSError:
        pass
    return {"engine": build, "serve_args": args}


def send(payload: dict, url: str = REPORT_URL) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": f"jit-launch/{__version__}"},
    )
    try:
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except HTTPError as exc:  # the registry says why it refused the report
        try:
            reason = json.loads(exc.read()).get("error")
        except ValueError:
            reason = None
        raise RuntimeError(reason or f"HTTP {exc.code}") from exc


def main(argv: list[str], discover) -> int:
    ap = argparse.ArgumentParser(
        prog="jit-launch bench",
        description="Time one user's decode and first token on the running engine, as our budget sweep did.",
    )
    ap.add_argument("--server", default="http://127.0.0.1:8080", help="the engine's URL")
    ap.add_argument("--report", action="store_true", help=f"also send the numbers to {REPORT_URL}")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--report-url", default=REPORT_URL, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        served = discover(a.server)
        print(
            f"Timing {served.model} at {served.origin}: a warmup, then {RUNS} runs of {MAX_TOKENS} tokens",
            file=sys.stderr,
        )
        result = measure(served.origin, served.model)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"jit-launch bench: {exc}", file=sys.stderr)
        return 1
    payload = {
        **mac(),
        **engine(),
        "model": served.model,
        **result,
        "client": f"jit-launch {__version__}",
    }
    if a.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"decode {result['decode_tok_s']} tok/s, first token {result['first_token_s']} s "
            f"({result['prompt_tokens']} prompt tokens; runs {result['runs_decode_tok_s']})"
        )
    if not a.report:
        print("Add --report to send these numbers to jitengine.ai's registry.", file=sys.stderr)
        return 0
    if not payload["chip"] or not payload["mem_gb"]:
        print(
            "jit-launch bench: --report needs an Apple silicon Mac (its chip and memory)",
            file=sys.stderr,
        )
        return 1
    print(f"Sending to {a.report_url}:\n{json.dumps(payload, indent=2)}", file=sys.stderr)
    try:
        answer = send(payload, a.report_url)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"jit-launch bench: the report did not go through: {exc}", file=sys.stderr)
        return 1
    print(answer.get("message") or json.dumps(answer), file=sys.stderr)
    return 0
