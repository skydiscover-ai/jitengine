"""Does the engine answer exactly as stock mlx-lm does?

Both servers start cold, one after the other, and get the same requests in the same order, one at
a time at temperature 0: the scored and held-out prompts whole and streamed, and a few with
logprobs and top_logprobs. Stock keeps a prompt cache, so its output can depend on the requests
before it; the same order for both keeps that equal. Exit status 0 when every answer is identical.

    python benchmarks/accuracy.py --model-path ~/models/Qwen3-1.7B-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request

import _servers

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOKENS = 128
STREAMED = 3
WITH_LOGPROBS = 3


def prompts() -> list[str]:
    out = []
    for name in ("prompts_scored.txt", "prompts_heldout.txt"):
        out += [line.strip() for line in open(os.path.join(HERE, name)) if line.strip()]
    return out


def post(server, body: dict, stream: bool = False):
    req = urllib.request.Request(
        server.chat_url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=300)
    if not stream:
        return json.load(r)
    chunks = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def collect(arm: str, model_path: str, log_dir: str) -> dict:
    server = _servers.Server(arm, model_path, os.path.join(log_dir, f"{arm}.server.log"))
    try:

        def base(text, **extra):
            return {
                "model": "default_model",
                "temperature": 0,
                "messages": [{"role": "user", "content": text}],
                **extra,
            }

        answers = {}
        texts = prompts()
        for i, text in enumerate(texts):
            reply = post(server, base(text, max_tokens=MAX_TOKENS))
            answers[f"whole-{i}"] = reply["choices"][0]["message"]["content"]
        for i, text in enumerate(texts[:STREAMED]):
            chunks = post(server, base(text, max_tokens=64, stream=True), stream=True)
            answers[f"stream-{i}"] = "".join(
                c["choices"][0].get("delta", {}).get("content") or ""
                for c in chunks
                if c.get("choices")
            )
        for i, text in enumerate(texts[:WITH_LOGPROBS]):
            reply = post(server, base(text, max_tokens=12, logprobs=True, top_logprobs=3))
            content = reply["choices"][0]["logprobs"]["content"]
            answers[f"logprobs-{i}"] = [
                (x["token"], x["logprob"], [(t["token"], t["logprob"]) for t in x["top_logprobs"]])
                for x in content
            ]
        return answers
    finally:
        server.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", help="JSON file for both sets of answers")
    a = ap.parse_args(argv)
    log_dir = tempfile.mkdtemp(prefix="jit-accuracy-")
    stock = collect("stock", a.model_path, log_dir)
    engine = collect("engine", a.model_path, log_dir)
    differing = [k for k in stock if stock[k] != engine[k]]
    print(f"{len(stock)} answers compared; differing: {differing or 'none'}")
    if a.out:
        _servers.dump({"stock": stock, "engine": engine, "differing": differing}, a.out)
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
