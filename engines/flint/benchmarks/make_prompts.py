"""Build the benchmark prompt sets once, so every system measured uses identical ids.

  python make_prompts.py [code heldout raw]   # writes $FLINT_BENCH_DIR/bench/prompts/

$FLINT_BENCH_DIR defaults to ~/flint-bench; the corpora are read from its repos/ directory.

code/<ctx>.json     Flint chat template; fixed coding-agent system line; user turn = real source code from
                    repos/llama.cpp (src/ then ggml/src/, C/C++ files, sorted path order, '// file: <path>'
                    headers) truncated at a line boundary so the rendered prompt is ctx +-16 tokens, then the
                    fixed task.  Deterministic.
heldout/<ctx>.json  same construction over repos/mlx-lm mlx_lm/*.py ('# file: <path>' headers). Confirmation only.
raw/<ctx>.json      seeded uniform random ids from the regular vocabulary [0, 150000) (no special tokens),
                    exactly ctx ids, plus the externally fixed 1,024-id decode stream (separate seed).
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from transformers import AutoTokenizer

RUN = os.path.abspath(os.path.expanduser(os.environ.get("FLINT_BENCH_DIR") or "~/flint-bench"))
OUT = os.path.join(RUN, "bench", "prompts")
MODEL = os.path.expanduser(os.environ.get("FLINT_MODEL", "~/models/flint-int4"))
CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 258048]
TOL = 16
SYSTEM = "You are a coding agent. You read the user's source code carefully and make precise, well-explained changes."
TASK = "Rewrite the last function above with clearer names and comments, then explain the change."
SEED = 20261023
REGULAR_VOCAB = 150000          # ids 150000..150143 are added/special tokens (tokenizer_config.json)
DECODE_STEPS = 1024
CORPORA = {
    "code": {"repo": "repos/llama.cpp", "dirs": ["src", "ggml/src"],
             "ext": (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"), "header": "// file: {}\n"},
    "heldout": {"repo": "repos/mlx-lm", "dirs": ["mlx_lm"], "ext": (".py",), "header": "# file: {}\n"},
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def corpus_lines(spec):
    root = os.path.join(RUN, spec["repo"])
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines, files = [], []
    for d in spec["dirs"]:
        paths = []
        for dp, _, fns in os.walk(os.path.join(root, d)):
            for fn in fns:
                if fn.endswith(spec["ext"]):
                    paths.append(os.path.relpath(os.path.join(dp, fn), root))
        for rel in sorted(paths):
            text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
            if not text.endswith("\n"):
                text += "\n"
            files.append((rel, len(lines)))
            lines.append(spec["header"].format(rel))
            lines.extend(text.splitlines(keepends=True))
    return commit, lines, files


def render(tok, lines, n):
    user = "".join(lines[:n]) + "\n" + TASK
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    return msgs, text, list(ids)


def build_code(tok, name, spec, manifest):
    commit, lines, files = corpus_lines(spec)
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    cache = {}

    def n_tok(n):
        if n not in cache:
            cache[n] = len(render(tok, lines, n)[2])
        return cache[n]

    for ctx in CONTEXTS:
        lo, hi = 1, len(lines)
        if n_tok(hi) <= ctx:
            raise SystemExit(f"{name}: corpus too small for {ctx}")
        while lo < hi:                        # largest n with n_tok(n) <= ctx
            mid = (lo + hi + 1) // 2
            if n_tok(mid) <= ctx:
                lo = mid
            else:
                hi = mid - 1
        n = min((lo, lo + 1), key=lambda k: abs(n_tok(k) - ctx))
        msgs, text, ids = render(tok, lines, n)
        if abs(len(ids) - ctx) > TOL:
            raise SystemExit(f"{name} {ctx}: closest line boundary gives {len(ids)} tokens (> +-{TOL})")
        last_file = max((f for f in files if f[1] < n), key=lambda f: f[1])[0]
        rec = {"set": name, "context": ctx, "n_tokens": len(ids), "corpus_repo": spec["repo"], "corpus_commit": commit,
               "corpus_dirs": spec["dirs"], "corpus_lines_used": n, "last_file": last_file,
               "system": SYSTEM, "task": TASK, "messages": msgs, "ids": ids}
        js = json.dumps(rec).encode()
        open(os.path.join(d, f"{ctx}.json"), "wb").write(js)
        open(os.path.join(d, f"{ctx}.txt"), "w").write(text)
        manifest[f"{name}/{ctx}"] = {"n_tokens": len(ids), "delta": len(ids) - ctx, "lines": n, "last_file": last_file,
                                      "ids_sha256": sha(json.dumps(ids).encode()), "json_sha256": sha(js)}
        print(name, ctx, len(ids), len(ids) - ctx, n, last_file, flush=True)


def build_raw(tok, manifest):
    d = os.path.join(OUT, "raw"); os.makedirs(d, exist_ok=True)
    for ctx in CONTEXTS:
        prompt = np.random.default_rng([SEED, ctx, 0]).integers(0, REGULAR_VOCAB, size=ctx).tolist()
        stream = np.random.default_rng([SEED, ctx, 1]).integers(0, REGULAR_VOCAB, size=DECODE_STEPS).tolist()
        rec = {"set": "raw", "context": ctx, "n_tokens": ctx, "seed": [SEED, ctx, 0], "decode_stream_seed": [SEED, ctx, 1],
               "vocab_range": [0, REGULAR_VOCAB], "generator": "numpy.random.default_rng(seed).integers(0, 150000, size)",
               "ids": prompt, "decode_stream": stream}
        js = json.dumps(rec).encode()
        open(os.path.join(d, f"{ctx}.json"), "wb").write(js)
        open(os.path.join(d, f"{ctx}.txt"), "w").write(tok.decode(prompt))
        manifest[f"raw/{ctx}"] = {"n_tokens": ctx, "ids_sha256": sha(json.dumps(prompt).encode()),
                                  "decode_stream_sha256": sha(json.dumps(stream).encode()), "json_sha256": sha(js)}
        print("raw", ctx, flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    import transformers
    manifest = {"model_dir": MODEL, "tokenizer_json_sha256": sha(open(os.path.join(MODEL, "tokenizer.json"), "rb").read()),
                "chat_template_sha256": sha(open(os.path.join(MODEL, "chat_template.jinja"), "rb").read()),
                "transformers": transformers.__version__, "numpy": np.__version__, "contexts": CONTEXTS, "tolerance": TOL,
                "system": SYSTEM, "task": TASK, "seed": SEED, "decode_steps": DECODE_STEPS,
                "builder": "make_prompts.py", "sets": {}}
    sets = sys.argv[1:] or ["code", "heldout", "raw"]
    for name in sets:
        if name == "raw":
            build_raw(tok, manifest["sets"])
        else:
            build_code(tok, name, CORPORA[name], manifest["sets"])
    path = os.path.join(OUT, "manifest.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({k: v for k, v in manifest.items() if k != "sets"})
    old.setdefault("sets", {}).update(manifest["sets"])
    json.dump(old, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
