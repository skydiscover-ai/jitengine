# Qwen3.6-35B-A3B on a 24 GB M2: details

The [README](README.md) has the quick start and the results. This page has the rest.

- [Install](#install)
- [Server](#server)
- [Options](#options)
- [Results](#results)
- [Why it's fast, in more detail](#why-its-fast-in-more-detail)
- [Measure it yourself](#measure-it-yourself)
- [Tested on](#tested-on)

## Install

From this directory, with Python 3.14 or newer:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install '.[download]'          # mlx 0.32.2, numpy, transformers, jinja2 and the hf CLI
hf download andrevp/Qwen3.6-35B-A3B-3bit-MLX --revision 794408c4b69a90c9a7b96d3fce6949cccf08df85 \
    --local-dir ~/models/Qwen3.6-35B-A3B-3bit                                     # 15.2 GB
jitengine-qwen3.6-35b-a3b-m2 doctor                    # checks the Mac
```

Or convert it yourself with mlx-lm 0.31.3, which `pip install '.[benchmark]'` installs. The quantization is the
same, but the numbers here come from the download:
`mlx_lm.convert --hf-path Qwen/Qwen3.6-35B-A3B -q --q-bits 3 --q-group-size 64`.

## Server

```bash
jitengine-qwen3.6-35b-a3b-m2 serve --model-path ~/models/Qwen3.6-35B-A3B-3bit --profile 24gb-static
```

The server listens on `127.0.0.1:8080` (`/v1/chat/completions`, `/v1/completions`, `/health`,
`/stats`). It prints its memory plan before reading any weight and refuses a plan that does not fit.
`--log-level debug` prints more, and `warning` only what needs attention.

### Coding agents

A chat request's `tools` go to the checkpoint's template, and the calls in the
reply come back as OpenAI `tool_calls`. The same port serves the Anthropic Messages API
(`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI's Responses API (`/v1/responses`) on the
same chat, so Claude Code, Codex, OpenCode and Pi run on it ([`launch/`](../../launch) has the setup, and `jit-launch`). An agent needs more than the 8K default: `--max-context 32768`
raises the most tokens (prompt plus output) a request may hold to 32768. `/v1/models` reports what a request may
hold as `max_model_len`, which the launcher sizes the agent's context to.

Each turn of an agent re-sends the conversation so far. The prefix cache keeps each prompt's snapshot at its last
256-token boundary, and the next turn reads only what is new. A snapshot costs about 20 KB a token plus 61 MiB of
recurrent state. The default 128 MiB holds prompts of about 3K tokens, so for an agent's 8K to 13K-token prompts,
serve with `--prefix-cache-mib 1024`. In four appended turns on the M2, the fourth turn's 10,561-token prompt took
10.7 s with it instead of 47.1 s, with the same reply. The larger cache leaves memory for only part of the model:
`--build-expert-pack` builds a 14 GB expert pack in `~/.cache/jitengine` on the first start (14 GB more disk;
`JITENGINE_QWEN3_6_35B_A3B_M2_PACK_DIR` moves it), and the experts that do not fit stream from it.

### Profiles

`--profile 24gb-static` gives the engine 18 GiB, up to 8 requests at once and 8K context. If macOS cannot free
18 GiB within a minute, `serve` refuses and says how much is missing.

### Less memory, or 4-bit weights

A lower `--memory-budget-gib` streams experts from a pack on the SSD instead, more slowly. At the 18 GiB
`24gb-static` budget, a 4-bit checkpoint (20.4 GB) does not fit in memory and needs a pack:

```bash
hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
    --local-dir ~/models/Qwen3.6-35B-A3B-4bit   # 20.4 GB
jitengine-qwen3.6-35b-a3b-m2 pack --model-path ~/models/Qwen3.6-35B-A3B-4bit --output ~/models/qwen36-4bit.experts
jitengine-qwen3.6-35b-a3b-m2 serve --model-path ~/models/Qwen3.6-35B-A3B-4bit --profile 24gb-static --expert-pack-path ~/models/qwen36-4bit.experts
```

## Options

`jitengine-qwen3.6-35b-a3b-m2 serve --help` lists every option. The ones you would set:

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path` | required | the checkpoint directory |
| `--profile` | none | `24gb-static` (18 GiB, static memory policy, and the plan picks resident or offload) or `24gb-adaptive` (expert offload with an adaptive memory policy) |
| `--config` | none | a JSON configuration file; precedence is defaults, then `--profile`, then `--config`, then flags |
| `--print-config` | off | print the resolved configuration and exit |
| `--host`, `--port` | `127.0.0.1`, `8080` | where the server listens |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--mode` | `auto` | `resident`, `offload` or `auto` |
| `--memory-policy` | `static` | `static` or `adaptive` |
| `--memory-budget-gib` | 18 | the process footprint ceiling in GiB (`--memory-budget-gb` takes decimal GB) |
| `--reserve-system-gib` | 6 | memory left to macOS and other apps |
| `--max-slots` | 8 | concurrent generations |
| `--max-queued` | 16 | waiting requests before the server answers 429 |
| `--max-context` | 8192 | prompt plus output tokens |
| `--prefill-chunk` | 256 | prompt tokens processed per step |
| `--prefix-cache-mib` | 128 | memory for the prefix cache, in MiB |
| `--expert-pack-path` | `~/.cache/jitengine/<checkpoint>-<sha12>.v2.experts` | the expert pack file to read experts from; `<checkpoint>` is the checkpoint directory's name and `<sha12>` the first 12 hex digits of the SHA-256 of its `model.safetensors.index.json` |
| `--build-expert-pack` | off | build a missing or stale expert pack at startup instead of refusing |
| `--preflight-wait` | 60 | seconds to wait for free memory before `serve` refuses |
| `--log-level` | `info` | `debug`, `info` or `warning` |

Other subcommands: `doctor` (checks the Mac), `inspect --model-path M` (checkpoint inventory and
memory plan, without loading weights), `pack` (builds the expert pack) and `status` (health and
memory state of a running server).

Two environment variables say where expert packs live:

- `JITENGINE_QWEN3_6_35B_A3B_M2_PACK_DIR`: the directory for expert packs (default `~/.cache/jitengine`).
- `JITENGINE_QWEN3_6_35B_A3B_M2_EXPERT_PACK`: one expert pack file, in place of the one the directory would name.

## Results

How the [README](README.md#results)'s numbers were measured: decode in tok/s on an M2 Mac,
24 GB, 3-bit, 18 GiB limit (8 agents: total, held-out prompts). The engine scores 42/42 on the
same-output check (`benchmarks/accuracy.py`, see [Measure it yourself](#measure-it-yourself)).

The other servers ran in separate sessions on the same Mac, through the same benchmark client and the same
18 GiB limit. llama.cpp (`9a3b91c`) and Ollama run 3-bit GGUFs of about the same size. SGLang and mistral.rs need
more than 18 GiB, so they have no row.

## Why it's fast, in more detail

The [README](README.md#why-its-fast) has the short version. The M2's memory delivers about 93 GB/s, and stock
MLX's 3-bit weight multiply moved weights at 68 GB/s of it. With 3-bit weights, the M2's GPU spends its time
unpacking them and waiting for work to arrive:

- MLX's 3-bit kernel prepares each input again in every threadgroup. This engine prepares it once, in a small
  kernel ahead of the main one, and decodes each weight byte once. One user gained 2.5%.
- On this GPU, MLX commits a command buffer every 40 operations or 40 MB of declared inputs, and each expert call
  declared the whole 235 MB expert stack, so it committed a buffer on its own. Here each call declares only its own
  expert's weights and a buffer holds up to 100 operations: one user's decode step takes 20.03 ms instead of 22.05,
  for about 1 GB more memory.

The run kept eleven changes, each with outputs bit-identical to the build before. Eight agents' total decode after
each change, on the scored prompts and on held-out prompts (the start, 32.58 tok/s scored, has no held-out run):

| Change, in the order the run kept it | Scored | Held out |
|:--|--:|--:|
| Each input prepared once for the dense 3-bit multiplies, and exact kernels for each request's experts | 34.43 | 34.72 |
| One fused kernel for a single request's experts | 34.62 | 34.93 |
| Exact top-k routing and exact decode attention | 35.85 | 36.09 |
| Exact prefill experts, aligned to segments | 36.80 | 36.96 |
| Waiting prompts read together, layer by layer, in one pass | 37.00 | 36.90 |
| Command-buffer pacing, expert inputs that name one expert, larger prefill tiles | 38.55 | 38.42 |
| Radix argsort and a batched exact sampler | 39.65 | 39.24 |
| Batched-decode rows in one launch | 41.11 | 40.68 |
| Grouped prefill, 8 request slots sharing a 4-row KV pool, prefix protection that grows with reuse | 42.92 | 42.47 |
| Gate and up projections at 16 outputs per simdgroup (a 32-thread warp) | 43.97 | 43.49 |
| One threadgroup per pair of request and expert | 44.90 | 44.36 |

On held-out prompts, eight agents passed llama.cpp's best for eight requests (41.4, 8 slots) at the ninth change.

The coding agents also built multi-token prediction (MTP) with exact verification and measured it here: checking one extra
draft token adds 17 to 19 ms to a 26 ms step (65 to 73%), so it does not pay, and this build leaves it out. The [M4 Pro build](../qwen3.6-35b-a3b-m4pro), where the same check costs about 21%, turns it
on.

## Measure it yourself

[`benchmarks/`](benchmarks) has the client behind every number above (its README lists the file
digests), and the same-output check.

With the server from [Server](#server) running in another terminal:

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-3bit --mode interactive     # 1 user
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-3bit --heldout --repeats 1  # 8 agents
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-3bit --mode interactive \
    --temperature 0 --max-tokens 512 --output-json greedy.json    # text_sha 44e24c591fc59187 on the M2
```

The same-output check compares greedy output with mlx-lm on the same checkpoint. mlx-lm and the engine
do not fit in 24 GB together, so it caches the reference first, then checks the running server:

```bash
python benchmarks/accuracy.py reference --model-path ~/models/Qwen3.6-35B-A3B-3bit
python benchmarks/accuracy.py check --model-path ~/models/Qwen3.6-35B-A3B-3bit --url http://127.0.0.1:8080
```

## Tested on

The engine checks each custom kernel against stock MLX when it loads, and runs stock MLX in place of
any that differs.

| Mac | Memory | Configuration | Greedy-512 hash | Decode, 1 user |
|:--|:--|:--|:--|--:|
| M2 Mac (`applegpu_g14g`), the measured Mac | 24 GB | `24gb-static`, resident | `44e24c591fc59187`, all kernel checks passed | 44.9 tok/s |
| M2 Mac | 24 GB | offload, 18 GiB and 12 GiB | `44e24c591fc59187`, all kernel checks passed | 15.8 and 7.7 tok/s |
| M1 Pro (`applegpu_g13s`) | 32 GB | `24gb-static`, resident | `44e24c591fc59187`, 3 of 3 repeats | 47.6 tok/s |
| M1 Pro | 32 GB | offload, 8 GiB, an earlier build | `c2f1bffd3e568617`, differs from resident (offload then used other kernels) | 5.99 tok/s |

On 2026-11-03 we reran it from the release tree on the 24 GB M2: its unit and Metal tests passed (543 passed, 7
skipped), and the end-to-end test gave `44e24c591fc59187` on 3 of 3 repeats at 45.7 tok/s.

Offload reads experts from disk, so disk speed sets the rate: the M2's offload rows used a cloud
Mac's network volume. The M1 Pro rows come from a reviewer's runs of an earlier build, not re-run on this one. If you run it on another Mac,
send `doctor --json` and the end-to-end test's output, and we will add a row.
