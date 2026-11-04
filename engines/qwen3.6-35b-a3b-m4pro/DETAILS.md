# Qwen3.6-35B-A3B on a 48 GB M4 Pro: details

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
hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
    --local-dir ~/models/Qwen3.6-35B-A3B-4bit                                     # 20.4 GB
hf download mlx-community/Qwen3.6-35B-A3B-MTP-4bit --revision 0295b81421bf4d0fccca9a7c0fcfb1418dda3516 \
    --local-dir ~/models/Qwen3.6-35B-A3B-MTP-4bit                                 # 0.5 GB
jitengine-qwen3.6-35b-a3b-m4pro doctor                          # checks the Mac
```

The 4-bit conversion drops the MTP head, so the second download supplies it. The engine looks for
it beside the checkpoint (`…-4bit` next to `…-MTP-4bit`); `mtp_path` in a `--config` file sets
another path.

The model takes 20.4 GB in memory. On a Mac with 40 GiB or more the default budget keeps it there
([tested on](#tested-on)). On a 32 or 36 GB Mac, give it memory less 6 GiB (26 or 30), and on a 32 GB Mac also
raise the GPU wired limit to memory less 2 GiB, as [`install.sh`](../../install.sh) does. We have not run it on a 32 or
36 GB Mac:

```bash
sudo sysctl iogpu.wired_limit_mb=30720     # on a 32 GB Mac; lasts until the next restart
jitengine-qwen3.6-35b-a3b-m4pro serve --model-path ~/models/Qwen3.6-35B-A3B-4bit --memory-budget-gib 26
```

Instead we measured on our 48 GB M4 Pro with its GPU wired limit set to 21.3 GiB (two-thirds of 32 GiB), 27 GiB and
30 GiB. We did not read these limits from real 32 or 36 GB Macs. At the agent setting (`--max-context 32768`) it decoded
166.4 tok/s in memory at 30 GiB and 166.3 at 27 GiB, against 20.1 streaming experts from the SSD at 21.3 GiB. The
[M2 build](../qwen3.6-35b-a3b-m2) decoded 111.0 at 21.3 GiB, in memory.
We ran these on 2026-11-03.

## Server

```bash
jitengine-qwen3.6-35b-a3b-m4pro serve --model-path ~/models/Qwen3.6-35B-A3B-4bit
```

The server listens on `127.0.0.1:8080` (`/v1/chat/completions`, `/v1/completions`, `/health`,
`/stats`). On a Mac with 40 GiB or more it plans a 34 GiB budget and keeps the model in memory. It
prints its memory plan before reading any weight and refuses a plan that does not fit.
`--log-level debug` prints more, and `warning` only what needs attention.

### Coding agents

A chat request's `tools` go to the checkpoint's template, and the calls in the
reply come back as OpenAI `tool_calls`. The same port serves the Anthropic Messages API
(`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI's Responses API (`/v1/responses`) on the
same chat, which is what Claude Code, Codex, OpenCode and Pi speak ([`launch/`](../../launch) has the setup, and `jit-launch`). The tests here drive these APIs with the OpenAI and Anthropic SDKs, not with the agents themselves. An agent needs more than the 8K default: `--max-context 32768`
lets a request hold up to 32768 prompt plus output tokens. `/v1/models` reports what a request may hold as
`max_model_len`, which the launcher sizes the agent's context to.

## Options

`jitengine-qwen3.6-35b-a3b-m4pro serve --help` lists every option. The full table:

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path` | required | the 4-bit checkpoint directory |
| `--host`, `--port` | `127.0.0.1`, `8080` | where the server listens |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--mode` | `auto` | `resident` (model in memory), `offload` (experts read from an expert pack on disk), or `auto` |
| `--memory-policy` | `static` | `static`, or `adaptive` (resizes caches to follow memory pressure) |
| `--memory-budget-gib` | 34 on a Mac with 40 GiB or more, else 18 | ceiling on the server's memory footprint (`--memory-budget-gb` takes decimal GB) |
| `--reserve-system-gib` | 6 | memory left to the OS and other apps |
| `--max-slots` | 8 | concurrent generations |
| `--max-queued` | 16 | waiting requests before the server answers 429 |
| `--max-context` | 8192 | prompt plus output tokens per request |
| `--prefill-chunk` | 256 | prompt tokens per prefill step |
| `--prefix-cache-mib` | 1152 on a Mac with 40 GiB or more, else 768 | cache of shared prompt prefixes |
| `--expert-pack-path` | `~/.cache/jitengine/<checkpoint>-<sha12>.v2.experts` | the expert pack file `offload` mode reads; `<checkpoint>` is the checkpoint directory's name and `<sha12>` the first 12 hex digits of the SHA-256 of its `model.safetensors.index.json` |
| `--build-expert-pack` | off | build a missing or stale expert pack at startup instead of refusing |
| `--preflight-wait` | 60 | seconds `serve` waits for free memory before it refuses |
| `--profile`, `--config` | none | a packaged profile or a JSON file of the same settings |
| `--print-config` | off | print the resolved settings and exit |
| `--log-level` | `info` | `debug`, `info`, `warning`, ... |

Other subcommands: `doctor` (checks the Mac), `inspect --model-path M` (reads the checkpoint headers
and prints the memory plan), `pack --model-path M` (builds the expert pack for `offload` mode) and
`status` (health and memory of a running server).

### Profiles

- `relaxed` trades exactness for speed, at oMLX's level: verify kernels that are not
  row-exact, two drafts per round, and probabilistic acceptance for sampled requests. Its outputs can
  differ from plain decoding, and it fails the same-output check (see [Measure it yourself](#measure-it-yourself)), as oMLX's MTP mode does.
- `24gb-static` and `24gb-adaptive` are for Macs with 24 GB (an 18 GiB budget and a 768 MiB
  prefix cache). `24gb-adaptive` always reads experts from an expert pack on disk and lets the
  server resize its caches as memory pressure changes; `24gb-static` keeps the sizes fixed and lets
  the plan pick `resident` or `offload`.

### Environment variables

Two environment variables say where expert packs live:

- `JITENGINE_QWEN3_6_35B_A3B_M4PRO_PACK_DIR`: the directory of expert packs, one per checkpoint (default `~/.cache/jitengine`).
- `JITENGINE_QWEN3_6_35B_A3B_M4PRO_EXPERT_PACK`: one expert pack file, in place of the one `PACK_DIR` would name.

## Results

How the [README](README.md#results)'s numbers were measured: decode in tok/s on an M4 Pro, 48 GB,
4-bit, with the model in memory (1 user: greedy, 512 tokens; 8 agents: total, on the benchmark's
prompts and on held-out prompts).

- One user's speed depends on whether the reply repeats its prompt, which MTP and prompt-copy drafts
  turn into accepted tokens: for the exact version, 172 to 179 tok/s when it does and about 125 when
  it does not.
- The other servers ran in other sessions, through the same benchmark. Four sessions
  were interleaved with oMLX's VLM engine with MTP, each measuring one user's sampled reply to the benchmark's `dedent`
  prompt. The mean of the four session medians was 185.8 tok/s for the relaxed version and 180.4 for oMLX. The README
  table's 225.1 and 217.2 come from the greedy 512-token reply instead.

## Why it's fast, in more detail

The [README](README.md#why-its-fast) has the short version.

- Reading the weights one token needs, 1.10 GB shared by every token and 0.57 GB of experts, takes about 6.4 ms at
  the 254 to 262 GB/s a large matrix-vector multiply reaches on this Mac, and a one-token pass takes 11.1 ms. A right
  guess adds a second token for about 2.3 ms more, a fifth of a pass.
- With two tokens at once, the exact checking kernels are limited by arithmetic, at about 2.6 instructions per
  multiply-add. Exactness has that price, which `--profile relaxed` shows.
- The coding agents read oMLX's server logs and saw that most of its lead on interactive replies came from copying
  text out of the prompt. They wrote the same idea into this engine from scratch, with no oMLX code.
- Checking a second token adds about 21% to a decode pass (13.4 against 11.1 ms).
- The engine drafts with the model's MTP head and checks the draft in the next pass, committing
  about 1.85 tokens per pass. When a reply repeats earlier context, drafts copied from that context
  add more.
- The verify kernels keep MLX's per-row arithmetic, so every output is bit-identical to plain
  decoding. MLX's own multi-row kernels are faster but not row-exact, and only `--profile relaxed`
  uses kernels like them: MLX's below 6 rows and its own from 6.
- The [M2 build](../qwen3.6-35b-a3b-m2) this one started from leaves MTP out, because there the
  check costs 65 to 73% of a step.

## Measure it yourself

[`benchmarks/`](benchmarks) has the client behind every number above, and the same-output check. The benchmark needs the server running. The same-output check first needs a reference from mlx-lm (`python benchmarks/accuracy.py reference --model-path ~/models/Qwen3.6-35B-A3B-4bit`, run once per checkpoint, before the server starts), then the running server:

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-4bit                    # 8 agents
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-4bit --heldout --repeats 1  # held out
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-4bit --mode interactive \
    --temperature 0 --max-tokens 512 --output-json greedy.json    # text_sha 44d9aee27f3f890e on the M4 Pro
python benchmarks/accuracy.py check --model-path ~/models/Qwen3.6-35B-A3B-4bit             # same-output check
```

## Tested on

The engine checks each custom kernel against stock MLX when it loads, and runs stock MLX in place of
any that differs.

| Mac | Memory | Configuration | Greedy-512 hash | Decode, 1 user |
|:--|:--|:--|:--|--:|
| M4 Pro, the measured Mac | 48 GB | resident, 4-bit, exact | `44d9aee27f3f890e`, 3 of 3 repeats | 194.6 tok/s |
| M2 (AWS EC2 Mac) | 24 GB | Metal suite: 279 passed | no e2e | |
| M1 Pro | 32 GB | Metal suite: 186 of 189 passed (see below) | no e2e | |

With its default budget the 4-bit model stays in memory only from 40 GiB, so the 24 GB M2 and the 32 GB M1 Pro ran
the Metal tests, not the end-to-end test. On the M1 Pro, three tests failed because the second pass of
two-pass attention launched 1024 threads; it now launches 256, a fix not yet re-run on an M1 Pro. The M2 count is from an earlier commit's tree, and the M1 Pro count
is a reviewer's report. Neither was re-run for this tree. On 2026-11-03 we reran it from the release tree on the 48 GB
M4 Pro: its unit and Metal tests passed (569 passed, 7 skipped), and the end-to-end test gave `44d9aee27f3f890e` on 3
of 3 repeats at 195.1 tok/s.
If you run it on another Mac, send `doctor --json` and the end-to-end test's output, and we will add a row.
