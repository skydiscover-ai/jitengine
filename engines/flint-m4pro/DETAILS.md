# Flint on a 48 GB M4 Pro: details

The [README](README.md) has the quick start and the results for Flint, Recursive's int4 mixture-of-experts model. This page has the rest.

- [Install](#install)
- [Server](#server)
- [Options](#options)
  - [Less memory](#less-memory)
- [Results](#results)
- [Why it's fast, in more detail](#why-its-fast-in-more-detail)
- [Measure it yourself](#measure-it-yourself)
- [Tested on](#tested-on)
  - [Other Macs](#other-macs)

## Install

From this directory, with Python 3.14 or newer:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install '.[download]'          # mlx 0.32.2, numpy, transformers, jinja2 and the hf CLI
hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 \
    --local-dir ~/models/flint-int4                                               # 22.6 GB
```

## Server

```bash
jitengine-flint-m4pro serve --model-path ~/models/flint-int4
```

The server listens on `127.0.0.1:8080` (`/v1/chat/completions`, `/v1/completions`, `/health`,
`/metrics`). It runs one request at a time and queues up to 8 more (`--max-queued N`); past the
queue it answers 429. A request field it cannot serve gets a 400 that names it: `n` or `best_of`
above 1, `logit_bias`, `suffix`, `functions`, a `tool_choice` other than `none` or `auto`, or a
`response_format` other than text. Tools go to the chat template, and `tool_choice: "none"` sends
none. `--log-level warning` prints only warnings and errors; `--version` prints the version.

### Coding agents

With `tools`, the reply is read as Recursive's `flint` parsers for SGLang read it:
a `<think>…</think>` that opens it comes back as `reasoning_content` and its calls as OpenAI
`tool_calls`; without tools the reply is the text Flint wrote, as before. The same port serves the
Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI's Responses API
(`/v1/responses`) on the same chat, its reasoning as a thinking block and a reasoning item, so
Claude Code, Codex, OpenCode and Pi run on it ([`launch/`](../../launch) has the setup, and
`jit-launch`), with the context limit `/v1/models` reports as `max_model_len`.

Before it reads a weight, it checks the GPU, the checkpoint's small files against its
`flint-int4.json` manifest, and the safetensors headers against `config.json`, and it sizes the
context limit to the GPU wired limit. `--context-limit auto`, the default, is the number of
tokens the wired limit holds beside the weights and 3 GiB of headroom, at most 262,144, rounded down
to a multiple of 1,024. 262,144 tokens need 31,764 MiB (33.3 GB) and 4,096 need 24,708 MiB; below
4,096 the engine keeps part of its experts on the SSD ([less memory](#less-memory)). On the M4 Pro
with its wired limit lowered to five values from 24,708 to 31,764 MiB (six runs), `auto` chose the predicted
limit each time, and the process stayed under the wired limit, at the same tokens and speed as at
45 GiB ([tested on](#tested-on)). When the wired
limit is not raised, it is the working set Metal recommends: 40.2 GB on a 48 GB M4 Pro, which holds
262,144. The measurements here raised it to 45 GiB (`sudo sysctl iogpu.wired_limit_mb=46080`); a 48 GB Mac does not need this to serve 262,144 tokens. An explicit
`--context-limit N` that does not fit beside the whole model is never truncated: the engine keeps it
with experts on the SSD, or, with `--decode-backend mk`, `--mk-grid` or profile kv8, which need the
whole model, refuses it, and the message gives the most that fits and the `sysctl` value that would
hold N.

`--profile kv8` keeps the 14 full-attention layers' history at 8 bits in groups of 64, as the
checkpoint's config declares, which halves the history read per token at long context. Its outputs
differ slightly from the default's 16-bit history, so its numbers are reported separately.

`--max-output-tokens N` and `--request-deadline-s S`, both off by default: the first caps
`max_tokens`; the second stops a request S seconds after it was submitted and answers it with the
tokens so far (`finish_reason` "length", `x_metrics.deadline_exceeded` true).

`--decode-backend auto|mk|graph`, default `auto`: the megakernel (the persistent kernel that runs a whole decode pass) if its load check passes on
this GPU, else stock MLX ops ([other Macs](#other-macs)). `mk` refuses rather than fall back;
`graph` decodes with stock MLX ops from the start. `--mk-grid G` pins the kernel's grid (a
multiple of 4 from 16 to 64 threadgroups): it is probed at G only and refused if not resident.

A port already in use is refused, also when the other server listens at `0.0.0.0` or `[::]`. The server
binds without `SO_REUSEADDR`, so it cannot sit on `127.0.0.1` beside a live server and take its
local traffic. It rebinds with `SO_REUSEADDR` only when
no live server answers on the port, so a quick restart over a previous server's closed connections
still works.

## Options

`jitengine-flint-m4pro serve --help` lists every option; `doctor --help` and `inspect --help` list
theirs. The subcommands are `serve`, `doctor` and `inspect`.

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path DIR` | required | the checkpoint directory |
| `--host`, `--port` | `127.0.0.1`, `8080` | where `serve` listens |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--context-limit auto\|N` | `auto` | history positions |
| `--memory-budget-gib N`, `--memory-budget-gb N` | the whole model, or the wired limit | the memory the engine may use |
| `--mode auto\|resident\|offload` | `auto` | whether experts stream from the SSD |
| `--max-queued N` | 8 | requests that may wait; more get 429 |
| `--max-output-tokens N`, `--request-deadline-s S` | off | a cap on `max_tokens`, a per-request time limit |
| `--decode-backend auto\|mk\|graph`, `--mk-grid G` | `auto`, tuned | the decode path and the kernel grid |
| `--profile main\|kv8` | `main` | 16-bit or 8-bit history for the full-attention layers |
| `--one-token-per-pass` | off | one committed token per pass, no speculation |
| `--no-prefix-cache` | off | do not keep the last prompt's prefill state |
| `--verify-weights` | off | sha256 every safetensors file at start |
| `--log-level` | `info` | the least severe log line printed |
| `doctor --smoke-tokens N` | 16 | greedy tokens decoded after loading (0: none) |

Environment variables: `FLINT_GRID_CACHE` (the `doctor --tune` cache file). The tests and
benchmarks read `JITENGINE_FLINT_M4PRO_E2E_CHECKPOINT`, `JITENGINE_FLINT_M4PRO_E2E_SHA`,
`JITENGINE_FLINT_M4PRO_E2E_MIN_TOKS` and `JITENGINE_FLINT_M4PRO_TEST_CHECKPOINT`
([tests/README.md](tests/README.md)), and `FLINT_BENCH_DIR`, `FLINT_IMPL_DIR`, `FLINT_MODEL` and
`FLINT_PYTHON` ([benchmarks/README.md](benchmarks/README.md)).

### Less memory

`--memory-budget-gib N` (or `--memory-budget-gb N`, in decimal GB) caps the memory the engine
uses. A budget that holds the whole model, 22.6 GB of weights plus the history and 3 GiB of
headroom, changes only the context limit, which `auto` then sizes to the budget. A smaller budget,
or a Mac whose GPU wired limit cannot hold the weights, runs Flint with its experts offloaded:

- The non-expert weights (1.3 GB) stay in memory, with as many of the 12,032 routed experts (47
  layers of 256, 1.77 MB each) as the budget has room for.
- An expert that is not in memory is read from the checkpoint when the router picks it, in place of
  the one used least lately.
- Decode runs on stock MLX ops: no megakernel and no speculation.
- The context limit defaults to 32,768 tokens; `--context-limit N` sets more, which leaves fewer
  experts in memory.
- The prompt is prefilled layer by layer across its 2,048-token chunks, so each layer's experts
  are read once per prompt rather than once per chunk.

Without a budget, offload takes the GPU wired limit, at most physical memory less 6 GiB: 18 GiB on
a 24 GB Mac. A budget too small for one layer's experts and 4,096 tokens of history, about 3.7 GiB,
is refused with the smallest that fits. Profile kv8, `--decode-backend mk` and `--mk-grid` need the
whole model in memory. `--mode offload` keeps the experts on the SSD even when the whole model
fits, and `--mode resident` refuses rather than offload. The text does not depend on the budget: an expert in the pool is the same
bytes, gathered by the same MLX kernel as when the whole model is in memory, and the unit tests
check the MoE layer's output bit for bit against it.

On a 24 GB M2 (an AWS EC2 Mac) we raised the GPU wired limit to 22 GiB with
`sudo sysctl iogpu.wired_limit_mb=22528`, which makes the default budget 18 GiB. With the default
limit, the default budget is that limit, which `doctor` reports. At 18 GiB the pool keeps 70% of the experts
(8,455 of 12,032). This build's offload mode decodes code at a 1K-token prompt at 8.5 tok/s in 18 GiB and 5.0 tok/s in
12 GiB (median of 3, greedy), measured as the starting point of the [M2 build](../flint-m2/README.md#results); its
README table has the 8K and raw rates.

## Results

Decode speed in tok/s on a 48 GB M4 Pro, for one stream with prefill excluded: 1,024 decode steps
per run, median of 5. In the table, 252K tokens of context is 258,048 tokens.

- The raw column runs one model pass per token: a random prompt and a fixed stream of random tokens fed in,
  as llama-bench measures. The code columns use a real coding prompt (llama.cpp's source, then a task) and
  the model's own continuation, greedy or seeded at the model card's 1.0 / 0.95 / 20.
- ¹ [rec](https://github.com/recursive-oss/rec), Recursive's own inference engine for Flint, release 0.1.0 with its defaults
  (mlx 0.32.0; its kernels are those of commit 8af99ab), the higher of its published chart and our re-measurement
  on the same Mac. From 32K up the chart is the higher; at 252K it shows 22.6 tok/s and we measured 20.4. rec does not guess ahead: its greedy code rate is its raw rate, and its sampled
  rate is lower (123.6 at 1K), since its sampler at these settings leaves its resident kernel. In
  the same session, alternating with rec over three rounds, raw decode is 1.019-1.022× rec's at 1K
  and 1.024-1.026× at 2K, code greedy 1.142-1.144× at 1K, and code sampled 1.264-1.266×.
- ² llama.cpp as plotted in rec's README chart, which does not give the llama.cpp version or flags. Our own run of
  llama.cpp 184f653 gave 75.6 tok/s at 1K and 9.0 at 252K (raw).
- On held-out code prompts (mlx-lm's source), greedy decodes 1.096×, 1.171× and 1.352× rec's rate
  at 1K, 8K and 32K, and sampled 1.132×, 1.156× and 1.270×.
- Every run in the table ran alone on the Mac, with no swap growth and the same tokens on every repeat.
- Peak memory, as the footprint of the engine's whole process tree sampled from outside, is 25.0 GB at 1K and 33.4 GB at
  252K, with no swap (rec: 27.8 and 42.3). `--profile
  kv8` decodes 137.1 / 107.1 / 64.9 / 43.0 tok/s raw at 1K / 32K / 128K / 252K, 1.90× rec at 252K,
  and peaks at 30.5 GB.

The table is in the [README](README.md#results).

## Why it's fast, in more detail

Each decode pass is one persistent Metal kernel over all 49 layers, the head and the token choice. On code, it guesses ahead: when the latest tokens match earlier text in the prompt or the reply, it
takes the tokens that followed there as guesses for the next few, and one pass checks them all. The
rows that check the guesses are bit-identical to one-token decoding, and a guess is kept only when
it equals the token the model picks, so every token in the output is the one plain decoding would
give.

At long context, each new token reads far more history than weights: at 252K tokens, about 7.4 GB
of the 14 full-attention layers' history against 1.45 GB of weights. Like rec, the engine runs each
decode pass as one persistent Metal kernel. Its attention reads each slice of history once, on the
GPU's matrix units, for all eight query heads that share it. In profiling at 128K to 252K,
it streams the history at 251 to 255 GB/s, 97 to 98% of the 259 GB/s measured for the M4
Pro's memory, against about 195 to 200 GB/s estimated from rec's published numbers. The two engines are level at
1K, and the lead grows with the context. On code, it also guesses ahead from earlier text and checks every guess
exactly (above). rec does not, so its greedy code rate equals its raw rate, and on code the 252K lead over that rate
grows to 48%.

Two sets of measurements give the 251 to 255 GB/s:

- The run's phase profile of the kernel measured 252.1 GB/s at 131,584 tokens and 255.0 at 258,560. The full-attention
  layers' attention took 14.96 and 29.07 ms per token.
- A re-run on this package's kernel on 2026-11-03 read 252.4 and 251.3 GB/s with
  `benchmarks/kernels/megakernel.py`.

The re-run's command:

```bash
python benchmarks/kernels/megakernel.py --model ~/models/flint-int4 --L 1536 131584 --quick --T 16 --reps 5 --out mk.json
```

What the run measured along the way:

- A chain of separate kernels needs 293 GPU launches per pass, and those launches cost 2.09 to 2.32 ms even when
  empty. The persistent kernel also draws sampled tokens itself, so sampling adds no round trip to the host; rec's
  sampled rate on code at 1K is 123.56 tok/s against 132.92 greedy.
- In the persistent kernel, 240 simdgroups (6 in each of the 40 threadgroups) split the history, and each computes its eight heads as one small matrix
  multiply. That layout was the largest gain at long context, +11.9% at 32K.
- At short context, rec led at first. Profiling both engines showed rec's lead sat in its expert block and output
  projection, while this engine was already ahead in attention. The coding agents kept that attention, fused the router into
  a rebuilt expert block, and had each stage pass a flag to the next instead of waiting at a barrier, where every
  threadgroup stops until the slowest arrives. Barriers fell from 342 to 150 per pass, and that change drew level with
  rec.


## Measure it yourself

[`benchmarks/`](benchmarks) has the benchmark harness, and its README has the commands:
it builds the prompts from llama.cpp's source, then times each mode and context.
`benchmarks/kernels/megakernel.py` times the megakernel alone, phase by phase.

For all 32 megakernel variants the engine compiles, the unit tests
check the rendered Metal code, comments aside, against the measured build's.

## Tested on

Each start checks the megakernel against its per-phase chain, stock MLX and one-token passes
([other Macs](#other-macs)). On the M4 Pro with the checkpoint, `tests/metal` also checks the
megakernel against its one-dispatch-per-phase chain and stock MLX phase by phase over many passes,
and multi-token decoding against one token per pass.

| Mac | Memory | What ran | Greedy-512 hash | Decode |
|:--|:--|:--|:--|--:|
| M4 Pro, the measured Mac | 48 GB | teacher-forced log-probabilities, greedy fidelity, kv8 fidelity and peak memory without swap at all nine contexts; admission by capability over simulated Macs, the fallback through 15 engines on the GPU, and every resident grid from 16 to 60 against grid 40's tokens and top-20 ids | | the table above |
| M4 Pro, wired limit lowered | 24.1 to 31.0 GiB | the engine at five wired limits (six runs) with `--context-limit auto`: context 4,096 to 262,144 as predicted, 1,024 tokens each, and three refusals | same tokens as at 45 GiB | ×0.9985 to ×1.0025 of 45 GiB |
| M4 Pro, the measured Mac | 48 GB | this package's `tests/unit` and `tests/metal` with the checkpoint (435 passed, 2 skipped) and the end-to-end test, run on 2026-11-03 on the release tree; the end-to-end test gave the same hash on 3 of 3 repeats, with the tuned grid 40, context 262,144 and multi-token decoding on | `22f2f2ab929aa000` | 140.0 tok/s (median of 3), greedy-512 chat |
| M2 (AWS EC2 Mac) | 24 GB | offload at 18 and 12 GiB ([less memory](#less-memory)), measured as the starting point of the [M2 build](../flint-m2/DETAILS.md#tested-on), whose logits (11,019 rows) and greedy tokens match this offload mode at both budgets | | 8.5 and 5.0 tok/s, code at 1K |
| Other Apple silicon Macs | | on the M4 Pro only: their device facts and wired limits simulated (M1 Pro, M3 Pro, M4, M2 Max, M2 Ultra, a binned M4 Pro, an unknown generation), forced compile and probe failures, load-check defects, and every grid from 16 to 60 | not run | unmeasured |

The M4 Pro rows ran with nothing else on the Mac; the memory checks need the Mac to themselves.

In the third row, `tests/unit` and `tests/metal` (the megakernel against its one-dispatch-per-phase
chain, the load check and its fallback with the weights, the engine, identity and the server) passed
and `tests/e2e` gave the same hash on every repeat. An earlier run on 2026-10-28 gave
140.9 to 141.3 tok/s, with a load check that passed every stage in 2.6 to 2.8 s, top-1 agreement on 32
of 32 tokens and a median log-probability gap of 0.59 nat.

The second row lowered the M4 Pro's GPU wired limit (`sudo sysctl iogpu.wired_limit_mb=B`) to
24,708, 25,600, 26,624 (main and kv8), 28,672 and 31,764 MiB and ran one code request per limit
with `auto`, at prompts from 2K to 252K tokens. Every run picked the predicted context limit and
decoded the tokens it decodes at 45 GiB at the same speed, with no swap. At 24,576 MiB the engine
refused in under 0.2 s, before reading a weight, and named `iogpu.wired_limit_mb=24708`. At 31,764
MiB with the full 262,144-token history the process peaks at 31,175 MiB, 589 MiB under the limit.
MLX's buffer cache would otherwise hold 1.2 GB of the prefill's freed buffers, which decode never
reuses, so the engine returns them when the prefill ends and before a shorter last chunk. A
long-lived server pays about 15 ms more for each new 2K prefill. The lowered limit caps the GPU's
memory, not the Mac's RAM, so these runs stand in for a smaller Mac's GPU budget only.

The last row is not a run on another Mac: the engine was given other Macs' facts on this one, and
this M4 Pro's GPU ran every kernel. If you run it on another Mac, send the output of
`jitengine-flint-m4pro doctor --model-path ~/models/flint-int4 --json` and of the end-to-end test with
`JITENGINE_FLINT_M4PRO_E2E_SHA=report` ([tests/README.md](tests/README.md)), and we will add a row.

### Other Macs

Everything that adapts to the Mac happens when the engine loads, and `/health` reports what it
chose (`device`, `mk`, `mk_unavailable`, `context_limit_source`).

- On the M4 Pro the kernel runs the tuned grid, 40 threadgroups (two per GPU core),
  with no search. On any other GPU it starts at two threadgroups per core, a multiple of 4 from 16
  to 64 (40 when the core count is unknown), and steps down by 4 until the residency probe finds a
  grid whose threadgroups are all resident at once.
- Before serving, the kernel is compared with its own one-dispatch-per-phase form
  over 8 passes, bit for bit, and with a repeat of itself; with stock MLX on a fixed text, where
  its top-1 must agree on at least 22 of 32 forced tokens (the median log-probability gap is
  reported, not required); and its verify rows with one-token passes. On the M4 Pro the check takes
  about 2.7 s.
- If the kernel does not build, no grid is resident, or the chain or stock-MLX
  check fails, the default profile decodes with stock MLX ops, with no speculation. In one 64-token greedy request at a short
  prompt it ran at 78.3 tok/s against 137.9 for the tuned kernel (0.57×). One WARNING line and `/health` `mk_unavailable` say which stage failed.
  `--profile kv8` has no stock form and refuses instead, as do `--decode-backend mk` and a pinned
  `--mk-grid`. If only the verify rows differ, speculation is turned off and the kernel stays.
- Every Metal kernel is compiled with math mode "safe" set explicitly, the mode
  the measurements were made under, rather than left to MLX's default; another MLX
  cannot change the arithmetic quietly. An MLX that cannot take the setting is refused at startup.
  Under MLX 0.32.2 the pinned build's outputs are bit-identical to the unpinned one's.

`doctor` loads the engine as `serve` would, decodes 16 greedy tokens, and reports the device, the
grid and how it was chosen, each check stage, the backend, the context limit and the decode. It
exits 0 when the kernel serves with speculation, 1 with stock MLX ops or without speculation, and 2
when the engine refuses.

```bash
jitengine-flint-m4pro doctor --model-path ~/models/flint-int4           # a summary
jitengine-flint-m4pro doctor --model-path ~/models/flint-int4 --json    # the full report on stdout
jitengine-flint-m4pro doctor --model-path ~/models/flint-int4 --tune    # time every resident grid first
```

`--json FILE` writes the report to a file. `--tune` is opt-in: it times every resident grid on this
Mac and caches the fastest in `~/.cache/flint-engine/grid-v1.json` (`FLINT_GRID_CACHE` sets
another file), under this GPU, the kernel's source hash, the profile, MLX and macOS. Later starts on
any Mac but the M4 Pro try that grid first. On the M4 Pro the tuned grid always wins; there 40
measured fastest, with 60 and 56 within about 1% and 44 6% slower. `doctor` takes the same
`--context-limit`, `--memory-budget-gib`, `--mode`, `--profile`, `--decode-backend` and `--mk-grid`
options as `serve`.

`inspect` takes the same options and reports what `serve` would plan on this Mac without reading a
weight: the memory plan, the context limit, the backend and the grid. It exits 0 when `serve` would
start and 2, with the reason, when it would refuse.

```bash
jitengine-flint-m4pro inspect --model-path ~/models/flint-int4 --memory-budget-gib 12
```
