# Flint on a 24 GB M2: details

The [README](README.md) has the quick start and the results for Flint, Recursive's int4 mixture-of-experts model. This page has the rest.

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
hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 \
    --local-dir ~/models/flint-int4                                               # 22.6 GB
```

Put the checkpoint on the internal SSD: every expert the pool (the experts held in memory) lacks is read from it during decode.

## Server

```bash
jitengine-flint-m2 serve --model-path ~/models/flint-int4
```

The server listens on `127.0.0.1:8080` (`/v1/chat/completions`, `/v1/completions`, `/health`, `/metrics`), runs one request at a time and queues up to 8 more (`--max-queued N`); past the queue it answers 429. It prints its memory plan before it reads a weight.

### Coding agents

With `tools`, the reply is read as Recursive's `flint` parsers for SGLang read it: a `<think>…</think>` that opens it comes back as `reasoning_content` and its calls as OpenAI `tool_calls`; without tools the reply is the text Flint wrote. The same port serves the Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI's Responses API (`/v1/responses`) on the same chat, its reasoning as a thinking block and a reasoning item, with the context limit `/v1/models` reports as `max_model_len`. These are the APIs Claude Code, Codex, OpenCode and Pi use ([`launch/`](../../launch) has the setup, and `jit-launch`); we ran Pi and Codex on this engine, on the 24 GB M2.

Within an 18 GiB budget it keeps 73% of the 12,032 routed experts in memory and reads the rest from the SSD as the router picks them. It serves one user at a time.

### Memory budget

- Without `--memory-budget-gib`, the budget is the GPU wired limit, at most physical memory less 6 GiB: 18 GiB on a 24 GB Mac whose wired limit is raised (`sudo sysctl iogpu.wired_limit_mb=22528`), else the limit Metal recommends, which `doctor` reports.
- A smaller budget keeps fewer experts in memory and decodes more slowly (the 12 GiB rows in the table).
- A budget too small for a layer's 256 experts and 4,096 tokens of history, about 3.2 GiB, is refused with the smallest that fits.
- On a Mac whose GPU holds the whole model, the engine runs it in memory, as the M4 Pro build does.

## Options

The table lists the options you would set; `jitengine-flint-m2 serve --help` lists every option (also `--profile`, `--decode-backend`, `--mk-grid` and `--one-token-per-pass`); `doctor --help` and `inspect --help` list theirs. The subcommands are `serve`, `doctor` (loads the engine as `serve` would and reports what this Mac runs) and `inspect` (the plan `serve` would make, without reading a weight).

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path DIR` | required | the checkpoint directory |
| `--host`, `--port` | `127.0.0.1`, `8080` | where `serve` listens |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--memory-budget-gib N`, `--memory-budget-gb N` | the wired limit, at most memory less 6 GiB | the memory the engine may use |
| `--mode auto\|resident\|offload` | `auto` | whether experts stream from the SSD |
| `--context-limit auto\|N` | `auto` (32,768 in offload) | history positions |
| `--max-queued N` | 8 | requests that may wait; more get 429 |
| `--max-output-tokens N`, `--request-deadline-s S` | off | a cap on `max_tokens`, a per-request time limit |
| `--no-prefix-cache` | off | do not keep the last prompt's prefill state |
| `--verify-weights` | off | sha256 every safetensors file at start |
| `--log-level` | `info` | the least severe log line printed |

`FLINT_ANNEX=0` turns the decode annex (extra expert slots that hold experts the pool evicts) off. `FLINT_SPECULATE=0` turns the speculated expert submission off. `FLINT_GPU_KEEPALIVE=0` turns the GPU keep-alive off and `=1` forces it on; unset, it runs in offload mode only. The tests read the variables in [tests/README.md](tests/README.md), and the benchmarks those in [benchmarks/README.md](benchmarks/README.md).

## Results

Decode tokens per second on a 24 GB M2 (an AWS EC2 Mac, checkpoint on its internal SSD), one stream: 1,024 decode steps after an untimed prefill, nothing else running on the Mac. The README table is the median of 5 repeats for this engine and of 3 for the M4 Pro build. **Code** is a real coding prompt and the model's greedy continuation; **raw** is a random prompt and a fixed stream of random tokens, one model pass per token. The [README](README.md#results) has the table.

- Peak memory of the whole process tree is 19.2 GB, under the 18 GiB budget (19.3 GB), and 12.8 GB, under the 12 GiB budget (12.9 GB), with no swap growth.
- Every logits row of three fixed prompts (11,019 rows) is bit-identical to the M4 Pro build's offload mode at both budgets, and greedy decoding gives the same tokens.
- [rec](https://github.com/recursive-oss/rec), Recursive's own inference engine for Flint, needs 27.8 GB at 1K of context and does not fit.
- llama.cpp (e6192de0a, Metal and BLAS, 4 threads) ran Recursive's own GGUF, `RecursiveSI/flint-int4-gguf` `flint-int4-q4_1.gguf` (26.0 GB, revision 9c471510), with `llama-bench -p 0 -n 128` (raw: fixed tokens, 128 decode steps). Among the GPU-layer counts we tried (0, 10, 15, 20, 30, 35 and 40; 45 and more ran out of GPU memory), it was fastest with every layer on the CPU (`-ngl 0`); its 4 threads were not varied. Median of 5, on 2026-11-03: 13.6 tok/s at 1K and 7.35 tok/s at 8K.
  - More GPU layers were slower at 1K: 7.56 tok/s with 20 layers on the GPU, 2.89 with 30 and 1.15 with 40.
  - With 45 or more, it ran out of GPU memory.
  - It reads the model through the file cache, so it may use all 24 GB, while this engine stays within its 18 GiB budget. Its repeats at 1K ranged from 11.6 to 16.1 tok/s, rising as the file cache warmed.
  - Against this engine's raw 28.2 and 24.9 tok/s, that is 2.1× and 3.4×.
- The README's chart is raw decoding at 18 GiB, median of 3, with both builds back to back in one session through the benchmark harness. The table below gives its tok/s. The 32K row ran with the context limit raised to 34,816. The M4 Pro build's speed varies with each context's prompt. The chart's session ran this engine's raw 1K at 26.0 against 28.2 in the run's scored session, so compare builds within one source, not across them.

| Context | This engine | M4 Pro build |
|:--|--:|--:|
| 1K | 26.0 | 12.9 |
| 2K | 26.1 | 14.4 |
| 4K | 25.6 | 14.1 |
| 8K | 23.4 | 12.7 |
| 16K | 21.1 | 13.5 |
| 32K | 17.9 | 11.1 |

## Why it's fast, in more detail

The M4 Pro build already offloads; this build changes how its offload path spends time. At 18 GiB, code at 1K, a token took 116 ms. A profile of the same setting, at 124 ms per token, put 45 ms in blocking reads of missing experts, 16 ms in choosing which expert to evict, and 18.5 ms in extra GPU time that followed the reads. Each change's gain below is the run's own measurement at 18 GiB, code at 1K unless noted.

- Offload decode reaches the GPU in bursts, one evaluation per MoE layer with reads and bookkeeping between them, and under that pattern the M2's GPU holds about 800 of its 1,398 MHz. While a request runs, a helper process streams tiny kernels on its own Metal queue, and the GPU stays at full clock. It shares no memory with the engine, so it cannot change the output. This is the largest gain: 9.3 to 13.6 tok/s.
- A missing expert is nine checkpoint slices. They are read on eight workers at once rather than one after another, and the expert to evict comes from a heap rather than a sort over every slot in the pool: 8.6 to 9.3 tok/s.
- A long prompt touches nearly every expert of every layer; counting that sweep evicted the experts decode keeps returning to. Prefill now refreshes recency only, and the first decode after an 8K prompt is 8 to 12% faster.
- The budget keeps room for a prefill's transient arrays. While decoding, the expert pool opens an annex there: an extra tier of slots that holds experts the pool evicts, so a later miss on one is a memory copy rather than a drive read. Before every prefill or history growth the annex closes and the engine waits until the process tree's footprint is back where it was, so the plan's memory is not exceeded; in the benchmark runs the process tree's footprint stayed under the budget (19.2 GB against 19.3 GB). `FLINT_ANNEX=0` turns it off. With it: +3.0 to +7.1% over an earlier build across code and raw at 1K and 8K; a smaller reserve for MLX's buffer cache later gave the annex about 220 more slots, +2.0 to +2.7%.
- In offload decode, each layer's output is submitted to the GPU as soon as its graph is built (`mx.async_eval`), so the GPU runs that layer's experts while the host builds the next layer up to its router: about +3% in an A/B test on the Mac.
- Each layer's experts are submitted on a device-side copy of the slot map before the host's lookup finishes; when the lookup finds every expert resident, that work is kept, and otherwise the layer is rebuilt on the real slots. Same graph, same kernels, so the output is unchanged: +9.4% on code and +18.0% raw at 1K over an earlier build, in the same session.
- The loader's staging buffer is an anonymous mapping, released after loading, which gave the annex 339 more slots: +6.9% over an earlier build. MLX's buffer cache (a prefill's transients, which decode does not refill) is emptied before the annex is sized: +1.8%. The reserve for MLX's buffer cache went from 384 to 128 MiB (+1.7%) and the prefill headroom from 2 to 1.5 GiB (+1.5%), each checked against the measured peak, which stays under the budget.
- When the speculated submission finds experts missing, only the missing experts' rows are rebuilt on the real slots, not the whole layer: +1.2 to +2.2% at 18 GiB.
- The decode route's integer part (pack each score with its expert id into one key, take the top k, unpack) and the weighted sum of the experts' outputs each run as one compiled graph, fewer launches per layer. Integer ops and the combine's bf16 multiply-add compile to the eager bits (checked on every bf16 pair); float transcendentals do not: a compiled fp32 sigmoid is one unit in the last place off on 1,949 of the 65,280 finite bf16 logits, which changed a raw decode at step 637, so the sigmoid stays eager. +2.5 to +3.5% over the build before, then +1.6 to +2.2% for the integer select: 4 to 6% in all at 18 GiB.

## Measure it yourself

[`benchmarks/`](benchmarks) has the harness behind the table; its README has the commands.

## Tested on

| Mac | Memory | What ran | Greedy-512 hash | Decode |
|:--|:--|:--|:--|--:|
| M2 (AWS EC2 Mac) | 24 GB | the README's table (the run's scored medians), the run's bit-identity check at both budgets, the context curve, the unit and Metal suites (452 passed, 12 skipped, run on 2026-11-03 on the release tree), and the end-to-end test at the default 18 GiB on that run (offload, 8,758 of 12,032 experts in memory, context 32,768) | `46cc7a346d0ee2fb`, 3 of 3 repeats | 22.8 tok/s (median of 3), greedy-512 chat |

The engine is the M4 Pro build with the offload changes above, so everything that build checks at startup ([other Macs](../flint-m4pro/DETAILS.md#other-macs)) applies here too.
