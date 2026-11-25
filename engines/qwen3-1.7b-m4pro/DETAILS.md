# Qwen3-1.7B on a 24 GB M4 Pro: details

## Install

You need a Mac with Apple silicon, Python 3.14, and one virtual environment for this engine:
it pins MLX 0.32.4, and the other engines in this repository pin 0.32.2, so the two cannot share one (`install.sh`
gives each build an environment of its own).

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install ../../runtime '.[download]'         # add ,benchmark for the benchmark client, ,dev for the tests
hf download mlx-community/Qwen3-1.7B-4bit --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
    --local-dir ~/models/Qwen3-1.7B-4bit        # about 1 GB
```

`pip` fetches mlx-lm from GitHub at commit `c39139d`, a main-branch snapshot that is not a PyPI
release. No API key is needed.

## Server

```bash
jitengine-qwen3-1.7b-m4pro serve --model-path ~/models/Qwen3-1.7B-4bit
```

The server listens on `127.0.0.1:8080` and serves:

| Path | What |
|:--|:--|
| `POST /v1/chat/completions` | OpenAI chat completions, streamed or whole, with tools, `logprobs` and `top_logprobs` |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages, streamed or whole, with tool use |
| `POST /v1/responses` | OpenAI Responses (stateless: send the whole conversation) |
| `GET /v1/models` | the served model and its `max_model_len`, which `jit-launch` reads |
| `GET /health` | `ok`, or `unavailable` when the generation thread has died |
| `GET /stats` | what was patched, the startup self-check, the token budget, the prompt cache, the power state |

Messages and Responses requests are translated into chat-completions requests and sent to the
server's own chat-completions endpoint over the loopback, so generation, batching and the prompt
cache are the path that was measured. The translators are copied from the Qwen3.6 engine in this
repository. Tool calls use mlx-lm's own parser for Qwen3. A request naming any model other than the
served one gets a 404 and is not fetched from the Hugging Face Hub, which stock mlx-lm would try.
A client that disconnects mid-decode stops its generation; one that leaves during prefill does not
(stock behaviour: the prompt is prefilled first).

## Options

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path` | required | the model directory; config and quantization are checked first |
| `--host`, `--port` | `127.0.0.1`, `8080` | where the server listens |
| `--max-context` | 32768 | tokens one request may hold, prompt plus output; more gets a 400 `context_length_exceeded` |
| `--max-inflight-tokens` | 32768 | padded KV tokens of the running requests (their number × the longest one, prompt plus `max_tokens`); others wait in arrival order |
| `--prompt-cache-gib` | 6 | memory the KV caches of finished (cached) requests may hold (`--prompt-cache-bytes` of mlx-lm); running requests are limited by `--max-inflight-tokens` |
| `--mlx-cache-gib` | 2 | memory MLX keeps for reuse after arrays are freed |
| `--served-model-name` | `mlx-community/Qwen3-1.7B-4bit` | the model name clients send |
| `--chat-template-args` | `{"enable_thinking": false}` | as measured: Qwen3's thinking mode off |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); any other host, or a browser page of another origin, gets 403 (DNS-rebinding protection: the API has no authentication) |
| `--adapter-path` | none | a LoRA adapter; the projections then stay unfused |

`status [--url URL] [--json]` reads `/health` and `/stats` of a running server: tokens in flight and waiting,
the prompt cache, the start-up check and the power state. `doctor [--model-path DIR] [--json]` reports the Mac, whether it is the one the numbers were measured
on, the power source and mode, the installed versions against the measured ones, whether the two
mlx-lm functions the engine edits are the pinned source, and runs the kernel check.

## Environment

The numbers were measured with Python 3.13.5, MLX 0.32.4, transformers 5.20.2, mlx-lm commit
`c39139d70f0798834fc721c28a1aa8e1374ea33c`, macOS 15.1, on an Apple M4 Pro with 24 GB. The model is
`mlx-community/Qwen3-1.7B-4bit` at revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`
(affine, 4 bits, group size 64, bf16 scales).

## Results

Measured on Apple M4 Pro with 24 GB, macOS 15.1, AC power (power mode 0),
Python 3.13.5, MLX 0.32.4, mlx-lm `c39139d`, one sitting right after a restart, with only the terminal
running. Each run starts a fresh server with no warmup, and the two arms alternate who goes first.
Throughput is total tokens over total time. All runs are listed; none was dropped or rerun, and no request
failed in any of them.

**Two sets.** The first set of runs was made while macOS was still indexing after the restart:
`spotlightknowledged` used about one core before almost every c=16 run, and the tool recorded it. Stock was
barely affected (480 to 482 tok/s at 16 requests) but the engine's speed varied from 695 to 763. Once the
indexing had finished the c=16 and guard runs were repeated as a second set. Both are given.

| Workload, set | Engine tok/s, median (range) | Stock tok/s, median (range) | Ratio of medians | Engine total tokens | Stock total tokens |
|:--|--:|--:|--:|:--|:--|
| 16 requests, scored prompts, first set | 727.9 (695–763) | 480.8 (480–482) | 1.51× | 8086–8092 | 8088–8091 |
| 16 requests, scored prompts, second set | 759.0 (741–760) | 474.4 (473–477) | 1.60× | 8085–8092 | 8087–8088 |
| 16 requests, scored prompts, third set (after the memory limits were added) | 762.1 (762–771) | 475.5 (474–480) | 1.60× | 8084–8092 | 8085–8090 |
| 16 requests, held-out prompts, first set | 625.2 (620–633) | 441.8 (438–446) | 1.41× | 7567–7681 | 7675–7686 |
| 16 requests, held-out prompts, second set | 626.1 (610–645) | 437.7 (431–439) | 1.43× | 7678–7705 | 7610–7681 |
| 1 request, first set | 181.0 (181–181) | 157.4 (157–158) | 1.15× | 8083–8083 | 8083–8083 |
| 1 request, second set | 180.0 (180–180) | 154.9 (155–155) | 1.16× | 8083–8083 | 8083–8083 |
| 4 requests, first set | 333.4 (333–333) | 323.8 (323–328) | 1.03× | 8092–8096 | 8090–8095 |
| 4 requests, second set | 332.7 (333–333) | 328.2 (312–328) | 1.01× | 8087–8090 | 8090–8090 |
| 8 requests, first set | 411.4 (408–420) | 366.5 (362–367) | 1.12× | 8088–8090 | 8090–8091 |
| 8 requests, second set | 409.1 (405–428) | 378.7 (368–384) | 1.08× | 8088–8092 | 8086–8092 |
| 32 requests, first set | 827.8 (822–829) | 718.8 (718–719) | 1.15× | 8088–8089 | 8090–8090 |
| 32 requests, second set | 822.4 (819–827) | 728.6 (718–734) | 1.13× | 8087–8090 | 8086–8089 |

The ratio at 16 requests is 1.51× in the first set, 1.60× in the second and 1.60× in the third (`c16-scored-recheck.json`, three runs per arm). The README quotes the range.

**Time to first token** (average, median of runs): 0.05 s for both at 1 request; 0.37 to 0.38 s for the
engine against 0.19 s for stock at 4 requests (the engine's 95th percentile is lower, 0.43 to 0.46 s against 0.61 s);
0.38 to 0.41 s against 0.34 to 0.42 s at 8; 0.45 to 0.47 s against 0.42 to 0.46 s at 16. The engine does not
improve it. Mean request latency is lower than stock's at every concurrency except 4, where it is about equal
(1.50 s against 1.52 to 1.55 s): at 16 requests it is 2.6 to 2.7 s against 4.1 to 4.2 s.

**The speed moved between levels.** At 16 requests the engine's runs fell between 695 and 771 tok/s, and
the runs do not explain why. Stock stayed within 9 tok/s (473 to 482). Three observations: the first set
(a busy `spotlightknowledged` before every run) had the slowest run (695) and also a 763; one second-set run
at 741 had a busy process before it, while its three neighbours with none ran at 759; and in the third set
the 762 and 771 runs had a busy process before them (`softwareupdated` at 86% in one). So a process seen busy before a run does not
sort the runs into fast and slow, and these runs do not support any one cause. Four of the five
second-set runs ran at 759 to 760, the third set at 762 to 771. The SkyDiscover run this engine came from
measured 818 to 820 tok/s in its three delivered runs ([record](benchmarks/results/research-run.md)); that level did not appear in
these runs. The original code, measured in the same way a few days earlier on this Mac, reached 826 once in
ten runs (705 to 826, median 744; from the session record, not stored as a file).

**A third set, after the memory limits.** When the memory limits below were added, the 16-request
workload was measured once more, 3 runs each, on the same Mac: the engine 762.1 tok/s (762 to 771), stock
475.5 (474 to 480), a ratio of medians of 1.60×
(`benchmarks/results/c16-scored-recheck.json`). A few system processes (photo library, software update) were flagged
busy before three of its runs; they are listed with each run.

<details><summary>Every run</summary>


**16 requests, scored prompts, first set** (`benchmarks/results/c16-scored.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 743.2 | 8089 | 0.400 | 2.64 | spotlightknowled 97%, iTerm2 19% |
| 2 | stock | 480.9 | 8091 | 0.468 | 4.13 | spotlightknowled 97%, iTerm2 19% |
| 3 | stock | 482.2 | 8091 | 0.461 | 4.12 | STExtractionServ 80%, signpost_reporte 78%, spotlightknowled 75%, fseventsd 32% |
| 4 | engine | 763.2 | 8086 | 0.451 | 2.59 | spotlightknowled 98%, signpost_reporte 97% |
| 5 | engine | 725.2 | 8092 | 0.485 | 2.71 | spotlightknowled 94%, apfsd 73%, spindump 28% |
| 6 | stock | 480.4 | 8088 | 0.453 | 4.13 | spotlightknowled 98%, apfsd 89% |
| 7 | stock | 480.8 | 8090 | 0.455 | 4.13 | spotlightknowled 91%, apfsd 84% |
| 8 | engine | 727.9 | 8088 | 0.500 | 2.71 | spotlightknowled 88%, apfsd 72% |
| 9 | engine | 694.7 | 8091 | 0.471 | 2.77 | spotlightknowled 97%, apfsd 40% |
| 10 | stock | 480.7 | 8090 | 0.456 | 4.13 | apfsd 98%, spotlightknowled 76%, kernel_task 16% |

**16 requests, scored prompts, second set** (`benchmarks/results/c16-scored-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 759.6 | 8086 | 0.448 | 2.60 | iTerm2 18% |
| 2 | stock | 474.4 | 8088 | 0.423 | 4.17 | none |
| 3 | stock | 474.2 | 8088 | 0.424 | 4.17 | none |
| 4 | engine | 759.0 | 8086 | 0.462 | 2.60 | none |
| 5 | engine | 759.1 | 8085 | 0.448 | 2.61 | none |
| 6 | stock | 473.3 | 8088 | 0.424 | 4.18 | none |
| 7 | stock | 474.5 | 8088 | 0.410 | 4.17 | none |
| 8 | engine | 759.0 | 8088 | 0.449 | 2.61 | none |
| 9 | engine | 741.1 | 8092 | 0.530 | 2.62 | searchpartyusera 98% |
| 10 | stock | 476.5 | 8087 | 0.432 | 4.15 | searchpartyusera 41% |

**16 requests, held-out prompts, first set** (`benchmarks/results/c16-heldout.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 620.0 | 7567 | 0.504 | 2.80 | spotlightknowled 98%, apfsd 88% |
| 2 | stock | 441.8 | 7686 | 0.451 | 4.03 | spotlightknowled 94%, apfsd 82% |
| 3 | stock | 445.8 | 7675 | 0.455 | 4.07 | spotlightknowled 97% |
| 4 | engine | 625.2 | 7658 | 0.531 | 2.84 | spotlightknowled 92% |
| 5 | engine | 633.4 | 7681 | 0.446 | 2.80 | spotlightknowled 98% |
| 6 | stock | 437.7 | 7678 | 0.482 | 4.09 | spotlightknowled 92% |

**16 requests, held-out prompts, second set** (`benchmarks/results/c16-heldout-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 645.1 | 7705 | 0.463 | 2.76 | trustd 29%, syspolicyd 17% |
| 2 | stock | 437.7 | 7681 | 0.491 | 4.10 | XProtectRemediat 85% |
| 3 | stock | 431.4 | 7610 | 0.537 | 4.12 | none |
| 4 | engine | 610.3 | 7678 | 0.608 | 2.85 | none |
| 5 | engine | 626.1 | 7695 | 0.589 | 2.79 | none |
| 6 | stock | 438.5 | 7640 | 0.576 | 4.04 | none |

**1 request, first set** (`benchmarks/results/c1-scored.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 181.1 | 8083 | 0.049 | 0.70 | spotlightknowled 98%, ScreenTimeAgent 33% |
| 2 | stock | 157.4 | 8083 | 0.050 | 0.80 | spotlightknowled 97% |
| 3 | stock | 157.4 | 8083 | 0.050 | 0.80 | spotlightknowled 97% |
| 4 | engine | 181.0 | 8083 | 0.049 | 0.70 | spotlightknowled 98% |
| 5 | engine | 180.7 | 8083 | 0.049 | 0.70 | spotlightknowled 98% |
| 6 | stock | 157.6 | 8083 | 0.050 | 0.80 | spotlightknowled 98% |

**1 request, second set** (`benchmarks/results/c1-scored-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 180.0 | 8083 | 0.049 | 0.70 | iTerm2 19% |
| 2 | stock | 154.9 | 8083 | 0.051 | 0.82 | none |
| 3 | stock | 154.9 | 8083 | 0.050 | 0.82 | none |
| 4 | engine | 179.7 | 8083 | 0.049 | 0.70 | none |
| 5 | engine | 180.4 | 8083 | 0.049 | 0.70 | none |
| 6 | stock | 155.0 | 8083 | 0.050 | 0.81 | none |

**4 requests, first set** (`benchmarks/results/c4-scored.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 333.4 | 8094 | 0.381 | 1.50 | spotlightknowled 99% |
| 2 | stock | 323.5 | 8093 | 0.191 | 1.55 | spotlightknowled 98% |
| 3 | stock | 323.8 | 8095 | 0.190 | 1.55 | spotlightknowled 98% |
| 4 | engine | 333.4 | 8096 | 0.382 | 1.50 | spotlightknowled 98% |
| 5 | engine | 333.1 | 8092 | 0.373 | 1.50 | spotlightknowled 98% |
| 6 | stock | 328.1 | 8090 | 0.187 | 1.52 | none |

**4 requests, second set** (`benchmarks/results/c4-scored-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 332.7 | 8090 | 0.367 | 1.50 | none |
| 2 | stock | 328.2 | 8090 | 0.186 | 1.52 | none |
| 3 | stock | 328.3 | 8090 | 0.186 | 1.52 | none |
| 4 | engine | 332.7 | 8089 | 0.368 | 1.50 | none |
| 5 | engine | 332.6 | 8087 | 0.367 | 1.50 | none |
| 6 | stock | 312.1 | 8090 | 0.272 | 1.60 | none |

**8 requests, first set** (`benchmarks/results/c8-scored.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 411.4 | 8090 | 0.376 | 2.39 | none |
| 2 | stock | 362.4 | 8090 | 0.412 | 2.69 | none |
| 3 | stock | 366.5 | 8091 | 0.419 | 2.68 | none |
| 4 | engine | 407.9 | 8088 | 0.393 | 2.40 | none |
| 5 | engine | 420.3 | 8089 | 0.273 | 2.36 | none |
| 6 | stock | 366.8 | 8091 | 0.421 | 2.68 | none |

**8 requests, second set** (`benchmarks/results/c8-scored-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 404.8 | 8088 | 0.441 | 2.42 | none |
| 2 | stock | 378.7 | 8088 | 0.344 | 2.62 | XProtectRemediat 96% |
| 3 | stock | 383.5 | 8086 | 0.277 | 2.60 | XProtectRemediat 89% |
| 4 | engine | 427.7 | 8092 | 0.322 | 2.30 | XProtectRemediat 35% |
| 5 | engine | 409.1 | 8089 | 0.409 | 2.38 | XProtectRemediat 95% |
| 6 | stock | 368.0 | 8092 | 0.423 | 2.66 | XProtectRemediat 98% |

**32 requests, first set** (`benchmarks/results/c32-scored.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 828.6 | 8089 | 0.806 | 4.70 | none |
| 2 | stock | 717.8 | 8090 | 0.773 | 5.40 | none |
| 3 | stock | 718.8 | 8090 | 0.774 | 5.39 | none |
| 4 | engine | 821.8 | 8088 | 0.816 | 4.74 | none |
| 5 | engine | 827.8 | 8088 | 0.791 | 4.70 | none |
| 6 | stock | 718.8 | 8090 | 0.772 | 5.39 | none |

**32 requests, second set** (`benchmarks/results/c32-scored-second.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |
|--:|:--|--:|--:|--:|--:|:--|
| 1 | engine | 818.8 | 8090 | 0.817 | 4.75 | none |
| 2 | stock | 718.4 | 8089 | 0.772 | 5.39 | none |
| 3 | stock | 733.5 | 8086 | 0.800 | 5.32 | none |
| 4 | engine | 822.4 | 8087 | 0.799 | 4.72 | none |
| 5 | engine | 827.3 | 8087 | 0.811 | 4.71 | WindowServer 32% |
| 6 | stock | 728.6 | 8087 | 0.802 | 5.36 | none |

**c16 scored, third set** (`benchmarks/results/c16-scored-recheck.json`)

| # | arm | tok/s | total tokens | TTFT avg (s) | mean latency (s) | busy before the run |\n|--:|:--|--:|--:|--:|--:|:--|\n| 1 | engine | 771.0 | 8092 | 0.361 | 2.56 | iTerm2 16% |\n| 2 | stock | 475.5 | 8085 | 0.436 | 4.16 | none |\n| 3 | stock | 474.2 | 8088 | 0.412 | 4.17 | none |\n| 4 | engine | 762.1 | 8084 | 0.461 | 2.59 | photolibraryd 17% |\n| 5 | engine | 761.6 | 8086 | 0.463 | 2.59 | softwareupdated 86%, mdwrite 76%, com.apple.photos 47%, MessagesBlastDoo 22%, photolibraryd 21% |\n| 6 | stock | 480.2 | 8090 | 0.456 | 4.14 | mdwrite 93%, com.apple.photos 51%, photolibraryd 30%, MessagesBlastDoo 27%, storagekitd 16% |
</details>

`benchmarks/accuracy.py` found all 36 answers identical to stock
(`benchmarks/results/accuracy.json`); `doctor.json` is the Mac's `doctor --json`.

## Why it's fast, in more detail

**The matrix multiply.** For 13 to 16 rows MLX's `quantized_matmul` runs a tiled kernel: it
dequantizes the weights to bf16, accumulates in fp32 with 8x8x8 matrix products in natural K order,
and, for small output widths, splits K into P partitions it adds in order in fp32. P follows the
output width N alone: 1 for N of 12288 or more, 2 for 6144 to 8192, 4 for 3072 to 4096, 8 for 2048
and 16 for 1024. The engine's kernel reads each weight once and does the same arithmetic in the same
order, splitting K over the grid and adding the partitions in a second small kernel. It covers
bf16, affine 4-bit, group size 64, K in {2048, 6144, 8192}, and the tied output head. Anything else
runs on stock.

**Fusion.** Fusing Q, K and V changes N from 2048 / 1024 / 1024 to 4096, and stock would pick a
different P for the joined width. The kernel therefore takes the unfused widths and applies each
projection's own P to its rows, so a fused call equals three stock calls. For batches above 16 rows
or prefill the fused arrays are sliced back into the original projections and stock runs them,
because stock's joined-width kernel differs from the unfused one by about 1 ulp there.

**Log-probabilities.** The batch step normally computes `logits - logsumexp(logits)` for every row
and the server reads a value from it per token. With greedy sampling the next token is the argmax of
the raw logits, the same token. The engine keeps a lazy handle to the logits and computes the
log-softmax only for a request that set `logprobs` or `top_logprobs`; it compares equal to stock's
when asked. This, not the argmax, is where the saving comes from.

**How it is applied.** The installed mlx-lm is never written. At start-up the engine replaces
`GenerationBatch._step` and `ResponseGenerator._generate` in memory with edited copies, but only if
each function's source hashes to the value recorded for the pinned commit and every edited text
appears exactly once. If not, it leaves the stock function and says why in the log and in `/stats`.

## Correctness

- **Kernel against stock:** `tests/metal/test_qmm_exact.py` compares the kernel with
  `mx.quantized_matmul` bit for bit on Qwen3-1.7B's five decode shapes at 13, 14, 15 and 16 rows with
  several random weight sets, including the fused shapes against three unfused calls, and checks that
  other row counts and shapes are left to stock.
- **Real weights:** `tests/metal/test_fusion_real_model.py` compares the patched model's attention and
  MLP outputs with the unpatched model's on the real checkpoint at 1, 5, 12, 13 to 16, 17 and 32 rows.
- **Start-up check:** the same comparison runs on the real layer-0 weights each time the server loads.
  A shape that differs, or raises, is switched off and a warning is logged. It samples one layer and
  random inputs, so it is a safety net and not a proof for every weight.
- **Answers:** `benchmarks/accuracy.py` sends the scored and held-out prompts to stock and to the
  engine one at a time at temperature 0, whole, streamed and with `logprobs`, and compares: 36 of 36
  identical.
- **Concurrent answers are not required to match stock** token for token: stock's own output depends
  on the batch sizes it runs at, because MLX uses different arithmetic below 13 rows.

## Why the token totals vary

A run's total generated tokens varies with the speed level. Each request ends at its own end-of-sequence
token, and which token that is can depend on the batch sizes the request passed through, because stock MLX
uses different arithmetic at 12 rows and below than at 13 and above. The engine's kernel equals stock at each
batch size, so this is a property of the schedule, which the timing of the 16 clients' arrivals sets.

Across the thirteen engine runs at 16 requests (three sets), the eight that ran at 759 tok/s or more
totalled 8,084 to 8,092 tokens (six of them 8,084 to 8,086, the other two 8,088 and 8,092) and the five
slower ones 8,088 to 8,092.
Stock's thirteen runs totalled 8,085 to 8,091, one of them (8,085) below 8,087. The SkyDiscover run this
engine came from used a floor of 8,087 tokens per run, which its three delivered runs met
([record](benchmarks/results/research-run.md)); six of these thirteen
engine runs (8,084 to 8,086) fell just below it, and one of stock's did. All requests succeeded; on the scored prompts every request returned 122 to 128
tokens, and on the held-out prompts some replies were shorter (as few as 30 tokens; none over 128), and the answers are identical to stock's when one request runs at a time. On the
held-out prompts the engine's totals were 7,567 to 7,705 against stock's 7,610 to 7,686.

## Memory and context limits

The 24 GB Mac runs out of Metal memory at about 16 GiB of MLX's peak. Stock mlx-lm does not protect against
that, and the engine adds limits.

**What goes wrong in stock.** One prompt of 15,000 tokens peaks at 3.1 GiB (the weights, its KV cache and a
little working memory), the same for stock and the engine, in 13 s. Three such prompts arriving together
should peak near 7.5 GiB, but measured 16.1 GiB on unpatched mlx-lm: the generation thread died with a Metal
out-of-memory error, all three requests were lost and the server stayed down. The engine survived the same
wave at 11.7 GiB. So several long prompts at once need 6 to 10 GiB more than their KV caches; the likely cause
is that mlx-lm concatenates the freshly prefilled caches into the running batch, holding old and new copies,
but that is not confirmed. Separately, stock keeps up to 10 finished requests' KV caches with no limit on their
size (a Claude Code session of 15k to 30k-token prompts filled 9.6 GB), and MLX keeps freed buffers for
reuse with a limit as large as the Mac's memory (10 GiB of them at one moment).

**What the engine does.** A request may hold 32,768 tokens, prompt plus output (`--max-context`); more gets a 400
`context_length_exceeded`. The running requests are charged `their number × the longest of them` against 32,768 tokens
(`--max-inflight-tokens`), because mlx-lm keeps a running batch as one left-padded tensor (112 KiB per token);
others wait in arrival order, so two 15k-token prompts run together and a third waits, and while a 20k-token
request runs, a short one waits as well. The KV caches of finished (cached)
requests may hold 6 GiB (`--prompt-cache-gib`; mlx-lm trims the cache when the next request starts, so it can
briefly hold one more entry), and MLX keeps at most 2 GiB of freed buffers (`--mlx-cache-gib`). All four numbers are
proposals, chosen with margin below the measured limit and not tuned. The 16-request workload keeps about 2,400
tokens in flight, so they do not touch it.

| Test on this Mac | Result | MLX peak |
|:--|:--|--:|
| 3 waves of 3 concurrent 15k-token prompts, with the limits | all 9 requests succeeded, the server stayed up | 9.0 GiB |
| 3 waves of 3 concurrent 15k-token prompts, stock mlx-lm, one wave | the server died, all 3 requests lost | 16.1 GiB |
| 1 request of about 20k tokens plus 8 of about 1k, both arrival orders, with the padded-cache budget | all 9 succeeded in each order, the server stayed up; the test asserts the padded peak at most 32,768 tokens and the MLX peak under 14 GiB (the values were not recorded) | < 14 GiB |
| 4 prompts of about 23k tokens, with the limits | queue one after another, all succeed (130 s) | |
| 5 repeats of two Claude Code tasks through `jit-launch claude`, with the limits | 190 responses, all status 200, no thread death | 12.7 GiB |

The printed output behind the first, second and last rows, the 3.1 GiB figure for one prompt and the 11.7 and
15.6 GiB figures above is in [benchmarks/results/step-time-and-memory-log.md](benchmarks/results/step-time-and-memory-log.md); the
third row is asserted by `tests/e2e/test_limits.py`. The 23k-token row, the three-of-four server deaths below and
the 43 s prefill are from the session record and have no stored output. The stock run in the log had MLX's
cache limit set to 2 GiB as well. Stock's out-of-memory death was seen once, in one wave.

Before the 32,768-token in-flight limit was set, the same Claude Code repeats killed the server in three of four
attempts: with stock's unbounded prompt cache, with the 6 GiB cap, and with the 2 GiB MLX cap added. They did not in
the one repeat made afterwards, a small sample. A single 32,000-token prompt prefills in 43 s.

## Measure it yourself

See [benchmarks/README.md](benchmarks/README.md). The client is mlx-lm's server benchmark, unchanged,
with its hash recorded; `run.py` alternates the engine and stock on fresh servers and keeps every
run with the Mac's power state. A MacBook's speed depends on its state: on battery, or with other
apps busy, the same code ran 10% slower.

## Known limits

- One Mac was measured. Other Apple silicon Macs run the engine and check the kernels at start-up.
- MLX 0.32.2 gives different split-K behaviour: four of the five kernel shapes then differ from
  stock, and the start-up check falls back to stock for them.
- A LoRA adapter disables fusion; the exact kernel still applies.
- A client that leaves during prefill is not noticed until prefill ends, as in stock.
- Anthropic and Responses requests are forwarded to the server's own chat endpoint over the loopback. A client
  that leaves while its request waits in the queue or prefills is noticed only when the server first writes
  to it, so that request still runs. A streamed reply sends nothing but its headers until prefill ends (up
  to 43 s for a 32,000-token prompt), so a client with a short read timeout may give up. A client that leaves a
  request that is not streamed is not noticed at all: the reply is generated to the end and holds its share of
  `--max-inflight-tokens` until then.
- A client that stops reading a stream keeps its share of `--max-inflight-tokens` until the server can write the
  rest of the reply, which may be never: there is no write timeout, as in stock.
- A client's own stop sequence ends the reply, but mlx-lm does not say which sequence matched, so an
  Anthropic reply reports `stop_reason: "end_turn"` and `stop_sequence: null` instead of `stop_sequence`.
- `max_tokens` (or `max_completion_tokens`) must be a positive whole number; anything else is a 400.
- The prompt cache and admission limits are for one Mac's memory: two long prompts run together and the rest wait.

## Provenance

`engine/_step_edits.py` holds text of mlx-lm (Copyright Apple Inc., Apache-2.0) that the engine
edits in memory; see NOTICE. The Anthropic and Responses translators are jitengine-runtime's (`runtime/`),
which every engine here serves through. The benchmark client is mlx-lm's.

## Tested on

| Mac | Memory | Configuration | Result |
|:--|:--|:--|:--|
| M4 Pro, the measured Mac | 24 GB | Python 3.13.5, MLX 0.32.4 | see Results |
| M4 Pro (an AWS EC2 Mac) | 48 GB | Python 3.14.10 and 3.13.15, MLX 0.32.4, macOS 26.6.2 | the start-up check passed every shape; 36 of 36 answers equal stock's; 16 at once, scored prompts: 922 against 495 tok/s (1.86×) on 3.14, 920 against 519 (1.77×) on 3.13; held-out prompts 743 against 472 (1.57×); one at a time 188 against 155 (1.21×) |

The 48 GB rows are medians of three runs per arm (two at one request), interleaved and cold as in Results, in one sitting
on 2026-11-20; macOS background services kept the load average at 3.8 to 5.9 throughout, and every run recorded them. The
files are `benchmarks/results/m4pro48-*.json`.

Ran it on another Mac? Send `doctor --json` and the output of `benchmarks/run.py`, and we will add a row.
