# Qwen3-1.7B on a 24 GB M4 Pro

A serving engine for [Qwen3-1.7B (4-bit)](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit) on
one Mac, an Apple M4 Pro with 24 GB, tuned for one workload: **16 requests in flight at once**, each
decoding a few hundred tokens. It is stock [mlx-lm](https://github.com/ml-explore/mlx-lm)'s server
with four changes that were found by SkyDiscover and checked against stock MLX (the matrix multiply bit for bit). It speaks
the OpenAI chat-completions and Responses APIs and Anthropic's Messages API, so coding agents such as
Claude Code can use it.

> **Status: alpha, work in progress.** Measured on one Mac. This is not a claim of production
> readiness or of support for other Macs.

## Quick start

On a Mac with 24 GB of memory or more, one command installs it, its model and a `jitengine` command:

```bash
curl -fsSL https://jitengine.ai/install.sh | bash -s -- qwen3-1.7b
jitengine start      # serve it on port 8080, in the background
```

By hand, you need Python 3.14 and about 1 GB of disk for the model. From this directory:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install ../../runtime '.[download]'
hf download mlx-community/Qwen3-1.7B-4bit --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
    --local-dir ~/models/Qwen3-1.7B-4bit
jitengine-qwen3-1.7b-m4pro serve --model-path ~/models/Qwen3-1.7B-4bit
```

Or skip the download: `serve` with no `--model-path` fetches the pinned checkpoint into the Hugging Face cache, and
`--model-path` also takes a Hugging Face id (`org/name` or `org/name@revision`).

Then point any OpenAI client at `http://127.0.0.1:8080/v1`:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

To drive it with Claude Code, run `jitengine claude`, or [`jit-launch claude`](../../launch) in another terminal
after installing by hand.
`jitengine-qwen3-1.7b-m4pro doctor` checks the Mac, the installed packages and the kernels, and `status` shows a running server.

## Results

Throughput in tokens per second (total tokens over total time), 4-bit, 64 requests of up to 128 tokens each,
cold server, on a 24 GB M4 Pro on AC power. Each cell is the median of 3 to 5 runs; a range spans the sets of runs (three for 16 requests at once, two for the rest), the later ones
made after macOS background indexing had finished. "Stock" is the mlx-lm commit this engine is built on, unpatched, measured in the same sitting
with the runs interleaved.

| Server | 16 requests at once | 16 at once, held-out prompts | 1 | 4 | 8 | 32 |
|:--|--:|--:|--:|--:|--:|--:|
| mlx-lm (stock) | 474–481 | 438–442 | 155–157 | 324–328 | 367–379 | 719–729 |
| **This engine** | **728–762** | 625–626 | 180–181 | 333 | 409–411 | 822–828 |
| Engine ÷ stock | 1.51–1.60× | 1.41–1.43× | 1.15–1.16× | 1.01–1.03× | 1.08–1.12× | 1.13–1.15× |

The gain is for the workload it is tuned for: about 1.5 to 1.6 times stock at 16 requests in flight
(about 1.4 times on prompts it was not tuned on). With a single user it is about 15%, and at 4 requests it
is within noise of stock. Time to first token is not improved: equal to stock at 1, 8 and 16
requests, and at 4 requests the average is higher (0.37 s against 0.19 s) with a lower 95th percentile.
At 16 requests the engine's speed moved between levels from run to run (695 to 771 tok/s across all
its runs) while stock barely moved (473 to 482), and its total generated tokens ran from 8,084 to 8,092
against stock's 8,085 to 8,091, because the batches it passes through change where a few replies end ([DETAILS.md](DETAILS.md#why-the-token-totals-vary)). Every request
succeeded in every run. [DETAILS.md](DETAILS.md#results) has every run and the machine state.

## Why it's fast

At 16 requests a decode step multiplies a 16-row activation by each layer's 4-bit weights. Stock MLX
runs that on a kernel that reads the weights at about a quarter of the Mac's memory bandwidth. The
engine changes four things:

- **A faster, exact matrix multiply for 13 to 16 rows.** It reads each weight once and repeats stock
  MLX's arithmetic exactly, including how stock splits the work along K, so its output equals
  `mx.quantized_matmul` bit for bit. A decode step at 16 rows drops from about 27.5 ms to 17.6 ms ([measured output](benchmarks/results/step-time-and-memory-log.md)). Below
  13 rows stock runs a different kernel and an exact replica was not faster, so those rows stay on
  stock.
- **Fused projections.** The Q, K and V weights, and the gate and up weights, are joined at load, so
  decode launches one matmul where stock launched three or two. Prefill and any batch above 16 rows
  use the original shapes.
- **No log-probability work nobody asked for.** Stock computes a full-vocabulary log-softmax for
  every token and waits on it. The engine does it only when a request asks for `logprobs`.
- **A deeper listen backlog.** A burst of 16 clients overflowed stock's backlog of 5 and some were
  reset.

At startup the engine checks each custom shape against stock MLX on the Mac it runs on and uses stock
for any that differ, so a different MLX version or GPU gives correct answers at stock speed, with a
warning in the log.

## Options

`serve --help` lists every option. The ones you would set:

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path` | `mlx-community/Qwen3-1.7B-4bit` at `3b1b176` | the checkpoint: a directory, or a Hugging Face id fetched into the Hugging Face cache |
| `--max-context` | 32768 | tokens one request may hold, prompt plus output |
| `--max-inflight-tokens` | 32768 | padded KV tokens of the running requests (their number × the longest one, prompt plus `max_tokens`); others wait in arrival order |
| `--prompt-cache-gib` | 6 | memory the KV caches of finished (cached) requests may hold; running requests are limited by `--max-inflight-tokens` |
| `--mlx-cache-gib` | 2 | memory MLX keeps for reuse after arrays are freed |
| `--served-model-name` | `mlx-community/Qwen3-1.7B-4bit` | the model name clients send |
| `--host`, `--port` | `127.0.0.1`, `8080` | where the server listens |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--adapter-path` | none | a LoRA adapter; the projections then stay unfused |

The limits are measured on this Mac and are proposals (see [DETAILS.md](DETAILS.md#memory-and-context-limits)).

## What it is not

- Tuned for 16 concurrent requests on a 24 GB M4 Pro. Other Macs run it and check themselves at
  startup, but nothing here is measured on them.
- Pinned to MLX 0.32.4 and one mlx-lm commit. The other engines in this repository pin MLX 0.32.2, so
  give this one its own virtual environment.
- Not a from-scratch engine: it is mlx-lm's server with process-local patches. Nothing is written to
  the installed mlx-lm.

## Test

```bash
pip install ../../runtime '.[dev]' && pytest
```

`tests/unit` needs nothing, `tests/metal` needs a Mac, and `tests/e2e` starts the server on the real
checkpoint; set `JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT` to its directory. See
[benchmarks/README.md](benchmarks/README.md) to measure it yourself.
