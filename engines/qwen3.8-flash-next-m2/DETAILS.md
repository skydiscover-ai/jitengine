# Qwen3.8-Flash-Next on a 24 GB M2: details

The [README](README.md) has the quick start and the results. This page has the rest.

- [Install](#install)
- [Server](#server)
- [Options](#options)
- [Results](#results)
- [Why it's fast, in more detail](#why-its-fast-in-more-detail)
- [Measure it yourself](#measure-it-yourself)
- [Tested on](#tested-on)

## Install

The [README](README.md#quick-start) has the install commands. `pack` checks 64 records of each pack byte for byte against the checkpoint (`--verify N` changes the count); the engine only reads the packs. The packs
have the 48 GB M4 Pro engine's format and come from the same tools.

## Server

```bash
jitengine-qwen3.8-flash-next-m2 serve --model-path ~/models/Qwen3.8-Flash-Next-4bit
# 18 GB (16.8 GiB) by default; add --memory-budget-gb 16 for the other measured budget
```

The server listens on `127.0.0.1:8080`. The default budget, 18 GB (16.8 GiB), is the one the M2 was
measured at. `--memory-budget-gb` sets the budget in decimal GB, the unit the budgets were measured
in, and `--memory-budget-gib` in GiB. A budget above 80% of memory is refused unless `--force`.
Nothing needs sudo. `--log-level debug` prints the whole memory plan and every kernel choice, and
`warning` only what needs attention.

### Coding agents

A chat request's `tools` go to the checkpoint's template, and the calls in the
reply come back as OpenAI `tool_calls`. The same port serves the Anthropic Messages API
(`/v1/messages`, `/v1/messages/count_tokens`) and OpenAI's Responses API (`/v1/responses`) on the
same chat. We tested these endpoints with the OpenAI and Anthropic SDKs, and we have not run Claude Code, Codex,
OpenCode or Pi on this engine. [`launch/`](../../launch) and `jit-launch` set those agents up. One request holds at most the KV budget (`--kv-budget-mb`) and a
32K-token prompt; `/v1/models` reports the smaller as `max_model_len`, which the launcher sizes the
agent's context to.

## Options

`jitengine-qwen3.8-flash-next-m2 <command> --help` lists every option. The commands are `serve`,
`pack`, `inspect` (the memory plan for a budget, from the checkpoint headers), `status` (health and
memory of a running server) and `doctor` (the platform, MLX and Metal; `--json` for a report).

| Option | Default | What it sets |
|:--|:--|:--|
| `--model-path` | required | the checkpoint directory |
| `--host`, `--port` | `127.0.0.1`, `8080` | where to listen |
| `--allowed-host NAME` | none | a host name clients may use besides `localhost`, IP literals and `--host` (repeatable); a request naming any other host, or sent by a browser page of another origin, gets 403 |
| `--memory-budget-gb`, `--memory-budget-gib` | 18 GB (16.8 GiB) | the memory budget |
| `--force` | off | allow a budget above 80% of memory |
| `--expert-pack-path`, `--ple-pack-path` | none | the pack files, in place of the ones the pack directory names |
| `--max-slots` | 8 | concurrent request slots |
| `--kv-budget-mb` | planned from the memory budget | the live attention KV budget, in MB |
| `--prefix-grid-tokens`, `--prefix-min-divergence-tokens` | 2048, 256 | the prefix store's snapshot grid and minimum divergence |
| `--log-level` | `info` | `debug`, `info`, `warning` or `error` |

`pack --verify N` checks N records of each pack against the checkpoint (default 64).

Environment variables (all are paths; `~` is expanded):

- `JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR`: the directory holding the expert and PLE packs.
- `JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK`: one expert pack file, in place of the one `PACK_DIR` names.
- `JITENGINE_QWEN3_8_FLASH_NEXT_M2_PLE_PACK`: one PLE pack file, in place of the one `PACK_DIR` names.
- `JITENGINE_QWEN3_8_FLASH_NEXT_M2_MTP_PATH`: the multi-token-prediction head's file, in place of `mtp.safetensors` beside the checkpoint.

## Results

Decode speed in tok/s on a 24 GB M2 Mac, with the 112 GB 4-bit model streamed from the internal
SSD. The budgets are 18 GB (16.8 GiB) and 16 GB (14.9 GiB); 1 user is greedy, 512 tokens. The
[README](README.md#results) shows the 18 GB column.

| Server | 1 user, 18 GB | 1 user, 16 GB |
|:--|--:|--:|
| **This engine** | **4.48** | **3.98** |
| M4 Pro engine, unmodified | 4.07 | 3.55 |
| slotstream 0.2.30, MTP off | 2.76 | 2.55 |

- The unmodified engine is the 48 GB M4 Pro engine, this build's starting point. The rows
  come from different sessions on the same M2, slotstream's from an earlier one. MTP off is
  slotstream's fastest setting.
- The budgets were measured in decimal GB: `--memory-budget-gb 18` gives the 18 GB budget's plan to
  the byte.

## Why it's fast, in more detail

The [README](README.md#why-its-fast) has the short version. The M2 has a smaller budget than the M4 Pro (18 GB against 40 GB) and a slower SSD (1.79 against
4.66 GB/s), so SSD reads set the pace: each expert missing from memory costs about 1.71 ms. Every expert that is not in memory is read from the internal SSD, so the engine is built
around which experts to keep in memory and when to read the rest. The M4
Pro engine, moved over unchanged, read about 113 experts per token from the SSD at the 18 GB budget;
this build reads about 95.

- It streams the token embedding from the SSD as well, which leaves more room for experts.
- It evicts by expected next use. Each pass runs through the layers in order, so the experts of a
  layer that just ran are needed again last, and those go first. In a trace replay at 3,561 pool slots, this
  rule (looking back over 64 tokens) misses 110.3 times per step against 112.6 for the least-used rule the M4 Pro build then had (it
  now evicts by its own next-use rule).
- It keeps the M4 Pro build's prefetching: while one layer's reads are in flight, it starts reading
  the experts the next layer's router predicts.

Batching backfires here. The eight agents barely share experts, so a batched step loads eight sets
that evict one another. The engine takes turns instead, decoding one agent for 64 tokens before
switching, which cut SSD reads per token from 243.5 to 140.9 in an A/B on the same Mac.

## Measure it yourself

[`benchmarks/`](benchmarks) has the client that measured the decode rates above. Run it against a
running server, as the numbers above were measured
(`--server-pid` samples the server's memory):

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --url http://127.0.0.1:8080 --model-path ~/models/Qwen3.8-Flash-Next-4bit \
    --mode interactive --temperature 0 --max-tokens 512 --repeats 3 --server-pid <pid> --output-json out.json
```

## Tested on

The engine checks each custom kernel against stock MLX when it loads, and runs stock MLX in place of any that
fails its check. On a GPU family other than the M2's it also prints a note that the build was tuned on a 24 GB M2. All rows are at the 18 GB (16.8 GiB) default
budget.

| Mac | Memory | Configuration | Greedy-512 hash | Decode, 1 user |
|:--|:--|:--|:--|--:|
| M2 Mac (`applegpu_g14g`), the Mac the numbers above come from | 24 GB | internal SSD | `af6575a8e19ef6d5` | 4.48 tok/s |
| AWS EC2 M2 Mac | 24 GB | network-attached (EBS) disk | `af6575a8e19ef6d5`, 3 of 3 repeats | 1.82 tok/s |
| M1 Pro | 32 GB | Metal suite: reported passed in a review of an earlier commit | no e2e | |

On 2026-11-03 we reran it from the release tree on the EC2 Mac: its unit and Metal tests passed (630 passed, 4
skipped), and the end-to-end test gave `af6575a8e19ef6d5` on 3 of 3 repeats at 1.83 tok/s. On the EC2 Mac the
greedy-512 text is the same on every repeat. Its model sits on a network-attached disk, which we
did not time. The results sweep used a different workload (256 tokens) and measured 1.46 tok/s on this Mac, against
the 1.82 here. On the M1 Pro only the Metal suite, which checks each
kernel at this model's geometry, was run, and a reviewer reported that it passed; we have not rerun it on this tree. The end-to-end test was not run. If you run it on
another Mac, send `doctor --json` and the end-to-end test's output, and we will add a row.
