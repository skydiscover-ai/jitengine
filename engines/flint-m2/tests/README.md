# Tests

From the engine's directory:

```bash
pip install '.[dev]' && pytest
```

A pass is every test passing or skipping. The end-to-end test passes when its three greedy repeats give the same text hash, which on the 24 GB M2 with the checkpoint is `46cc7a346d0ee2fb`.

| Suite | Checks | Needs |
|:--|:--|:--|
| `tests/unit` | the startup checks, memory plan, expert pool, kernel plan, speculation loop, server and APIs, on a fake GPU and simulated Macs | nothing; MLX's CPU build on a machine without an Apple GPU (`pip install 'mlx[cpu]==0.32.2'`) |
| `tests/metal` | the kernels and the persistent kernel (the megakernel) on the GPU, against stock MLX and an fp64 reference | a Mac with Metal; the M4 Pro and the checkpoint for `kernels/test_megakernel.py`, `engine/` and `test_server.py`; the code prompts for `engine/test_identity.py` |
| `tests/e2e` | `serve` end to end: three greedy 512-token requests and the recorded text hash | a Mac with Metal and the checkpoint; the 24 GB M2 for the recorded hash |

A suite skips what it cannot run here. `tests/data/flint-int4` holds the checkpoint's `config.json`,
its `flint-int4.json` manifest and the headers of its five safetensors files without their tensor
data (RecursiveSI/flint-int4 at `3b93f905`, Apache-2.0); the unit tests read only those.

```bash
JITENGINE_FLINT_M2_E2E_CHECKPOINT=~/models/flint-int4 pytest tests/e2e -s
JITENGINE_FLINT_M2_E2E_CHECKPOINT=~/models/flint-int4 JITENGINE_FLINT_M2_E2E_SHA=report pytest tests/e2e -s   # another Mac
```

## What each suite checks

### `tests/unit`

- The expert pool's parallel slice reads, its heap victims (in the order a sort over every slot gives) and prefill lookups that add no use counts, the GPU keep-alive's start, pause and exit with its parent.
- The startup checks and the weight-bank layout.
- The context limit sized to the wired limit: `auto`, an explicit limit that fits, and one refused with what fits and the sysctl.
- Admission by capability and the grid per device over simulated Macs: the G0 rule, the tuned table, a pinned grid, and the `doctor --tune` cache and its key.
- The memory plan: resident exactly when the whole model fits the budget or the wired limit, offload below it, and the refusals.
- The expert pool against stacked experts bit for bit: decode's and prefill's gathers, both landings, and evictions.
- The megakernel plan and parameters, and the rendered Metal code, comments aside, against the measured build's.
- The verify pass's MoE block per row count (and `HEADER_V` as the one source of every verify render), the M = 1 render's phase-mask specialization, and the residency probe on a stub GPU.
- On a fake GPU and a stub model: the load check's stages and statistics, and each injected defect caught by its stage.
- The grid search, and the fallback table (`auto` + main decodes with stock MLX ops, kv8 refuses, `mk` or a pinned grid raises).
- Offload on simulated Macs that do not hold the model, and `--mode`.
- `doctor`'s and `inspect`'s reports and exit status, and the CLI's options (`--max-queued`, `--log-level`, `--version`).
- The draft index and depth policy, and the pipelined speculation loop on a fake megakernel.
- The sampler's seeded uniforms, stop strings, the fault seam, and the kernel-source hash `/health` reports.
- The server's port binding: a live listener on the port refuses the bind, also a wildcard or dual-stack one; a previous server's TIME_WAIT ends do not.
- The 400 that names each request field the server cannot serve.
- A chat reply to a request with tools, read into its reasoning, content and tool calls, whole and streamed, and one without tools left as Flint wrote it.
- The same reply on the Anthropic Messages API and OpenAI's Responses API.

### `tests/metal`

- On the GPU, from the headers in `tests/data`: the rendered `pos_grid` against the plan for every position, and the kv8 primitives (quantise-on-write against `mx.quantize`, the in-stream dequantisation against `mx.dequantize`, kv8 attention and `att6` against an fp64 reference).
- On the M4 Pro with the real checkpoint: the megakernel against its one-dispatch-per-phase chain and stock MLX, per phase, over many passes, for greedy and forced streams and for every verify row count, in both profiles (`kernels/test_megakernel.py`).
- Multi-token decoding against one token per pass, bit for bit, in both profiles (`engine/test_identity.py`).
- `Engine` construction: request validation, profiles, limits and refusals, and the tuned path (G 40, no search, the context limit sized to the wired limit).
- The load check with the weights (every stage passes; an injected chain defect falls back to stock MLX ops, `engine/test_loadcheck.py`).
- The request deadline and a restart on the same port through the HTTP server on a stub model (`test_server.py`).

### `tests/e2e`

- The end-to-end test: `serve` (the default configuration), one warmup and three greedy 512-token chat requests, every repeat's text hash equal (and equal to the M2's recorded hash, `46cc7a346d0ee2fb`, or the one `JITENGINE_FLINT_M2_E2E_SHA` names), the measured configuration in `/health` (offload, the graph backend, profile main), a clean exit on SIGTERM.
- With `JITENGINE_FLINT_M2_E2E_SHA=report`, the hash and the configuration the server chose are printed instead of required.

## Variables

| Variable | What it sets | Default |
|:--|:--|:--|
| `JITENGINE_FLINT_M2_E2E_CHECKPOINT` | the end-to-end test's checkpoint directory | unset: the test skips |
| `JITENGINE_FLINT_M2_E2E_SHA` | the greedy-512 hash the test expects; `report` records the hash and the configuration instead | `46cc7a346d0ee2fb`, the 24 GB M2's |
| `JITENGINE_FLINT_M2_E2E_MIN_TOKS` | a decode rate, in tok/s, below which the test fails | unset: the rate is only reported |
| `JITENGINE_FLINT_M2_TEST_CHECKPOINT` | the checkpoint `tests/metal/engine`, `tests/metal/test_server.py` and `tests/metal/kernels/test_megakernel.py` load | `~/models/flint-int4` |
| `FLINT_GRID_CACHE` | the grid cache `doctor --tune` writes and the engine reads off the tuned M4 Pro (the unit tests point it at a temporary file) | `~/.cache/flint-engine/grid-v1.json` |
| `FLINT_BENCH_DIR` | the directory whose `bench/prompts` holds the code prompts `benchmarks/make_prompts.py` builds, for `tests/metal/engine/test_identity.py` | unset: that test skips |

`test_megakernel.py`, `test_kv8_src.py` and `test_identity.py` each
also run alone as a script (`--help`), writing its full report as JSON. The megakernel checks
need about 25 GB and the Mac to themselves.
