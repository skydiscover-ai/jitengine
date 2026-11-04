# Engines

Each folder is one standalone package: one model, tuned on one Mac. Its README has the results, why
the engine is fast, the install steps and how to measure it.

| Engine | Tuned on | Memory it needs | Requests at once |
|:--|:--|:--|:--|
| [qwen3.6-35b-a3b-m2](qwen3.6-35b-a3b-m2) | 24 GB M2 | 24 GB; less with offload, slower | 8 by default (`--max-slots`) |
| [qwen3.6-35b-a3b-m4pro](qwen3.6-35b-a3b-m4pro) | 48 GB M4 Pro | 32 GB, with a budget of memory less 6 GiB below 40 GB and the GPU wired limit raised | 8 by default (`--max-slots`) |
| [qwen3.8-flash-next-m4pro](qwen3.8-flash-next-m4pro) | 48 GB M4 Pro | 24 GB, and 290 GB free on the SSD (220 GB without the optional pack16 file) | 8 by default (`--max-slots`) |
| [qwen3.8-flash-next-m2](qwen3.8-flash-next-m2) | 24 GB M2 | 24 GB, and 220 GB free on the SSD | 8 by default (`--max-slots`) |
| [flint-m4pro](flint-m4pro) | 48 GB M4 Pro | the whole model: a GPU wired limit of 24.1 GiB (4K context) to 31.02 GiB (262K context); with less memory, experts stream from the SSD, slower (offload mode tested on a 24 GB M2) | 1; up to 8 more wait (`--max-queued`) |
| [flint-m2](flint-m2) | 24 GB M2 | 24 GB; below the whole model, experts stream from the SSD | 1; up to 8 more wait (`--max-queued`) |

We ran every build on both of our Macs; [`install.sh`](../install.sh) picks one by memory and chip from those runs.
It gives a Mac the M4 Pro build of Qwen3.6 from 32 GB of memory and of Flint from 40 GB, the M2 builds below, and
Flash-Next's M2 build on a base M1 or M2 and its M4 Pro build on other chips.
Most baselines were measured on the same Mac in separate sessions. For Flint on the M4 Pro, llama.cpp's figures, and
rec's from 32K tokens of context up, are the ones rec published in its chart; llama.cpp on the M2 was measured on our
M2.

Each engine plans its memory within `--memory-budget-gib` and prints the plan before loading. The default budgets:

| Engine | Default budget |
|:--|:--|
| Qwen3.6 | 18 GiB; 34 GiB for the M4 Pro build with 40 GB or more |
| Flash-Next | 16.8 GiB on the M2, 62% of memory on the M4 Pro |
| Flint | the whole model if the GPU holds it, else the GPU's wired limit, at most memory less 6 GiB (18 GiB on a 24 GB Mac once the limit is raised, see the Flint M2 README) |

On a smaller budget, an engine streams experts from the SSD (Qwen3.6 from an expert pack, which its `pack` command
builds). Decode slows; in our tests the output stayed the same. A budget too small for the model is refused at
startup.

## What differs between a model's two builds

Each model's second build started from its first and was tuned on the other Mac.

Qwen3.6 guesses ahead on the M4 Pro and not on the M2. Multi-token prediction (MTP) has the model's draft head guess
the next token and the next pass check it. On the M2, checking one guess costs 65 to 73% of a step, so the M2 build
leaves MTP off; on the M4 Pro it costs about 21%, and the build commits about 1.85 tokens per pass. For exact decode
attention, the M2 build has a custom one-pass kernel up to MLX's switch to two passes and stock MLX beyond it; the M4
Pro build copies MLX's one- and two-pass kernels line for line, so it keeps stock's arithmetic at every length.

Flash-Next streams most of its 112 GB from the SSD on both Macs, and the M2's smaller budget and slower SSD change
three choices. On the M4 Pro, MTP is on at 40 GB, with one guessed token checked per pass in one-token arithmetic; on
the M2 it stays off. For one request, the M4 Pro build drops the expert whose next use is expected farthest away (its
layer's place in the cycle and how often it is routed), and the least-used expert when batched; the M2 build drops the
experts whose layer comes up last. With eight agents, the M4 Pro decodes all eight rows in each batched step, while
the M2 decodes one row at a time, in turns. On the M2, guessing ahead raised SSD misses per generated token by 14 to
33%, and batching eight requests read 243.5 experts per token from the SSD against 140.9 in turns.

Flint, at 22.6 GB, fits in the M4 Pro's memory and overflows the M2's 18 GiB budget. The M4 Pro build keeps the
whole model in memory and runs one persistent kernel per pass. The M2 build reads experts from the SSD on stock MLX operations: it reads each expert's
pieces in parallel, runs a helper process that keeps the GPU clock up, and starts each layer's experts on the GPU
before the CPU confirms they are in memory.

## Custom kernels and how they are checked

Each engine checks its custom kernels against stock MLX when it loads and runs the stock operations for any that
disagree. We ran the engines on a 24 GB M2 and a 48 GB M4 Pro; on a reviewer's 32 GB M1 Pro, an earlier version of the
M2 Qwen3.6 build reproduced its recorded greedy output token for token; each engine's end-to-end test checks that output on the Mac it was tuned on.
Flint has one decode kernel, so it checks that kernel as a whole: it sizes the kernel's grid to the GPU (the M4 Pro
keeps its tuned one), compares the kernel with its own per-phase form and with stock MLX, and its default profile
decodes with stock MLX ops if any check fails. We tested Flint's kernel only on the M4 Pro, simulating other Macs
there; the M2 build, which streams its experts on stock MLX ops, was measured on a 24 GB M2.

The kernels READMEs say what each kernel replaces and why:
[Qwen3.6 M2](qwen3.6-35b-a3b-m2/src/jitengine_qwen3_6_35b_a3b_m2/kernels),
[Qwen3.6 M4 Pro](qwen3.6-35b-a3b-m4pro/src/jitengine_qwen3_6_35b_a3b_m4pro/kernels),
[Flash-Next M4 Pro](qwen3.8-flash-next-m4pro/src/jitengine_qwen3_8_flash_next_m4pro/kernels),
[Flash-Next M2](qwen3.8-flash-next-m2/src/jitengine_qwen3_8_flash_next_m2/kernels). Flint's kernel is described in
[flint-m4pro/DETAILS.md](flint-m4pro/DETAILS.md#why-its-fast-in-more-detail).

## Metal and MLX terms for CUDA readers

| Term | Meaning |
|:--|:--|
| threadgroup | a thread block |
| simdgroup | a warp of 32 threads |
| threadgroup memory | shared memory |
| command buffer | a batch of GPU work the CPU submits at once; MLX closes one after a set number of operations or bytes of inputs |
| unified memory | CPU and GPU share one DRAM, so there are no host-to-device copies |
| GPU wired limit | how much memory macOS lets the GPU keep pinned (`sysctl iogpu.wired_limit_mb`) |
| persistent kernel | one GPU program that runs a whole decode pass, instead of one launch per operation (Flint's megakernel) |
| expert pack | the model's experts rewritten into one file, so a missing expert is one read from the SSD |
| GDN | Gated DeltaNet, a layer type in Qwen3.6 and Flash-Next that updates a running state instead of appending to a KV history |
| MTP | multi-token prediction, the model's own draft head |
| PLE | Flash-Next's per-layer hashed n-gram embedding, whose rows are read from their own pack on the SSD |

## How far apart the builds are

The second builds came out thousands of lines apart for Qwen3.6 and Flash-Next, while Flint's M2 build changes mostly
its offload path:

| | Qwen3.6, M2 → M4 Pro | Flash-Next, M4 Pro → M2 | Flint, M4 Pro → M2 |
|:--|:--|:--|:--|
| Identical | 30 files, 36% of the lines | 18 files, 12% of the lines | 30 files, 60% of the lines |
| Changed | 34 files, 3,044 lines differ | 60 files, 6,625 lines differ | 15 files, 711 lines differ |
| Only in the second build | 7 files, 1,115 lines: MTP drafting, relaxed kernels, exact GDN, fast top-k | none; the M4 Pro build has 8 files the M2 build lacks: the compact expert format, the prefix-store spill, the draft vocabulary, the MTP head builder | 1 file, 89 lines: the GPU keep-alive |

The changes sit on the hot path, so one shared engine would need the switches these builds removed;
each build stays a separate package. Only 6 files, 1,962 lines, are identical in all four Qwen builds. (Counted over the Python and
Metal files in each package's `src/`, after normalizing package names and leaving out empty `__init__.py` files.)

The files that builds still share, tests included, are listed in [copies.txt](copies.txt). [check_copies.py](check_copies.py)
runs in pre-commit and in CI's lint job, and fails when one copy changes and the others do not.
