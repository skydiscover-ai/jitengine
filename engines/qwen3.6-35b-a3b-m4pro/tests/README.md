# Tests

From the engine's directory:

```bash
pip install '.[dev]' && pytest
```

A pass is every test passing or skipping. On the M4 Pro with the checkpoint, the end-to-end test checks that greedy output hashes to `44d9aee27f3f890e`.

| Suite | Checks | Needs |
|:--|:--|:--|
| `tests/unit` | the planner, CLI, configuration, server, scheduler, speculative decoding, memory and packs | nothing; the tests that run MLX use its CPU build on a machine without an Apple GPU (`pip install 'mlx[cpu]==0.32.2'`) |
| `tests/metal` | each custom kernel against stock MLX, and the engine on tiny generated checkpoints, token for token: batched against solo, a prefix hit against a cold prompt | a Mac with Metal |
| `tests/e2e` | the greedy-512 end-to-end test: it starts `serve` (the exact default) and `benchmarks/benchmark.py`, and checks every repeat's text hash, every load-time kernel check, and each dense kernel's per-shape checks, from `/stats` | the 4-bit checkpoint with its MTP head beside it, and `pip install '.[dev,benchmark]'` |

A suite skips what it cannot run here. The tiny checkpoints have the real tensor names, layouts and
quantization over random weights, and copy their tokenizer from a real checkpoint when one is found.

```bash
JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_CHECKPOINT=~/models/Qwen3.6-35B-A3B-4bit pytest tests/e2e -s
```

## Variables

| Variable | What it sets | Default |
|:--|:--|:--|
| `JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_CHECKPOINT` | the end-to-end test's 4-bit checkpoint directory | unset: the test skips |
| `JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_SHA` | the greedy-512 hash the end-to-end test expects; on another Mac, that Mac's | `44d9aee27f3f890e`, the M4 Pro's |
| `JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_MIN_TOKS` | a decode rate, in tok/s, below which the test fails | unset: the rate is only reported |
| `JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_CHECKPOINT` | a real checkpoint, which a unit test scans and whose tokenizer the tiny checkpoints copy | `~/models/Qwen3.6-35B-A3B-4bit` |
| `JITENGINE_QWEN3_6_35B_A3B_M4PRO_TEST_TOKENIZER` | another tokenizer directory for the tiny checkpoints | the test checkpoint's |
