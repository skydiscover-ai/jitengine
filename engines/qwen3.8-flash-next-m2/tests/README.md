# Tests

From the engine's directory:

```bash
pip install '.[dev]' && pytest
```

| Suite | Checks | Needs |
|:--|:--|:--|
| `tests/unit` | the memory planner, CLI, packs, expert pool, step rollback, scheduler, server, and the load-time checks of the MTP verify pass's two kernels, which run only when an MTP head file is found | nothing; the tests that run MLX use its CPU build on a machine without an Apple GPU (`pip install 'mlx[cpu]==0.32.2'`) |
| `tests/metal` | each kernel against stock MLX, bit for bit where its module claims it and otherwise no less accurate, and the engine on a tiny generated checkpoint, token for token: a prefix hit against a cold prompt, a small expert pool against a full one, a batched row beside different rows | a Mac with Metal |
| `tests/e2e` | the greedy-512 end-to-end test: it starts `serve` with the two packs at the measured 18 GB (16.8 GiB) and `benchmarks/benchmark.py`, and checks every repeat's text hash, every load-time kernel check (the two verify kernels' only when an MTP head file is found), and that the measured kernels and decode paths are on | the checkpoint and its packs, and `pip install '.[dev,benchmark]'` |

A pass is every test passing or skipping, and a suite skips what it cannot run here. The end-to-end test passes when
every repeat's text hash is `af6575a8e19ef6d5`.

```bash
JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_CHECKPOINT=~/models/Qwen3.8-Flash-Next-4bit \
    JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_PACKS=$JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR pytest tests/e2e -s
```

## Variables

| Variable | What it sets | Default |
|:--|:--|:--|
| `JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_CHECKPOINT` | the e2e test's checkpoint directory | unset: the test skips |
| `JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_PACKS` | the directory holding that checkpoint's expert and PLE packs | required with the checkpoint |
| `JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_BUDGET_GB` | the e2e test's budget, in decimal GB, as the budgets were measured | `18` (16.8 GiB) |
| `JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_SHA` | the greedy-512 hash the e2e test expects; on another Mac, that Mac's | `af6575a8e19ef6d5`, the M2's |
| `JITENGINE_QWEN3_8_FLASH_NEXT_M2_E2E_MIN_TOKS` | a decode rate, in tok/s, below which the e2e test fails | unset: the rate is only reported |
