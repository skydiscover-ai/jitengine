# Tests

From the engine's directory:

```bash
pip install ../../runtime '.[dev]' && pytest
```

| Suite | Checks | Needs |
|:--|:--|:--|
| `tests/unit` | the CLI, the model-directory checks, hardware parsing, the split-K table, source patching, admission, the host and origin checks and the benchmark files' hashes (the Anthropic and Responses translators are jitengine-runtime's, tested in `runtime/tests`) | nothing; the tests that run MLX work on its CPU build off a Mac (`pip install mlx==0.32.2`), and the tests that import mlx-lm skip without it |
| `tests/metal` | the exact kernel against `mx.quantized_matmul` bit for bit on the model's five decode shapes at 13 to 16 rows, the start-up check (including a deliberately broken kernel), and the patched model layers against the unpatched ones at 1 to 32 rows on the real checkpoint | a Mac with Metal on **MLX 0.32.4**; on 0.32.2 the kernel tests fail by design (see `kernels/README.md`); the last one also needs the checkpoint |
| `tests/e2e` | the server on the real checkpoint: serial answers equal to stock's, the 16-client workload, streaming and logprobs, request identity, bad requests, disconnects, the context limit, queueing, the prompt-cache cap, the connection burst, host and origin checks | the checkpoint, and `pip install ../../runtime '.[dev,benchmark]'` |

A suite skips what it cannot run here. The end-to-end tests start one server at a time, each on its own
free port, and hold a lock on the GPU for the whole session, so they never compete for it or for a port.

```bash
JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT=~/models/Qwen3-1.7B-4bit pytest tests/metal tests/e2e -s
```

## Variables

| Variable | What it sets | Default |
|:--|:--|:--|
| `JITENGINE_QWEN3_1_7B_M4PRO_E2E_CHECKPOINT` | the 4-bit checkpoint directory for the Metal and end-to-end tests | unset: they skip |
| `JITENGINE_QWEN3_1_7B_M4PRO_E2E_SLOW` | set to run the slow tests: four 23k-token prompts queueing (about 2 minutes) and the c=1/4/8/32 comparisons with stock (about 15 minutes, and speed depends on the state of the Mac) | unset: they skip |
