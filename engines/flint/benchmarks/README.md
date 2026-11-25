# Benchmarks

The harness that measured the published numbers. `benchmark.py` times the engine from outside,
`driver.py` runs the requests, and `engine.py` is the entry point the driver loads the engine
through (`create_engine(model)`, default options, the budget from `FLINT_MEMORY_BUDGET_GIB`).

| File | Does |
|:--|:--|
| `make_prompts.py` | builds the prompt sets once: `code/<ctx>.json` (the chat template, a fixed coding-agent system line, real C/C++ source from llama.cpp at `184f653`, truncated so the prompt is ctx ±16 tokens, then a fixed task), `heldout/<ctx>.json` (the same over mlx-lm's Python, for confirmation only), `raw/<ctx>.json` (seeded random ids and a fixed 1,024-id decode stream) |
| `benchmark.py` | takes an exclusive lock on `$FLINT_BENCH_DIR/box.lock`, so two benchmark runs never share the Mac, waits for free memory, starts the driver, samples the whole process tree's footprint and swap from outside, and writes `result.json` |
| `driver.py` | loads the engine through `engine.py` and runs each mode: `code_greedy`, `code_sampled` (seeded, 1.0 / 0.95 / 20), and `raw` (the forced stream, one pass per token) |
| `kernels/megakernel.py` | the megakernel alone, on real weights and a random history of length L: each phase's time from phase-mask arms of the persistent kernel, its GB/s, and the verify pass at each row count (`--help`) |

The protocol, in every mode: exactly 1,024 decode steps after a prefill that is not timed,
end-of-sequence ignored, 5 repeats (the default; the M2 build's README column for the M4 Pro build was run with `--repeats 3`, and the M4 Pro build's table with 5), median. A run with a co-tenant over 2 GB or any swap growth is
marked invalid. A pass is `"valid": true` in `result.json`, which `benchmark.py` also prints when it finishes.

From the repository root, in the Python 3.14 environment from the engine's quick start:

```bash
pip install ./runtime ./engines/flint
export FLINT_BENCH_DIR=~/flint-bench FLINT_MODEL=~/models/flint-int4
git clone https://github.com/ggml-org/llama.cpp "$FLINT_BENCH_DIR/repos/llama.cpp"
git -C "$FLINT_BENCH_DIR/repos/llama.cpp" checkout 184f653
python engines/flint/benchmarks/make_prompts.py code raw          # writes $FLINT_BENCH_DIR/bench/prompts
for b in 18 12; do
  FLINT_MEMORY_BUDGET_GIB=$b python engines/flint/benchmarks/benchmark.py --impl engines/flint/benchmarks \
      --model "$FLINT_MODEL" --python "$(which python)" --contexts 1024,8192 --modes code_greedy,raw
done
```

`FLINT_BENCH_DIR` defaults to `~/flint-bench`, `FLINT_MODEL` to `~/models/flint-int4` and `FLINT_PYTHON`
to the current Python. Run it with nothing else on the Mac: expert reads share the SSD, and the
engine's memory plan assumes the budget is free.
