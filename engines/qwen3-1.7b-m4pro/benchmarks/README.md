# Benchmark client and workload

`benchmark.py` is mlx-lm's own server benchmark (Apache-2.0, Copyright Apple Inc.), unchanged from
the client that measured the SkyDiscover run this engine came from, so ruff leaves it alone.
`run.py` starts a fresh server for each run and drives it with that client; `accuracy.py` checks the
engine's answers against stock mlx-lm's.

| file | sha256 |
|---|---|
| `benchmark.py` | `5b6b3f679e54a4483f2adea0171330ffd0f2003dc5b680d7cb62f73b2175784c` |
| `prompts_scored.txt` | `942f120e41845cd2f52d09864123147fb656780722a9dd4d2ebf5c1052e15508` |
| `prompts_heldout.txt` | `d835255342e2320fb10197b5bc026d55b792df92e14b8fa00ef470a55430bc87` |

Check them with `shasum -a 256`. `workload.json` records the same hashes and the settings.

## The workload

16 requests in flight, 64 requests in all, each asking for 128 tokens at the server's default
(greedy) sampling, drawn from 15 short prompts of 18 to 23 tokens. Decode dominates. Every request
returns 122 to 128 tokens on the scored prompts, the figure the client counts as SSE chunks; the
held-out prompts can end sooner (as few as 30 tokens), so only the cap of 128 applies there.

## Run it

```bash
pip install ../../runtime '.[benchmark]'
MODEL=~/models/Qwen3-1.7B-4bit     # the pinned checkpoint, see the engine README
python benchmarks/run.py --model-path $MODEL --arms engine stock --concurrency 16 --repeats 5 \
    --out results/c16.json
python benchmarks/run.py --model-path $MODEL --arms engine stock --concurrency 16 --repeats 5 \
    --prompts heldout --out results/c16-heldout.json
python benchmarks/accuracy.py --model-path $MODEL
```

`run.py` takes the stock arm from the installed mlx-lm, unpatched. Each run starts the server cold
with no warmup, and the arms alternate who goes first. Every run is saved with the Mac's power
source and mode and with any process using more than 15% of a core before it started. Nothing is
dropped or rerun. The summary gives each arm's median, range and token totals, and the ratio of the
medians.

## Measure on a quiet Mac

Throughput on a MacBook depends on its state. On battery, or with other apps busy, the same code
ran 10% slower, and the speed moved between a few discrete levels from run to run. Plug in, close
other apps, and read the saved machine and contention fields before trusting a number.
