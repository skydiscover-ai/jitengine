# Benchmark client

`benchmark.py` is a client for any OpenAI-compatible server (agent and interactive modes, with
memory sampling), and `workload.json` is its workload: a shared system prompt and many user turns.
They measured this engine's published decode rates. Since then, only `benchmark.py`'s docstring, help text and
printed labels were reworded, and ruff leaves it alone so it stays as measured.

| file | sha256 |
|---|---|
| `benchmark.py` | `a9f95493aae90e1150ee4e648b1b4faf70aee1ac3423158f1f1e5bbbb515fc85` |
| `workload.json` | `ea00f78ad6544bdb7fa2fd485e76a46df2de4abb711835a3ba4da2bcd4fa1bc2` |

## Run

Against a running server, as the published numbers were measured (`--server-pid` samples the
server's memory):

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --url http://127.0.0.1:8080 --model-path ~/models/Qwen3.8-Flash-Next-4bit \
    --mode interactive --temperature 0 --max-tokens 512 --repeats 3 --server-pid <pid> --output-json out.json
```

On the 48 GB M4 Pro at `--memory-budget-gb 40`, every repeat's `text_sha` was `07a676778e3a6720`.
