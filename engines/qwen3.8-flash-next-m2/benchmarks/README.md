# Benchmark client

`benchmark.py` and `workload.json` are the client and workload that measured this engine's
published decode rates. Since then only `benchmark.py`'s docstring, help text and printed labels
were reworded; ruff leaves it alone, and the hashes below identify the files.

| file | sha256 |
|---|---|
| `benchmark.py` | `a9f95493aae90e1150ee4e648b1b4faf70aee1ac3423158f1f1e5bbbb515fc85` |
| `workload.json` | `ea00f78ad6544bdb7fa2fd485e76a46df2de4abb711835a3ba4da2bcd4fa1bc2` |

The same files are the client of the Flash-Next engine for the 48 GB M4 Pro and of the Qwen3.6
engines, so decode rates are comparable across them.

## Run

Against a running server, as the published numbers were measured (`--server-pid` samples the
server's memory):

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --url http://127.0.0.1:8080 --model-path ~/models/Qwen3.8-Flash-Next-4bit \
    --mode interactive --temperature 0 --max-tokens 512 --repeats 3 --server-pid <pid> --output-json out.json
```

On the 24 GB M2 at `--memory-budget-gb 18`, every repeat's `text_sha` was `af6575a8e19ef6d5`.
