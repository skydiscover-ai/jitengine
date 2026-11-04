# Benchmark client

`benchmark.py` is the client that measured this engine's published numbers, and `workload.json` is its workload.
Since those runs, only the client's docstring, help text and printed labels were reworded, and ruff leaves the file
alone so it stays as measured. `accuracy.py` and `accuracy_prompts.json` are the same-output check.

| file | sha256 |
|---|---|
| `benchmark.py` | `a9f95493aae90e1150ee4e648b1b4faf70aee1ac3423158f1f1e5bbbb515fc85` |
| `workload.json` | `ea00f78ad6544bdb7fa2fd485e76a46df2de4abb711835a3ba4da2bcd4fa1bc2` |
| `accuracy.py` | `370a497c15a13a82e8df41ce8fb0ae3a609c5a67d68cb34890dab2123707edfe` |

Check them with `shasum -a 256`. A changed client or workload no longer
matches the published numbers.

## Run

Against a running server (`--server-pid` samples the server's memory):

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-3bit --mode interactive \
    --temperature 0 --max-tokens 512 --repeats 3 --server-pid <pid> --output-json greedy.json
```

On the 24 GB M2, every repeat's `text_sha` was `44e24c591fc59187`. The engine's [DETAILS.md](../DETAILS.md#measure-it-yourself) lists the other
measurements and their commands.
