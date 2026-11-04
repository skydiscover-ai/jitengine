# Benchmark client

`benchmark.py` is the client that measured this engine's published numbers, and `workload.json` is its workload.
Since those runs, only the client's docstring, help text and printed labels were reworded, and ruff leaves the file
alone so it stays as measured. `accuracy.py` and `accuracy_prompts.json` are the same-output check.

| file | sha256 |
|---|---|
| `benchmark.py` | `a9f95493aae90e1150ee4e648b1b4faf70aee1ac3423158f1f1e5bbbb515fc85` |
| `workload.json` | `ea00f78ad6544bdb7fa2fd485e76a46df2de4abb711835a3ba4da2bcd4fa1bc2` |
| `accuracy.py` | `d33e1f31539bd805e28e0365b8c79181f37a04f5e9175573d050238be4c11831` |

`benchmark.py` and `workload.json` are byte-identical to the 24 GB M2 engine's, so the numbers of
the two Qwen3.6 engines come from one client. Check the digests with
`shasum -a 256`.

## Run

Against a running server (`--server-pid` samples the server's memory):

```bash
pip install '.[benchmark]'
python benchmarks/benchmark.py --model-path ~/models/Qwen3.6-35B-A3B-4bit --mode interactive \
    --temperature 0 --max-tokens 512 --repeats 3 --server-pid <pid> --output-json greedy.json
```

[DETAILS.md](../DETAILS.md#measure-it-yourself) lists the other measurements and their commands.
