# Flint's engine

The engine both Flint builds run: [`flint-m4pro`](../flint-m4pro), tuned on a 48 GB M4 Pro, and
[`flint-m2`](../flint-m2), tuned on a 24 GB M2. Each build's folder holds its command (`jitengine-flint-m4pro`,
`jitengine-flint-m2`), its results, the Mac it was measured on and its end-to-end test; this one holds the engine.

- With the whole model in memory, it decodes with one persistent kernel per pass (the megakernel), tuned on the 20-core
  M4 Pro and checked against stock MLX when it loads on any other Mac
  ([the M4 Pro build's DETAILS](../flint-m4pro/DETAILS.md#why-its-fast-in-more-detail)).
- When the model does not fit the memory budget, the experts stream from the SSD on stock MLX operations, the path
  tuned on the 24 GB M2 ([the M2 build's DETAILS](../flint-m2/DETAILS.md#why-its-fast-in-more-detail)).

[`install.sh`](../../install.sh) installs this package with the build it picks. By hand, from this directory:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install ../../runtime . '../flint-m4pro[download]'      # or ../flint-m2
```

[`tests/README.md`](tests/README.md) has the engine's unit and Metal suites, and
[`benchmarks/README.md`](benchmarks/README.md) the harness behind both builds' tables.
