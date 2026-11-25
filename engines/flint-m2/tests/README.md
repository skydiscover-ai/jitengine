# Tests

This build's end-to-end test. The engine's unit and Metal suites are in [`../flint/tests`](../../flint/tests/README.md).

```bash
pip install ../../runtime ../flint '.[dev]'
JITENGINE_FLINT_M2_E2E_CHECKPOINT=~/models/flint-int4 pytest tests/e2e -s
JITENGINE_FLINT_M2_E2E_CHECKPOINT=~/models/flint-int4 JITENGINE_FLINT_M2_E2E_SHA=report pytest tests/e2e -s   # another Mac
```

It starts `jitengine-flint-m2 serve` with the default configuration, sends one warmup and three greedy 512-token chat
requests with end-of-sequence ignored, and requires every repeat's text hash to be equal, and on the 24 GB M2 with the
checkpoint to be `46cc7a346d0ee2fb`, with the measured configuration in `/health` (offload, the graph backend, profile main). `JITENGINE_FLINT_M2_E2E_SHA` names
another hash, or `report` prints the hash and the configuration the server chose instead; `JITENGINE_FLINT_M2_E2E_MIN_TOKS`
fails the test below a decode rate. Without `JITENGINE_FLINT_M2_E2E_CHECKPOINT` it skips.
