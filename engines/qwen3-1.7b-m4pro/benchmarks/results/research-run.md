# The SkyDiscover run this engine came from

Copied word for word from that run's own delivery record (`DELIVERY.md`, line 4) and its
`best/score.json` (the delivered snapshot). Nothing was rerun for this file; the
numbers were measured on 2026-10-27 on the same Mac with the same stock mlx-lm commit.

## DELIVERY.md, measurement line

> - Measurement (not rerun): c=16, 3 cold repeats each, cold server, no warmup. Candidate 819.69 / 818.18 / 819.21 tok/s (median 819.21, totals 8091/8088/8091); same-procedure stock 489.71 / 490.82 / 492.08 (median 490.82, totals 8088/8093/8091). Ratio 1.669. Every run 64/64, 122-128 tokens/request, total >= 8087, median total >= 8088. Earlier confirmations: evidence/bench/final (5 repeats, 818.90 vs 491.73) and set7.

## score.json of the delivered checkpoint

```json
{
  "checkpoint": 1,
  "created_at": "2026-10-27T21:08:12.293720Z",
  "score": {
    "total_tokens_per_sec": 819.2073565230821
  },
  "baselines": [
    {
      "name": "stock_cold_c16_final9",
      "score": {
        "total_tokens_per_sec": 490.81880234183825
      }
    }
  ],
  "became_best": true
}
```

Reading: three cold runs of the delivered candidate at 16 requests gave 818.18 to 819.69 tok/s
(median 819.21); the same procedure on stock gave 489.71 to 492.08 (median 490.82); the ratio of
medians is 1.669. Every run had a total of at least 8,087 tokens, the gate that run used.
