# Step-time and memory measurements (terminal output)

These numbers are quoted in `README.md` and `DETAILS.md`. They come from scripts that printed to the
terminal; the scripts and their log files lived in `/tmp` and did not survive a reboot. What follows
is that printed output, copied word for word from the session transcript (timestamps are UTC), so it
is a transcription, not a re-run, and it comes with no script that reproduces it. The
Metal tests (`tests/metal`) and the end-to-end limit tests (`tests/e2e/test_limits.py`) check the
same behaviour, with bounds instead of these measured values.

## Decode step at 16 rows (2026-11-04 02:19, plugged in)

One decode step of a batch of 16 on the real model, five repetitions per line, in milliseconds.
`stock` is mlx-lm unchanged, `art` the SkyDiscover run's delivered code, `pkg` this engine.

```
stock fused False ms/step [27.59, 27.41, 27.66, 27.54, 27.14] med 27.54
art fused True ms/step [17.54, 17.61, 17.76, 17.64, 17.7] med 17.64
pkg fused True ms/step [17.73, 17.56, 17.63, 17.63, 17.62] med 17.63
art fused True ms/step [17.58, 17.56, 17.59, 17.59, 17.6] med 17.59
pkg fused True ms/step [17.61, 17.61, 17.55, 17.6, 17.82] med 17.61
stock fused False ms/step [27.51, 27.55, 27.32, 27.46, 27.63] med 27.51
```

The README's "27.6 ms to 17.6 ms" is the stock medians (27.51, 27.54) against this engine's (17.61,
17.63), rounded.

## One 15,000-token prompt (2026-11-08 03:56)

```
stock          prompt 15000 tokens  weights 0.90 GiB  prefill peak 3.08 GiB  active after 0.90 GiB  13.0s
engine         prompt 15000 tokens  weights 0.90 GiB  prefill peak 3.08 GiB  active after 0.90 GiB  12.8s
unfused        prompt 15000 tokens  weights 0.90 GiB  prefill peak 3.08 GiB  active after 0.90 GiB  12.8s
```

## Three concurrent 15k-token prompts (2026-11-08)

One wave of three concurrent prompts of about 15,000 tokens each (the later engine line is three waves).
The first two lines are this engine as it was then, with an in-flight limit of 49,152 tokens, so all
three prompts were admitted together, and the prefill chunk at 2,048 (the default, kept) and at 512
(tried and removed). The `stock` line is unpatched mlx-lm with only MLX's cache limit set to 2 GiB. The
last line is the final engine, with an in-flight limit of 32,768 tokens and the other limits in
`DETAILS.md`.

```
prefill-step 2048: waves [(59, [200, 200, 200])]  health ok  MLX peak 11.65 GiB  max active 11.44 GiB  deaths 0
prefill-step 512: waves [(58, [200, 200, 200])]  health ok  MLX peak 15.62 GiB  max active 15.40 GiB  deaths 0
stock: waves [(60, ["RemoteDisconnected('Remote end closed connection without res", "RemoteDisconnected('Remote end closed connection without res", "RemoteDisconnected('Remote end closed connection without res"])]  health down  MLX peak 16.13 GiB  max active 15.62 GiB  deaths 7
engine: waves [(57, [200, 200, 200]), (53, [200, 200, 200]), (57, [200, 200, 200])]  health ok  MLX peak 9.01 GiB  max active 9.00 GiB  deaths 0
```

## Claude Code tasks through `jit-launch claude`, with the limits (2026-11-08 20:06 local)

```
trial 1 20:06:26 read_ok=1 write_ok=0 deaths=0
trial 2 20:06:58 read_ok=2 write_ok=1 deaths=0
trial 3 20:07:34 read_ok=3 write_ok=1 deaths=0
trial 4 20:08:10 read_ok=4 write_ok=2 deaths=0
trial 5 20:08:46 read_ok=5 write_ok=3 deaths=0
RESULT read 5/5 write 3/5 deaths 0
max footprint 14.0 GB
max MLX freed-buffer cache 2.03 GiB
MLX peak 12.66 GiB
 190 200
```

The last line is `uniq -c` output over the server's response statuses: 190 responses with status
200, and no other status.
