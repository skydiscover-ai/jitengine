# Kernels

One custom Metal kernel, in two launches, that replaces a stock MLX op and claims its output
bit-identical. The check that guards it at start-up is `selfcheck.py`. Kernel paths are relative to
this directory, tests to the engine root.

| op | file | stock op | covers | contract | claim | check | test |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `qmm_exact` (main kernel) | `qmm_exact.py`, source in `_sources.py` | `mx.quantized_matmul` on 13 to 16 rows | bf16, affine 4-bit, group size 64, K in {2048, 6144, 8192}, N a multiple of 16 and at least 1024, decode (one token per row); the fused q\|k\|v and gate\|up joins and the tied output head | returns `None` for any other shape, row count, dtype or an uncovered N, and stock runs it | bit-identical to `mx.quantized_matmul` on the unfused widths | `selfcheck.run`: each shape at 13 to 16 rows on the model's layer-0 weights (generated weights if those cannot be read) | `tests/metal/test_qmm_exact.py`, `tests/metal/test_selfcheck.py`, `tests/metal/test_fusion_real_model.py` |
| `qmm_exact` (reduce kernel) | `qmm_exact.py`, source in `_sources.py` | the in-order fp32 sum over K partitions that stock's tiled kernel makes | the shapes whose N asks for more than one partition (P = 2, 4, 8, 16) | called only by the main kernel's wrapper | the sum is sequential in fp32 with one rounding, as stock's | same as above | same as above |

**What the kernel does.** Stock runs a tiled kernel at 13 to 16 rows: it dequantizes the weights to
bf16, accumulates in fp32 with 8x8x8 matrix products in natural K order, and for small N splits K into
P partitions it adds in order. P follows N alone (`split_k` in `qmm_exact.py`; found by probing the
pinned MLX). This kernel reads each weight once and repeats that arithmetic. A fused projection gives
each of its original projections' rows that projection's own P (`plan`), so a fused call equals the
unfused calls.

**Fallbacks.** `plan` returns `None` outside the table, and the call goes to stock. At start-up
`selfcheck.run` compares each shape with stock on this Mac; a shape that differs, or raises, is added
to `DISABLED` (`qmm_exact.disable`) and runs on stock, and a warning is logged. A fused decode that
differs from the unfused one sets `fusion.FUSED_MAX_ROWS` to 0, so decode runs the original
projections.

**MLX version.** The table was probed on MLX 0.32.4. On 0.32.2 the qkv, o, gate_up and down shapes
differ from stock and the check turns them off; only the output head stays on the kernel.

The call sites are `engine/patches.py` (`_install_exact_qmm`, which replaces
`nn.QuantizedLinear.__call__` and `nn.QuantizedEmbedding.as_linear`) and `engine/fusion.py`.
