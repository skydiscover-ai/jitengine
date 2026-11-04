# Qwen3.6 kernels

Custom Metal kernels, one row per Python entry point (or pair), each replacing a stock MLX op (mlx
0.32.2) and claiming its output bit-identical (except the GDN recurrence and the two relaxed
ones). The mechanism (checks, flags, log, warning, `/stats`) is in the docstring of `selfcheck.py`.
This engine serves 4-bit weights, so 3-bit rows never run here. Kernel paths are relative to this
directory, tests to the engine root.

## What each kernel is for

This directory holds no per-kernel speed measurements; the engine's numbers are in its README (a one-token pass
takes 11.1 ms, and a pass that also checks a guessed token 13.4 ms). Most of them make a pass that checks several
tokens give each token exactly the bits of one-token decoding. Stock `mx.quantized_matmul` changes its accumulation
order past a row count that depends on the shape, and MLX's multi-token kernels round differently. A threadgroup is a
CUDA thread block and a simdgroup a warp of 32 threads. This engine serves 4-bit weights, so the 3-bit kernels it
shares with the M2 build (`qmv`, `qmm_t`, the prescaled experts, `seg_gather_qmm`) never run here.

- The attention kernels (`decode`, `decode_rows`) copy MLX's one-pass kernel (`sdpa_vector`) line for line and its two-pass
  kernel with the same arithmetic (its second pass runs 256 threads, not MLX's 1024), with MLX's own switch between them, which depends on the GPU and the number of keys.
  They read each request's KV buffer in place, and `decode_rows` runs a batch or a verify window in one launch.
- The Gated DeltaNet kernels run the recurrence of Qwen3.6's linear-attention layers, which MLX has no operation for.
  `recurrence_rows` runs a batch in one launch, and `gdn_recurrence_steps` stores the state after every position, so a
  verify pass can stop at the first rejected guess. `gdn_pre_rows` and `gdn_post_rows` fuse the operations around the
  recurrence while reproducing MLX's chain of operations bit for bit.
- The multi-row dense kernels (`qmv_fast_mrow`, `qmv_fast_mrow_rl4`) keep stock's one-row arithmetic (`qmv_fast`)
  while several input rows share each weight read, so a verify window or a batch matches one-row decoding at any row
  count.
- The router kernels reproduce `mx.argpartition`'s top-k exactly, tie order included (`topk`), and run the whole
  router (softmax, top-k, gather, sum and divide) in one launch (`router_topk`).
- The expert kernels compute one token's gate, up, SwiGLU, down and weighted sum in two launches
  (`expert_gate_up_exact`, `expert_down_exact`), and the `_rows` and `_par` forms do the same for verify windows of 4 to
  16 rows, each row bit-equal to its one-request result.
- `radix_argsort` is an exact argsort over the vocabulary for sampled requests.
- Two kernels give up exactness for speed: `qmm` in `relaxed_qmm.py` (rows reduced in another order) and `topk` in
  `topk_fast.py` (no tie order). Only `--profile relaxed` uses them, and that profile fails the same-output check.

## Contracts

| op | file | stock op | covers (flag, gate) | contract (raises, assumes) | claim | check (result) | test |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `decode` | `sdpa_decode.py` | `mx.fast.scaled_dot_product_attention`, one query | `exact_sdpa`; `supported`: D 64/96/128/192/256, H/Hkv<=32 | ValueError unless q [1,H,1,D], H%Hkv==0, 1<=n<=cap | bit-identical, one and two passes | `check_sdpa` (`exact_sdpa`) | `test_sdpa_decode_bitwise_vs_stock` |
| `decode_rows` | `sdpa_decode.py` | B one-query SDPA calls | `sdpa_rows`; B<=8, else per-row `decode` | ValueError as `decode`, per row | bit-identical; 2nd pass 256 threads | `check_sdpa_rows` (`sdpa_rows`) | `test_two_pass_rows_bitwise_vs_stock` |
| `gdn_recurrence` | `gdn_recurrence.py` | none (mlx_lm's `_make_gated_delta_kernel` layout) | always; batched: where `recurrence_rows` is off | Dk%32==0, Hv%Hk==0, fp32 state | none | none; reference of the rows checks | `test_the_rows_launch_equals_row_by_row` |
| `recurrence_rows` | `gdn_recurrence.py` | `gdn_recurrence` per row | `gdn_rows`; 2-8 float32 states | ValueError unless T==1, 1<=B<=8 | bit-identical to `gdn_recurrence` | `check_gdn_rows` (`gdn_rows`) | `test_the_rows_launch_equals_row_by_row` |
| `gdn_recurrence_steps` | `gdn_recurrence.py` | `gdn_recurrence` chained over T positions | every verify window | ValueError unless fp32 state | bit-identical to the chain | none | `test_gdn_recurrence_steps_equals_chained_single_steps` |
| `gdn_pre_rows`, `gdn_post_rows` | `gdn_exact.py` | the composed conv, norm and gate ops | `gdn_exact`; batched decode, verify | none; Dk=Dv=128 | bit-identical | `check_gdn_exact` (`gdn_exact`) | none; e2e gate |
| `qmv` | `qmv_prescaled.py` | `mx.quantized_matmul`, per row | 3-bit only: `dense_prescaled` | no row cap | bit-identical | `check_dense` (`dense_v4`), `calibrate_prescaled` | `test_dense_v4_rows_bitwise_vs_stock_solo` |
| `qmv_fast_mrow` | `qmv_mrow.py` | `mx.quantized_matmul`, per row | per shape: 1 row if weight>=64 MiB; 2+ rows in 2-row blocks where rl4 is off | 3/4/8-bit, biases, K%512; no row cap | bit-identical | `calibrate_solo`, `calibrate_mrow` | e2e gate |
| `qmv_fast_mrow_rl4` | `qmv_mrow.py` | `mx.quantized_matmul`, per row | per 4-bit shape, 2+ rows | ValueError unless 4-bit; g32/64/128 | bit-identical (bit patterns) | `calibrate_rl3` (`rl3`) | e2e gate |
| `qmm_t` | `qmm_prefill.py` | `mx.quantized_matmul`, prefill chunk | 3-bit only, per shape: m>=192 | biases, bf16 scales | bit-identical | `calibrate_pf` (`prefill`) | `test_dense_prefill_qmm_bitwise_vs_stock` |
| `qmm` | `relaxed_qmm.py` | `mx.quantized_matmul`, verify window | relaxed only: M>=6, N>=2048 | 4-bit g64 biases, K%64, M<=8 per launch | none; last-ulp bf16 | none | none |
| `topk` | `router_topk.py` | `mx.argpartition(gates, kth=-k)[..., -k:]` | `exact_topk`; n<=1024 gates | ValueError unless 0<k<=n | bit-identical (same order) | `check_topk` (`exact_topk`) | `test_topk_stable_equals_argpartition_tail` |
| `router_topk` | `router_topk.py` | softmax, `topk`, take, normalise | `fused_router`, `exact_topk`; n%32==0 | none (`router_supported`) | bit-identical to the chain | `check_router` (`fused_router`) | `test_fused_router_bitwise_vs_stock_chain` |
| `expert_gate_up_exact`, `expert_down_exact` | `expert_solo.py` | one token's `mx.gather_qmm` experts | `exact_experts`: resident, 1-3 rows | one-row x; g64, K%512, 3/4/8-bit | bit-identical | `check_experts` (`exact_experts`) | `test_expert_exact_bitwise_vs_stock` |
| `expert_gate_up_exact_rows`, `expert_down_exact_par` | `expert_solo.py` | as above, B rows | resident, 4-16 rows, same gate | as above, x [R,K] | bit-identical per row | none of its own | none |
| `expert_gate_up_prescaled(_rows)`, `experts_rows_exact` | `expert_qmv_prescaled.py`, `expert_rows.py` | as above | 3-bit only: `moe_prescaled`, `rows_prescaled`, `rows_exact` | 3-bit, g64, K%512 | bit-identical | `check_experts` (`moe_v4`, `rows_v4`) | `test_expert_v4_solo_and_rows_bitwise_vs_expert_exact` |
| `seg_gather_qmm` | `gather_qmm_prefill.py` | sorted `mx.gather_qmm` | 3-bit only: `seg_prefill`, resident prefill | rows sorted by expert | bit-identical | `check_experts` (`seg_prefill`) | `test_seg_gather_tall_tiles_bitwise` |
| `rms_norm_pre` | `rms_norm_prescale.py` | `mx.fast.rms_norm`, then `prescale` | `fused_norm`: resident solo decode | D%16==0, D<=4096 | bit-identical; at 4 bits the pre-scale goes unread, and the kernel stays on (no slower in an A/B, see `engine.py`) | `check_rms_pre` (`fused_norm`) | e2e gate |
| `radix_argsort` | `radix_argsort.py` | `mx.argsort` of bf16 log-probs | `radix_sort`: sampled requests | bf16 [R,N], no NaN | same permutation | `check_sort` (`radix_sort`), `check_sampler_tail` | `test_radix_argsort_is_mx_argsort` |
| `topk` | `topk_fast.py` | `mx.argpartition`, vocabulary row | relaxed only: sampled verify | ValueError unless 1<=k<=256 | none; ties may differ | none | `test_top_values_equal_a_full_sort` |

## Kernel by path

`kernel`: runs once its check passed; `fallback`: only where the preferred kernel does not;
`a/b`: resident, then offload (stock `gather_qmm` experts). Verify is resident only.

| op | solo decode | batched decode | prefill | verify |
|:--|:--|:--|:--|:--|
| `decode` | kernel | fallback | stock | 1 row |
| `decode_rows` | n/a | kernel | n/a | kernel |
| `gdn_recurrence` | kernel | fallback | kernel | n/a |
| `recurrence_rows` | n/a | kernel | n/a | n/a |
| `gdn_recurrence_steps` | n/a | n/a | n/a | kernel |
| `gdn_pre_rows`, `gdn_post_rows` | stock | kernel | stock | kernel |
| `qmv`, `qmm_t`, prescaled and `seg_gather_qmm` experts | 3-bit | 3-bit | 3-bit | 3-bit |
| `qmv_fast_mrow` | kernel | fallback | n/a | fallback |
| `qmv_fast_mrow_rl4` | n/a | kernel | n/a | kernel |
| `qmm` | n/a | n/a | n/a | relaxed |
| `router_topk` | kernel | kernel | kernel | kernel |
| `topk` | fallback | fallback | fallback | fallback |
| `expert_gate_up_exact`, `expert_down_exact` | kernel/stock | 2-3 rows/stock | n/a | 2-3 rows |
| `expert_gate_up_exact_rows` | n/a | 4-16 rows/stock | n/a | 4-16 rows |
| `rms_norm_pre` | kernel/stock | stock | stock | stock |
| `radix_argsort` | kernel | kernel | n/a | kernel |
| `topk_fast.topk` | n/a | n/a | n/a | relaxed |

## Tests

From the engine root: `pytest tests/unit/kernels tests/metal/kernels` (Metal tests need MLX on
Metal; unit tests also run on its CPU build). Helpers: `constants.py`, `source.py`, `elementwise.py`
(`swiglu`, `gated_silu_f32`) and `selfcheck.py`. The tests the table cites:

```
tests/metal/kernels/test_sdpa_decode.py::test_sdpa_decode_bitwise_vs_stock
tests/metal/kernels/test_sdpa_decode.py::test_two_pass_rows_bitwise_vs_stock
tests/metal/kernels/test_gdn_recurrence.py::test_the_rows_launch_equals_row_by_row
tests/metal/kernels/test_gdn_recurrence.py::test_gdn_recurrence_steps_equals_chained_single_steps
tests/metal/kernels/test_qmm_prefill.py::test_dense_prefill_qmm_bitwise_vs_stock
tests/metal/kernels/test_qmv_prescaled.py::test_dense_v4_rows_bitwise_vs_stock_solo
tests/metal/kernels/test_expert_qmv_prescaled.py::test_expert_v4_solo_and_rows_bitwise_vs_expert_exact
tests/metal/kernels/test_expert_solo.py::test_expert_exact_bitwise_vs_stock
tests/metal/kernels/test_gather_qmm_prefill.py::test_seg_gather_tall_tiles_bitwise
tests/metal/kernels/test_router_topk.py::test_topk_stable_equals_argpartition_tail
tests/metal/kernels/test_router_topk.py::test_fused_router_bitwise_vs_stock_chain
tests/metal/kernels/test_radix_argsort.py::test_radix_argsort_is_mx_argsort
tests/metal/kernels/test_topk_fast.py::test_top_values_equal_a_full_sort
tests/e2e/test_gate.py::test_greedy_512_hash_kernel_checks_and_decode
tests/unit/engine/test_kernel_checks.py::test_a_four_bit_model_runs_only_the_checks_that_apply_to_it
```
