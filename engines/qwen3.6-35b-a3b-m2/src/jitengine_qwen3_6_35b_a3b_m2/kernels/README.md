# Qwen3.6 kernels

Custom Metal kernels, one row per Python entry point (or pair), each replacing a stock MLX op (mlx
0.32.2) and claiming its output bit-identical (except the GDN recurrence). The mechanism (load-time
and first-use checks, which result sets which flag, log, warning and `/stats`, what bit-identical
means) is in the docstring of `selfcheck.py`. Kernel paths are relative to this directory, tests to
the engine root.

## What each kernel is for

Most of these kernels keep batched decoding bit-identical to one-request decoding, or cut GPU launches. The speed
numbers below are those measured for one change alone; the engine's [DETAILS.md](../../../DETAILS.md#why-its-fast-in-more-detail)
gives the eight-agent throughput after each of its eleven kept changes. Stock `mx.quantized_matmul` changes its accumulation order past a
row count that depends on the shape, so stock batched decode would have to split its rows to match one-request
decoding. The multi-row kernels here keep one-row arithmetic at any row count. In the table, a threadgroup is a CUDA
thread block and a simdgroup a warp of 32 threads.

- The dense 3-bit multiplies (`qmv` with its pre-scale kernel, `rms_norm_pre`, and the fallback `qmv_fast_mrow`)
  replace `mx.quantized_matmul` in decode. Stock re-prepares each activation slice in every threadgroup. Here a small
  kernel divides each input once by the power-of-two constants the 3-bit unpack needs, and the main kernel reuses each
  weight read for several input rows. An 8192 × 2048 weight multiply drops from 0.108 to 0.0886 ms.
  `rms_norm_pre` does that preparation inside the RMS norm, so a one-request step skips a launch.
- The expert kernels (`expert_gate_up_*`, `expert_down_*`, `experts_rows_exact`) replace one token's `mx.gather_qmm`
  experts, the SwiGLU and the weighted sum with two kernels. They take one-expert views, so a call declares only the
  expert it reads; with up to 100 operations per command buffer, one user's step went from 22.05 to 20.03 ms. For
  batched decode, each pair of request and expert gets its own threadgroups, and a served eight-row step went from
  127.1 to 123.65 ms.
- The prefill kernels (`qmm_t`, `seg_gather_qmm`) run MLX's own tiled matrix multiplies with other tiles: 128 × 64
  output tiles for dense layers, and tiles that never cross an expert boundary for the experts. Both are bit-identical
  to stock on the shapes they accept. On the M2, a dense 256-row `in_proj_qkv` multiply went from 4.41 to 3.57 ms, and
  the expert gate and up projections at 256 tokens from 12.2 to 10.0 ms.
- The decode attention kernels (`decode`, `decode_rows`) are MLX's one-pass kernel (`sdpa_vector`), line for line,
  reading each request's preallocated KV buffer in place, with its length as a runtime value. `decode_rows` runs up
  to 8 requests in one launch. Past MLX's switch to two passes, stock runs.
- The Gated DeltaNet kernels (`gdn_recurrence`, `recurrence_rows`) run the recurrence of Qwen3.6's linear-attention
  layers, which MLX has no operation for. Each simdgroup keeps one row of the fp32 state in registers across all time
  steps, and `recurrence_rows` handles 2 to 8 requests in one launch. With no stock operation to match, the check
  compares the rows launch with single rows.
- `topk` and `radix_argsort` reproduce `mx.argpartition` and `mx.argsort` exactly, tie order included, because the
  down projection sums experts in slot order and the sampler's draw depends on the sort. The radix argsort takes
  about 235 us per row instead of 405.

## Contracts

| op | file | stock op | covers (flag, gate) | contract (raises, assumes) | claim | check (result) | test |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `decode` | `sdpa_decode.py` | `mx.fast.scaled_dot_product_attention`, one query | `exact_sdpa`; `supported`: D 64/96/128/192/256, H/Hkv<=32, `single_pass` | ValueError unless q [1,H,1,D], K=V, H%Hkv==0, one dtype, 1<=n<=cap | bit-identical where stock runs `sdpa_vector`; none at D 192 | `check_sdpa` (`exact_sdpa`) | `test_sdpa_decode_bitwise_vs_stock` |
| `decode_rows` | `sdpa_decode.py` | B one-query SDPA calls | `exact_sdpa`, `sdpa_rows`; B<=8, rows `supported`, else per-row `decode` | ValueError as `decode`, per row | bit-identical as `decode` (none at D 192) | `check_sdpa_rows` (`sdpa_rows`) | `test_sdpa_rows_and_gdn_rows_match_single_row` |
| `gdn_recurrence` | `gdn_recurrence.py` | none (mlx_lm's `_make_gated_delta_kernel` layout) | no flag or check; batched decode: only where `recurrence_rows` does not run | Dk%32==0, Hv%Hk==0, fp32 state | none | none; reference of `check_gdn_rows` | none; reference in `test_the_rows_launch_equals_row_by_row` |
| `recurrence_rows` | `gdn_recurrence.py` | `gdn_recurrence` per row | `gdn_rows`; `supported_rows`: 2-8 float32 states, else per-row `gdn_recurrence` | `gdn_recurrence_rows`: ValueError unless T==1, 1<=B<=8, one state per row | bit-identical to `gdn_recurrence` | `check_gdn_rows` (`gdn_rows`; fixed geometry) | `test_the_rows_launch_equals_row_by_row` |
| `qmv` | `qmv_prescaled.py` | `mx.quantized_matmul`, per row | `dense_prescaled`; `supports`: 3-bit, biases, K%512==0, g32/64/128 | no row cap | bit-identical | `check_dense` (`dense_v4`; 3 fixed shapes, 1-8 rows), then per shape at first use `calibrate_prescaled` (`prescaled`) | `test_dense_v4_rows_bitwise_vs_stock_solo`, `test_v10_rows_bitwise_vs_stock_solo` |
| `qmv_fast_mrow` | `qmv_mrow.py` | `mx.quantized_matmul`, per row | per shape, where `qmv` is off; one-row kernel: weight+scales+biases>=64 MiB | 3/4/8-bit, biases, K%group, K%512 (8-bit: 256); no row cap | bit-identical | `calibrate_solo`, `calibrate_mrow` (<=16 rows), per shape at build (`single_row`, `multi_row`) | `test_greedy_512_hash_kernel_checks_and_decode` |
| `qmm_t` | `qmm_prefill.py` | `mx.quantized_matmul`, prefill chunk | per shape; bf16 x; `eligible`: m>=192, m*n>2^18, n%64==0, k%32==0, 3-bit | biases, bf16 scales | bit-identical | `calibrate_pf` (`prefill`), first eligible call; probes m 192-300, so n<=832 passes unprobed | `test_dense_prefill_qmm_bitwise_vs_stock` |
| `topk` | `router_topk.py` | `mx.argpartition(gates, kth=-k)[..., -k:]` | `exact_topk`; `supported`: n<=1024 gates | ValueError unless 0<k<=n | bit-identical (same order) | `check_topk` (`exact_topk`) | `test_topk_stable_equals_argpartition_tail` |
| `expert_gate_up_exact`, `expert_down_exact` | `expert_solo.py` | one token's `mx.gather_qmm` experts | `exact_experts`: resident solo decode, prescaled pair off; `supported`: 3/4/8-bit, g64, biases, K%512, bf16/fp16 | one-row x | bit-identical | `check_experts` (`exact_experts`; 3-bit only) | `test_expert_exact_bitwise_vs_stock` |
| `expert_gate_up_prescaled`, `expert_down_exact_par` | `expert_qmv_prescaled.py`, `expert_solo.py` | as above, pre-scaled | `moe_prescaled`: where the exact pair's gate holds, shared gate/up on `qmv`; 3-bit, `expert_rows.supported` | one-row x | bit-identical to the exact pair | `check_experts` (`moe_v4`, needs `exact_experts`) | `test_expert_v4_solo_and_rows_bitwise_vs_expert_exact` |
| `expert_gate_up_prescaled_rows`, `expert_down_exact_par` | `expert_qmv_prescaled.py`, `expert_solo.py` | as above, B rows | `rows_prescaled`: resident batched decode, `expert_rows.supported` | x [M,K] | bit-identical (exact pair) | `check_experts` (`rows_v4`, `moe_v4`); gate/up rps 4 in check and test, served 8-16 | `test_expert_v4_solo_and_rows_bitwise_vs_expert_exact` |
| `experts_rows_exact` | `expert_rows.py` | as above, B rows | `rows_exact` without `rows_prescaled`; `supported`: 3-bit, g64, biases, K%512, H, D %8 | no row repeats an expert | claimed bit-identical | none of its own: `rows_v4` tests the prescaled kernels | `test_expert_v4_solo_and_rows_bitwise_vs_expert_exact` |
| `seg_gather_qmm` | `gather_qmm_prefill.py` | sorted `mx.gather_qmm` (`gather_qmm_rhs`) | `seg_prefill`: prefill pairs P>=16, P//E>=4 (E: experts per layer), 3-bit, g64 (merged path skips g64) | rows sorted by expert; N%tile, K%32 | bit-identical | `check_experts` (`seg_prefill`; gate/up, `CFG` tile) | `test_seg_gather_tall_tiles_bitwise` |
| `rms_norm_pre` | `rms_norm_prescale.py` | `mx.fast.rms_norm`, then `prescale` (custom) | `fused_norm` (needs `moe_prescaled`): resident solo decode | D%16==0, D<=4096 | norm to stock, pre-scale to `prescale`, bit-identical | `check_rms_pre` (`fused_norm`) | `test_greedy_512_hash_kernel_checks_and_decode` |
| `radix_argsort` | `radix_argsort.py` | `mx.argsort` of bf16 log-probs | `radix_sort`: sampled requests, 0<top_k, 4*top_k<=vocab | bf16 [R,N], no NaN | same permutation; -0 never drawn by check or test | `check_sort` (`radix_sort`; V 248320) | `test_radix_argsort_is_mx_argsort` |

## Kernel by path

`fallback`: runs only where the preferred kernel does not; `n/a`: no such op on the path; `a/b`:
resident, then offload (pool stacks: stock `gather_qmm` in decode, `seg_gather_qmm` in prefill).

| op (Metal kernels) | solo decode | batched decode | prefill |
|:--|:--|:--|:--|
| `decode` (`sdpa_decode_exact`) | kernel | fallback | n/a |
| `decode_rows` (`sdpa_decode_exact_rows`) | n/a | kernel | n/a |
| `gdn_recurrence` (`moe_engine_gdn_recurrence`) | kernel | fallback | kernel |
| `recurrence_rows` (`moe_engine_gdn_recurrence_rows`) | n/a | kernel | n/a |
| `qmv` (`qmv_mrow_v4_pro`, `qmv_mrow_v4`, `qmv_rows_v10`) | kernel | kernel | n/a |
| `qmv_fast_mrow` (`affine_qmv_mrow`) | fallback | fallback | n/a |
| `qmm_t` (`dense_prefill_qmm_t`) | n/a | n/a | kernel |
| `topk` (`moe_topk_stable`) | kernel | kernel | kernel |
| `expert_gate_up_exact`, `expert_down_exact` (`moe_gate_up_exact`, `moe_down_exact`) | fallback/stock | n/a | n/a |
| `expert_gate_up_prescaled` (`moe_gate_up_v4`) | kernel/stock | n/a | n/a |
| `expert_gate_up_prescaled_rows` (`moe_gate_up_v4_rows`) | n/a | kernel/stock | n/a |
| `expert_down_exact_par` (`moe_down_exact_par`) | kernel/stock | kernel/stock | n/a |
| `experts_rows_exact` (`moe_gate_up_rows_exact`, `moe_down_rows_exact`) | n/a | fallback/stock | n/a |
| `seg_gather_qmm` (`seg_gather_qmm`, `seg_tiles`) | n/a | n/a (<128 rows) | kernel |
| `rms_norm_pre` (`rms_norm_pre`) | kernel/stock | stock | stock |
| `radix_argsort` (`rx2_keys`, `rx2_hist`, `rx2_scatter`) | kernel | kernel | n/a |

## Tests

From the engine root: `pytest tests/unit/kernels tests/metal/kernels` (Metal tests need MLX on
Metal; unit tests also run on its CPU build). Helpers: `constants.py`, `source.py`, `elementwise.py`
(`swiglu`, `gated_silu_f32`; no Metal kernel) and `selfcheck.py`. The kernel tests, and the e2e test
the table cites:

```
tests/metal/kernels/test_selfcheck.py::test_kernel_selfchecks_pass_on_this_device
tests/metal/kernels/test_selfcheck.py::test_sdpa_rows_and_gdn_rows_match_single_row
tests/metal/kernels/test_sdpa_decode.py::test_sdpa_decode_bitwise_vs_stock
tests/metal/kernels/test_gdn_recurrence.py::test_the_rows_launch_equals_row_by_row
tests/metal/kernels/test_qmm_prefill.py::test_dense_prefill_qmm_bitwise_vs_stock
tests/metal/kernels/test_qmm_prefill.py::test_qlinear_routes_prefill_rows_and_keeps_stock_bits
tests/metal/kernels/test_qmv_prescaled.py::test_dense_v4_rows_bitwise_vs_stock_solo
tests/metal/kernels/test_qmv_prescaled.py::test_v10_rows_bitwise_vs_stock_solo
tests/metal/kernels/test_expert_qmv_prescaled.py::test_expert_v4_solo_and_rows_bitwise_vs_expert_exact
tests/metal/kernels/test_expert_qmv_prescaled.py::test_one_expert_views_change_no_bits
tests/metal/kernels/test_expert_solo.py::test_expert_exact_bitwise_vs_stock
tests/metal/kernels/test_gather_qmm_prefill.py::test_seg_gather_tall_tiles_bitwise
tests/metal/kernels/test_router_topk.py::test_topk_stable_equals_argpartition_tail
tests/metal/kernels/test_radix_argsort.py::test_radix_argsort_is_mx_argsort
tests/unit/kernels/test_constants.py::test_a_constant_is_built_once_per_value
tests/unit/kernels/test_selfcheck.py::test_a_kernel_that_fails_its_check_is_named_in_words
tests/unit/kernels/test_selfcheck.py::test_a_check_that_raises_turns_its_kernel_off_and_says_why
tests/unit/kernels/test_selfcheck.py::test_one_check_of_several_kernels_gives_each_its_result
tests/unit/kernels/test_sdpa_decode.py::test_exact_sdpa_stops_where_stock_switches_to_two_passes
tests/unit/kernels/test_sdpa_decode.py::test_rows_outside_the_contract_raise
tests/unit/kernels/test_sdpa_decode.py::test_operands_outside_the_contract_raise
tests/unit/kernels/test_sdpa_decode.py::test_decode_rows_keeps_the_contract_in_every_row
tests/unit/kernels/test_gdn_recurrence.py::test_operands_outside_the_contract_raise
tests/unit/kernels/test_gdn_recurrence.py::test_the_rows_kernel_covers_two_to_eight_float32_states
tests/unit/kernels/test_expert_solo.py::test_expert_exact_unsupported_shapes_fall_back
tests/unit/kernels/test_router_topk.py::test_top_k_outside_the_contract_raises
tests/unit/kernels/test_router_topk.py::test_stock_runs_where_the_kernel_does_not
tests/e2e/test_gate.py::test_greedy_512_hash_kernel_checks_and_decode
```
