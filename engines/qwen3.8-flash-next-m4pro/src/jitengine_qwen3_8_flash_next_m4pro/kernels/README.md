# Flash-Next kernels

Custom Metal kernels, one row per entry point (or pair), each replacing the stock MLX ops of its
`stock op` cell (mlx 0.32.2). Entry points validate nothing; callers gate them. The mechanism
(load-time checks, switching a failing kernel off, log, warning and `/stats`, the tolerance
`no_worse_than_stock`) is in the docstring of `selfcheck.py`. Kernel paths are relative to this
directory, tests to the engine root. Helpers here: `constants.py`, `source.py`.

## What each kernel is for

Most of these kernels exist for one of two reasons. Batched and verify rows must equal one-row decoding bit for bit,
and stock `mx.quantized_matmul` and `gather_qmm` change their summation order with the row count and the pool size.
Or several small operations become one launch, which keeps the layers that run ahead of the CPU cheap enough to build
in time. Two have a measured effect of their own. A threadgroup is a CUDA thread block and a simdgroup a warp of 32
threads.

- The sparse-attention kernels (`qsa_mask`, `indexer_mask`, `prefill_mask`, `decode_masked`) serve Flash-Next's
  sparse attention, where an indexer scores blocks of history and attention reads only the top-scoring blocks. They
  build the block mask on the GPU in one launch, without sorting, and `decode_masked` reads each key and value row once
  for all query heads that share it, skipping masked keys.
- The Gated DeltaNet kernels (`gdn_pre`, `gdn_post`, `gdn_recurrence`, `gdn_recurrence_staged`, `qmv_multi`) run 36 of
  the model's 48 layers, which update a running state instead of appending to a KV history. The recurrence keeps the
  state in registers across steps, and `gdn_recurrence_pair` runs both positions of a verify pass in one launch,
  building the rollback state only when a guess is wrong: 3% faster at 40 GB than writing the state every pass.
- `hc_fused` runs the hyper-connection math (the model's mixing of several residual streams) in 3 launches instead of
  about 9. In an earlier run, on a build decoding about 16 tok/s, that added 5% on its own. It is not one of the changes that took
  the shipped build from 23.8 to 32.9 tok/s.
- The expert kernels (`grouped_qmv`, `grouped_swiglu`, `grouped_qmv_x`, `grouped_qmv_u2`) read each routed expert's
  weights from its cache slot once for every row that picked it. `grouped_qmv_x` looks the slot up on the GPU, which
  lets a layer's experts run before the CPU knows the router's choice.
- `topk_softmax` and `moe_combine` run the router and the weighted sum of the experts in one launch each, with ties
  going to the lower expert id.
- The `2` and `_pair` forms (`topk_softmax2`, `moe_combine2`, `decode_masked2`, `qmv_multi_v`) run the two positions of
  a multi-token-prediction verify pass in one launch, each row with the one-row kernel's bits.
- `qmv_wide_rb` is MLX's 4-bit matrix-vector multiply with a fixed reduction order, so a row's value does not depend on
  how many rows share the launch.

## Contracts

| op | file | stock op | covers (flag, gate) | contract (assumes) | claim | check | test |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `qsa_mask` | `qsa_decode_mask.py` | packed-key argsort mask (`select_device`) | `attn_fused`; nb<=4096; decode past `prune_from` where `indexer_mask` does not run | sc [nb] fp32, P>=nb*cr | bit-identical, ties included | `check_qsa_mask` | `test_equals_the_packed_key_argsort_on_tied_scores` |
| `indexer_mask` | `qsa_indexer_decode.py` | `Qwen4ExpTextQSAIndexer.scores` and mask | `indexer_fused`; `supports` (nh<=32, hd%32==0, rope even <=hd, nb<=4096); decode past `prune_from` | q [nh,hd] raw, bk fp32 | mask = tie rule on its own scores, bit-identical; scores tolerance | `check_indexer` | `test_mask_is_the_tie_rule_on_its_own_scores` |
| `prefill_mask` | `indexer_prefill.py` | `Qwen4ExpTextQSAIndexer.select`: stock `scores`, numpy mask | `indexer_prefill`; `supports` (nh<=32, hd%32==0, rope even <=hd, nb<=4096); chunk of L>1 past `prune_from` | q [L,nh,hd] raw, bk fp32 | tolerance (near-ties only differ) | `check_prefill_mask` | `test_matches_the_numpy_mask_up_to_rounding_ties` |
| `decode_masked` | `sdpa_decode_masked.py` | `mx.fast.scaled_dot_product_attention`, bool mask | `sdpa_decode_masked`; one row, one token, with a mask; no `supports` gate (only in the check) | q [1,H,1,D], P valid KV rows | tolerance | `check_decode_masked` | `test_as_accurate_as_stock_sdpa_with_the_same_mask` |
| `qmv_multi` | `gdn_decode.py` | `QLinear.__call__` per weight set (`qmv_wide_rb` where served), `swiglu` | `gdn_fused` (GDN as `gdn_pre`; shared expert also `moe_fused`, `hc_kernel`); `supports_multi`; one row, one token | x [M,K], or [M,2K] with `swiglu` | row = set alone, bit-identical; tolerance | `check_glue` | `test_qmv_multi_rows_do_not_depend_on_the_other_weight_sets` |
| `gdn_pre`, `gdn_post` | `gdn_decode.py` | `_decode_body`'s ops around the recurrence | `gdn_fused`; `supports_gdn`, `supports_multi`; one row, one token, split in_proj layout | sigmoid gate; Hv<=128 | conv state bit-identical; rest tolerance | `check_glue` | `test_gdn_pre_as_accurate_as_the_stock_ops`, `test_gdn_post_as_accurate_as_the_stock_ops` |
| `gdn_recurrence` | `gdn_recurrence.py` | `gdn_recurrence_ops` | `gdn_kernel`, `GDN_IMPL` "kernel"; one position (decode, `step`) | Dk%32==0, Hv%Hk==0, fp32 state | tolerance | `check_recurrence` | `test_kernel_as_accurate_as_the_plain_ops` |
| `gdn_recurrence_staged` | `gdn_recurrence.py` | `gdn_recurrence_ops`, several positions | `gdn_kernel`, `GDN_IMPL` "kernel"; T>1 (prefill), `GDN_STAGED_TB` 8 | as above, Dv%4==0; g fp32; k, v, beta in q's dtype | bit-identical to `gdn_recurrence` | `check_recurrence` | `test_staged_equals_the_single_launch` |
| `qmv_wide_rb` | `qmv_row_blocked.py` | `mx.quantized_matmul` in `QLinear.__call__` | `dense_kernel` (`DENSE_KERNELS`, `DENSE_KERNEL_DEFAULT`); `supports`, `config_for`; compute-dtype x; <=`MAX_M` rows | KL per shape (`CONFIGS`, `CONFIG_BASIS`); `vecs_for` | row = row alone, bit-identical; tolerance | `check_dense` (`served` shapes) | `test_rows_in_a_batch_equal_the_row_alone` |
| `hc_fused` | `hc_decode.py` | `Qwen4ExpTextGatedResidual` (`_decode_body`; `__call__(rows=True)`), `hyper_inject` | `hc_kernel`; `supports`: 4-bit, hc 2/4/8; one token, or <=`HC_FUSED_ROWS` rows; not the terminal mixer | h [M,hc*hid]; `block_out`, `inject_in` together | row = row alone, bit-identical; tolerance | `check_hc` | `test_rows_in_a_batch_equal_the_row_alone` |
| `topk_softmax` | `router_combine.py` | `mx.softmax(precise=True)`, `mx.argpartition`, renormalise | `moe_fused`; `supports_topk`; one row, one token, with a pool and `hc_kernel` | logits [E] | ids = descending, ties to lower id, bit-identical; probabilities tolerance | `check_router_combine` | `test_topk_softmax_ties_go_to_the_lower_expert_id` |
| `moe_combine` | `router_combine.py` | `(y * scores).sum(axis=-2)`, shared expert, `hyper_inject` | `moe_fused`, `hc_kernel`; as `topk_softmax`; one capacity wave (always when speculative) | y1 [P1,hid], y2, `sel`; fp32 sum in pair order | split reads bit-identical; tolerance | `check_router_combine` | `test_moe_combine_reads_split_pairs_in_place` |
| `grouped_qmv`, `grouped_swiglu` | `grouped_expert_qmv.py` | `mx.gather_qmm` (unsorted), `swiglu` | `moe_kernel` (`MOE_KERNELS`); `grouped_for` (plan: `supports_geometry`); swiglu only in variant "fused" | sorted pairs; `config_for`, `dtype_name` | `grouped_qmv`: pair = alone, bit-identical; both tolerance | `check_grouped` | `test_each_pair_equals_the_pair_alone_in_any_slot` |
| `grouped_qmv_x` | `grouped_expert_qmv.py` | none (the pooled gather instead) | `moe_kernel`, `moe_fused`, `hc_kernel`; one token; `spec_ok` ("split", no `moe_sync_each_layer`), `grouped_for` | int32 device slots; slot<0 clamped; three array sets (main, extension >= MAIN, annex >= MAIN2); `solo_config_for` (VECS `SOLO_VECS`, `SOLO_VECS_BASIS`) | bit-identical to the `grouped_qmv` call | `check_grouped` (`_Pool.check_device`) | `test_device_slot_variant_equals_the_canonical_call` |
| `qmv_multi_v` | `gdn_decode.py` | two `qmv_multi` calls | `qmv_multi_v` (the MTP verify pass: GDN and attention input projections, shared expert); checked only with an MTP head | x [2,K] (or [2,2K] with `swiglu`), VECS=2 | row = `qmv_multi` on that row alone, bit-identical | `check_multi_v` | `test_verified_tokens_and_rows_are_the_solo_paths` |
| `grouped_qmv_u2` | `grouped_expert_qmv.py` | two `grouped_qmv_x` calls | `grouped_qmv_u2` (the MTP verify pass's routed experts); checked only with an MTP head | expert ids resolved through the device table at a layer offset; a shared expert streamed once for both rows | pair = `grouped_qmv_x` on its row alone, bit-identical | `check_grouped_u2` | `test_with_misses_and_evictions` |
| `topk_softmax2` | `router_combine.py` | two `topk_softmax` calls | `moe_fused` (the MTP verify pass's routers, `router_pair`); `supports_pair` (>= 8 experts) | two rows' logits, threadgroup r on row r | row = `topk_softmax` on that row alone, bit-identical | `check_router_combine` | `test_topk_softmax2_rows_are_the_one_row_kernel` |
| `moe_combine2` | `router_combine.py` | per row: `take(tab, inds + off)`, `where(slot >= 0, scores, 0)`, `moe_combine` | `moe_fused` (the MTP verify pass's speculative combine, `spec_pair_tail`); as `topk_softmax2` | y [2k,hid] as `grouped_qmv_u2` writes it; the device slot table at the layer offset | row = that per-row chain, bit-identical | `check_router_combine` | `test_moe_combine2_rows_are_the_masked_one_row_combine` |
| `decode_masked2` | `sdpa_decode_masked.py` | two `decode_masked` calls, row 1 after row 0's KV append | `sdpa_decode_masked2` (the MTP verify pass's attention, both rows masked); checked only with an MTP head | q0, q1 [1,H,1,D]; row 1 over P0+1 keys of one K/V buffer; each row its own mask | row = `decode_masked` on that row alone, bit-identical | `check_decode_masked2` | `test_two_rows_in_one_launch_are_each_the_row_alone` |
| `gdn_recurrence_pair` | `gdn_recurrence.py` | two `gdn_recurrence_step` launches, row 1 on the state row 0 left | `gdn_pair` (the MTP verify pass's compiled DeltaNet pair build under `gdn.GDN_LAZY`, which keeps the state after row 0 as `LazyRow0`); checked only with an MTP head | as `gdn_recurrence`; rows unexpanded; fp32 state (refused otherwise) | y0, y1, final state = the two launches', bit-identical; the state after row 0 not written | `check_pair` | `test_pair_is_two_single_launches_bit_for_bit` |

## Kernel by path

Solo decode is one row and one token; `a/b`: with the speculative gather (the default for a lone
row), without it. A one-token chunk takes the decode branches (in a prefill wave, attention and GDN
only). `fallback`: runs only where the preferred kernel does not. Mask kernels run once the indexer
prunes (`prune_from`). `qmv_wide_rb` needs compute-dtype input: with a different `--residual-dtype`,
linears after `hc_fused` run stock.

| op (Metal) | solo decode | batched decode | prefill |
|:--|:--|:--|:--|
| `qsa_mask` (`moe_engine_qsa_mask`) | fallback | fallback | n/a |
| `indexer_mask` (`moe_engine_indexer_fused`) | kernel | kernel | n/a |
| `prefill_mask` (`moe_engine_indexer_prefill_mask`) | n/a | n/a | kernel |
| `decode_masked` (`moe_engine_sdpa_decode2_part`, `moe_engine_sdpa_decode2_reduce`) | kernel | stock | stock |
| `qmv_multi` (`moe_engine_qmv_multi`) | kernel | stock | stock |
| `gdn_pre`, `gdn_post` (`moe_engine_gdn_pre`, `moe_engine_gdn_post`) | kernel | stock | stock |
| `gdn_recurrence` (`moe_engine_gdn_recurrence`) | kernel | kernel | n/a |
| `gdn_recurrence_staged` (`moe_engine_gdn_recurrence_staged`) | n/a | n/a | kernel |
| `qmv_wide_rb` (`moe_engine_affine_qmv_wide_rb`) | kernel | kernel | kernel, chunk<=8 tokens |
| `hc_fused` (`moe_engine_hc_pre`, `moe_engine_hc_down`, `moe_engine_hc_up`) | kernel | kernel, <=8 rows | stock |
| `topk_softmax` (`moe_engine_topk_softmax`) | kernel | stock | stock |
| `moe_combine` (`moe_engine_moe_combine`) | kernel/kernel, one wave | stock | stock |
| `grouped_qmv` (`moe_engine_grouped_qmv`) | recomputed pairs/kernel | kernel | kernel |
| `grouped_swiglu`, variant "fused" (default `DEFAULT_VARIANT` "split", `VARIANT_BASIS`; `moe_engine_grouped_qmv_dual`) | no/kernel | kernel | kernel |
| `grouped_qmv_x` (`moe_engine_grouped_qmv_x`) | kernel/no | n/a | n/a |
| `qmv_multi_v`, `grouped_qmv_u2`, `decode_masked2` (`moe_engine_qmv_multiv`, `moe_engine_grouped_qmv_u2`, `moe_engine_sdpa_decode2_part_rows2`, `moe_engine_sdpa_decode2_reduce_rows2`) | MTP verify pass only | n/a | n/a |
| `topk_softmax2`, `moe_combine2` (`moe_engine_topk_softmax2`, `moe_engine_moe_combine2`) | MTP verify pass only (compiled pair build) | n/a | n/a |
| `gdn_recurrence_pair` (`moe_engine_gdn_recurrence_pair`) | MTP verify pass only (compiled pair build, `GDN_LAZY`) | n/a | n/a |

## Tests

From the engine root: `pytest tests/unit/kernels tests/metal/kernels` (Metal tests need MLX on
Metal; unit tests need MLX, its CPU build on Linux). The kernel tests, and the e2e test (run
separately, needs the checkpoint):

```
tests/metal/kernels/test_indexer_prefill.py::test_matches_the_numpy_mask_up_to_rounding_ties
tests/metal/kernels/test_qsa_decode_mask.py::test_equals_the_packed_key_argsort_on_stock_scores
tests/metal/kernels/test_qsa_decode_mask.py::test_equals_the_packed_key_argsort_on_tied_scores
tests/metal/kernels/test_qsa_indexer_decode.py::test_mask_is_the_tie_rule_on_its_own_scores
tests/metal/kernels/test_qsa_indexer_decode.py::test_scores_as_accurate_as_stock
tests/metal/kernels/test_sdpa_decode_masked.py::test_as_accurate_as_stock_sdpa_with_the_same_mask
tests/metal/kernels/test_gdn_decode.py::test_qmv_multi_rows_do_not_depend_on_the_other_weight_sets
tests/metal/kernels/test_gdn_decode.py::test_qmv_multi_as_accurate_as_the_stock_projections
tests/metal/kernels/test_gdn_decode.py::test_gdn_pre_as_accurate_as_the_stock_ops
tests/metal/kernels/test_gdn_decode.py::test_gdn_post_as_accurate_as_the_stock_ops
tests/metal/kernels/test_gdn_recurrence.py::test_staged_equals_the_single_launch
tests/metal/kernels/test_gdn_recurrence.py::test_kernel_as_accurate_as_the_plain_ops
tests/metal/kernels/test_gdn_recurrence.py::test_pair_is_two_single_launches_bit_for_bit
tests/metal/kernels/test_qmv_row_blocked.py::test_rows_in_a_batch_equal_the_row_alone
tests/metal/kernels/test_qmv_row_blocked.py::test_qlinear_rows_past_max_m_equal_each_row_alone
tests/metal/kernels/test_qmv_row_blocked.py::test_as_accurate_as_stock_quantized_matmul
tests/metal/kernels/test_hc_decode.py::test_rows_in_a_batch_equal_the_row_alone
tests/metal/kernels/test_hc_decode.py::test_the_launch_shape_moves_no_bit_outside_the_reduction_orders
tests/metal/kernels/test_hc_decode.py::test_as_accurate_as_the_stock_ops
tests/metal/kernels/test_grouped_expert_qmv.py::test_each_pair_equals_the_pair_alone_in_any_slot
tests/metal/kernels/test_grouped_expert_qmv.py::test_device_slot_variant_equals_the_canonical_call
tests/metal/kernels/test_grouped_expert_qmv.py::test_as_accurate_as_stock_gather_qmm
tests/metal/kernels/test_grouped_expert_qmv.py::test_fused_swiglu_variant_as_accurate_as_the_split_stock_path
tests/metal/kernels/test_router_combine.py::test_topk_softmax_ties_go_to_the_lower_expert_id
tests/metal/kernels/test_router_combine.py::test_topk_softmax_as_accurate_as_stock_route
tests/metal/kernels/test_router_combine.py::test_topk_softmax_selection_is_the_round_by_round_argmax
tests/metal/kernels/test_router_combine.py::test_moe_combine_reads_split_pairs_in_place
tests/metal/kernels/test_router_combine.py::test_moe_combine_as_accurate_as_the_stock_tail
tests/metal/kernels/test_router_combine.py::test_topk_softmax2_rows_are_the_one_row_kernel
tests/metal/kernels/test_router_combine.py::test_moe_combine2_rows_are_the_masked_one_row_combine
tests/unit/kernels/test_selfcheck.py::test_run_records_each_result_and_the_error_of_a_raise
tests/unit/kernels/test_selfcheck.py::test_run_many_records_every_name_and_a_raise_fails_them_all
tests/unit/kernels/test_selfcheck.py::test_report_names_every_kernel_that_is_off_and_why
tests/unit/kernels/test_selfcheck.py::test_no_worse_than_stock_allows_another_summation_order_only
tests/unit/kernels/test_selfcheck.py::test_a_16_bit_stock_may_be_two_ulps_off
tests/unit/kernels/test_selfcheck.py::test_exact_matmul_reads_the_stored_values
tests/unit/kernels/test_selfcheck.py::test_quantized_codes_are_uniform_and_entries_scale_with_1_over_k
tests/unit/kernels/test_selfcheck.py::test_sample_spreads_over_the_range
tests/e2e/test_gate.py::test_greedy_512_hash_paths_and_decode
```
