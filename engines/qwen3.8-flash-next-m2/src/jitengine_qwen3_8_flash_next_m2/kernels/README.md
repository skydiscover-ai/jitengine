# Flash-Next M2 kernels

Custom Metal kernels, one row per entry point (or pair), each replacing the stock MLX ops of its
`stock op` cell (mlx 0.32.2). Entry points validate nothing; callers gate them. The last two rows
serve only the MTP verify pass, which runs only when an MTP head file is found, and must reproduce the solo kernel's rows. The mechanism (checks, switching
a kernel off, log, `/stats`, the tolerance `no_worse_than_stock`) is in `selfcheck.py`'s docstring. Kernel paths are relative to this
directory, tests to the engine root. Helpers here: `constants.py`, `source.py`.

## What each kernel is for

This build's kernels come from the [M4 Pro build](../../../../qwen3.8-flash-next-m4pro/src/jitengine_qwen3_8_flash_next_m4pro/kernels),
minus four that serve its multi-token-prediction verify pass (`topk_softmax2`, `moe_combine2`, `decode_masked2` and
`gdn_recurrence_pair`). It keeps the verify pass's other two, `qmv_multi_v` and `grouped_qmv_u2`. They launch only when
the checkpoint's MTP head file is found, the kernel checks pass and the plan holds at least 30% of the expert records.
The M2 `pack` command does not build the head. On the M2, SSD reads set the pace. The run record has timings of whole decode-layer pieces on this Mac (every large
piece ran at 1.02 to 1.14 times its memory-bandwidth floor), not speed measurements of these kernels one by one. They exist so that batched rows equal one-row decoding bit for bit, or
the result is no less accurate than stock, and so that chains of small operations become one launch. A threadgroup is
a CUDA thread block and a simdgroup a warp of 32 threads.

- The sparse-attention kernels (`qsa_mask`, `indexer_mask`, `prefill_mask`, `decode_masked`) serve Flash-Next's
  sparse attention, where an indexer scores blocks of history and attention reads only the top-scoring blocks. They
  build the block mask on the GPU in one launch, and `decode_masked` reads each key and value row once for all query
  heads that share it.
- The Gated DeltaNet kernels (`qmv_multi`, `gdn_pre`, `gdn_post`, `gdn_recurrence`, `gdn_recurrence_staged`) run the
  layers that update a running state, with the state kept in registers across steps.
- `hc_fused` runs the hyper-connection math (the mixing of several residual streams) in fused launches.
- The expert kernels (`grouped_qmv`, `grouped_swiglu`, `grouped_qmv_x`, `grouped_qmv_u2`) read each routed expert's
  weights from its cache slot once for every row that picked it, and `qmv_multi_v` streams the weights once for
  several activation rows.
- `topk_softmax` and `moe_combine` run the router and the weighted sum of the experts in one launch each, with ties
  going to the lower expert id.
- `qmv_wide_rb` is MLX's 4-bit matrix-vector multiply with a fixed reduction order, so a row's value does not depend on
  how many rows share the launch.

## Contracts

op (file) | stock op | covers (flag, gate) | contract (assumes) | claim | check, test
:--|:--|:--|:--|:--|:--
`qsa_mask` (`qsa_decode_mask.py`) | packed-key argsort mask (`select_device`) | `attn_fused`; nb<=4096; decode past `prune_from` where `indexer_mask` does not run | sc [nb] fp32, P>=nb*cr | bit-identical, ties included | `check_qsa_mask`; `test_equals_the_packed_key_argsort_on_tied_scores`
`indexer_mask` (`qsa_indexer_decode.py`) | `Qwen4ExpTextQSAIndexer.scores` and mask | `indexer_fused`; `supports` (nh<=32, hd%32==0, rope even <=hd, nb<=4096); decode past `prune_from` | q [nh,hd] raw, bk fp32 | mask = tie rule on its own scores, bit-identical; scores tolerance | `check_indexer`; `test_mask_is_the_tie_rule_on_its_own_scores`
`prefill_mask` (`indexer_prefill.py`) | `Qwen4ExpTextQSAIndexer.select`: stock `scores`, numpy mask | `indexer_prefill`; `supports` (nh<=32, hd%32==0, rope even <=hd, nb<=4096); chunk of L>1 past `prune_from` | q [L,nh,hd] raw, bk fp32 | tolerance (near-ties only differ) | `check_prefill_mask`; `test_matches_the_numpy_mask_up_to_rounding_ties`
`decode_masked` (`sdpa_decode_masked.py`) | `mx.fast.scaled_dot_product_attention`, bool mask | `sdpa_decode_masked`; one row, one token, with a mask; no `supports` gate (only in the check) | q [1,H,1,D], P valid KV rows | tolerance | `check_decode_masked`; `test_as_accurate_as_stock_sdpa_with_the_same_mask`
`qmv_multi` (`gdn_decode.py`) | `QLinear.__call__` per weight set (`qmv_wide_rb` where served), `swiglu` | `gdn_fused` (GDN as `gdn_pre`; shared expert also `moe_fused`, `hc_kernel`); `supports_multi`; one row, one token | x [M,K], or [M,2K] with `swiglu` | row = set alone, bit-identical; tolerance | `check_glue`; `test_qmv_multi_rows_do_not_depend_on_the_other_weight_sets`
`gdn_pre`, `gdn_post` (`gdn_decode.py`) | `_decode_body`'s ops around the recurrence | `gdn_fused`; `supports_gdn`, `supports_multi`; one row, one token, split in_proj layout | sigmoid gate; Hv<=128 | conv state bit-identical; rest tolerance | `check_glue`; `test_gdn_pre_as_accurate_as_the_stock_ops`, `test_gdn_post_as_accurate_as_the_stock_ops`
`gdn_recurrence` (`gdn_recurrence.py`) | `gdn_recurrence_ops` | `gdn_kernel`, `GDN_IMPL` "kernel"; one position (decode, `step`) | Dk%32==0, Hv%Hk==0, fp32 state | tolerance | `check_recurrence`; `test_kernel_as_accurate_as_the_plain_ops`
`gdn_recurrence_staged` (`gdn_recurrence.py`) | `gdn_recurrence_ops`, several positions | `gdn_kernel`, `GDN_IMPL` "kernel"; T>1 (prefill), `GDN_STAGED_TB` 8 | as above, Dv%4==0; g fp32; k, v, beta in q's dtype | bit-identical to `gdn_recurrence` | `check_recurrence`; `test_staged_equals_the_single_launch`
`qmv_wide_rb` (`qmv_row_blocked.py`) | `mx.quantized_matmul` in `QLinear.__call__` | `dense_kernel` (`DENSE_KERNELS`, `DENSE_KERNEL_DEFAULT`); `supports`, `config_for`; compute-dtype x; <=`MAX_M` rows | KL per shape (`CONFIGS`, `CONFIG_BASIS`); `vecs_for` | row = row alone, bit-identical; tolerance | `check_dense` (`served` shapes); `test_rows_in_a_batch_equal_the_row_alone`
`hc_fused` (`hc_decode.py`) | `Qwen4ExpTextGatedResidual` (`_decode_body`; `__call__(rows=True)`), `hyper_inject` | `hc_kernel`; `supports`: 4-bit, hc 2/4/8; one token, or <=`HC_FUSED_ROWS` rows; not the terminal mixer | h [M,hc*hid]; `block_out`, `inject_in` together | row = row alone, bit-identical; tolerance | `check_hc`; `test_rows_in_a_batch_equal_the_row_alone`
`topk_softmax` (`router_combine.py`) | `mx.softmax(precise=True)`, `mx.argpartition`, renormalise | `moe_fused`; `supports_topk`; one row, one token, with a pool and `hc_kernel` | logits [E] | ids = descending, ties to lower id, bit-identical; probabilities tolerance | `check_router_combine`; `test_topk_softmax_ties_go_to_the_lower_expert_id`
`moe_combine` (`router_combine.py`) | `(y * scores).sum(axis=-2)`, shared expert, `hyper_inject` | `moe_fused`, `hc_kernel`; as `topk_softmax`; one capacity wave (always when speculative) | y1 [P1,hid], y2, `sel`; fp32 sum in pair order | split reads bit-identical; tolerance | `check_router_combine`; `test_moe_combine_reads_split_pairs_in_place`
`grouped_qmv`, `grouped_swiglu` (`grouped_expert_qmv.py`) | `mx.gather_qmm` (unsorted), `swiglu` | `moe_kernel` (`MOE_KERNELS`); `grouped_for` (plan: `supports_geometry`); swiglu only in variant "fused" | sorted pairs; `config_for`, `dtype_name` | `grouped_qmv`: pair = alone, bit-identical; both tolerance | `check_grouped`; `test_each_pair_equals_the_pair_alone_in_any_slot`
`grouped_qmv_x` (`grouped_expert_qmv.py`) | none (the pooled gather instead) | `moe_kernel`, `moe_fused`, `hc_kernel`; one token; `spec_ok` ("split", no `moe_sync_each_layer`), `grouped_for` | int32 device slots; slot<0 clamped | bit-identical to the `grouped_qmv` call | `check_grouped` (`_Pool.check_device`); `test_device_slot_variant_equals_the_canonical_call`
`qmv_multi_v` (`gdn_decode.py`) | none: two solo `qmv_multi` launches | `qmv_multi_v`; MTP head file found only; verify pass, two rows (`build_spec2`) | x [2,K], or [2,2K] with `swiglu`; `vecs`=2 | each row = `qmv_multi` alone, bit-identical | `check_multi_v`; `test_qmv_multi_v_rows_equal_qmv_multi_on_each_row_alone`
`grouped_qmv_u2` (`grouped_expert_qmv.py`) | none: two solo `grouped_qmv_x` gathers | `grouped_qmv_u2`; MTP head file found only; verify pass (`spec_build_pair`) | int32 slots, `inds2` for both rows; slot<0 clamped | each pair = `grouped_qmv_x` alone, bit-identical | `check_grouped_u2`; `test_grouped_qmv_u2_equals_the_solo_gather_when_both_rows_share_experts`

## Kernel by path

Solo decode is one row and one token; `a/b`: with the speculative gather (the default for a lone
row), without it. Rows decode in turns (`decode_turn` 64, each turn a solo-decode forward), so
batched decode runs only with `decode_turn` 0. The MTP verify pass (two positions per layer, only with a head file) runs
the solo column's kernels, with `qmv_multi_v` (`moe_engine_qmv_multiv`) for `qmv_multi` and
`grouped_qmv_u2` (`moe_engine_grouped_qmv_u2`) for `grouped_qmv_x`. A one-token chunk takes the decode branches (in a prefill wave, attention and GDN
only). `fallback`: runs only where the preferred kernel does not. Mask kernels run once the indexer
prunes (`prune_from`). `qmv_wide_rb` needs compute-dtype input: with a different `--residual-dtype`,
linears after `hc_fused` run stock.

op (Metal) | solo decode | batched decode | prefill
:--|:--|:--|:--
`qsa_mask` (`moe_engine_qsa_mask`) | fallback | fallback | n/a
`indexer_mask` (`moe_engine_indexer_fused`) | kernel | kernel | n/a
`prefill_mask` (`moe_engine_indexer_prefill_mask`) | n/a | n/a | kernel
`decode_masked` (`moe_engine_sdpa_decode2_part`, `moe_engine_sdpa_decode2_reduce`) | kernel | stock | stock
`qmv_multi` (`moe_engine_qmv_multi`) | kernel | stock | stock
`gdn_pre`, `gdn_post` (`moe_engine_gdn_pre`, `moe_engine_gdn_post`) | kernel | stock | stock
`gdn_recurrence` (`moe_engine_gdn_recurrence`) | kernel | kernel | n/a
`gdn_recurrence_staged` (`moe_engine_gdn_recurrence_staged`) | n/a | n/a | kernel
`qmv_wide_rb` (`moe_engine_affine_qmv_wide_rb`) | kernel | kernel | kernel, chunk<=8 tokens
`hc_fused` (`moe_engine_hc_pre`, `moe_engine_hc_down`, `moe_engine_hc_up`) | kernel | kernel, <=8 rows | stock
`topk_softmax` (`moe_engine_topk_softmax`) | kernel | stock | stock
`moe_combine` (`moe_engine_moe_combine`) | kernel/kernel, one wave | stock | stock
`grouped_qmv` (`moe_engine_grouped_qmv`) | recomputed pairs/kernel | kernel | kernel
`grouped_swiglu`, variant "fused" (default `DEFAULT_VARIANT` "split", `VARIANT_BASIS`; `moe_engine_grouped_qmv_dual`) | no/kernel | kernel | kernel
`grouped_qmv_x` (`moe_engine_grouped_qmv_x`) | kernel/no | n/a | n/a

## Tests

From the engine root: `pytest tests/unit/kernels tests/metal/kernels` (Metal tests need Metal; unit
tests need MLX, its CPU build on Linux). The kernel tests and the e2e test (needs the checkpoint):

```
tests/metal/kernels/test_indexer_prefill.py
tests/metal/kernels/test_qsa_decode_mask.py
tests/metal/kernels/test_qsa_indexer_decode.py
tests/metal/kernels/test_sdpa_decode_masked.py
tests/metal/kernels/test_gdn_decode.py
tests/metal/kernels/test_gdn_recurrence.py
tests/metal/kernels/test_qmv_row_blocked.py
tests/metal/kernels/test_hc_decode.py
tests/metal/kernels/test_grouped_expert_qmv.py
tests/metal/kernels/test_router_combine.py
tests/unit/kernels/test_selfcheck.py
tests/unit/kernels/test_row_blocked_template.py
tests/unit/kernels/test_verify_checks.py (the verify checks on CPU stand-ins)
tests/e2e/test_gate.py::test_greedy_512_hash_paths_and_decode
```
