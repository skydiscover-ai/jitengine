"""The Engine: one model, one worker thread that owns all GPU work, and a continuous slot batcher.

Public calls (generate / stream / generate_batch / score) submit a job and wait on its own queue,
so any number of client threads may call concurrently. The worker loop drains the queue, admits
what fits into free slots, runs prompt work, steps every decoding row together and retires
finished rows; KV, recurrent state, sampler, EOS and max_tokens stay per row.

Exactness: a lone row decodes through `Qwen4ExpForCausalLM.forward` exactly as without the batcher.
Prefill is computed per row (by default a wave shares expert fetches, not tensors), so prefix-store
snapshots and warm runs match cold ones bit for bit; stock `mx.quantized_matmul` is not batched ==
solo for M >= 2. `score` never touches the store or a decode batch, so it stays cold-exact.

`Engine` holds the constructor (the `init_*` steps) and the public calls. Its other parts are
mixins, each in the subsystem it serves: the model thread (`scheduler.scheduler`), admission
(`scheduler.admission`), prompt processing (`scheduler.prefill`), decoding (`scheduler.decode`),
retirement (`scheduler.retire`), scoring (`scheduler.score`) and expert offloading
(`moe.offload`).

The state the mixins share: the `init_*` steps set each attribute below first; after that its owner
keeps it consistent, and the classes in the last column write it only to reset it when a step fails
or the engine stops, unless noted (tests/unit/engine/test_engine_state.py checks the table against
the code).

| attributes | owner | also written by |
|:--|:--|:--|
| `_batch`, `_y`, `_telem`, `_reads` | DecodingMixin | RetirementMixin, SchedulingMixin |
| `_step_no`, `_telem_tokens` | DecodingMixin | |
| `row_faults` | DecodingMixin | PromptProcessingMixin counts its own |
| `decode_rows_active_max` | DecodingMixin | |
| `_turn_order`, `_turn_owner`, `_turn_used`, `turns`, `parked_rows_peak` | DecodingMixin | |
| `mtp_verifies`, `mtp_accepted` | DecodingMixin | |
| `_pending` | AdmissionMixin | SchedulingMixin |
| `_pending_dirty` | AdmissionMixin | SchedulingMixin marks arrivals |
| `_group_points` | AdmissionMixin | |
| `waves_run`, `wave_rows_total`, `wave_rows_max`, `wave_tokens_max` | PromptProcessingMixin | |
| `_dead`, `_work`, `_scores` | SchedulingMixin | |
| `admit_windows`, `admit_warm_windows`, `admit_window_collected` | SchedulingMixin | |
| `admit_window_seconds` | SchedulingMixin | |
| `turn_ext_slots` | SchedulingMixin | |
| `mtp`, `mtp_gen`, `mtp_loads`, `mtp_on`, `mtp_bytes` | SchedulingMixin | |
| `expert_store`, `expert_pool` | OffloadingMixin | |
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator, Sequence

import mlx.core as mx

from jitengine_qwen3_8_flash_next_m2.budget import (
    GIB,
    WAVE_TOKENS_SERVED,
    declared_budget_bytes,
    plan_budget,
)
from jitengine_qwen3_8_flash_next_m2.checkpoint.loader import load_language_weights
from jitengine_qwen3_8_flash_next_m2.engine.io_struct import GenRequest, GenResult
from jitengine_qwen3_8_flash_next_m2.kernels.grouped_expert_qmv import (
    DEFAULT_VARIANT as _MOE_DEFAULT_VARIANT,
)
from jitengine_qwen3_8_flash_next_m2.kernels.qmv_row_blocked import (
    DENSE_KERNEL_DEFAULT,
    DENSE_KERNELS,
)
from jitengine_qwen3_8_flash_next_m2.kvcache.prefix_cache import MIN_DIVERGENCE_TOKENS, PrefixCache
from jitengine_qwen3_8_flash_next_m2.model.config import Qwen4ExpTextConfig, load_config
from jitengine_qwen3_8_flash_next_m2.model.hyper_connections import HC_PATH_DEFAULT, HC_PATHS
from jitengine_qwen3_8_flash_next_m2.model.model import WAVE_EVAL, Qwen4ExpForCausalLM
from jitengine_qwen3_8_flash_next_m2.model.moe import MOE_KERNEL_DEFAULT, MOE_KERNELS, MOE_VARIANTS
from jitengine_qwen3_8_flash_next_m2.moe.offload import OffloadingMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.admission import AdmissionMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.batch import Job, Req, SlotPool, kv_bytes_for
from jitengine_qwen3_8_flash_next_m2.scheduler.decode import DecodingMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.faults import FaultSeam
from jitengine_qwen3_8_flash_next_m2.scheduler.prefill import PromptProcessingMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.retire import RetirementMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.scheduler import SchedulingMixin
from jitengine_qwen3_8_flash_next_m2.scheduler.score import SCORE_READOUT_ROWS, ScoringMixin
from jitengine_qwen3_8_flash_next_m2.server.stats import PhaseMeter, Stats, StepTelemetry
from jitengine_qwen3_8_flash_next_m2.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# the decode-in-turn quantum (Engine.decode_turn_q)
DECODE_TURN = 64
# Greedy single-stream decoding drafts with the MTP head (mtp.py, Model.verify2) when the head
# matches the model and the plan without it holds at least this share of the expert records; the
# head comes out of the pool, and below that it costs more misses than the verify saves.
MTP_MIN_CAPACITY = 0.30
# The kernels whose failed check keeps MTP off (`kernel_checks.NAMES`): the verify pass's own two,
# and the three the speculative build behind `Model.forward_hidden` and `Model.verify2` launches
# whatever its flags say (`hc_attn.fused`, `moe.fused_decode`, `_gather_dev`), where the solo
# step, with them off, runs the stock ops: MTP then could not match the solo step bit for bit.
MTP_KERNELS = ("qmv_multi_v", "grouped_qmv_u2", "hc_kernel", "moe_fused", "moe_kernel")
# The absolute chunk grid that generate rows and `score` both partition prompts on.
WAVE_CHUNK = 2048
# Bounded admission window: an idle scheduler waits briefly so an arrival group is admitted together
# and forms one wave. It closes after ADMIT_QUIET_S of silence or ADMIT_WINDOW_S in total.
ADMIT_WINDOW_S = 5.0
ADMIT_QUIET_S = 1.0
# Uncovered prefill that opens the window (the store's min-divergence unit); below it a row's own
# prefill is shorter than the wait.
ADMIT_MIN_TOKENS = 256
# Quiet timeout of the warm window, which opens when every pending job is fully covered by the
# store: holding such a row delays only its own first token, and a short wait lets a burst form one
# wave. 0 disables the warm clause.
ADMIT_WARM_QUIET_S = 0.1
# The wave's in-flight token budget lives in the memory plan (`plan.wave_tokens`): its transient
# scales with the tokens in flight, not the rows, and is charged to the plan's scratch bucket.


class Engine(
    OffloadingMixin,
    SchedulingMixin,
    AdmissionMixin,
    PromptProcessingMixin,
    DecodingMixin,
    RetirementMixin,
    ScoringMixin,
):
    """One model, a worker thread that runs every request, and a continuous slot batcher.

    `generate`, `stream`, `generate_batch` and `score` submit a job and wait on its own queue, so
    any thread may call them.
    """

    def __init__(self, model_path: str, **options):
        self.init_options(model_path, options)

        t0 = time.monotonic()
        self.init_kernels(model_path)
        scan = self.init_memory_plan(model_path, options)
        self.init_tokenizer(model_path)
        self.init_schedule(options)
        self.init_store_and_stats()

        self.init_memory_limits(options)

        self.init_model(model_path, options, scan)
        self.init_slots()

        self.init_scheduler()

        self.init_worker()
        self.init_store_and_stats()  # warm-up entries never count
        self.meter = PhaseMeter(self.expert_pool)  # warm-up traffic never counts either
        if self.expert_pool is not None:
            logger.info(
                f"[engine] expert pool: {self.expert_pool.slots} slots "
                f"({100 * self.expert_pool.capacity_fraction():.2f}% of {self.expert_pool.num_ids}), "
                f"landing={self.expert_pool.landing}, "
                f"seeded {self.expert_pool.seed_records} records "
                f"({self.expert_pool.seed_bytes / GIB:.2f} GiB)"
            )
        logger.info(
            f"[engine] ready in {time.monotonic() - t0:.1f}s; active memory {mx.get_active_memory() / GIB:.2f} GiB"
            + ("; WARNING: some kernels are off (see above)" if self.kernels_off else "")
        )

    def init_options(self, model_path: str, options: dict) -> None:
        """The engine options, checked; each one becomes an attribute."""
        self.model_path = model_path
        self.options = dict(options)
        self.memory_budget_bytes = declared_budget_bytes(options)
        self.model_name = options.get("model_name") or model_path
        self.log_requests = bool(options.get("log_requests", True))
        # Router-lookahead prefetch (0 = off). `prefetch_reserve` is the rotating pool of
        # never-published slots that speculative reads land in.
        self.prefetch_lookahead = max(0, int(options.get("prefetch_lookahead", 0)))
        # the miss chain reads into the prefetch reserve, so it needs one
        _CHAIN_RESERVE = 32
        # The reserve scales with the lookahead (16 slots per layer, floor 32; a dry reserve drops
        # predictions) and comes out of the pool, so it is only allocated when something uses it.
        self.prefetch_reserve = (
            max(0, int(options.get("prefetch_reserve", max(32, 16 * self.prefetch_lookahead))))
            if self.prefetch_lookahead
            else max(0, int(options.get("prefetch_reserve", _CHAIN_RESERVE)))
        )
        # Landing region for demand fetches, so eviction can happen after the bytes arrive; like the
        # prefetch reserve, it comes out of the pool's residency.
        from jitengine_qwen3_8_flash_next_m2.moe.offload_cache import (
            LANDING_SLOTS as _LANDING_SLOTS,
        )

        # The `landing_slots` option lets tests exercise the region on small fixtures, where the
        # ratio-sized default resolves to 0. It changes no operand.
        self.landing_slots = max(0, int(options.get("landing_slots", _LANDING_SLOTS)))
        # prefill/score work above this many uncovered tokens first releases the extension pool
        self.ext_scratch_tokens = int(options.get("ext_scratch_tokens", "256"))
        self.compute_dtype_name = str(options.get("compute_dtype", "float32"))
        if self.compute_dtype_name not in ("bfloat16", "float32", "float16"):
            raise ValueError(f"compute_dtype {self.compute_dtype_name!r}")
        # the routed-expert kernel (layers.MOE_KERNELS) and its variant, refused before any work
        self.moe_kernel = str(options.get("moe_kernel") or MOE_KERNEL_DEFAULT)
        if self.moe_kernel not in MOE_KERNELS:
            raise ValueError(f"moe_kernel {self.moe_kernel!r} not in {MOE_KERNELS}")
        self.moe_kernel_variant = str(options.get("moe_kernel_variant") or _MOE_DEFAULT_VARIANT)
        if self.moe_kernel_variant not in MOE_VARIANTS:
            raise ValueError(
                f"moe_kernel_variant {self.moe_kernel_variant!r} not in {MOE_VARIANTS}"
            )
        # Hyper-connection elementwise path (layers.HC_PATHS): "compiled" or "stock" (the per-site
        # fallback). Both are bit-identical, so this changes launches, never arithmetic.
        self.hc_path = str(options.get("hc_path") or HC_PATH_DEFAULT)
        if self.hc_path not in HC_PATHS:
            raise ValueError(f"hc_path {self.hc_path!r} not in {HC_PATHS}")
        # Dense kernel (qmv_row_blocked.DENSE_KERNELS): "rb" runs the row-blocked kernel on the shapes
        # its table serves, else "stock". Unlike hc_path it changes the accumulation order.
        self.dense_kernel = str(options.get("dense_kernel") or DENSE_KERNEL_DEFAULT)
        if self.dense_kernel not in DENSE_KERNELS:
            raise ValueError(f"dense_kernel {self.dense_kernel!r} not in {DENSE_KERNELS}")
        # `prefill_chunk` no longer exists (score walks the grid); refuse it rather than ignore it.
        if options.get("prefill_chunk") is not None:
            raise ValueError(
                "prefill_chunk is not an option: score() runs the served partition (the absolute "
                "prefix grid, option prefix_grid_tokens, default 2048), with no separate score "
                "chunk. Set prefix_grid_tokens if you mean the grid."
            )
        # the fault-injection seam is parsed first so a malformed option is refused before any work
        self.fault_seam = FaultSeam.from_option(options.get("fault_inject"))

    def init_kernels(self, model_path: str) -> None:
        """The model config, and each custom kernel the model would run checked against stock MLX
        on this device at the config's geometry, before the memory plan prices the dense and expert
        kernels: a kernel whose check fails or raises is switched off and its stock ops run."""
        from jitengine_qwen3_8_flash_next_m2.kernels.selfcheck import report, tuned_note
        from jitengine_qwen3_8_flash_next_m2.model import mtp as _mtp
        from jitengine_qwen3_8_flash_next_m2.model.kernel_checks import run_kernel_checks

        self.cfg: Qwen4ExpTextConfig = load_config(model_path)
        _mtp.resolve(model_path)
        mtp_head = _mtp.present(self.cfg)
        checks = run_kernel_checks(
            self.cfg,
            getattr(mx, self.compute_dtype_name),
            self.dense_kernel,
            self.moe_kernel,
            self.moe_kernel_variant,
            mtp=mtp_head,
        )
        self.kernel_checks = dict(checks.results)
        self.kernel_check_errors = dict(checks.errors)
        self.kernels_off = frozenset(name for name, ok in self.kernel_checks.items() if not ok)
        logger.info(
            f"[engine] kernel self-checks vs stock MLX on this device: {self.kernel_checks}"
        )
        device = mx.device_info()
        gpu = str(device.get("architecture"))
        warning = report(self.kernel_checks, mx.__version__, gpu, self.kernel_check_errors)
        if warning:
            logger.warning(f"[engine] WARNING: {warning}")
        note = tuned_note(str(device.get("device_name", "")), gpu, checks_passed=warning is None)
        if note:
            logger.info(f"[engine] {note}")
        if "dense_kernel" in self.kernels_off:
            self.dense_kernel = "stock"
        if "moe_kernel" in self.kernels_off:
            self.moe_kernel = "stock"
        # The verify pass is exact only if its kernels give the solo step's values bit for bit;
        # with one of MTP_KERNELS off, MTP stays off and the lone row decodes on the solo path.
        self.mtp_kernels_off = tuple(n for n in MTP_KERNELS if n in self.kernels_off)
        if mtp_head and self.mtp_kernels_off:
            logger.warning(
                f"[engine] MTP off: {', '.join(self.mtp_kernels_off)} failed the load-time "
                f"check, so the lone greedy row decodes on the solo path"
            )

    def init_memory_plan(self, model_path: str, options: dict) -> None:
        """A header-only inventory of the checkpoint, the MTP head's size, and the memory plan from
        them and the config.

        The plan comes before the tokenizer, MLX limits, weights, packs and worker thread, so a
        refused budget costs only the kernel checks and header I/O. The checkpoint scan comes back
        because the streamed embedding table reads it again.
        """
        from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_reader import CheckpointScan

        scan = CheckpointScan(model_path)
        _plan_opts = dict(
            options,
            compute_dtype=self.compute_dtype_name,
            moe_kernel=self.moe_kernel,
            dense_kernel=self.dense_kernel,
        )
        # Pass the served wave budget explicitly; `plan_budget` steps it down if it does not fit.
        _plan_opts.setdefault("wave_tokens", WAVE_TOKENS_SERVED)
        # Greedy single-stream decoding drafts with the checkpoint's MTP head and verifies two
        # positions per pass (Model.verify2: every emitted token's logits bit-identical to a solo
        # step's).
        from jitengine_qwen3_8_flash_next_m2.model import mtp as _mtp

        _mtp.resolve(model_path)
        self.plan = plan_budget(self.memory_budget_bytes, scan, self.cfg, _plan_opts)
        self.mtp_on = (
            _mtp.present(self.cfg)
            and not self.mtp_kernels_off
            and self.plan.capacity_fraction >= MTP_MIN_CAPACITY
        )
        # The head is not in the plan: it lives in the single-row extension (memory the plan reserves
        # for other rows and for prefill scratch, idle while one row decodes), loaded when the
        # extension grows for a lone row and released before it shrinks, so the main pool and every
        # multi-row mode are exactly the head-free plan's. Its bytes are the file (an upper bound)
        # plus the speculative round's transients.
        self.mtp_bytes = (_mtp.resident_bytes() + _mtp.MTP_TRANSIENT_BYTES) if self.mtp_on else 0
        return scan

    def init_tokenizer(self, model_path: str) -> None:
        """The tokenizer and the end-of-sequence token ids."""
        self.tokenizer = Tokenizer(model_path)
        self.eos_ids = [int(t) for t in self.cfg.eos_token_ids]

    def init_schedule(self, options: dict) -> None:
        """The sizes the plan fixes, and how prompts are admitted and prefilled: the chunk grid, the
        waves, the decode turn and the admission window."""
        self.delta_bytes = self.plan.state_per_row
        self.kv_reserve = int(self.plan.kv_reserve)  # decode tokens pre-sized in the KV
        self.expert_pool = None
        self.expert_store = None
        self.ple_store = None
        # the readout group the plan sized (budget.score_scratch_bytes charges exactly it)
        self.score_readout_rows = min(SCORE_READOUT_ROWS, int(self.plan.score_readout_rows))
        # `prefix_grid_tokens`: the absolute chunk grid that generate rows, their snapshots and
        # `score` all use. `prefix_min_divergence_tokens`: the store's "worth a snapshot" unit.
        self.wave_chunk = max(
            1, int(options.get("prefix_grid_tokens", options.get("wave_chunk", WAVE_CHUNK)))
        )
        self.prefix_min_divergence = max(
            1, int(options.get("prefix_min_divergence_tokens", MIN_DIVERGENCE_TOKENS))
        )
        # A wave may carry at most this many rows; 1 disables waves.
        self.wave_width = max(1, int(options.get("wave_width", self.plan.max_slots)))
        # Decode in turns: with several rows decoding, issue one row at a time (b == 1, the solo
        # Model.forward path with lookahead, miss chain and decay_solo) for a quantum of `decode_turn`
        # tokens, rotating FIFO in decode-entry order, while the prefill wave stays batched. 0
        # restores the batched decode bit for bit. A parked row waits at most (rows-1)*Q tokens.
        self.decode_turn_q = max(0, int(options.get("decode_turn", DECODE_TURN)))
        # The admission window (see ADMIT_WINDOW_S). 0 disables it.
        self.admit_window_s = max(0.0, float(options.get("admit_window_s", ADMIT_WINDOW_S)))
        self.admit_quiet_s = max(0.0, float(options.get("admit_quiet_s", ADMIT_QUIET_S)))
        self.admit_min_tokens = max(
            1,
            int(options.get("admit_min_tokens", min(ADMIT_MIN_TOKENS, self.prefix_min_divergence))),
        )
        # quiet timeout while every pending job is fully covered (see ADMIT_WARM_QUIET_S); 0 = off
        self.admit_warm_quiet_s = max(
            0.0, float(options.get("admit_warm_quiet_s", ADMIT_WARM_QUIET_S))
        )
        self.wave_tokens = self.plan.wave_tokens
        self.prefix_store_mb = self.plan.prefix_store_mb
        self.max_slots = self.plan.max_slots
        self.kv_budget_mb = self.plan.kv_budget_mb
        logger.info(f"[engine] plan {self.plan.describe()}")
        for line in self.plan.table().splitlines():
            logger.debug(line)

    def init_store_and_stats(self) -> None:
        """An empty prefix store and empty request statistics."""
        self.store = PrefixCache(
            cap_bytes=int(self.plan.prefix_store_bytes), min_divergence=self.prefix_min_divergence
        )
        self.stats_ = Stats()

    def init_memory_limits(self, options: dict) -> None:
        """MLX's memory, wired and cache limits, from the plan."""
        # An over-allocation must raise a catchable exception, not fault Metal. The MLX limit sits
        # below the budget by the non-MLX footprint; `iogpu.wired_limit_mb` is never touched.
        info = mx.device_info()
        try:
            self.memory_limit_bytes = int(
                options.get("memory_limit_bytes") or self.plan.memory_limit_bytes
            )
            wired = min(int(info["max_recommended_working_set_size"]), self.memory_limit_bytes)
            mx.set_memory_limit(self.memory_limit_bytes)
            mx.set_wired_limit(wired)
        except Exception as exc:  # noqa: BLE001
            self.memory_limit_bytes = 0
            logger.warning(f"[engine] set_wired_limit / set_memory_limit failed: {exc}")
        logger.debug(
            f"[engine] MLX memory limit {self.memory_limit_bytes / GIB:.2f} GiB "
            f"(over-allocation raises rather than faulting Metal)"
        )
        mx.set_cache_limit(int(self.plan.cache_bytes))

    def init_model(self, model_path: str, options: dict, scan) -> None:
        """The streamed embedding table (untied checkpoints), the model built from the weights with
        the expert and PLE packs open, and the kernels it runs, which go to the log."""
        # an untied embedding table is streamed from the checkpoint, not resident; the plan already
        # left its bytes to the pool (budget.plan_budget, embed_streamed_bytes)
        from jitengine_qwen3_8_flash_next_m2.model import embed_store as _es

        self.embed_store = None
        if _es.embed_streamed_bytes(scan) > 0:
            _eb, _eg = self.cfg.quant_of(_es.EMBED)
            self.embed_store = _es.StreamedEmbedding(
                scan, _eb, _eg, nocache=bool(options.get("embed_io_nocache", True))
            )
            logger.info(
                f"[engine] embed_tokens streamed from the checkpoint: {self.embed_store.out_dims} rows x "
                f"{self.embed_store.row_bytes} B = {self.embed_store.table_bytes / 1e9:.3f} GB not resident"
            )
        weights = load_language_weights(
            model_path, skip_experts=True, skip_embed=self.embed_store is not None
        )
        self._open_expert_pool(model_path)
        self.ple_store = self._open_ple_store(model_path)
        self.compute_dtype = {
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
            "float16": mx.float16,
        }[self.compute_dtype_name]
        _rd = options.get("residual_dtype")
        self.residual_dtype = {"bfloat16": mx.bfloat16, "float32": mx.float32, None: None}[
            None if _rd is None else str(_rd)
        ]
        _rt = options.get("router_dtype")
        self.router_dtype = {"bfloat16": mx.bfloat16, "float32": mx.float32, None: None}[
            None if _rt is None else str(_rt)
        ]
        self.meter = PhaseMeter(self.expert_pool)
        # Where a wave's activation scratch dies (layers.WAVE_EVAL): it changes when arrays are
        # evaluated, never what is computed, so every setting is bit-identical.
        self.wave_eval = str(options.get("wave_eval", WAVE_EVAL))
        self.model = Qwen4ExpForCausalLM(
            self.cfg,
            weights,
            pool=self.expert_pool,
            dense_kernel=self.dense_kernel,
            sync_each_layer=bool(options.get("moe_sync_each_layer", False)),
            wave_eval=self.wave_eval,
            ple_store=self.ple_store,
            compute_dtype=self.compute_dtype,
            residual_dtype=self.residual_dtype,
            router_dtype=self.router_dtype,
            moe_kernel=self.moe_kernel,
            moe_kernel_variant=self.moe_kernel_variant,
            hc_path=self.hc_path,
            embed=self.embed_store,
            kernels_off=self.kernels_off,
        )
        self.model.prefetch_lookahead = self.prefetch_lookahead
        self.mtp = None  # loaded only while the single-row extension is live (_ext_policy)
        self.mtp_gen = 0
        self.mtp_loads = 0
        self.moe_kernel_info = self.model.moe_kernel_info()
        self.hc_info = self.model.hc_info()
        self.dense_kernel_info = self.model.dense_kernel_info()
        logger.debug(
            f"[engine] hyper-connection path: {self.hc_info['option']} over {self.hc_info['sites']} sites, "
            f"regions {self.hc_info['regions']}; not fused: {self.hc_info['not_fused']}"
        )
        logger.debug(
            f"[engine] moe kernel: {self.moe_kernel_info['kernel']} (option {self.moe_kernel}, variant "
            f"{self.moe_kernel_info['variant']}, fallback layers {self.moe_kernel_info['fallback_layers']}); "
            f"configs {self.moe_kernel_info['configs']}"
        )
        if self.moe_kernel_info["kernel"] != self.plan.moe_kernel:
            # the plan decided the fp32 upcast charge from header geometry (budget.moe_kernel_for);
            # the model must run the kernel the plan priced, or the plan is not this engine's
            raise RuntimeError(
                f"memory plan priced moe kernel {self.plan.moe_kernel!r} but the model "
                f"runs {self.moe_kernel_info['kernel']!r}"
            )

    def init_slots(self) -> None:
        """The KV slot pool and the context bounds it implies."""
        self.vocab = self.model.lm_head.out_dims
        # Token ids index the embedding table, and MLX's gather does not bounds-check; lm_head
        # matches its row count here only because the weights are tied.
        self.token_id_limit = min(int(self.vocab), int(self.model.embed.out_dims))
        self.pool = SlotPool(self.max_slots, int(self.plan.kv_bytes))
        # the per-row KV ceiling, derived from the plan
        self.act_bytes = 4 if self.compute_dtype == mx.float32 else 2
        self.kv_bytes_per_position = kv_bytes_for(self.cfg, 1, self.act_bytes)
        self.max_kv_positions = int(self.pool.kv_budget_bytes // self.kv_bytes_per_position)
        logger.info(
            f"[engine] state plan: {self.max_slots} slots x {self.delta_bytes / 2**20:.1f} MiB recurrent "
            f"= {self.max_slots * self.delta_bytes / GIB:.2f} GiB max, KV budget "
            f"{self.kv_budget_mb:.0f} MB ({self.max_kv_positions} positions), prefix store "
            f"{self.prefix_store_mb:.0f} MB, grid {self.wave_chunk} / min divergence {self.prefix_min_divergence}"
        )

    def init_scheduler(self) -> None:
        """The scheduler's state, which only the worker thread touches: queues, the decode turn, the
        MTP counters and the liveness flag."""
        # ------------------------------------------------------------ scheduler state (worker only)
        self._queue: queue.Queue = queue.Queue()
        self._pending: list[Job] = []  # admitted-to-the-engine, not yet holding a slot
        self._pending_dirty = True  # group prefix points need recomputing
        self._group_points: dict[int, int] = {}
        self._scores: list[Job] = []
        self._rows: list[Req] = []  # rows holding a slot (prefilling or decoding)
        self._batch: list[Req] | None = None  # rows of the in-flight decode step
        self._y = None  # its lazy [B] sampled tokens
        self._telem: StepTelemetry | None = None
        self._reads: list = []  # its lazy per-row drawn-from readouts
        self._step_no = 0
        self._telem_tokens = 0  # row-tokens decoded since the last armed step
        self._work = 0
        self.waves_run = 0  # multi-row prefill waves executed
        self.wave_rows_total = 0  # rows summed over them (mean width = total / waves_run)
        self.wave_rows_max = 0  # the widest wave that ran (published as scheduler.wave_rows)
        self.wave_tokens_max = 0  # the most tokens ever in flight in one wave (the memory driver)
        self.admit_windows = 0  # admission windows opened
        # ...of which opened on the warm clause (every pending job fully covered)
        self.admit_warm_windows = 0
        self.admit_window_seconds = 0.0
        self.admit_window_collected = 0  # jobs that landed inside a window (the wave's members)
        self.row_faults = 0  # rows failed by an injected / real read fault
        # decode in turns: `_turn_order` is the FIFO of decoding rows (the owner moves to the back
        # when its quantum ends); `_turn_used` counts the owner's steps in the current turn
        self._turn_order: list[Req] = []
        self._turn_owner: Req | None = None
        self._turn_used = 0
        self.turns = 0  # turns started (an owner change, or the first owner)
        self.parked_rows_peak = 0  # most rows parked behind an owner at once
        self.turn_ext_slots = 0  # the last turn-mode extension operating point computed
        self.decode_rows_active_max = 0  # widest decode step ever issued (1 in turn mode)
        self.mtp_verifies = 0  # speculative rounds (two-position verifies) and accepted drafts
        self.mtp_accepted = 0
        # Liveness: the worker's `finally` sets `_dead` (to the fault) before failing every waiter;
        # `/health` and `_submit` read it, and nothing else writes it.
        self._dead: BaseException | None = None

    def init_worker(self) -> None:
        """Start the model worker thread and warm it up."""
        self._worker = threading.Thread(
            target=self._worker_loop, name="moe-engine-model", daemon=True
        )
        self._worker.start()
        self._warmup()

    # ================================================================== public contract
    def _submit(self, job: Job) -> Job:
        """Hand a job to the model thread, or fail it at once if that thread is gone.

        `_dead` is re-read after the put, so a job submitted during teardown is either drained by
        the dying worker or failed here, never left on a queue nobody serves.
        """
        if self._dead is not None:
            job.out.put(("error", self._dead))
            return job
        self._queue.put(job)
        if self._dead is not None:
            self._reap_queue(self._dead)
        return job

    def health(self) -> dict:
        """Observed liveness: the model thread is alive and the engine has not failed."""
        alive = bool(self._worker.is_alive()) and self._dead is None
        out = {
            "status": "ok" if alive else "error",
            "memory_bytes": self.memory_bytes(),
            "worker_alive": bool(self._worker.is_alive()),
        }
        if not alive:
            out["error"] = (
                repr(self._dead) if self._dead is not None else "the model thread is not running"
            )
        return out

    def _warmup(self) -> None:
        req = GenRequest(prompt_ids=[self.eos_ids[-1], 1, 2], max_tokens=2, temperature=0.0)
        self.generate(req)
        req = GenRequest(
            prompt_ids=[1, 2], max_tokens=1, temperature=0.7, top_p=0.95, top_k=40, seed=1
        )
        self.generate(req)

    def events(self, req: GenRequest, logprobs: int = 0) -> Iterator[tuple[str, object]]:
        """Yield ("token", id) per decode step, then ("done", GenResult). Raises on engine error.

        With `logprobs` > 0, `extra["logprobs"]` holds one score-shaped entry per decode step for
        the distribution that token was drawn from; the forward and sampling are unchanged.
        """
        job = self._submit(Job("generate", req, logprobs=int(logprobs or 0)))
        try:
            while True:
                kind, payload = job.out.get()
                if kind == "error":
                    raise payload
                yield kind, payload
                if kind == "done":
                    return
        finally:
            job.cancelled = True

    def generate(self, req: GenRequest, logprobs: int = 0) -> GenResult:
        """Run one request to its end; `logprobs` is as for `events`. A `ValueError` does not
        raise but gives a result with `finish_reason` "error" and the message in
        `extra["error"]`; any other failure raises."""
        try:
            for kind, payload in self.events(req, logprobs=logprobs):
                if kind == "done":
                    return payload
        except ValueError as exc:
            return GenResult(
                token_ids=[],
                finish_reason="error",
                prompt_tokens=len(req.prompt_ids),
                extra={"error": str(exc)},
            )
        raise RuntimeError("generate ended without a result")

    def stream(self, req: GenRequest) -> Iterator[int]:
        """The token ids of one request, each as the worker decodes it; raises on any refusal or
        failure."""
        for kind, payload in self.events(req):
            if kind == "token":
                yield payload

    def generate_batch(self, reqs: Sequence[GenRequest]) -> list[GenResult]:
        """Submit every request before waiting, so the worker can batch them; the results come
        back in request order. A request that fails, whatever the failure, gives a `GenResult`
        with `finish_reason` "error" and the message in `extra["error"]` instead of raising."""
        jobs = [self._submit(Job("generate", r)) for r in reqs]
        results = []
        for r, job in zip(reqs, jobs):
            res = None
            while True:
                kind, payload = job.out.get()
                if kind == "done":
                    res = payload
                    break
                if kind == "error":
                    res = GenResult(
                        token_ids=[],
                        finish_reason="error",
                        prompt_tokens=len(r.prompt_ids),
                        extra={"error": str(payload)},
                    )
                    break
            results.append(res)
        return results

    def score(
        self, prompt_ids: Sequence[int], continuation_ids: Sequence[int] = (), top_k: int = 5
    ) -> list[dict]:
        """One entry per position of `prompt_ids + continuation_ids`; see `_run_score`."""
        job = self._submit(
            Job("score", payload=(list(prompt_ids), list(continuation_ids), int(top_k)))
        )
        kind, payload = job.out.get()
        if kind == "error":
            raise payload
        return payload

    def memory_bytes(self) -> int:
        """Bytes of memory MLX is actively using, not counting the buffers it caches."""
        return int(mx.get_active_memory())

    def stats(self) -> dict:
        """Cumulative counters, prefix-store and batcher counters, and the ring of recent records."""
        ps = self.store.stats()
        out = {
            "memory_bytes": self.memory_bytes(),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "memory_limit_bytes": int(self.memory_limit_bytes),
            "cache_memory_bytes": int(mx.get_cache_memory()),
            "prefix_store": ps,
            "mtp": {
                "enabled": bool(self.mtp_on),
                "loaded": self.mtp is not None,
                "loads": self.mtp_loads,
                "verifies": self.mtp_verifies,
                "accepted": self.mtp_accepted,
            },
            "prefix_cache_entries": ps["entries"],
            "prefix_cache_bytes": ps["bytes"],
            "phases": self.meter.snapshot(),
            "batcher": {
                **self.pool.stats(),
                "delta_bytes_per_row": self.delta_bytes,
                "queued": len(self._pending),
            },
            "scheduler": {
                "wave_chunk": self.wave_chunk,
                "wave_width": self.wave_width,
                # score walks the served grid; there is no second chunk size
                "score_chunk": self.wave_chunk,
                "score_readout_rows": int(self.score_readout_rows),
                "prefill_waves": self.waves_run,
                # `wave_rows`: the widest wave that ran; `wave_rows_total` sums them
                "wave_rows": self.wave_rows_max,
                "wave_rows_max": self.wave_rows_max,
                "wave_rows_total": self.wave_rows_total,
                "wave_rows_mean": (
                    self.wave_rows_total / self.waves_run if self.waves_run else 0.0
                ),
                # one per layer per multi-row wave; unlike `expert_wave_calls`, which
                # counts capacity-partitioned segment waves within a layer
                "expert_wave_multi_calls": (
                    self.expert_pool.wave_multi_calls if self.expert_pool is not None else 0
                ),
                "expert_wave_calls": (
                    self.expert_pool.wave_calls if self.expert_pool is not None else 0
                ),
                "admit_window_s": self.admit_window_s,
                "admit_quiet_s": self.admit_quiet_s,
                "admit_warm_quiet_s": self.admit_warm_quiet_s,
                "admit_min_tokens": self.admit_min_tokens,
                "wave_tokens": self.wave_tokens,
                "wave_tokens_max": self.wave_tokens_max,
                "admit_windows": self.admit_windows,
                "admit_warm_windows": self.admit_warm_windows,
                "admit_window_seconds": round(self.admit_window_seconds, 3),
                "admit_window_collected": self.admit_window_collected,
                # decode in turns
                "decode_turn_q": int(self.decode_turn_q),
                "turns": int(self.turns),
                "parked_rows_peak": int(self.parked_rows_peak),
                "turn_ext_slots": int(self.turn_ext_slots),
                "decode_rows_active_max": int(self.decode_rows_active_max),
            },
            **self.stats_.snapshot(),
        }
        out["memory_budget_bytes"] = self.memory_budget_bytes
        out["memory_budget_gib"] = self.memory_budget_bytes / GIB
        out["memory_plan"] = self.plan.as_dict()
        out["memory_plan_summary"] = self.plan.describe()
        out["memory_plan_table"] = self.plan.table()
        out["prefix_cache_cap_bytes"] = int(self.plan.prefix_store_bytes)
        out["prefix_cache_evictions"] = ps["evictions"]
        out["prefix_snapshot_grid_tokens"] = int(self.wave_chunk)
        out["prefix_min_divergence_tokens"] = int(self.prefix_min_divergence)
        out["max_context_tokens"] = int(self.plan.max_context)
        out["compute_dtype"] = self.compute_dtype_name
        out["prefetch_lookahead"] = int(self.prefetch_lookahead)
        out["prefetch_reserve"] = int(self.prefetch_reserve)
        out["landing_slots"] = int(self.landing_slots)
        # the readout's row-exact group size per weight shape, at this compute dtype
        out["score_readout"] = {
            "rows_per_group": int(self.score_readout_rows),
            "rows_per_group_max": int(SCORE_READOUT_ROWS),
            "row_limits": self.model.row_limits(),
            "grid_tokens": int(self.wave_chunk),
        }
        out["moe_kernel"] = dict(self.moe_kernel_info)
        out["hc_path"] = dict(self.hc_info)
        out["dense_kernel"] = dict(self.dense_kernel_info)
        out["kernel_checks"] = dict(self.kernel_checks)
        out["kernel_check_errors"] = dict(self.kernel_check_errors)
        out["fault_injections"] = int(self.fault_seam.fired) if self.fault_seam is not None else 0
        out["row_faults"] = int(self.row_faults)
        if self.ple_store is not None:
            out.update(self.ple_store.stats())
        out.update(
            self.embed_store.stats() if self.embed_store is not None else {"embed_streamed": False}
        )
        if self.expert_pool is not None:
            # cumulative expert counters
            out.update(self.expert_pool.stats())
        return out

    def serve(self, host: str = "127.0.0.1", port: int = 8080, allowed_hosts=()) -> None:
        """Serve the OpenAI-compatible HTTP API on `host:port`; blocks until the server stops
        (`server.api_server.serve`)."""
        from jitengine_qwen3_8_flash_next_m2.server.api_server import serve

        serve(self, host, port, allowed_hosts=allowed_hosts)


def create_engine(model_path: str, **options) -> Engine:
    """Build an `Engine` for the checkpoint at `model_path`, warmed up and ready for requests;
    `options` are the engine options, checked by `Engine.init_options`."""
    return Engine(model_path, **options)
