"""The Engine: one model, one worker thread that owns all GPU work, and a continuous slot batcher.

Public calls submit a job and wait on its own queue, so any thread may call. Prefill is always a
single-row forward, which keeps a prefix-store hit bit-identical to a cold run: the dense kernel
path depends on the row count, so nothing a batched step produces may enter the store.

`Engine` holds the constructor (the `init_*` steps) and the public calls. Its other parts are
mixins, each in the subsystem it serves: the model thread (`scheduler.scheduler`), admission
(`scheduler.admission`), prompt processing (`scheduler.prefill`), decoding (`scheduler.decode`),
retirement (`scheduler.retire`), scoring (`scheduler.score`), expert offloading (`moe.offload`)
and memory control (`memory.control`).

The state the mixins share: the `init_*` steps set each attribute below first; after that its owner
keeps it consistent, and the classes in the last column write it only to reset it when a step fails
or the engine stops, unless noted (tests/unit/engine/test_engine_state.py checks the table against
the code).

| attributes | owner | also written by |
|:--|:--|:--|
| `_batch`, `_y`, `_telem` | DecodingMixin | RetirementMixin, MemoryControlMixin, SchedulingMixin |
| `_step_no` | DecodingMixin | |
| `_pending` | AdmissionMixin | MemoryControlMixin, SchedulingMixin |
| `_pending_dirty` | AdmissionMixin | MemoryControlMixin; SchedulingMixin marks arrivals |
| `_group_points` | AdmissionMixin | |
| `_prefilling` | PromptProcessingMixin | MemoryControlMixin; RetirementMixin clears a retiring row's |
| `_dead`, `_work`, `_scores` | SchedulingMixin | |
| `expert_store` | OffloadingMixin | |
| `expert_pool` | OffloadingMixin | MemoryControlMixin swaps it on a resize |
| `plan`, `limits`, `memory_limit_bytes`, `pool_generation`, `last_resize` | MemoryControlMixin | |
| `_admission_paused`, `controller`, `monitor` | MemoryControlMixin | |
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Iterator, Sequence

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.loader import load_language_weights
from jitengine_qwen3_6_35b_a3b_m4pro.engine.args import (  # noqa: F401  (re-exported)
    DEFAULTS,
    resolve_options,
)
from jitengine_qwen3_6_35b_a3b_m4pro.engine.errors import (
    AdmissionPausedError,
    MemoryPressureError,
    QueueFullError,
)
from jitengine_qwen3_6_35b_a3b_m4pro.engine.io_struct import GenRequest, GenResult
from jitengine_qwen3_6_35b_a3b_m4pro.kvcache.prefix_cache import PrefixCache
from jitengine_qwen3_6_35b_a3b_m4pro.linear import dense_kernel_report
from jitengine_qwen3_6_35b_a3b_m4pro.memory.budget import BudgetError, make_plan
from jitengine_qwen3_6_35b_a3b_m4pro.memory.control import MemoryControlMixin
from jitengine_qwen3_6_35b_a3b_m4pro.model.config import Qwen3_5MoeConfig, load_config
from jitengine_qwen3_6_35b_a3b_m4pro.model.kernel_choice import KernelChoice, Strategies
from jitengine_qwen3_6_35b_a3b_m4pro.model.model import Qwen3_5MoeForCausalLM
from jitengine_qwen3_6_35b_a3b_m4pro.model.weight import FUSIONS
from jitengine_qwen3_6_35b_a3b_m4pro.moe.offload import OffloadingMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.admission import AdmissionMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.batch import (
    Job,
    Req,
    SlotPool,
    delta_bytes_for,
    kv_bytes_for,
)
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.decode import DecodingMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.prefill import PromptProcessingMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.retire import RetirementMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.scheduler import SchedulingMixin
from jitengine_qwen3_6_35b_a3b_m4pro.scheduler.score import ScoringMixin
from jitengine_qwen3_6_35b_a3b_m4pro.server.stats import Stats, StepTelemetry
from jitengine_qwen3_6_35b_a3b_m4pro.tokenizer import Tokenizer


def run_load_checks(cfg, offload: bool) -> tuple:
    """Run every custom-kernel self-check at the served quantization width; returns the `Checks` and the expert results.

    A kernel that does not take `cfg.q_bits` is neither checked nor reported (the prescaled GEMV and
    the prescaled, rows and segmented expert kernels are 3-bit only). The expert results are {} when
    offloading."""
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.expert_solo import check_experts, expert_checks
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_exact import check_gdn_exact
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.gdn_recurrence import check_gdn_rows
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.qmv_prescaled import check_dense, covers
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import check_sort
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.rms_norm_prescale import check_rms_pre
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.router_topk import check_router, check_topk
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.sdpa_decode import check_sdpa, check_sdpa_rows
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels.selfcheck import Checks
    from jitengine_qwen3_6_35b_a3b_m4pro.sampler import check_sampler_tail

    bits, group_size = cfg.q_bits, cfg.q_group_size
    checks = Checks()
    if covers(bits, group_size):
        checks.run("dense_v4", check_dense, bits, group_size)
    exp_ok = (
        {}
        if offload
        else checks.run_many(
            expert_checks(bits),
            check_experts,
            cfg.moe_intermediate_size,
            cfg.hidden_size,
            bits,
            group_size,
        )
    )
    attention = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    checks.run("exact_sdpa", check_sdpa, *attention)
    checks.run("exact_topk", check_topk, cfg.num_experts, cfg.num_experts_per_tok)
    checks.run("fused_norm", check_rms_pre, cfg.hidden_size)
    checks.run("gdn_exact", check_gdn_exact, cfg)
    checks.run("fused_router", check_router, cfg.num_experts, cfg.num_experts_per_tok)
    checks.run("radix_sort", check_sort)
    checks.run("sampler_tail", check_sampler_tail, bool(checks.results["radix_sort"]))
    checks.run("sdpa_rows", check_sdpa_rows, *attention)
    checks.run("gdn_rows", check_gdn_rows)
    return checks, exp_ok


logger = logging.getLogger(__name__)


class Engine(
    OffloadingMixin,
    SchedulingMixin,
    AdmissionMixin,
    PromptProcessingMixin,
    DecodingMixin,
    RetirementMixin,
    ScoringMixin,
    MemoryControlMixin,
):
    """One model, a worker thread that runs every request, and a continuous slot batcher.

    `generate`, `stream`, `generate_batch` and `score` submit a job and wait on its own queue, so
    any thread may call them.
    """

    def __init__(self, model_path: str, **options):
        opt = self.init_options(model_path, options)

        t0 = time.monotonic()
        self.init_checkpoint(model_path)

        info = self.init_memory_plan(model_path, opt)
        self.init_store_and_stats(opt)

        self.init_memory_limits(opt, info)

        self.init_model(model_path, opt)
        self.init_prefill_schedule(opt)
        self.vocab = self.model.lm_head.out_dims
        self.init_kernels()
        self.init_speculation(model_path, opt)
        self.init_slots(opt)

        self.init_scheduler()

        self.init_worker()
        self.init_store_and_stats(opt)  # warm-up entries never count
        self.ready_seconds = time.monotonic() - t0
        if self.memory_policy == "adaptive" or opt.get("monitor"):
            self._start_monitor()
        logger.info(
            f"[engine] ready in {self.ready_seconds:.1f}s; active memory {mx.get_active_memory() / 2**30:.2f} GiB"
            + ("; WARNING: some kernels are off (see above)" if self.kernel_warning else "")
        )

    def init_options(self, model_path: str, options: dict) -> dict:
        """The engine options, resolved against the defaults; the scalar ones become attributes."""
        self.model_path = model_path
        self.options = resolve_options(options)
        opt = self.options
        self.model_name = opt.get("model_name") or model_path
        self.kv_reserve = int(opt.get("kv_reserve", 512))  # decode tokens pre-sized in the KV
        self.log_requests = bool(opt.get("log_requests", True))
        self.max_context_tokens = int(opt["max_context_tokens"])
        self.max_queued_requests = int(opt["max_queued_requests"])
        self.prefill_chunk = int(opt["prefill_chunk"])
        self.max_slots = int(opt["max_slots"])
        self.memory_policy = opt["memory_policy"]
        return opt

    def init_checkpoint(self, model_path: str) -> None:
        """The model config, a header-only inventory of the checkpoint, and the tokenizer."""
        self.cfg: Qwen3_5MoeConfig = load_config(model_path)
        # header-only validation before any tensor is read; an unsupported checkpoint stops here
        from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint.inventory import inventory

        self.inventory = inventory(model_path, self.cfg)
        self.tokenizer = Tokenizer(model_path)
        self.eos_ids = [int(t) for t in self.cfg.eos_token_ids]
        self.delta_bytes = delta_bytes_for(self.cfg)

    def init_memory_plan(self, model_path: str, opt: dict) -> dict:
        """Plan memory in integer bytes before anything is allocated; returns the device info."""
        from jitengine_qwen3_6_35b_a3b_m4pro.memory.budget import capacity_for, limits_for

        (self.capacity, _, _, self.scan, self.kv_budget_bytes, _) = capacity_for(
            model_path, opt, cfg=self.cfg, inv=self.inventory
        )
        info = mx.device_info()
        self.device_info = {k: (int(v) if isinstance(v, int) else v) for k, v in info.items()}
        self.limits = limits_for(opt, self.device_info)
        self.plan = make_plan(self.capacity, self.limits, opt["mode"])
        self.configured_limits = self.limits
        self.state = "serving"
        self.offload = self.plan.mode == "offload"
        self.max_buffer_bytes = int(info.get("max_buffer_length") or 0)
        self.expert_pool = None
        self.expert_store = None
        logger.info(f"[engine] plan {self.plan.describe()}")
        return info

    def init_store_and_stats(self, opt: dict) -> None:
        """An empty prefix store and empty request statistics."""
        self.store = PrefixCache(cap_bytes=int(opt["prefix_cache_bytes"]))
        self.stats_ = Stats()

    def init_memory_limits(self, opt: dict, info: dict) -> None:
        """MLX's memory, wired and cache limits, from the plan."""
        # `set_memory_limit` is a guideline inside MLX's allocator (an allocation above it clears
        # the cache, then raises), not an OS reservation; the wired limit keeps buffers resident
        self.limit_status = {
            "memory_limit_bytes": None,
            "wired_limit_bytes": None,
            "cache_limit_bytes": None,
            "errors": [],
            "memory_limit_is_guideline": True,
        }
        try:
            mx.set_memory_limit(int(self.plan.mlx_cap_bytes))
            self.limit_status["memory_limit_bytes"] = int(self.plan.mlx_cap_bytes)
        except Exception as exc:
            raise BudgetError(
                f"mx.set_memory_limit({self.plan.mlx_cap_bytes}) failed: {exc!r}"
            ) from exc
        wired = min(int(self.plan.planned_mlx_bytes), int(info["max_recommended_working_set_size"]))
        try:
            mx.set_wired_limit(wired)
            self.limit_status["wired_limit_bytes"] = wired
        except Exception as exc:  # noqa: BLE001  performance setting: record, continue unwired
            self.limit_status["errors"].append(f"set_wired_limit({wired}): {exc!r}")
            logger.warning(
                f"[engine] set_wired_limit({wired}) failed, continuing without it: {exc!r}"
            )
        mx.set_cache_limit(int(opt["allocator_cache_bytes"]))
        self.limit_status["cache_limit_bytes"] = int(opt["allocator_cache_bytes"])
        self.memory_limit_bytes = int(self.plan.mlx_cap_bytes)

    def init_model(self, model_path: str, opt: dict) -> None:
        """The model, built from the weights (the experts come from a pool when offloading)."""
        lazy = bool(opt.get("fuse_projections", True))
        weights = load_language_weights(model_path, skip_experts=self.offload, materialize=not lazy)
        self.weight_bytes = sum(v.nbytes for v in weights.values())
        want = self.capacity.dense_bytes + (0 if self.offload else self.capacity.all_expert_bytes)
        if self.weight_bytes != want:
            raise RuntimeError(
                f"loaded {self.weight_bytes} B of weights, the header scan planned "
                f"{want} B: the loader and the scan disagree about this checkpoint"
            )
        if self.offload:
            self._open_expert_pool(model_path)
        fp = opt.get("fuse_projections", True)
        if fp is True:
            # the fusions that are output-identical to the unfused engine (not the router's)
            fp = {"attn", "experts", "gdn", "shared"}
        if isinstance(fp, (list, tuple, set, frozenset)):
            fuse = frozenset(fp)
        else:
            fuse = FUSIONS if fp else frozenset()
        self.fuse_projections = bool(fuse)
        self.model = Qwen3_5MoeForCausalLM(self.cfg, weights, pool=self.expert_pool, fuse=fuse)
        if lazy:
            # evaluate what was not fused on this thread: a lazy graph is bound to the thread that
            # built it, so the model worker must only see evaluated weights
            mx.eval(*weights.values(), *self.model.arrays())
        del weights

    def init_prefill_schedule(self, opt: dict) -> None:
        """How prompts are prefilled: layer-major, with merged expert gathers, several at once."""
        self.speculative_gather = bool(opt.get("speculative_gather", True)) and self.offload
        merge = not self.offload
        self.model.use_strategies(
            Strategies(
                speculative=self.speculative_gather,
                merge_prefill_experts=merge,
                # layer-major prefill: async per-chunk evals, at most this many in flight (0 = blocking)
                prefill_async_depth=int(opt.get("prefill_async_depth", 4) or 0),
                verify_relaxed=bool(opt.get("verify_relaxed", False)),
            )
        )
        # resident: layer-major only buys something when the chunks' routed experts are merged
        self.layer_major_prefill = bool(opt.get("layer_major_prefill", True)) and (
            self.offload or merge
        )
        # while rows decode, a head up to this long still takes the bit-identical layer-major pass
        self.merge_decoding_max = 512
        # several pending prompts in one layer-major pass (merged expert gathers across prompts)
        self.group_prefill = (
            bool(opt.get("group_prefill", True)) and self.layer_major_prefill and not self.offload
        )
        self.group_prefill_max = int(opt.get("group_prefill_max_tokens", 4096) or 0)

    def init_kernels(self) -> None:
        """Check each custom kernel against stock MLX on this device; the model runs the ones that match."""
        checks, exp_ok = run_load_checks(self.cfg, self.offload)
        bits = self.cfg.q_bits
        dense_prescaled = checks.results.get("dense_v4", False)
        order = (
            "dense_v4",
            "exact_sdpa",
            "exact_topk",
            "fused_norm",
            "gdn_exact",
            "fused_router",
            "radix_sort",
            "sampler_tail",
            "sdpa_rows",
        )
        self.kernel_checks = {
            k: checks.results[k] for k in (*order, "gdn_rows", *exp_ok) if k in checks.results
        }
        self.kernel_check_errors = dict(checks.errors)
        logger.info(
            f"[engine] kernel self-checks vs stock MLX on this device: {self.kernel_checks}"
        )
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.selfcheck import report, tuned_note

        device = mx.device_info()
        gpu = str(device.get("architecture"))
        warning = report(
            self.kernel_checks, mx.__version__, gpu, checks.errors, dense_kernel_report()
        )
        if warning:
            logger.warning(f"[engine] WARNING: {warning}")
        self.kernel_warning = warning
        note = tuned_note(str(device.get("device_name", "")), gpu, checks_passed=warning is None)
        if note:
            logger.info(f"[engine] {note}")
        moe_prescaled = not self.offload and exp_ok.get("moe_v4", False)
        # one launch per decoder norm on the solo path: the norm and, where the prescaled GEMVs run
        # (3-bit weights), their prologue. At other widths nothing reads the prologue, and the kernel
        # still runs when its own check passes: in an A/B on the 48 GB M4 Pro (greedy 512, two rounds
        # each), one user decoded 195.94 and 195.63 tok/s with it
        # against 195.36 and 195.11 with the stock norm, with the same text
        fused_norm = not self.offload and bool(self.kernel_checks["fused_norm"])
        if bits == 3:
            fused_norm = fused_norm and moe_prescaled
        exact_topk = bool(self.kernel_checks["exact_topk"])
        self.model.use_kernel_choice(
            KernelChoice(
                dense_prescaled=bool(dense_prescaled),
                exact_experts=not self.offload and exp_ok.get("exact_experts", False),
                moe_prescaled=moe_prescaled,
                rows_exact=not self.offload and exp_ok.get("rows_v4", False),
                rows_prescaled=moe_prescaled and exp_ok.get("rows_v4", False),
                exact_topk=exact_topk,
                fused_router=exact_topk and bool(self.kernel_checks["fused_router"]),
                seg_prefill=not self.offload and exp_ok.get("seg_prefill", False),
                exact_sdpa=bool(self.kernel_checks["exact_sdpa"]),
                sdpa_rows=bool(self.kernel_checks["sdpa_rows"]),
                gdn_rows=bool(self.kernel_checks["gdn_rows"]),
                gdn_exact=bool(self.kernel_checks.get("gdn_exact")),
                fused_norm=fused_norm,
            )
        )
        self.radix_sort = bool(self.kernel_checks["radix_sort"])
        logger.info(f"[engine] sampler tail self-check: {self.kernel_checks['sampler_tail']}")
        self.fused_tail = bool(self.kernel_checks["sampler_tail"])

    def init_speculation(self, model_path: str, opt: dict) -> None:
        """Speculative decoding with the checkpoint's MTP head (single-stream rounds only), and the draft settings."""
        self.spec_k = int(opt.get("spec_k", 1) or 0)
        self.spec_sampling = str(
            opt.get("spec_sampling", "exact") or "exact"
        )  # "exact" | "leviathan" (relaxed)
        self.copy_max = 7
        self.copy_wide_max = 15
        self.mtp = None
        mtp_path = opt.get("mtp_path")
        if mtp_path is None:
            # find the checkpoint's own MTP head beside it (mlx-community's split `-MTP-4bit` repo) or
            # inside it (`mtp.safetensors`); an explicit mtp_path (or "" to disable) always wins
            import re as _re

            base = model_path.rstrip("/")
            for cand in (
                _re.sub(r"(-\d+bit)$", r"-MTP\1", base)
                + "/model.safetensors",  # X-4bit -> X-MTP-4bit
                base + "-MTP-4bit/model.safetensors",
                os.path.join(base, "mtp.safetensors"),
            ):
                if os.path.isfile(cand):
                    mtp_path = cand
                    break
        if mtp_path and self.spec_k > 0 and not self.offload:
            from jitengine_qwen3_6_35b_a3b_m4pro.model.mtp import MTPHead

            self.mtp = MTPHead(
                os.path.expanduser(mtp_path),
                self.model,
                model_path,
                draft_vocab=65536,
            )
            logger.info(
                f"[engine] speculative decoding: MTP head {self.mtp.nbytes() / 2**20:.0f} MiB, "
                f"k={self.spec_k}, draft vocab {self.mtp.V}"
            )
        self.model.mtp = self.mtp

    def init_slots(self, opt: dict) -> None:
        """The KV slot pool and the context bounds it implies."""
        # token ids index the embedding table and MLX's gather does not bounds-check, so bound by
        # its row count too (lm_head.out_dims only happens to match on this checkpoint)
        self.token_id_limit = min(int(self.vocab), int(self.model.embed.out_dims))
        self.pool = SlotPool(self.max_slots, int(self.kv_budget_bytes))
        self.kv_bytes_per_position = kv_bytes_for(self.cfg, 1)
        self.max_kv_positions = min(
            int(self.kv_budget_bytes // self.kv_bytes_per_position), self.max_context_tokens
        )
        logger.info(
            f"[engine] state plan: {self.max_slots} slot(s) x {self.delta_bytes / 2**20:.1f} MiB "
            f"recurrent, KV budget {self.kv_budget_bytes / 2**20:.0f} MiB, context "
            f"{self.max_context_tokens} tokens, prefix store {opt['prefix_cache_bytes'] / 2**20:.0f} MiB, "
            f"queue {self.max_queued_requests}"
        )

    def init_scheduler(self) -> None:
        """The scheduler's state, which only the worker thread touches."""
        self._queue: queue.Queue = queue.Queue()
        self._pending: list[Job] = []  # admitted-to-the-engine, not yet holding a slot
        self._pending_dirty = True  # group prefix points need recomputing
        self._group_points: dict[int, int] = {}
        self._scores: list[Job] = []
        self._rows: list[Req] = []  # rows holding a slot (prefilling or decoding)
        self._prefilling: Req | None = None
        self._batch: list[Req] | None = None  # rows of the in-flight decode step
        self._y = None  # its lazy [B] sampled tokens
        self._telem: StepTelemetry | None = None
        self._step_no = 0
        self._work = 0
        # set (to the fault) only by `_fail_everything`, before it fails every waiter; read by
        # `health` and `_submit`
        self._dead: BaseException | None = None
        self._closing = False  # close() asked the worker to stop
        # outstanding generate jobs (queued + pending + holding a slot), for the queue bound
        self._outstanding = 0
        self._outstanding_lock = threading.Lock()
        self.rejected_queue_full = 0
        # memory maintenance: the admission gate and the last resize result
        self._admission_paused: str | None = None
        self.last_resize: dict | None = None
        self.pool_generation = 0
        self.monitor = None
        self.controller = None

    def init_worker(self) -> None:
        """Start the model worker thread and warm it up."""
        self._worker = threading.Thread(
            target=self._worker_loop, name="moe-engine-model", daemon=True
        )
        self._worker.start()
        self._warmup()

    def _submit(self, job: Job) -> Job:
        """Hand a job to the model thread, or fail it at once if that thread is gone.

        `_dead` is set before the dying worker drains the queue and is re-read after the put, so a
        job submitted during teardown is either drained or failed here, never left waiting.
        """
        if self._dead is not None:
            job.out.put(("error", self._dead))
            return job
        if job.kind == "generate":
            # at most max_slots running + max_queued_requests waiting; excess gets a retryable error
            with self._outstanding_lock:
                paused = self._admission_paused
                if (
                    paused is None
                    and self._outstanding >= self.max_slots + self.max_queued_requests
                ):
                    paused = "queue_full"
                if paused is None:
                    self._outstanding += 1
                    job.counted = True
            if paused is not None:
                if paused == "queue_full":
                    self.rejected_queue_full += 1
                    exc = QueueFullError(
                        f"{self._outstanding} requests outstanding (limit "
                        f"{self.max_slots + self.max_queued_requests}); retry later"
                    )
                else:
                    exc = AdmissionPausedError(paused)
                job.out.put(("error", exc))
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
            "admission": self._admission_paused or "open",
            "memory_mode": self.plan.mode,
        }
        if not alive:
            out["error"] = (
                repr(self._dead) if self._dead is not None else "the model thread is not running"
            )
        return out

    def _warmup(self) -> None:
        """A greedy and a sampled request, unlogged; their statistics are reset after."""
        logged, self.log_requests = self.log_requests, False
        try:
            req = GenRequest(prompt_ids=[self.eos_ids[-1], 1, 2], max_tokens=2, temperature=0.0)
            self.generate(req)
            req = GenRequest(
                prompt_ids=[1, 2], max_tokens=1, temperature=0.7, top_p=0.95, top_k=40, seed=1
            )
            self.generate(req)
        finally:
            self.log_requests = logged

    def events(self, req: GenRequest) -> Iterator[tuple[str, object]]:
        """Yield ("token", id) per decode step, then ("done", GenResult). Raises on engine error."""
        job = self._submit(Job("generate", req))
        try:
            while True:
                kind, payload = job.out.get()
                if kind in ("error", "done"):
                    self._release_outstanding(job)
                if kind == "error":
                    raise payload
                yield kind, payload
                if kind == "done":
                    return
        finally:
            job.cancelled = True
            self._release_outstanding(job)

    def _release_outstanding(self, job: Job) -> None:
        with self._outstanding_lock:
            if getattr(job, "counted", False):
                job.counted = False
                self._outstanding -= 1

    def generate(self, req: GenRequest) -> GenResult:
        """Run one request to its end. `ValueError`, `QueueFullError`, `AdmissionPausedError` and
        `MemoryPressureError` do not raise but give a result with `finish_reason` "error", and
        `error`, `error_type` and `retryable` in `extra`; any other failure raises."""
        try:
            for kind, payload in self.events(req):
                if kind == "done":
                    return payload
        except (ValueError, QueueFullError, AdmissionPausedError, MemoryPressureError) as exc:
            return GenResult(
                token_ids=[],
                finish_reason="error",
                prompt_tokens=len(req.prompt_ids),
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "retryable": bool(getattr(exc, "retryable", False)),
                },
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
        with `finish_reason` "error" instead of raising."""
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
                        extra={
                            "error": str(payload),
                            "error_type": type(payload).__name__,
                            "retryable": bool(getattr(payload, "retryable", False)),
                        },
                    )
                    break
            self._release_outstanding(job)
            results.append(res)
        return results

    def score(
        self, prompt_ids: Sequence[int], continuation_ids: Sequence[int], top_k: int = 5
    ) -> list[dict]:
        """Teacher-forced logprobs of `continuation_ids` after `prompt_ids`, computed cold: one
        {"logprob": the token's, "top": the `top_k` best as [id, logprob]} per continuation
        token. Raises `ValueError` for an empty prompt, a bad token id or a sequence past the
        context limit."""
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

    def close(self, timeout_s: float = 30.0) -> None:
        """Stop the monitor and the worker; in-flight, pending and later requests then fail."""
        if self.monitor is not None:
            self.monitor.stop()
        if self._worker.is_alive() and self._dead is None:
            self._closing = True
            self._queue.put(Job("stop"))
            self._worker.join(timeout_s)

    def stats(self) -> dict:
        """Cumulative counters, prefix-store and batcher counters, and the ring of recent records."""
        out = {
            "memory_bytes": self.memory_bytes(),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "memory_limit_bytes": int(self.memory_limit_bytes),
            "cache_memory_bytes": int(mx.get_cache_memory()),
            "prefix_store": self.store.stats(),
            "batcher": {
                **self.pool.stats(),
                "delta_bytes_per_row": self.delta_bytes,
                "queued": len(self._pending),
                "outstanding": int(self._outstanding),
                "max_outstanding": self.max_slots + self.max_queued_requests,
                "rejected_queue_full": int(self.rejected_queue_full),
            },
            **self.stats_.snapshot(),
            "kernel_checks": dict(self.kernel_checks),
            "kernel_check_errors": dict(self.kernel_check_errors),
            "dense_kernels": dense_kernel_report(),
            "memory_plan": self.plan.describe(),
            "memory_mode": self.plan.mode,
            "process_budget_bytes": int(self.limits.process_bytes),
            "max_context_tokens": int(self.max_context_tokens),
            "memory": self.memory_snapshot(),
        }
        if self.expert_pool is not None:
            # `expert_cache_miss_bytes` is an alias of `expert_miss_bytes`
            ec = self.expert_pool.stats()
            out.update(ec)
            out["expert_cache_miss_bytes"] = ec["expert_miss_bytes"]
        return out

    def serve(self, host: str = "127.0.0.1", port: int = 8080, allowed_hosts=()) -> None:
        """Serve the OpenAI-compatible HTTP API on `host:port`; blocks until the server stops
        (`server.api_server.serve`)."""
        from jitengine_qwen3_6_35b_a3b_m4pro.server.api_server import serve

        serve(self, host, port, allowed_hosts=allowed_hosts)


def create_engine(model_path: str, **options) -> Engine:
    """Build an `Engine` for the checkpoint at `model_path`, warmed up and ready for requests;
    `options` are the engine options (`engine.args`)."""
    return Engine(model_path, **options)
