"""The engine: one model, one session, one worker thread that owns every MLX call.

Callers (in-process users, the benchmark harness, the HTTP handlers) validate and enqueue requests from their
own threads and read released steps from a per-request queue; nothing but plain Python values crosses
threads. One request runs at a time; up to `max_queue` wait, further ones are refused (QueueFull).

Decode loop: the graph of pass j+1 is built on the host (its input token is pass j's lazy argmax or
draw) while the GPU runs pass j; pass j is then synchronised and released, and only then is pass j+1
encoded. Encoding after the sync is what lets the history's in-place slice updates donate their
buffers instead of copying whole history arrays.

Prefix cache: the post-prefill state of the last prompt (ring copies + length + last logits) is kept;
an identical prompt restores it instead of re-running the prefill. Decode only writes positions at or
after the prompt length, so a restored run is bit-identical to a cold one.

Cancellation: every job carries a cancel event (set by closing its stream, by the HTTP server when the
client disconnects, or by stop_all). A queued job whose event is set never starts; a prefill stops between chunks
(Cancelled: no error, the history is invalidated); a decode stops at its next release group and settles what is in
flight. A client-side cancel ends the job silently (error_type "Cancelled"); stop_all (SIGTERM / close) answers the
running and every queued job with an explicit EngineUnavailable error, and refuses new ones.

Request limits (both off by default, and off they change nothing): max_output_tokens caps max_tokens like
the context room does (an absent or larger max_tokens becomes the cap; forced streams are not capped).
request_deadline_s is a wall-clock budget per generation request, counted from its submission (queue wait included):
when it runs out, the job's cancel event is set, so the request stops where a cancel stops it (in the queue, between
prefill chunks, or at its next released step, settling the passes in flight) and is answered cleanly with the tokens
released so far, finish_reason "length" and x_metrics.deadline_exceeded true. Echo scoring is not cut by it.
/health and /metrics report counts, timings, statuses and versions; a request's seed and request_id appear only in
its own response (x_metrics), never in last_request / recent.

Portability: any Apple silicon Mac is admitted by capability (device.py), the context limit defaults to what the GPU
wired limit holds (config.resolve_context_limit), and decode_backend "auto" (the default) brings the megakernel up
through its load check (loadcheck.py: compile, the residency probe over the device's grid plan, then the persistent
pass against its per-phase chain, agreement with stock MLX on a fixed text, and verify rows against M = 1 passes). On
the tuned M4 Pro that is G = 40 with no search, plus the check. If the compile, probe, chain or stock stage fails,
profile main decodes with stock MLX ops (the graph backend; one WARNING line, /health mk_unavailable), profile kv8 is
refused, and decode_backend "mk" or a pinned mk_grid raises: an engine never runs a grid or a backend other than the one
it was asked for. If only the verify stage fails, speculation is turned off and the megakernel serves one token per
pass. A grid measured by `doctor --tune` (device.read_grid_cache) is tried first on a device that is not in the tuned
table; without a cache entry nothing changes.

Memory (budget.py): resident, as above, when the budget (memory_budget_bytes) or, without one, the GPU wired limit holds
the whole model; otherwise profile main offloads the routed experts: bank 0 loads, an expert pool (expert_pool.py) holds
what the budget leaves room for and reads the rest from the checkpoint on demand, decode runs on the graph backend, and
the prefill runs layer-major (model.prefill_layer_major), so a layer's experts are read once per prompt. While decoding,
the pool opens an annex in the memory the budget keeps for prefill transients (the budget minus the process tree's
footprint minus budget.ANNEX_RESERVE_BYTES); it closes before every prefill and history growth, which wait until its
memory is back (FLINT_ANNEX=0 turns it off).
"""

from __future__ import annotations

import gc
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

from jitengine_flint_m2 import device, logs
from jitengine_flint_m2.engine import faults
from jitengine_flint_m2.engine.io_struct import (
    FORCED_TOP,
    TOP_LOGPROBS_MAX,
    Cancelled,
    ContextTooLarge,
    EngineUnavailable,
    GenResult,
    QueueFull,
    ScoredPosition,
    StepOutput,
)
from jitengine_flint_m2.kernels import mathmode
from jitengine_flint_m2.model.config import (
    CONTEXT_AUTO,
    KV_BITS,
    StartupError,
    check_checkpoint,
    check_context_limit,
    check_kv8_config,
    check_profile,
    load_config,
    tensor_index,
)
from jitengine_flint_m2.tokenizer import IncrementalDecoder, StopMatcher, Tokenizer

PREFILL_CHUNK = 2048
MAX_QUEUE = 8
ANNEX_RELEASE_SLACK_BYTES = (
    32 * 2**20
)  # a closed annex is released once the tree is back within this of its open
ANNEX_RELEASE_POLL_S = 0.002
ANNEX_RELEASE_CAP_S = 2.0  # stop waiting (and log) after this long
ANNEX_OPEN_CLEAR_CAP_S = (
    0.5  # at the open, wait at most this long for the cleared buffer cache to leave the footprint
)
VERSION = (
    "flint-engine 0.7.3 (decode: megakernel v1, layout-6d full-layer attention on a per-position grid "
    "(prefix-consistent: no request quantity reaches the reduction order), items dealt "
    "threadgroup-first, M = 1 prologues on a threadgroup descriptor copy and early weight loads; mode code: "
    "context-copy drafts + row-exact verify passes of 2..8 rows at every context, "
    "pipelined host loop, cost-fitted depth with an online acceptance guard; "
    "graph decode kept as decode_backend=graph; --profile kv8: full-layer K/V at 8 bits in groups of 64, "
    "quantised on write, layout 6d-kv8 with the dequantisation folded at the score / P level, on the "
    "twice-finer grid above position 32768; reliability: "
    "cancel on disconnect / in prefill / in queue, graceful SIGTERM, residency ring + real-pass probe, "
    "in-kernel failure propagation and the kernel_stall seam; portability: admission by capability, grid per "
    "device with a residency search (or a doctor --tune cache), context limit sized to the wired limit, a load "
    "check (compile, probe, persistent == chain, stock-MLX agreement, verify rows == M = 1) with the stock-MLX "
    "fallback for main; every Metal kernel compiled with math mode 'safe', pinned; the prefill's cached "
    "transients released before a shorter last chunk and at the prefill/decode boundary)"
)
DECODE_BACKENDS = ("auto", "mk", "graph")
FALLBACK_NOTE = "decoding with stock MLX ops, about 0.56x the tuned speed at 1K, no speculation"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class _Plan:
    prompt: list
    n_out: int  # outputs to release (forced: m + 1)
    passes: int  # decode passes
    clamped: bool
    temperature: float
    top_k: int
    top_p: float
    seed: int
    ignore_eos: bool
    stop: StopMatcher
    logprobs: int | None
    forced: list | None
    fault: faults.Fault | None
    request_id: str | None
    one_token_per_pass: bool | None = (
        None  # per-request override of the engine switch (None: the engine's)
    )
    timed_out: bool = False  # the request deadline cut the decode short (set by _finish)


@dataclass(eq=False)  # identity-hashed: the engine keeps its pending jobs in a set
class _Job:
    kind: str  # "generate" | "score"
    plan: _Plan | None = None
    score: tuple | None = None  # (prompt, positions, top_k)
    out: queue.Queue = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    reason: str = ""  # set by stop_all: the engine stopped it ("" = its client went away)
    timed_out: bool = False  # set by the deadline timer (request_deadline_s) when it fired first
    timer: threading.Timer | None = None


PRIVATE_METRICS = (
    "seed",
    "request_id",
)  # per-request values kept out of /health and /metrics


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in PRIVATE_METRICS}


def _check_limits(deadline_s, max_output) -> tuple:
    """The optional request limits: None (off), or a positive finite deadline in seconds / a positive token cap."""
    if deadline_s is not None and (
        isinstance(deadline_s, bool)
        or not isinstance(deadline_s, (int, float))
        or not math.isfinite(deadline_s)
        or deadline_s <= 0
    ):
        raise StartupError(
            f"request_deadline_s must be a positive number of seconds, got {deadline_s!r}"
        )
    if max_output is not None and (
        isinstance(max_output, bool) or not isinstance(max_output, int) or max_output < 1
    ):
        raise StartupError(f"max_output_tokens must be an integer >= 1, got {max_output!r}")
    return (None if deadline_s is None else float(deadline_s)), max_output


def _check_budget(nbytes) -> int | None:
    """The memory budget: None (the whole model when the GPU holds it), or a positive number of bytes."""
    if nbytes is not None and (
        isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0
    ):
        raise StartupError(f"memory_budget_bytes must be a positive integer, got {nbytes!r}")
    return nbytes


def _require_math_mode() -> None:
    """Refuse an MLX that cannot compile with the pinned Metal math mode (kernels/mathmode.py),
    whatever the decode backend, before any weight is allocated."""
    try:
        mathmode.require()
    except RuntimeError as exc:
        raise StartupError(str(exc)) from exc


def toolchain_versions() -> dict:
    """The toolchain every byte-identity claim depends on (reported by /health and /metrics)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "mlx": ver("mlx"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "numpy": ver("numpy"),
        "transformers": ver("transformers"),
        "kernel_source_sha256": kernel_source_sha256(),
        "metal_math_mode": mathmode.MATH_MODE,
    }


def kernel_source_sha256() -> str:
    """sha256 (16 hex digits) of the megakernel's sources: the four modules that render it, then
    its .metal files by name. A missing source raises."""
    import hashlib
    from importlib import resources

    kernels = resources.files("jitengine_flint_m2.kernels")
    metal = sorted(f.name for f in kernels.iterdir() if f.name.endswith(".metal"))
    h = hashlib.sha256()
    for name in ("metal_src.py", "vmoe_src.py", "kv8_src.py", "kv8_fold.py", *metal):
        h.update(kernels.joinpath(name).read_bytes())
    return h.hexdigest()[:16]


class Engine:
    def __init__(
        self,
        model_path: str,
        profile: str = "main",
        context_limit: int | str = CONTEXT_AUTO,
        one_token_per_pass: bool = False,
        allow_fault_injection: bool = False,
        prefix_cache: bool = True,
        verify_weights: bool = False,
        prefill_chunk: int = PREFILL_CHUNK,
        max_queue: int = MAX_QUEUE,
        wait: bool = True,
        start: bool = True,
        log=_log,
        decode_backend: str = "auto",
        mk_grid: int | None = None,
        mk_chunks: str | None = None,
        speculation: bool = True,
        request_deadline_s: float | None = None,
        max_output_tokens: int | None = None,
        sim: dict | None = None,
        memory_budget_bytes: int | None = None,
        memory_mode: str = "auto",
    ):
        """context_limit: "auto" (the tokens the GPU wired limit holds, at most 262,144) or an explicit 1..262,144
        (refused when it does not fit). decode_backend: "auto" (megakernel if its load check passes, else stock MLX ops
        for profile main), "mk" (megakernel or refuse) or "graph". mk_grid: pin the megakernel grid (probed at that
        value only; refused if not co-resident). sim: device / load-check simulation (tests; needs
        allow_fault_injection; device.Sim). memory_budget_bytes: the memory the engine may use (budget.py); None
        keeps the whole model resident when the GPU wired limit holds it, and streams the experts otherwise.
        memory_mode: "auto", "resident" (refuse rather than stream the experts) or "offload" (stream them even when
        the whole model fits)."""
        # ---- refusals before any weight is allocated
        check_profile(profile)
        self.request_deadline_s, self.max_output_tokens = _check_limits(
            request_deadline_s, max_output_tokens
        )
        self.kv_bits = KV_BITS[profile]
        pinned = self._check_backend(decode_backend, self.kv_bits, mk_grid)
        self.sim = device.parse_sim(sim, allow_fault_injection)
        _require_math_mode()
        self.versions = toolchain_versions()
        from jitengine_flint_m2.kernels.plan import check_policy, default_policy, is_6d

        if mk_chunks is None:  # the profile's partition: main "l6d", kv8 "l6dk"
            mk_chunks = default_policy(self.kv_bits)
        check_policy(mk_chunks)  # ValueError for an unknown name
        if not is_6d(mk_chunks):  # the served grid is per position, never request-fixed
            raise StartupError(
                f"mk_chunks {mk_chunks!r} is a request-fixed partition (kernel timing only); the "
                f"engine serves the per-position layout-6d partitions 'l6d' (main), 'l6dk' (kv8), 'l6d2'"
            )
        self.decode_backend_requested = decode_backend
        # "auto" resolves at load: "mk", or "graph" (mk_unavailable)
        self.decode_backend = decode_backend
        self.mk_chunks = mk_chunks
        self.mk = None
        # {G, G_source, G0, reason, tried, loadcheck} once the megakernel is up; {stage, reason, grid, tried,
        # loadcheck} when "auto" fell back to graph
        self.mk_info: dict | None = None
        self.mk_unavailable: dict | None = None
        if context_limit != CONTEXT_AUTO and context_limit is not None:
            check_context_limit(context_limit)
        self.model_path = model_path
        self.profile = profile
        self.log = log
        self.memory_budget_bytes = _check_budget(memory_budget_bytes)
        self.memory_mode = memory_mode
        self.cfg = load_config(model_path)
        if self.kv_bits == 8:
            check_kv8_config(self.cfg)
        self.index = tensor_index(model_path, self.cfg)
        self.manifest = check_checkpoint(model_path, verify_weights=verify_weights)
        self._check_device(context_limit, pinned)
        self.one_token_per_pass = bool(one_token_per_pass)
        # verify passes in mode `code` (the megakernel's; offload decodes on the graph backend)
        self.speculation = bool(speculation) and self.decode_backend != "graph"
        self._draft_base = None
        self.verify_probe: dict = {}
        self.allow_fault_injection = bool(allow_fault_injection)
        self.prefix_cache = bool(prefix_cache)
        self.prefill_chunk = int(prefill_chunk)
        self.max_queue = int(max_queue)
        self.vocab = self.cfg.vocab
        self.eos = tuple(self.cfg.eos_ids)
        self.tokenizer = Tokenizer(model_path)
        self.model_name = self.manifest.get("model_id", "flint-int4")
        self._init_worker_state()
        if start:
            self.start()
            if wait:
                self.wait_ready()

    @staticmethod
    def _check_backend(decode_backend: str, kv_bits: int, mk_grid) -> int | None:
        """The decode backend and a pinned grid, checked; returns the pinned grid (None: the device's plan)."""
        if decode_backend not in DECODE_BACKENDS:
            raise StartupError(
                f"unknown decode_backend {decode_backend!r}; one of {', '.join(DECODE_BACKENDS)}"
            )
        if kv_bits == 8 and decode_backend == "graph":
            raise StartupError(
                "profile 'kv8' runs on the megakernel decode backend only (decode_backend 'auto' or 'mk')"
            )
        pinned = None if mk_grid is None else device.check_grid(mk_grid)
        if pinned is not None and decode_backend == "graph":
            raise StartupError("mk_grid pins the megakernel grid; decode_backend 'graph' has none")
        return pinned

    def _check_device(self, context_limit, pinned: int | None) -> None:
        """Admission by capability (device.admit), the memory plan for the budget and the GPU wired limit (budget.py:
        resident with config.resolve_context_limit's context, or offload) and the megakernel grid plan
        (device.plan_grid)."""
        from jitengine_flint_m2.budget import plan_memory
        from jitengine_flint_m2.weight_banks import BankPlan

        bank_plan = BankPlan(self.cfg, self.index)
        # the real GPU (its working set is the MLX wired limit), and what admission, the budget and the grid see
        # (a simulation's injected facts, or the real ones)
        self.device = device.mlx_device_info()
        self.facts = device.read_facts(self.sim)
        # offload needs profile main on the graph backend without a pinned grid (kv8 and the megakernel read every
        # expert from the resident banks)
        offload_ok = self.kv_bits == 16 and self.decode_backend != "mk" and pinned is None
        banks_fit = max(bank_plan.bank_sizes) <= self.facts.max_buffer_length
        stream = offload_ok and (not banks_fit or self.memory_mode == "offload")
        sizes = bank_plan.bank_sizes[:1] if stream else bank_plan.bank_sizes
        self.device_notes = device.admit(self.facts, sizes)
        self.weights_bytes = bank_plan.total_bytes
        largest_field = max(f.nbytes for f in bank_plan.expert_fields.values())
        self.memory = plan_memory(
            self.memory_budget_bytes,
            context_limit,
            weights_bytes=self.weights_bytes,
            dense_bytes=bank_plan.bank_sizes[0],
            record_bytes=bank_plan.expert_stride,
            total_records=len(self.cfg.moe_layers) * self.cfg.n_experts,
            min_slots=self.cfg.n_experts,
            max_slots=self.facts.max_buffer_length
            // largest_field,  # one pool array is slots x a field
            bytes_per_token=self.cfg.history_bytes_per_token_bits(self.kv_bits),
            wired_bytes=self.facts.wired_bytes,
            memory_size=self.facts.memory_size,
            profile=self.profile,
            offload_ok=offload_ok,
            resident_ok=banks_fit,
            mode=self.memory_mode,
        )
        self.context_limit = self.memory.context_limit
        self.context_limit_source = self.memory.context_limit_source
        if self.memory.mode == "offload":
            self.decode_backend = "graph"
        self.log(f"[engine] memory plan: {self.memory.line()}")
        # a `doctor --tune` entry for this device and kernel source, if any (never read on the tuned device)
        self.grid_cache = None
        if self.decode_backend != "graph" and pinned is None and self.facts.tuned_grid is None:
            self.grid_cache = device.read_grid_cache(self.facts, self.versions, self.kv_bits)
        self.grid = (
            None
            if self.decode_backend == "graph"
            else device.plan_grid(self.facts, pinned, self.grid_cache)
        )
        # the grid in use once the megakernel is up
        self.mk_grid = self.grid.G0 if self.grid is not None else None
        for note in self.device_notes:
            self.log(f"[engine] {note}")
        if self.sim is not None:
            self.log(logs.warning(f"[engine] SIMULATED (test hook): {self.sim.as_dict()}"))

    def _init_worker_state(self) -> None:
        """The state shared with the worker: status, the job queue and its lock, counters, and the worker thread, not yet started."""
        self._status = "loading"
        self._fatal: str | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._waiting = 0
        self._lock = threading.Lock()
        self._pending: set = set()  # jobs submitted and not yet finished (stop_all cancels them)
        self._closing = False  # stop_all ran: new jobs are refused
        self.requests = 0
        self.errors = 0
        self.last_metrics: dict = {}
        self.recent: list = []
        self.load_s = 0.0
        self._ready = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, name="flint-engine", daemon=True)
        self.started = False

    def start(self) -> None:
        """Begin loading the weights on the worker thread (validation already happened)."""
        if not self.started:
            self.started = True
            self._worker.start()

    # ================================================================== public surface
    def wait_ready(self, timeout: float | None = None) -> None:
        self._ready.wait(timeout)
        if self._status == "unhealthy":
            raise EngineUnavailable(f"engine failed to start: {self._fatal}")

    def render_chat(self, messages, tools=None) -> list[int]:
        return self.tokenizer.render_chat(messages, tools)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def health(self) -> dict:
        pool = self._pool_stats()
        return {
            "status": self._status,
            "model": self.model_name,
            "profile": self.profile,
            "kv_bits": self.kv_bits,
            "context_limit": self.context_limit,
            "stop_token_ids": list(self.eos),
            "one_token_per_pass": self.one_token_per_pass,
            "multi_token": self.multi_token(),
            "verify": self.verify_info(),
            "request_deadline_s": self.request_deadline_s,
            "max_output_tokens": self.max_output_tokens,
            "max_queued": self.max_queue,
            "requests": self.requests,
            "errors": self.errors,
            "queue": self._waiting,
            "last_request": _public(self.last_metrics),
            "load_s": round(self.load_s, 2),
            "weights_bytes": self.weights_bytes,
            "version": VERSION,
            "versions": dict(self.versions),
            "decode_backend": self.decode_backend,
            "decode_backend_requested": self.decode_backend_requested,
            "context_limit_source": self.context_limit_source,
            "memory": self.memory.to_dict(),
            "memory_mode_requested": self.memory_mode,
            **({"expert_pool": pool} if pool else {}),
            "mk": self.mk_info,
            "device": self.facts.summary(),
            **({"mk_unavailable": self.mk_unavailable} if self.mk_unavailable else {}),
            **({"simulated": self.sim.as_dict()} if self.sim is not None else {}),
            **({"fatal": self._fatal} if self._fatal else {}),
        }

    def _pool_stats(self) -> dict | None:
        pool = getattr(getattr(self, "W", None), "pool", None)
        if pool is None or pool.experts is None:
            return None
        return {**pool.stats(), **getattr(self, "_annex_stats", {})}

    def multi_token(self) -> bool:
        """True when a default request may commit several tokens per pass (verify passes compiled and probed)."""
        return bool(
            self.speculation
            and not self.one_token_per_pass
            and self.mk is not None
            and self.mk.vrows
        )

    def verify_info(self) -> dict:
        from jitengine_flint_m2.kernels.plan import six_text

        return {
            "enabled": self.multi_token(),
            "rows": list(self.mk.vrows) if self.mk is not None else [],
            "band": "every context (full layers layout 6d, sliding layers strided)",
            "kernel_family": (
                f"layout 6d{'-kv8' if self.kv_bits == 8 else ''} full layers on the per-position "
                f"grid {six_text(self.mk_chunks)} (row q = last key); strided rings; "
                "a verify row = the M = 1 item at its position"
            ),
            "drafts": "context copy, longest earlier suffix match (n <= 64); depth per match bucket from measured verify "
            "costs by KV length and a prior acceptance model, backed off online per request by the "
            "acceptance it realises; M = 1 passes queued device-fed while certain to draft nothing",
            "probe_ms": dict(self.verify_probe),
        }

    def draft_base(self, prompt):
        """The prompt's n-gram index for drafting, kept for the last prompt (beside the prefix-cache snapshot)."""
        from jitengine_flint_m2.copy_draft import PromptIndex

        key = tuple(prompt)
        if self._draft_base is None or self._draft_base.prompt != key:
            self._draft_base = PromptIndex(key)
        return self._draft_base

    def stats(self) -> dict:
        h = self.health()
        h["recent"] = [_public(m) for m in self.recent]
        return h

    def generate(self, req) -> GenResult:
        result = None
        for item in self.stream(req):
            if isinstance(item, GenResult):
                result = item
        return result

    def stream(self, req, cancel: threading.Event | None = None):
        """cancel: an event the caller sets to cancel the job (the HTTP server's disconnect watcher); closing the
        returned iterator cancels it too."""
        plan = self._validate(req)
        job = _Job("generate", plan=plan)
        if cancel is not None:
            job.cancel = cancel
        self._arm_deadline(job)
        try:
            self._submit(job)
        except BaseException:
            self._disarm(job)
            raise
        return self._drain(job)

    def score(
        self, prompt_ids, positions=None, top_k: int = 20, cancel: threading.Event | None = None
    ) -> list[ScoredPosition]:
        prompt = self._check_ids(prompt_ids, "prompt")
        if len(prompt) > self.context_limit:
            raise ContextTooLarge(
                f"prompt of {len(prompt)} tokens exceeds the context limit {self.context_limit}"
            )
        if not 0 <= int(top_k) <= TOP_LOGPROBS_MAX:
            raise ValueError(f"top_k must be in 0..{TOP_LOGPROBS_MAX}")
        if positions is None:
            pos = list(range(1, len(prompt)))
        else:
            pos = sorted({int(p) for p in positions})
            if pos and (pos[0] < 1 or pos[-1] >= len(prompt)):
                raise ValueError(f"positions must lie in 1..{len(prompt) - 1}")
        job = _Job("score", score=(prompt, pos, int(top_k)))
        if cancel is not None:
            job.cancel = cancel
        self._submit(job)
        out = None
        for item in self._drain(job):
            out = item
        if isinstance(out, BaseException):
            raise out
        return out

    def serve(self, host: str = "127.0.0.1", port: int = 8080, allowed_hosts=()) -> None:
        from jitengine_flint_m2.server import serve

        serve(self, host, port, allowed_hosts=allowed_hosts)

    def stop_all(self, reason: str = "engine is stopping") -> None:
        """Stop serving: refuse new jobs and cancel the running one and every queued one. Each is still answered: a
        decode ends at its next release group, a prefill at its next chunk, a queued job never starts, each with
        finish_reason "error" / EngineUnavailable(reason) (HTTP 503, or an SSE error event then [DONE])."""
        with self._lock:
            self._closing = True
            jobs = list(self._pending)
        for j in jobs:
            if not j.reason:
                j.reason = reason
            j.cancel.set()

    def close(self, timeout: float = 60.0) -> None:
        """stop_all, then wait (up to `timeout` s) for the worker to answer every job and release the weights."""
        if self._status in ("stopping", "closed"):
            return
        self.stop_all("engine is closing")
        self._status = "stopping"
        if self.started:
            self._jobs.put(None)
            self._worker.join(timeout=timeout)
        self._status = "closed"

    # ================================================================== validation (caller thread)
    def _check_ids(self, ids, what: str) -> list:
        if ids is None or isinstance(ids, (str, bytes)):
            raise ValueError(f"{what} must be a list of token ids")
        out = []
        for t in ids:
            if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.vocab:
                raise ValueError(
                    f"{what} contains an invalid token id {t!r} (valid: integers in 0..{self.vocab - 1})"
                )
            out.append(int(t))
        return out

    def _validate(self, req) -> _Plan:
        g = lambda k, d=None: getattr(req, k, d)  # noqa: E731
        prompt = self._check_ids(g("prompt_ids"), "prompt")
        if not prompt:
            raise ValueError("prompt is empty")
        n = len(prompt)
        if n >= self.context_limit:
            raise ContextTooLarge(
                f"prompt of {n} tokens leaves no room under the context limit {self.context_limit}"
            )
        room = self.context_limit - n
        forced = g("forced_tokens")
        if forced is not None:
            forced = self._check_ids(forced, "forced_tokens")
            if n + len(forced) > self.context_limit:
                raise ContextTooLarge(
                    f"prompt + forced stream = {n + len(forced)} positions exceed the "
                    f"context limit {self.context_limit}"
                )
        mt = g("max_tokens")
        clamped = False
        if forced is not None:
            n_out, passes = len(forced) + 1, len(forced)
        else:
            if mt is None:
                mt, clamped = room, True
            if isinstance(mt, bool) or not isinstance(mt, int) or mt < 0:
                raise ValueError("max_tokens must be an integer >= 0")
            if mt > room:
                mt, clamped = room, True
            if self.max_output_tokens is not None and mt > self.max_output_tokens:
                mt, clamped = self.max_output_tokens, True
            n_out, passes = mt, max(0, mt - 1)
        T, tp, tk, seed = self._check_sampling(g)
        lp, stop, otp, fault = self._check_options(g)
        return _Plan(
            prompt,
            n_out,
            passes,
            clamped,
            float(T),
            int(tk),
            float(tp),
            seed,
            bool(g("ignore_eos", False)),
            StopMatcher(stop),
            lp,
            forced,
            fault,
            g("request_id"),
            otp,
        )

    def _check_sampling(self, g) -> tuple:
        """The request's temperature, top-p, top-k and seed, defaulted and checked (a fresh seed when none is given)."""
        T = g("temperature", 1.0)
        T = 1.0 if T is None else T
        tp = g("top_p", 0.95)
        tp = 0.95 if tp is None else tp
        tk = g("top_k", 20)
        tk = 20 if tk is None else tk
        for name, v in (("temperature", T), ("top_p", tp)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        if T < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < tp <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(tk, bool) or not isinstance(tk, int) or tk < 0:
            raise ValueError("top_k must be an integer >= 0")
        seed = g("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
        from jitengine_flint_m2.sampler import fresh_seed

        seed = fresh_seed() if seed is None else int(seed)
        return T, tp, tk, seed

    def _check_options(self, g) -> tuple:
        """The request's logprobs, stop strings, one-token-per-pass flag and fault, checked."""
        lp = g("logprobs")
        if lp is not None and (
            isinstance(lp, bool) or not isinstance(lp, int) or not 0 <= lp <= TOP_LOGPROBS_MAX
        ):
            raise ValueError(f"logprobs must be an integer in 0..{TOP_LOGPROBS_MAX}")
        stop = g("stop") or ()
        if isinstance(stop, str):
            stop = [stop]
        if not all(isinstance(s, str) for s in stop):
            raise ValueError("stop must be a string or a list of strings")
        otp = g("one_token_per_pass")
        if otp is not None and not isinstance(otp, bool):
            raise ValueError("one_token_per_pass must be a boolean")
        fspec = g("fault")
        if fspec is not None and not self.allow_fault_injection:
            raise ValueError(
                "fault injection is disabled (start the engine with allow_fault_injection)"
            )
        fault = faults.parse(fspec)
        if fault is not None and fault.kind == "kernel_stall" and self.decode_backend == "graph":
            raise ValueError("fault kind kernel_stall needs the megakernel decode backend")
        return lp, stop, otp, fault

    def _submit(self, job: _Job) -> None:
        if self._status in ("unhealthy", "stopping", "closed"):
            raise EngineUnavailable(f"engine is {self._status}")
        with self._lock:
            if self._closing:
                raise EngineUnavailable("engine is stopping")
            if self._waiting >= self.max_queue:
                raise QueueFull(f"{self._waiting} requests already waiting")
            self._waiting += 1
            self._pending.add(job)
        self._jobs.put(job)

    def _arm_deadline(self, job: _Job) -> None:
        """request_deadline_s: a timer that cancels the job as a deadline (answered cleanly, see _finish)."""
        if self.request_deadline_s is None:
            return
        job.timer = threading.Timer(self.request_deadline_s, self._expire, (job,))
        job.timer.daemon = True
        job.timer.start()

    @staticmethod
    def _expire(job: _Job) -> None:
        if not job.cancel.is_set():  # a client that already left (or stop_all) keeps its own reason
            job.timed_out = True
            job.cancel.set()

    @staticmethod
    def _disarm(job: _Job) -> None:
        if job.timer is not None:
            job.timer.cancel()

    @staticmethod
    def _deadline_cut(job: _Job) -> bool:
        """Was the job cancelled by its deadline (and not by stop_all)? Then it is answered as a clean truncation."""
        return job.timed_out and not job.reason

    @staticmethod
    def _cancel_reason(job: _Job) -> tuple:
        """(error_type, message) of a cancelled job: the engine stopped it, or its client went away."""
        return (
            ("EngineUnavailable", job.reason)
            if job.reason
            else ("Cancelled", "cancelled: the client went away")
        )

    def _answer_cancelled(self, job: _Job) -> None:
        """End a job cancelled before or during its prefill (nothing was released)."""
        et, msg = self._cancel_reason(job)
        if job.kind == "generate" and self._deadline_cut(
            job
        ):  # out of time in the queue or the prefill
            p = job.plan
            p.timed_out = True
            job.out.put(
                GenResult(
                    [],
                    "length",
                    len(p.prompt),
                    metrics={
                        "request_id": p.request_id,
                        "prompt_tokens": len(p.prompt),
                        "completion_tokens": 0,
                        "finish_reason": "length",
                        "deadline_exceeded": True,
                        "profile": self.profile,
                        "kv_bits": self.kv_bits,
                    },
                )
            )
        elif job.kind == "generate":
            p = job.plan
            job.out.put(
                GenResult(
                    [],
                    "error",
                    len(p.prompt),
                    error=msg,
                    metrics={
                        "error_type": et,
                        "request_id": p.request_id,
                        "prompt_tokens": len(p.prompt),
                        "completion_tokens": 0,
                        "profile": self.profile,
                        "kv_bits": self.kv_bits,
                    },
                )
            )
        else:
            job.out.put(EngineUnavailable(msg) if job.reason else Cancelled(msg))

    def _drain(self, job: _Job):
        try:
            while True:
                item = job.out.get()
                yield item
                if not isinstance(item, StepOutput):
                    return
        finally:
            job.cancel.set()

    # ================================================================== worker thread
    def _run_worker(self) -> None:
        try:
            self._load()
        except BaseException as err:  # noqa: BLE001
            self._fatal = f"{type(err).__name__}: {err}"
            self._status = "unhealthy"
            self.log(logs.error(f"[engine] load failed: {self._fatal}"))
            self._release_all()  # a refused load holds no weights, kernels or history
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                break
            with self._lock:
                self._waiting -= 1
            if (
                job.cancel.is_set()
            ):  # its client left (or the engine stopped) while it waited: never runs
                self._disarm(job)
                self._answer_cancelled(job)
                with self._lock:
                    self._pending.discard(job)
                self.requests += 1
                continue
            if self._status == "ready":
                self._status = "busy"
            ka = getattr(self, "_keepalive", None)
            if ka is not None:  # the GPU clock stays up only while a job runs (gpu_keepalive.py)
                ka.resume()
            try:
                self._run_job(job)
            finally:
                if ka is not None and self._jobs.empty():
                    ka.pause()
        self._release_all()

    def _load(self) -> None:
        """Load the weights, then, unless the graph backend was asked for, bring the megakernel up through its load
        check (or fall back), then build the graph backend's history if that is what decodes; then the engine is
        ready. In offload mode only the non-expert weights load, and the expert pool takes the routed experts."""
        import mlx.core as mx

        from jitengine_flint_m2.kv_cache import History, MKHistory, MKHistory8
        from jitengine_flint_m2.loader import load_weights

        mx.set_wired_limit(int(self.device["max_recommended_working_set_size"]))
        t0 = time.perf_counter()
        offload = self.memory.mode == "offload"
        if offload:
            from jitengine_flint_m2.budget import ALLOCATOR_CACHE_BYTES

            mx.set_memory_limit(int(self.memory.budget_bytes))
            mx.set_cache_limit(ALLOCATOR_CACHE_BYTES)
        if offload:
            self.W = load_weights(self.cfg, self.index, log=self.log, experts=False)
            self.W.pool = self._open_pool()
        else:
            self.W = load_weights(self.cfg, self.index, log=self.log)
        from jitengine_flint_m2 import gpu_keepalive

        if gpu_keepalive.enabled(offload):
            self._keepalive = gpu_keepalive.KeepAlive(log=self.log)
        self._annex_allowed = True
        self._annex_fp0 = 0  # the tree's footprint when the annex opened
        self._annex_stats = {
            "annex_release_wait_ms": 0.0,
            "annex_release_wait_ms_max": 0.0,
            "annex_release_capped": 0,
            "annex_open_clear_ms": 0.0,
        }
        self._snap = None
        self.hist = None
        if self.decode_backend != "graph":
            self._bring_up_mk(MKHistory8 if self.kv_bits == 8 else MKHistory)
        if self.decode_backend == "graph":
            self.hist = History(self.cfg, self.context_limit)
        self.load_s = time.perf_counter() - t0
        # a sporadic full (gen-2) collection of the load-time heap paused the worker 35-70 ms inside timed windows
        # (measured on the M4 Pro); collect once and move that heap out of the collector's generations (undone in
        # _release_all)
        gc.collect()
        gc.freeze()
        self._status = "ready"

    def _open_pool(self):
        """The expert pool of the memory plan, its landing proved and three records checked against the checkpoint."""
        from jitengine_flint_m2.expert_pool import ExpertPool

        annex_max = 0
        if os.environ.get("FLINT_ANNEX", "").strip() != "0":
            annex_max = self.memory.total_records - self.memory.pool_slots
        pool = ExpertPool(
            self.cfg,
            self.index,
            self.memory.pool_slots,
            log=self.log,
            annex_max=annex_max,
            speculate=os.environ.get("FLINT_SPECULATE", "").strip() != "0",
        )
        try:
            pool.selfcheck()
        except BaseException:
            pool.close()
            raise
        return pool

    # ------------------------------------------------------------------ expert-pool annex (offload decode)
    def _tree_footprint(self) -> int:
        """phys_footprint of this process plus the GPU keep-alive child: what the budget (and the benchmark) counts."""
        pids = [os.getpid()]
        ka = getattr(self, "_keepalive", None)
        if ka is not None:
            pids.append(ka.proc.pid)
        return sum(device.phys_footprint(p) or 0 for p in pids)

    def _annex_open(self) -> None:
        """At the first decode lookup: open the annex in what the budget leaves (budget.ANNEX_RESERVE_BYTES kept)."""
        import mlx.core as mx

        from jitengine_flint_m2.budget import ANNEX_RESERVE_BYTES

        pool = self.W.pool
        if pool is None or self.memory.budget_bytes is None:
            return
        if not self._annex_allowed or pool.annex_slots or not pool.annex_max:
            return
        fp = self._tree_footprint()
        cached = mx.get_cache_memory()
        if cached:
            # MLX's buffer cache (prefill transients) is in the footprint, and decode does not refill it:
            # hand it back first and size the annex on the footprint without it. The
            # release is asynchronous; wait on the footprint, polled, at most ANNEX_OPEN_CLEAR_CAP_S, and size on
            # whatever it then reads.
            mx.clear_cache()
            target = fp - cached + ANNEX_RELEASE_SLACK_BYTES
            t0 = time.perf_counter()
            while (
                fp := self._tree_footprint()
            ) > target and time.perf_counter() - t0 < ANNEX_OPEN_CLEAR_CAP_S:
                time.sleep(ANNEX_RELEASE_POLL_S)
            self._annex_stats["annex_open_clear_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        if pool.open_annex(self.memory.budget_bytes - fp - ANNEX_RESERVE_BYTES):
            self._annex_fp0 = fp

    def _annex_close(self) -> None:
        """Close the annex and wait until the tree's footprint is back to where it was when the annex opened, so the
        prefill or history growth that follows has the memory the plan gave it (the unmap returns it at once; the wait
        is the check). Waits on the footprint, polled, up to ANNEX_RELEASE_CAP_S."""
        pool = getattr(self.W, "pool", None)
        if pool is None or not pool.annex_slots:
            return
        t0 = time.perf_counter()
        pool.close_annex()
        target = self._annex_fp0 + ANNEX_RELEASE_SLACK_BYTES
        while (fp := self._tree_footprint()) > target:
            if time.perf_counter() - t0 > ANNEX_RELEASE_CAP_S:
                self._annex_stats["annex_release_capped"] += 1
                self.log(
                    f"[engine] annex release: footprint {fp / 1e9:.3f} GB still above {target / 1e9:.3f} GB "
                    f"after {ANNEX_RELEASE_CAP_S:.0f} s"
                )
                break
            time.sleep(ANNEX_RELEASE_POLL_S)
        ms = round((time.perf_counter() - t0) * 1e3, 1)
        st = self._annex_stats
        st["annex_release_wait_ms"] = ms
        st["annex_release_wait_ms_max"] = max(st["annex_release_wait_ms_max"], ms)

    def _reserve(self, need: int) -> None:
        """hist.reserve(need), closing the annex first when the history grows (the new arrays are allocated while the
        old ones are alive)."""
        if self.memory.mode == "offload" and (need > self.hist.capacity or not self.hist.rk):
            self._annex_close()
        self.hist.reserve(need)

    def _bring_up_mk(self, hist_cls) -> None:
        """The megakernel through its load check (loadcheck.py) on the grid plan (device.plan_grid): compile and probe;
        the verify passes compiled and probed; then the chain, stock and verify stages on a fixed text. A failed
        compile / probe / chain / stock stage: "auto" + main falls back to the graph backend; kv8, "mk" or a pinned grid
        raise. A failed verify stage turns speculation off (the megakernel serves one token per pass)."""
        from jitengine_flint_m2 import loadcheck
        from jitengine_flint_m2.kernels.plan import attn_params

        t0 = time.perf_counter()
        self.hist = hist_cls(self.cfg, self.context_limit)
        self.hist.reserve(1)
        att = attn_params(self.mk_chunks)
        report: dict = {}
        failed = None
        try:
            mk, pr, tried = loadcheck.bring_up(
                self.W, self.hist, att, self.grid, self.kv_bits == 8, self.sim, self.log, report
            )
        except loadcheck.MKUnavailable as exc:
            # keep its fields only: its traceback holds the failed kernels
            failed = (exc.stage, exc.reason, exc.tried, exc.report)
        if failed is not None:
            failed[3].setdefault("s", round(time.perf_counter() - t0, 3))
            self._mk_unavailable(*failed)
            return
        self.mk = mk
        del mk
        self.hist.invalidate()  # the probe's real passes wrote rows 0..3
        self.log(
            f"[engine] megakernel G={self.mk.G} residency probe ok: barrier-only pass "
            f"{pr['bar_ms']:.2f} ms, flag ring {pr['ring_ms']:.2f} ms, real passes {pr['real_ms']:.2f} ms each "
            f"({self.mk.nbar} grid barriers per pass)"
        )
        self._warm_verify(att)
        try:
            loadcheck.check(
                self.W,
                self.mk,
                self.hist,
                att,
                self.tokenizer,
                self.sim,
                self.log,
                t0,
                tried,
                report,
            )
        except loadcheck.MKUnavailable as exc:
            failed = (exc.stage, exc.reason, exc.tried, exc.report)
        if failed is not None:
            self._mk_unavailable(*failed)
            return
        if not report["verify"]["ok"] and self.mk.vrows:
            self.mk.vrows = ()
            self.log(
                logs.warning(
                    f"[engine] WARNING: verify passes DISABLED by the load check "
                    f"({report['verify']['reason']}); the megakernel serves one token per pass"
                )
            )
        self.mk_grid, self.decode_backend = self.mk.G, "mk"
        tuned_or_pinned = self.grid.source in ("tuned", "pinned")
        source = self.grid.source if (tuned_or_pinned or len(tried) == 1) else "search"
        self.mk_info = {
            "G": self.mk.G,
            "G_source": source,
            "G0": self.grid.G0,
            "reason": self.grid.reason,
            "tried": tried,
            "loadcheck": report,
        }
        self.log(f"[engine] megakernel G={self.mk_grid} ({source}) {loadcheck.summary(report)}")

    def _warm_verify(self, att: dict) -> None:
        """With speculation on, compile and probe the verify passes; a refused one turns speculation off, loudly."""
        from jitengine_flint_m2.kernels.runner import BarrierProbeError

        if not self.speculation:
            return
        try:
            self.verify_probe = self.mk.warm_verify(self.hist.arrays, att, self.hist.capacity)
            self.hist.invalidate()
            self.log(f"[engine] verify passes compiled and probed: {self.verify_probe} (M: ms)")
        except BarrierProbeError as exc:
            # refuse verify loudly, keep serving one token per pass
            self.mk.vrows = ()
            self.log(logs.warning(f"[engine] verify passes DISABLED: {exc}"))

    def _mk_unavailable(self, stage: str, reason: str, tried: list, report: dict) -> None:
        """A load-check stage failed. decode_backend "mk" or a pinned grid: raise (never another grid or backend).
        Profile kv8: refuse (it has no graph form; never serve main in its place). Otherwise ("auto", main): free the
        kernels, scratch and megakernel history, then decode with stock MLX ops."""
        import mlx.core as mx

        where = f"{stage}: {reason}"
        if self.decode_backend_requested == "mk" or self.grid.source == "pinned":
            what = (
                f"mk_grid={self.grid.G0} is pinned"
                if self.grid.source == "pinned"
                else "decode_backend 'mk' was requested"
            )
            raise RuntimeError(
                f"megakernel unavailable on this Mac ({where}); {what}, so the engine refuses rather "
                f"than search another grid or fall back to stock MLX ops"
            )
        if self.kv_bits == 8:
            raise StartupError(
                f"profile kv8 needs the megakernel, unavailable on this Mac ({where}); run --profile main"
            )
        self.mk_unavailable = {
            "stage": stage,
            "reason": reason,
            "grid": self.grid.reason,
            "tried": tried,
            "loadcheck": report,
        }
        self.mk = None
        self.hist.release()
        self.hist = None
        gc.collect()
        mx.clear_cache()
        self.decode_backend, self.speculation, self.mk_grid = "graph", False, None
        self.log(
            logs.warning(
                f"[engine] WARNING: megakernel unavailable on this Mac ({where}); {FALLBACK_NOTE}"
            )
        )

    def _run_job(self, job) -> None:
        """Run one job; every failure ends it with an error result, and the session state is reset."""
        try:
            if job.kind == "generate":
                self._run_generate(job)
            else:
                job.out.put(self._run_score(job))
        except Cancelled:  # stopped between prefill chunks: not a fault; the history is invalid
            self._snap = None
            self.hist.invalidate()
            self._answer_cancelled(job)
        except BaseException as exc:  # noqa: BLE001  (every failure ends the request, loudly)
            self.errors += 1
            msg = f"{type(exc).__name__}: {exc}"
            self.log(logs.error(f"[engine] request failed: {msg}"))
            self._reset_session()
            if job.kind == "generate":
                p = job.plan
                job.out.put(
                    GenResult(
                        [],
                        "error",
                        len(p.prompt),
                        error=msg,
                        metrics={
                            "error_type": type(exc).__name__,
                            "profile": self.profile,
                            "kv_bits": self.kv_bits,
                        },
                    )
                )
            else:
                job.out.put(RuntimeError(msg))
        finally:
            self._disarm(job)
            with self._lock:
                self._pending.discard(job)
            self.requests += 1
            if self._status == "busy":
                self._status = "ready"

    def _release_all(self) -> None:
        try:
            import mlx.core as mx

            self._snap = None
            if getattr(self, "_keepalive", None) is not None:
                self._keepalive.close()
                self._keepalive = None
            if getattr(self, "hist", None) is not None:
                self.hist.release()
            if getattr(getattr(self, "W", None), "pool", None) is not None:
                self.W.pool.close()
            self.W = None
            self.mk = None  # the kernel object holds the weight banks
            gc.unfreeze()  # the load-time heap is collectable again once the engine is closed
            mx.clear_cache()
        except Exception as exc:  # noqa: BLE001  (close() finishes; the failure is logged)
            self.log(logs.error(f"[engine] release failed: {type(exc).__name__}: {exc}"))

    def _reset_session(self) -> None:
        """After any failure the session state is untrusted: drop the cache, rebuild the rings."""
        try:
            import mlx.core as mx

            self._snap = None
            self.hist.release()
            mx.clear_cache()
        except BaseException as exc:  # noqa: BLE001
            self._fatal = f"session reset failed: {exc}"
            self._status = "unhealthy"

    # ------------------------------------------------------------------ prefill
    def _prefill(
        self, prompt: list, rows_for=None, on_rows=None, cancel: threading.Event | None = None
    ):
        """Chunked prefill from position 0 (chunk boundaries at multiples of prefill_chunk).

        rows_for(c0, m) -> rows of the chunk [c0, c0 + m) whose logits are wanted; on_rows(c0, rows,
        logits) consumes them chunk by chunk (so no chunk's [rows, vocab] logits outlive its chunk).
        cancel: checked before every chunk (each chunk is evaluated before the next starts), so a cancelled
        prefill stops within one chunk with Cancelled (the history is left invalid).
        Returns the bf16 logits [V] of the last prompt position (evaluated).

        MLX's buffer cache: it serves an allocation only from a cached buffer of about the same
        size, so the transients the full-length chunks leave in it cannot serve a shorter last chunk, and decode reuses
        none of them. They are returned to the system before a shorter last chunk and when the prefill ends, instead of
        sitting beside the decode's draft index (at full context they took the footprint past the auto budget)."""
        import mlx.core as mx

        from jitengine_flint_m2.model.model import prefill_chunk

        if self.memory.mode == "offload":
            return self._prefill_offload(prompt, rows_for, on_rows, cancel)
        hist, W = self.hist, self.W
        n = len(prompt)
        hist.invalidate()
        last = None
        for c0 in range(0, n, self.prefill_chunk):
            if cancel is not None and cancel.is_set():
                raise Cancelled(f"cancelled during the prefill, at position {c0} of {n}")
            c1 = min(n, c0 + self.prefill_chunk)
            m = c1 - c0
            if c0 and m != self.prefill_chunk:
                # the last chunk is shorter: the full-length chunks' cached transients cannot serve it
                mx.clear_cache()
            rows = set(rows_for(c0, m)) if rows_for else set()
            if c1 == n:
                rows.add(m - 1)
            rows = sorted(rows)
            toks = mx.array(prompt[c0:c1], dtype=mx.int32)
            lg = prefill_chunk(W, hist, toks, c0, rows or [m - 1])
            state = hist.state_arrays()
            if rows:
                mx.eval(lg, state)
                if on_rows is not None:
                    on_rows(c0, rows, lg)
                if c1 == n:
                    last = lg[len(rows) - 1]
                    mx.eval(last)
            else:
                mx.eval(state)
            del lg
            if self.kv_bits == 8:
                # kv8 only: each chunk attends over mx.dequantize of a 2,048-row-longer history
                # prefix (MKHistory8.full_kv), and MLX's buffer cache never reuses a buffer of another size, so the
                # chunk's freed transients are released here instead of accumulating across the prefill
                mx.clear_cache()
        # decode reuses none of the prefill's transients: return them now, not after the request (0.84 GB at
        # 2K, 1.2 GB at 258K; the next prefill allocates its own, ~15 ms at 2K)
        mx.clear_cache()
        hist.length = n
        hist.valid = True
        return last

    def _prefill_offload(self, prompt: list, rows_for, on_rows, cancel):
        """_prefill in offload mode: the same chunks, run layer-major (model.prefill_layer_major), so each layer's
        experts are read once per prompt; then each chunk's requested rows through the head, chunk by chunk."""
        import mlx.core as mx

        from jitengine_flint_m2.model.model import head, prefill_layer_major

        hist, W = self.hist, self.W
        n = len(prompt)
        hist.invalidate()
        chunks = [(c0, min(n, c0 + self.prefill_chunk)) for c0 in range(0, n, self.prefill_chunk)]

        def check():
            if cancel is not None and cancel.is_set():
                raise Cancelled(f"cancelled during the prefill of {n} positions")

        # the prefill's transients take the memory the decode annex borrowed
        self._annex_close()
        self._annex_allowed = False
        try:
            hs = prefill_layer_major(W, hist, mx.array(prompt, dtype=mx.int32), chunks, check)
        finally:
            self._annex_allowed = True
        last = None
        for (c0, c1), h in zip(chunks, hs):
            m = c1 - c0
            rows = set(rows_for(c0, m)) if rows_for else set()
            if c1 == n:
                rows.add(m - 1)
            if not rows:
                continue
            rows = sorted(rows)
            lg = head(W, mx.take(h.reshape(m, -1), mx.array(rows, dtype=mx.int32), axis=0))
            mx.eval(lg)
            if on_rows is not None:
                on_rows(c0, rows, lg)
            if c1 == n:
                last = lg[len(rows) - 1]
                mx.eval(last)
        hist.length = n
        hist.valid = True
        return last

    def _prefill_or_restore(self, prompt: list, cancel: threading.Event | None = None):
        """Logits [V] of the last prompt position, and the number of cached prompt positions."""
        snap = self._snap
        if self.prefix_cache and snap is not None and snap.prompt == tuple(prompt) and self.hist.rk:
            self.hist.restore(snap)
            return snap.logits, len(prompt)
        self._snap = None
        last = self._prefill(prompt, cancel=cancel)
        if self.prefix_cache:
            self._snap = self.hist.snapshot(prompt, last)
        return last, 0

    # ------------------------------------------------------------------ score (echo)
    def _run_score(self, job: _Job):
        import mlx.core as mx

        prompt, positions, top_k = job.score
        want = set(positions)
        n = len(prompt)
        self._snap = None
        self._reserve(n)
        out: list = []

        def rows_for(c0, m):
            # row r of the chunk starting at c0 predicts position c0 + r + 1
            return [p - c0 - 1 for p in want if c0 < p <= c0 + m]

        def on_rows(c0, rows, lg):
            sel = [(k, r) for k, r in enumerate(rows) if (c0 + r + 1) in want]
            if not sel:
                return
            for b0 in range(0, len(sel), 512):  # bound the fp32 [rows, vocab] transient
                part = sel[b0 : b0 + 512]
                idx = mx.array([k for k, _ in part], dtype=mx.int32)
                lf = mx.take(lg, idx, axis=0).astype(mx.float32)
                lp = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
                tgt = mx.array([prompt[c0 + r + 1] for _, r in part], dtype=mx.int32)
                chosen = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
                if top_k > 0:
                    ids = mx.argpartition(-lf, kth=top_k - 1, axis=-1)[:, :top_k]
                    vals = mx.take_along_axis(lp, ids, axis=-1)
                    order = mx.argsort(-vals, axis=-1)
                    ids = mx.take_along_axis(ids, order, axis=-1)
                    vals = mx.take_along_axis(vals, order, axis=-1)
                    mx.eval(chosen, ids, vals)
                    ids_l, vals_l = ids.tolist(), vals.tolist()
                else:
                    mx.eval(chosen)
                    ids_l = vals_l = [[] for _ in part]
                for k, ((_, r), c) in enumerate(zip(part, chosen.tolist())):
                    p = c0 + r + 1
                    out.append(
                        ScoredPosition(p, prompt[p], float(c), list(zip(ids_l[k], vals_l[k])))
                    )

        last = self._prefill(prompt, rows_for, on_rows, cancel=job.cancel)
        if self.prefix_cache:
            self._snap = self.hist.snapshot(prompt, last)
        out.sort(key=lambda s: s.position)
        return out

    # ------------------------------------------------------------------ generate
    def _step_graph(self, logits, j: int, plan: _Plan):
        """Lazy outputs for completion index j from its logits: (token array or None, outputs dict)."""
        import mlx.core as mx

        from jitengine_flint_m2.sampler import choose, logprob_outputs

        outs = {"finite": mx.all(mx.isfinite(logits))}
        if plan.forced is not None:
            tok = plan.forced[j] if j < len(plan.forced) else None
            tarr = mx.array([tok if tok is not None else 0], dtype=mx.int32)
            lp, ids, vals = logprob_outputs(logits, tarr, FORCED_TOP)
            outs.update(lp=lp, ids=ids, vals=vals)
            return (tarr if tok is not None else None), outs
        tarr = choose(logits, plan.temperature, plan.top_k, plan.top_p, plan.seed, j)
        outs["tok"] = tarr
        if plan.logprobs is not None:
            lp, ids, vals = logprob_outputs(logits, tarr, plan.logprobs)
            outs.update(lp=lp, ids=ids, vals=vals)
        return tarr, outs

    def _pass_graph(self, tok_in, j: int, plan: _Plan):
        """Decode pass j (feeds the token of index j-1 at position n + j - 1)."""
        import mlx.core as mx

        from jitengine_flint_m2.model.model import decode_step

        p = len(plan.prompt) + j - 1
        logits = decode_step(self.W, self.hist, tok_in, p, self.hist.ring_mask(p))
        if faults.poisons_logits(plan.fault, j):
            logits = logits * mx.array(float("nan"), dtype=logits.dtype)
        return self._step_graph(logits, j, plan)

    @staticmethod
    def _arrays(outs: dict) -> list:
        return [v for v in outs.values() if v is not None]

    def _sync(self, j: int, outs: dict, plan: _Plan) -> StepOutput:
        import mlx.core as mx

        mx.eval(self._arrays(outs))
        if not outs["finite"].item():
            raise faults.NonFiniteLogits(f"nonfinite logits at completion index {j}")
        if plan.forced is not None:
            tok = plan.forced[j] if j < len(plan.forced) else None
            lp = float(outs["lp"].item()) if tok is not None else None
        else:
            tok = int(outs["tok"].item())
            lp = float(outs["lp"].item()) if "lp" in outs else None
        top = (
            list(zip(outs["ids"].tolist(), outs["vals"].tolist()))
            if outs.get("ids") is not None
            else []
        )
        return StepOutput(j, tok, lp, top, time.perf_counter())

    def _run_generate(self, job: _Job) -> None:
        if self.decode_backend == "mk":
            from jitengine_flint_m2.decode import generate

            generate(self, job)
            return
        f = job.plan.fault
        # a kernel_stall request queued while "auto" was resolving, and then the engine fell back
        if f is not None and f.kind == "kernel_stall":
            raise ValueError(
                "fault kind kernel_stall needs the megakernel decode backend (this engine decodes with "
                "stock MLX ops)"
            )
        self._run_generate_graph(job)

    def _run_generate_graph(self, job: _Job) -> None:  # noqa: PLR0915
        import mlx.core as mx

        plan = job.plan
        n = len(plan.prompt)
        t0 = time.perf_counter()
        mx.reset_peak_memory()
        result = GenResult([], "length", n)
        if plan.n_out == 0:
            result.metrics = self._metrics(plan, result, t0, t0, None, None, 0)
            job.out.put(result)
            return
        self._reserve(n + plan.passes)
        logits0, cached = self._prefill_or_restore(plan.prompt, cancel=job.cancel)
        result.cached_prompt_tokens = cached
        t_prefill = time.perf_counter()
        tok, outs = self._step_graph(logits0, 0, plan)
        mx.async_eval(self._arrays(outs))
        step = self._sync(0, outs, plan)
        dec = IncrementalDecoder(self.tokenizer) if plan.stop else None
        t_first = step.t_release
        t_last = t_first
        finish = None
        nxt = None
        j = 0
        while True:
            # release step j
            job.out.put(step)
            result.steps.append(step)
            t_last = step.t_release
            if step.token_id is not None:
                result.token_ids.append(step.token_id)
            if plan.forced is None and not plan.ignore_eos and step.token_id in self.eos:
                finish = "stop"
            if dec is not None and step.token_id is not None:
                dec.push(step.token_id)
                cut = plan.stop.find(dec.text)
                if cut is not None:
                    finish = "stop"
                    result.text = dec.text[:cut]
            if finish or j + 1 >= plan.n_out or job.cancel.is_set():
                break
            # pass j+1: built on the host while nothing else is pending, encoded, then build-ahead j+2
            if nxt is None:
                if j == 0 and self.memory.mode == "offload":
                    self._annex_open()  # the first decode lookup is next
                nxt = self._pass_graph(tok, j + 1, plan)
                mx.async_eval(self._arrays(nxt[1]))
            cur = nxt
            nxt = None
            if j + 2 < plan.n_out and cur[0] is not None:
                ahead = self._pass_graph(cur[0], j + 2, plan)  # host graph build overlaps the GPU
            else:
                ahead = None
            faults.fire_host(plan.fault, j + 1)
            step = self._sync(j + 1, cur[1], plan)
            tok = cur[0]
            if ahead is not None:
                mx.async_eval(self._arrays(ahead[1]))
                nxt = ahead
            j += 1
        if nxt is not None:
            mx.eval(self._arrays(nxt[1]))  # settle the built-ahead pass so the history is concrete
        self.hist.length = n + len(result.steps) - 1
        self._finish(job, result, finish)
        if dec is not None and result.text is None:
            result.text = dec.text
        result.metrics = self._metrics(plan, result, t0, t_prefill, t_first, t_last, cached)
        if result.finish_reason == "error":
            result.metrics["error_type"] = self._cancel_reason(job)[0]
        job.out.put(result)

    def _finish(self, job: _Job, result: GenResult, finish: str | None) -> None:
        """The finish reason of a decode loop that ended: its own (stop / length), or a cancel that cut it short
        (finish_reason "error": EngineUnavailable when the engine stopped it, Cancelled when its client left; "length"
        with deadline_exceeded when its request deadline ran out)."""
        if (
            finish is None
            and job.cancel.is_set()
            and len(result.steps) < job.plan.n_out
            and self._deadline_cut(job)
        ):
            job.plan.timed_out = True
            result.finish_reason = "length"
        elif finish is None and job.cancel.is_set() and len(result.steps) < job.plan.n_out:
            result.finish_reason = "error"
            result.error = f"{self._cancel_reason(job)[1]} (after {len(result.token_ids)} tokens)"
        else:
            result.finish_reason = finish or "length"

    def _metrics(self, plan, result, t0, t_prefill, t_first, t_last, cached) -> dict:
        import mlx.core as mx

        steps = len(result.steps)
        dec_s = (t_last - t_first) if (t_first is not None and steps > 1) else 0.0
        m = {
            "request_id": plan.request_id,
            "prompt_tokens": len(plan.prompt),
            "completion_tokens": len(result.token_ids),
            "released_steps": steps,
            "decode_passes": max(0, steps - 1),
            "cached_prompt_tokens": cached,
            "prefill_s": round(t_prefill - t0, 6),
            "decode_s": round(dec_s, 6),
            "decode_tok_s": round((steps - 1) / dec_s, 3) if dec_s > 0 else None,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "seed": plan.seed,
            "profile": self.profile,
            "kv_bits": self.kv_bits,
            "finish_reason": result.finish_reason,
            "one_token_per_pass": True,
            "mode": "forced"
            if plan.forced is not None
            else ("greedy" if plan.temperature <= 0 or plan.top_k == 1 else "sampled"),
        }
        if plan.timed_out:
            m["deadline_exceeded"] = True
        self.last_metrics = m
        self.recent = (self.recent + [m])[-16:]
        return m


def create_engine(model_path: str, **options) -> Engine:
    return Engine(model_path, **options)
