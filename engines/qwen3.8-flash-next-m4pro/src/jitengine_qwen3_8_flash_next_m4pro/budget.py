"""Memory plan: split a declared budget into buckets derived from the checkpoint's geometry.

Buckets come from the safetensors headers and config.json (nothing is allocated here), and the
expert pool takes what is left; there is no resident mode, so a budget that fits every record
simply seeds each one once. A streamed token-embedding table (`model.embed_store`) is not dense:
only its row cache is charged. `BudgetError` refuses a budget before any weight or `mx` allocation.
"""

from __future__ import annotations

import math
from decimal import Decimal

# ----------------------------------------------------------------------------- constants
POOL_HAIRCUT = 0.98  # plan below the arithmetic maximum, never at it
GIB = 1 << 30
DEFAULT_BUDGET_BYTES = 40 * 10**9  # the measured operating point: 40 GB, 37.25 GiB
BUDGET_MAX_SLOTS = 8  # concurrent rows of recurrent state
BUDGET_KV_RESERVE = 512  # decode positions pre-sized at admission (engine.kv_reserve)
BUDGET_GRID_TOKENS = 2048  # the absolute prefix grid: a generate row's partition and score's
BUDGET_MAX_CONTEXT = (
    3400  # positions of KV + idx planned per row: prefix 2,582 + task + 256 + slack
)
BUDGET_STORE_ENTRIES = 2  # prefix-store entries the plan holds
BUDGET_STORE_PREFIX_TOKENS = 2582  # the workload's shared prefix (this checkpoint's tokenizer)
STORE_HEADROOM = 1.125  # an end-of-head entry is the prompt, a little longer than the prefix
KV_MIN_POSITIONS = 256  # kv_cache.KV_STEP: the smallest KV a row can hold (the floor's kv term)

CACHE_LIMIT_CAP = 0.128e9  # mx.set_cache_limit
CACHE_FRACTION = 0.02
# process bytes outside MLX's accounting (process peak minus MLX's own peak)
NON_MLX_MEASURED = 553_603_228
NON_MLX_FRACTION = 0.05
SAFETY_CAP = 0.4e9
SAFETY_FRACTION = 0.02

# The wave's transient footprint, fit per model geometry and compute dtype as
#     transient = fixed + per_token x tokens_in_flight + per_row_token x longest_row_tokens
# A geometry the table does not carry is scaled from the target's law by activation width.
WAVE_MEASURED_BY_DTYPE = {
    "bfloat16": {
        "model": "Qwen3.8-Flash-Next-4bit, bfloat16, measured 2026-09-11",
        "fixed_bytes": 365.7e6,
        "per_token_bytes": 0.0955e6,
        "per_row_token_bytes": 0.4151e6,
        "residual_bytes": 20.8e6,
        "geometry": 32768,
        "r2": 0.99888,
    },
    "float32": {
        "model": "Qwen3.8-Flash-Next-4bit, float32, measured 2026-09-11",
        "fixed_bytes": 2298.6e6,
        "per_token_bytes": 0.1012e6,
        "per_row_token_bytes": 0.3357e6,
        "residual_bytes": 220e6,
        "geometry": 32768,
        "r2": 0.9007,
    },
}
WAVE_MEASURED_BY_DTYPE["float16"] = WAVE_MEASURED_BY_DTYPE["bfloat16"]
# geometry -> dtype -> law; `wave_law` falls back to the target's law scaled by width.
WAVE_MEASURED_BY_GEOMETRY = {
    32768: WAVE_MEASURED_BY_DTYPE,
    704: {
        "float32": {
            "model": "tiny-qwen4exp fixture (12 cases, model-free)",
            "fixed_bytes": 0.1717e6,
            "per_token_bytes": 0.005025e6,
            "per_row_token_bytes": 0.044277e6,
            "residual_bytes": 0.084e6,
            "geometry": 704,
            "r2": 0.99961,
        },
        "bfloat16": {
            "model": "tiny-qwen4exp fixture (12 cases, model-free)",
            "fixed_bytes": 0.0433e6,
            "per_token_bytes": 0.002924e6,
            "per_row_token_bytes": 0.032348e6,
            "residual_bytes": 0.060e6,
            "geometry": 704,
            "r2": 0.99955,
        },
    },
}
WAVE_MEASURED_BY_GEOMETRY[704]["float16"] = WAVE_MEASURED_BY_GEOMETRY[704]["bfloat16"]
# Full-vocab fp32 temporaries charged per readout row, for geometries `SCORE_MEASURED` lacks.
READOUT_FACTOR = 9.5
# Measured transient of one score of grid + 1 tokens, keyed (geometry, dtype) then (grid tokens,
# readout rows). The target's readout cost is sublinear in rows, which the analytic model above
# does not capture; a grid missing here falls back to that model.
SCORE_MEASURED = {
    (32768, "float32"): {
        (2048, 1): 2_102_941_972,
        (2048, 32): 3_531_481_216,
        (2048, 128): 3_628_409_344,
        (512, 1): 703_642_048,
        (512, 32): 3_383_597_232,
        (512, 128): 3_480_525_492,
    },
}
SCORE_MEASURED_MARGIN = 1.10
# Measured wave transients keyed (geometry, dtype) then (wave tokens, longest row). The transient
# is monotone in both, so a dominating point bounds a query; `wave_scratch_bytes` takes the min
# with the law, so this table can only lower a charge and the wave-token ladder stays monotone.
WAVE_SCRATCH_MEASURED = {
    (32768, "float32"): {(6656, 2048): 2_912_553_192},
}


def wave_measured_bytes(
    wave_tokens: int, geometry: int, dtype: str, row_tokens: int
) -> float | None:
    """Smallest measured transient dominating (wave_tokens, row_tokens), with margin, or None."""
    tbl = WAVE_SCRATCH_MEASURED.get((int(geometry), str(dtype)))
    if not tbl:
        return None
    t, r = int(wave_tokens), int(row_tokens)
    cand = [v for (mt, mr), v in tbl.items() if mt >= t and mr >= r]
    return SCORE_MEASURED_MARGIN * min(cand) if cand else None


SCORE_READOUT_ROWS_MAX = 128  # score.SCORE_READOUT_ROWS: the largest group the plan may pick
SCORE_READOUT_ROW_LADDER = (128, 64, 32, 16, 8, 4, 2, 1)
WAVE_TOKENS_MAX = 7168
WAVE_TOKENS_STEP = 256
# Served wave budget. A wave admits rows FIFO until `tokens + n > wave_tokens`, counting the lead's
# own chunk; 8192 fits a burst of eight ~1k-token rows after the shared prefix in one wave. The
# scratch it reserves is lent to the extension pool whenever no prefill needs it.
WAVE_TOKENS_SERVED = 8192
WAVE_SCRATCH_SHARE = 0.10  # the wave may take at most this share of the pool's remainder


def wave_geometry(cfg) -> int:
    """Activation width per in-flight token: hyper-connection stream + GDN conv row + gated q."""
    return int(cfg.stream_dim + cfg.conv_dim + cfg.num_attention_heads * cfg.head_dim * 2)


def wave_law(geometry: int, dtype: str = "bfloat16") -> tuple[dict, float]:
    """(law, width factor) for `geometry`: its own measurement, else the target's law scaled."""
    by = WAVE_MEASURED_BY_GEOMETRY.get(int(geometry))
    if by is not None:
        m = by.get(str(dtype), by["bfloat16"])
        return m, 1.0
    m = WAVE_MEASURED_BY_DTYPE.get(str(dtype), WAVE_MEASURED_BY_DTYPE["bfloat16"])
    return m, float(geometry) / float(m["geometry"])


def wave_scratch_bytes(
    wave_tokens: int,
    geometry: int,
    row_tokens: int | None = None,
    act_bytes: int = 2,
    dtype: str = "bfloat16",
) -> float:
    """Transient bytes of a `wave_tokens` wave with longest row `row_tokens` (default: one row)."""
    m, k = wave_law(geometry, dtype)
    t = max(0, int(wave_tokens))
    r = t if row_tokens is None else max(0, min(t, int(row_tokens)))
    law = k * (
        m["fixed_bytes"]
        + m["per_token_bytes"] * t
        + m["per_row_token_bytes"] * r
        + m["residual_bytes"]
    )
    measured = wave_measured_bytes(t, geometry, dtype, r)
    return law if measured is None else min(law, measured)


def readout_scratch_bytes(rows: int, vocab: int) -> float:
    """The readout group's transient: READOUT_FACTOR full-vocab fp32 temporaries per row."""
    return READOUT_FACTOR * max(1, int(rows)) * max(1, int(vocab)) * 4


def score_measured_bytes(grid: int, geometry: int, dtype: str, readout_rows: int) -> float | None:
    """Measured transient of one score chunk at (grid, readout_rows), with margin, or None.

    `readout_rows` rounds up to the next measured row count (or the largest one measured).
    """
    tbl = SCORE_MEASURED.get((int(geometry), str(dtype)))
    if not tbl:
        return None
    rows_at = sorted({r for g, r in tbl if g == int(grid)})
    if not rows_at:
        return None
    pick = next((r for r in rows_at if r >= int(readout_rows)), rows_at[-1])
    return SCORE_MEASURED_MARGIN * tbl[(int(grid), pick)]


def score_scratch_bytes(
    grid: int,
    geometry: int,
    dtype: str = "bfloat16",
    vocab: int = 0,
    readout_rows: int = SCORE_READOUT_ROWS_MAX,
    act_bytes: int = 2,
) -> float:
    """One score chunk's transient: measured if known, else the grid forward plus one readout."""
    g = max(1, int(grid))
    m = score_measured_bytes(g, geometry, dtype, readout_rows)
    if m is not None:
        return m
    return wave_scratch_bytes(g, geometry, g, act_bytes, dtype) + readout_scratch_bytes(
        readout_rows, vocab
    )


def memory_limit_for(budget_bytes: float) -> int:
    """The budget minus the non-MLX footprint and the safety margin: the MLX memory limit."""
    b = float(budget_bytes)
    return int(
        b - min(NON_MLX_MEASURED, NON_MLX_FRACTION * b) - min(SAFETY_CAP, SAFETY_FRACTION * b)
    )


def _gib_up(b: float) -> str:
    """Bytes as GiB rounded up at the fourth decimal, so a budget declared at this value fits."""
    return f"{math.ceil(b * 10_000 / GIB) / 10_000:.4f}"


def declared_budget_bytes(options: dict) -> int:
    """The memory budget an engine's options declare, in bytes: `memory_budget_bytes`, or the
    `memory_budget_gb` in decimal GB, converted exactly (the published budgets were measured
    in GB: 40 GB is 40e9 bytes, 37.25 GiB); the measured 40 GB when neither is given. Only `None`
    means "not declared": an explicit 0 or a negative budget reaches `plan_budget`, which refuses
    it."""
    nbytes, gb = options.get("memory_budget_bytes"), options.get("memory_budget_gb")
    if nbytes is not None and gb is not None:
        raise ValueError("pass memory_budget_bytes or memory_budget_gb (decimal GB), not both")
    if gb is not None:
        return int(Decimal(str(gb)) * 10**9)
    return int(DEFAULT_BUDGET_BYTES if nbytes is None else nbytes)


class BudgetError(ValueError):
    """A declared budget that cannot hold the engine. Raised before anything is allocated."""


# ----------------------------------------------------------------------------- per-row geometry
def dtype_bytes_of(name: str) -> int:
    return 4 if str(name) == "float32" else 2


def gdn_state_bytes(cfg, act_bytes: int) -> int:
    """DeltaNet state of one row: fp32 ssm [Hv, Dv, Dk] + (K-1) x conv_dim conv window per layer."""
    n = sum(1 for i in range(cfg.num_hidden_layers) if not cfg.is_attention(i))
    ssm = cfg.linear_num_value_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4
    conv = (cfg.linear_conv_kernel_dim - 1) * cfg.conv_dim * act_bytes
    return n * (ssm + conv)


def ple_state_bytes(cfg, act_bytes: int) -> int:
    """PLE state of one row: the n-gram context (int64) and the conv window."""
    n = sum(1 for i in range(cfg.num_hidden_layers) if cfg.has_ple(i))
    return n * ((cfg.ngram_size - 1) * 8 + cfg.ple_conv_state_len * cfg.stream_dim * act_bytes)


def kv_bytes_per_position(cfg, act_bytes: int) -> int:
    """Attention KV + the QSA indexer's raw key per position, over the attention layers."""
    n = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_attention(i))
    return n * (2 * cfg.num_key_value_heads * cfg.head_dim + cfg.indexer_head_dim) * act_bytes


def dense_fp32_meta_bytes(scan, cfg, dtype: str, dense_kernel: str) -> int:
    """Bytes added by storing 16-bit dense scales and biases as float32, from the shard headers.

    Mirrors `linear.apply_dense_policy`: only under float32 activations, only for shapes
    the row-blocked kernel does not serve, and never for the embedding table.
    """
    from jitengine_qwen3_8_flash_next_m4pro.checkpoint.loader import EXPERTS, PLE_SHARDS
    from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as _dk

    if str(dtype) != "float32":
        return 0
    prefix = scan.prefix or ""
    total = 0
    for name, shard in scan.weight_map.items():
        if not (name.endswith(".scales") or name.endswith(".biases")):
            continue
        if EXPERTS in name or PLE_SHARDS in name:
            continue
        if prefix and not name.startswith(prefix):
            continue
        stripped = name[len(prefix) :]
        base = stripped.rsplit(".", 1)[0]
        if base.startswith("model.embed_tokens"):
            continue
        hdr, _off = scan.headers[shard]
        meta = hdr.get(name)
        if meta is None or meta["dtype"] not in ("BF16", "F16"):
            continue
        rows, cols = int(meta["shape"][0]), int(meta["shape"][1])
        bits, gs = cfg.quant_of(base)
        if (
            dense_kernel == "rb"
            and int(bits) == 4
            and _dk.config_for(rows, cols * int(gs), "float32") is not None
        ):
            continue  # the kernel serves this shape: no fp32 copy is stored for it
        total += rows * cols * 2  # 2 bytes -> 4 bytes
    return total


# ----------------------------------------------------------------------------- the plan
class BudgetPlan:
    __slots__ = (
        "budget_bytes",
        "dense_bytes",
        "record_bytes",
        "num_records",
        "experts_per_step",
        "step_bytes",
        "max_slots",
        "kv_reserve",
        "max_context",
        "kv_per_position",
        "kv_bytes",
        "state_per_row",
        "state_bytes",
        "store_entries",
        "store_prefix_tokens",
        "store_entry_bytes",
        "prefix_store_bytes",
        "score_chunk",
        "score_readout_rows",
        "score_scratch_bytes",
        "score_tokens",
        "vocab",
        "wave_tokens",
        "wave_geometry",
        "wave_row_tokens",
        "scratch_bytes",
        "cache_bytes",
        "non_mlx_bytes",
        "safety_bytes",
        "pool_bytes",
        "pool_slots",
        "floor_bytes",
        "need_bytes",
        "compute_dtype",
        "memory_limit_bytes",
        "pool_transient_per_slot",
        "moe_kernel",
        "dense_meta_fp32_bytes",
        "dense_kernel",
        "pool_codec",
        "pool_codec_basis",
        "slot_bytes",
        "staging_slots",
        "staging_bytes",
        "embed_streamed_bytes",
        "embed_cache_rows",
        "embed_cache_bytes",
        "store_spill_slots",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # the plan in the units the engine logs and reports
    @property
    def kv_budget_mb(self) -> float:
        return self.kv_bytes / 1e6

    @property
    def prefix_store_mb(self) -> float:
        return self.prefix_store_bytes / 1e6

    @property
    def delta_bytes(self) -> int:
        return self.state_per_row

    @property
    def reserve_bytes(self) -> int:
        return int(self.scratch_bytes + self.cache_bytes + self.non_mlx_bytes + self.safety_bytes)

    @property
    def fixed_bytes(self) -> int:
        """Everything that is not the expert pool (the pack16 staging set is fixed: it does not
        scale with the slots)."""
        return int(
            self.dense_bytes
            + self.state_bytes
            + self.kv_bytes
            + self.prefix_store_bytes
            + self.reserve_bytes
            + self.staging_bytes
            + self.embed_cache_bytes
        )

    @property
    def planned_bytes(self) -> int:
        return int(
            self.fixed_bytes + self.pool_slots * (self.slot_bytes + self.pool_transient_per_slot)
        )

    @property
    def capacity_fraction(self) -> float:
        return self.pool_slots / float(self.num_records)

    def buckets(self) -> list[tuple[str, int]]:
        return (
            [
                ("dense weights", int(self.dense_bytes) - int(self.dense_meta_fp32_bytes)),
                (
                    f"float32-stored dense metadata (dense kernel {self.dense_kernel})",
                    int(self.dense_meta_fp32_bytes),
                ),
            ]
            + self._embed_buckets()
            + [
                (
                    f"recurrent+PLE state ({self.max_slots} rows x {self.state_per_row} B)",
                    int(self.state_bytes),
                ),
                (
                    f"KV+idx ({self.max_slots} rows x {self.max_context} pos x {self.kv_per_position} B)",
                    int(self.kv_bytes),
                ),
                (
                    f"prefix store cap ({self.store_entries} x {self.store_entry_bytes} B x {STORE_HEADROOM})",
                    int(self.prefix_store_bytes),
                ),
                (
                    f"scratch: max(wave {self.wave_tokens} tok, longest row {self.wave_row_tokens}; score "
                    f"{self.score_tokens} tok of grid {self.score_chunk} + readout {self.score_readout_rows} rows"
                    f" = {self.score_scratch_bytes / GIB:.4f} GiB) width {self.wave_geometry}",
                    int(self.scratch_bytes),
                ),
                ("MLX cache limit", int(self.cache_bytes)),
                ("non-MLX process", int(self.non_mlx_bytes)),
                ("safety margin", int(self.safety_bytes)),
                (
                    f"expert pool ({self.pool_slots} of {self.num_records} records, {self.slot_bytes} B per "
                    f"slot, {self.pool_codec})",
                    int(self.pool_slots * self.slot_bytes),
                ),
                (
                    f"pack16 staging ({self.staging_slots} x {self.record_bytes} B whole-record reads)",
                    int(self.staging_bytes),
                ),
                (
                    f"fp32 pool-metadata upcast transient ({self.pool_transient_per_slot} B/slot; {self.upcast_case})",
                    int(self.pool_slots * self.pool_transient_per_slot),
                ),
            ]
        )

    def _embed_buckets(self) -> list[tuple[str, int]]:
        """The streamed embedding's row cache, the only part of the table the plan holds."""
        if not self.embed_streamed_bytes:
            return []
        return [
            (
                f"token-embedding row cache ({self.embed_cache_rows} rows; the "
                f"{self.embed_streamed_bytes / GIB:.3f} GiB table is streamed)",
                int(self.embed_cache_bytes),
            )
        ]

    @property
    def upcast_case(self) -> str:
        """Which case of the fp32 pool-metadata upcast rule this plan is in."""
        if self.compute_dtype != "float32":
            return f"{self.compute_dtype} activations: no upcast"
        if self.moe_kernel == "stock":
            return "stock gather_qmm + float32: charged"
        return "grouped kernel + float32: no upcast, not charged"

    def as_dict(self) -> dict:
        """The whole table, for `stats()["memory_plan"]`."""
        d = {k: getattr(self, k) for k in self.__slots__}
        d["buckets"] = [(n, b) for n, b in self.buckets()]
        d["planned_bytes"] = self.planned_bytes
        d["unallocated_bytes"] = int(self.budget_bytes - self.planned_bytes)
        d["capacity_fraction"] = self.capacity_fraction
        d["pool_haircut"] = POOL_HAIRCUT
        d["upcast_case"] = self.upcast_case
        law, k = wave_law(self.wave_geometry, self.compute_dtype)
        d["wave_measured"] = dict(law)
        d["wave_measured_width_factor"] = k
        d["scratch_binding_term"] = (
            "score" if self.score_scratch_bytes >= self.scratch_bytes else "wave"
        )
        d["readout_factor"] = READOUT_FACTOR
        d["borrowed"] = self.borrowed()
        return d

    def borrowed(self) -> list[dict]:
        """The memory lent across operating points that the plan declares: while one row decodes,
        the prefix store's entries are spilled to a scratch file and its cap holds
        `store_spill_slots` expert slots (the pool's annex), returned before any admission,
        prefill or score."""
        n = int(self.store_spill_slots) * int(self.slot_bytes)
        if not n:
            return []
        return [
            {
                "name": "expert-pool annex (one-row decode; the prefix store spilled)",
                "bytes": n,
                "sources": {"prefix store cap": n},
            }
        ]

    def table(self) -> str:
        lines = [
            f"[plan] memory budget {self.budget_bytes / GIB:.3f} GiB ({self.compute_dtype}, moe kernel {self.moe_kernel})"
        ]
        for name, b in self.buckets():
            lines.append(f"[plan]   {name:<62s} {b / GIB:8.4f} GiB")
        lines.append(
            f"[plan]   {'TOTAL PLANNED':<62s} {self.planned_bytes / GIB:8.4f} GiB "
            f"({(self.budget_bytes - self.planned_bytes) / GIB:.4f} GiB unallocated)"
        )
        lines.append(
            f"[plan]   expert capacity {100 * self.capacity_fraction:.2f}% "
            f"({self.pool_slots} / {self.num_records} records, {self.slot_bytes} B per slot, "
            f"{self.record_bytes} B per read, codec {self.pool_codec}); "
            f"floor {self.floor_bytes / GIB:.6g} GiB, this plan needs {self.need_bytes / GIB:.6g} GiB; "
            f"MLX limit {self.memory_limit_bytes / GIB:.3f} GiB"
        )
        return "\n".join(lines)

    def describe(self) -> str:
        g = GIB
        return (
            f"budget {self.budget_bytes / g:.2f} GiB: dense {self.dense_bytes / g:.3f} "
            + (
                f"(embedding streamed, cache {self.embed_cache_bytes / g:.3f}) "
                if self.embed_streamed_bytes
                else ""
            )
            + (
                f"(store spills to {self.store_spill_slots} annex slots in one-row decode) "
                if self.store_spill_slots
                else ""
            )
            + f"+ state "
            f"{self.state_bytes / g:.3f} ({self.max_slots} rows) + kv {self.kv_bytes / g:.3f} "
            f"+ store {self.prefix_store_bytes / g:.3f} + reserve {self.reserve_bytes / g:.3f} + "
            f"staging {self.staging_bytes / g:.3f} + pool "
            f"{self.pool_slots * self.slot_bytes / g:.3f} ({self.pool_slots} of {self.num_records} "
            f"experts, {100 * self.capacity_fraction:.2f}%, {self.pool_codec}) = {self.planned_bytes / g:.3f} GiB"
        )


def _capped(budget: float) -> tuple[float, float, float]:
    return (
        min(CACHE_LIMIT_CAP, CACHE_FRACTION * budget),
        min(NON_MLX_MEASURED, NON_MLX_FRACTION * budget),
        min(SAFETY_CAP, SAFETY_FRACTION * budget),
    )


def _score_rungs(grid: int) -> list[int]:
    """Reservation ladder for score: the grid, then halves down to WAVE_TOKENS_STEP."""
    g = max(1, int(grid))
    out = [g]
    t = g
    while t > WAVE_TOKENS_STEP:
        t = max(WAVE_TOKENS_STEP, t // 2)
        out.append(t)
    return out


def moe_kernel_for(scan, cfg, options: dict) -> str:
    """Routed-expert kernel from header geometry: `moe_kernel` (default "grouped") or "stock".

    "grouped" falls back to "stock" for unsupported shapes; the engine asserts
    `Qwen4ExpTextSparseMoeBlock` agrees.
    """
    from jitengine_qwen3_8_flash_next_m4pro.kernels.grouped_expert_qmv import supports_geometry

    want = str(options.get("moe_kernel") or "grouped")
    if want != "grouped":
        return want
    bits, gs = cfg.quant_of("model.layers.0.mlp.switch_mlp.gate_proj")
    by = {}
    for name, shape, dt in zip(
        (
            "gate_proj.weight",
            "gate_proj.scales",
            "gate_proj.biases",
            "up_proj.weight",
            "up_proj.scales",
            "up_proj.biases",
            "down_proj.weight",
            "down_proj.scales",
            "down_proj.biases",
        ),
        scan.field_shape,
        scan.field_dtype,
    ):
        by[name] = (shape, dt)
    in_dims = [
        int(by[f"{p}.scales"][0][-1]) * int(gs) for p in ("gate_proj", "up_proj", "down_proj")
    ]
    ok = supports_geometry(bits, gs, in_dims, by["gate_proj.scales"][1]) and all(
        by[f"{p}.weight"][1] == "U32" for p in ("gate_proj", "up_proj", "down_proj")
    )
    return "grouped" if ok else "stock"


def plan_budget(budget_bytes: float, scan, cfg, options: dict) -> BudgetPlan:  # noqa: PLR0915
    """Plan a budget of `budget_bytes` from a `CheckpointScan` and a `Qwen4ExpTextConfig`, or raise
    `BudgetError`."""
    budget = float(budget_bytes)
    dtype = str(options.get("compute_dtype", "float32"))
    act = dtype_bytes_of(dtype)
    max_slots = max(1, int(options.get("max_slots", BUDGET_MAX_SLOTS)))
    kv_reserve = int(options.get("kv_reserve", BUDGET_KV_RESERVE))
    max_context = max(KV_MIN_POSITIONS, int(options.get("max_context_tokens", BUDGET_MAX_CONTEXT)))
    entries = max(0, int(options.get("prefix_store_entries", BUDGET_STORE_ENTRIES)))
    prefix_tokens = max(
        1, int(options.get("prefix_store_prefix_tokens", BUDGET_STORE_PREFIX_TOKENS))
    )
    grid = int(options.get("prefix_grid_tokens", options.get("wave_chunk", BUDGET_GRID_TOKENS)))
    chunk = grid  # score's partition is the grid; there is no second chunk size
    dense_kernel = str(options.get("dense_kernel") or "rb")
    dense_meta_fp32 = dense_fp32_meta_bytes(scan, cfg, dtype, dense_kernel)
    streamed, EC, cache_rows = _embed_terms(scan, options)
    D = int(scan.dense_bytes) - streamed + dense_meta_fp32  # EC: the streamed table's row cache
    R = int(scan.record_bytes)
    L = int(scan.num_layers)
    experts_per_step = L * int(cfg.num_experts_per_tok)
    codec, codec_basis, S, staging = _pool_codec(scan, cfg, options)
    E = experts_per_step * S
    state_row = gdn_state_bytes(cfg, act) + ple_state_bytes(cfg, act)
    kv_pos = kv_bytes_per_position(cfg, act)
    geom = wave_geometry(cfg)
    state = max_slots * state_row
    kv = max_slots * max_context * kv_pos
    entry = prefix_tokens * kv_pos + state_row
    store = int(entries * entry * STORE_HEADROOM)
    store, kv = _apply_mb_overrides(options, store, kv)

    def fixed_without_scratch(b: float) -> float:
        c, n, s = _capped(b)
        return D + EC + state + kv + store + staging + c + n + s

    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    # The minimum this plan needs is a fixed point in the budget (three caps scale with it), with
    # the smallest score rung and one readout row.
    score_rungs = _score_rungs(grid)
    min_scratch = max(
        wave_scratch_bytes(WAVE_TOKENS_STEP, geom, grid, act, dtype),
        score_scratch_bytes(score_rungs[-1], geom, dtype, vocab, 1, act),
    )
    moe_kernel, upcast, meta_bf16_need, e_need = _moe_kernel_and_expert_need(
        scan, cfg, options, dtype, experts_per_step, S
    )
    need = _iterate_need(fixed_without_scratch, min_scratch, e_need)
    floor = D + EC + E / POOL_HAIRCUT + state_row + KV_MIN_POSITIONS * kv_pos

    remainder = budget - fixed_without_scratch(budget) - E / POOL_HAIRCUT
    if budget <= 0 or remainder < min_scratch:
        raise BudgetError(
            f"a budget of {budget / GIB:.4g} GiB is too small for this checkpoint: dense weights "
            f"{D} B + one decode step's experts {E} B ({L} layers x top-{cfg.num_experts_per_tok} x "
            f"{R} B) = {(D + E) / GIB:.6g} GiB is the geometry floor; with the pool haircut, one row of "
            f"state and {KV_MIN_POSITIONS} KV positions it is {floor / GIB:.6g} GiB, and this plan "
            f"({max_slots} rows x {max_context} positions, {entries} store entries, a "
            f"{WAVE_TOKENS_STEP}-token wave) needs {need / GIB:.6g} GiB. Declare at least "
            f"{_gib_up(need)} GiB."
        )

    allow = WAVE_SCRATCH_SHARE * remainder
    score_tokens = _pick_score_tokens(score_rungs, geom, dtype, vocab, act, remainder)
    readout_rows = _pick_readout_rows(options, allow, score_tokens, geom, act, dtype, vocab)
    score_scratch = score_scratch_bytes(score_tokens, geom, dtype, vocab, readout_rows, act)
    cache_b, non_mlx_b, safety_b = _capped(budget)
    per_slot = S + (2 * meta_bf16_need if upcast else 0)

    def slots_for(sc: float) -> int:
        """Pool slots left beside scratch `sc`; the same arithmetic as the final plan below."""
        pool = budget - (
            D + EC + state + kv + store + staging + sc + cache_b + non_mlx_b + safety_b
        )
        return int(POOL_HAIRCUT * pool // per_slot) if pool > 0 else 0

    wave_tokens = _pick_wave_tokens(
        options, slots_for, score_scratch, geom, grid, act, dtype, allow, experts_per_step
    )
    scratch = max(wave_scratch_bytes(wave_tokens, geom, grid, act, dtype), score_scratch)
    if remainder - scratch + E / POOL_HAIRCUT <= 0:
        raise BudgetError(
            f"a budget of {budget / GIB:.4g} GiB: wave_tokens={wave_tokens} needs {scratch / GIB:.6g} GiB "
            f"of scratch, more than the {(remainder + E / POOL_HAIRCUT) / GIB:.6g} GiB left for the "
            f"pool; declare at least {_gib_up(need)} GiB or lower wave_tokens"
        )

    fixed = D + EC + state + kv + store + staging + scratch + cache_b + non_mlx_b + safety_b
    pool_bytes = budget - fixed
    pool_transient_per_slot, slots = _pool_slots(scan, upcast, pool_bytes, S)
    if slots < experts_per_step:
        raise BudgetError(
            f"a budget of {budget / GIB:.4g} GiB is too small for this checkpoint: the expert pool would "
            f"hold {slots} records, below the {experts_per_step} one decode step routes (dense "
            f"{D} B + one step's experts {E} B = {(D + E) / GIB:.6g} GiB geometry floor; this plan "
            f"needs {need / GIB:.6g} GiB). Declare at least {_gib_up(need)} GiB."
        )

    return BudgetPlan(
        budget_bytes=budget,
        dense_bytes=D,
        record_bytes=R,
        num_records=int(scan.num_records),
        experts_per_step=experts_per_step,
        step_bytes=E,
        max_slots=max_slots,
        kv_reserve=kv_reserve,
        max_context=max_context,
        kv_per_position=kv_pos,
        kv_bytes=int(kv),
        state_per_row=int(state_row),
        state_bytes=int(state),
        store_entries=entries,
        store_prefix_tokens=prefix_tokens,
        store_entry_bytes=int(entry),
        prefix_store_bytes=int(store),
        score_chunk=chunk,
        score_readout_rows=int(readout_rows),
        score_scratch_bytes=int(score_scratch),
        score_tokens=int(score_tokens),
        vocab=int(vocab),
        wave_tokens=int(wave_tokens),
        wave_geometry=int(geom),
        wave_row_tokens=int(min(grid, wave_tokens)),
        scratch_bytes=int(scratch),
        cache_bytes=int(cache_b),
        non_mlx_bytes=int(non_mlx_b),
        safety_bytes=int(safety_b),
        pool_bytes=int(pool_bytes),
        pool_slots=int(slots),
        floor_bytes=int(floor),
        need_bytes=int(need),
        pool_transient_per_slot=int(pool_transient_per_slot),
        compute_dtype=dtype,
        memory_limit_bytes=memory_limit_for(budget),
        moe_kernel=moe_kernel,
        dense_meta_fp32_bytes=int(dense_meta_fp32),
        dense_kernel=dense_kernel,
        pool_codec=codec,
        pool_codec_basis=codec_basis,
        slot_bytes=int(S),
        staging_slots=int(staging // R) if staging else 0,
        staging_bytes=int(staging),
        embed_streamed_bytes=int(streamed),
        embed_cache_rows=int(cache_rows),
        embed_cache_bytes=int(EC),
        store_spill_slots=_store_spill_slots(options, store, S, entries, codec),
    )


def store_spill_wanted(options: dict) -> bool:
    """The `store_spill` option: "on" (default) or "off". Lending the store's whole cap put one-row
    decode at MLX's memory limits on a 24 GB budget, about 11% slower; the annex now borrows only the
    bytes the store's entries hold (`SchedulingMixin._annex_policy`)."""
    v = str(options.get("store_spill", "on")).strip().lower()
    if v not in ("on", "off"):
        raise ValueError(f"store_spill {v!r}: 'on' or 'off'")
    return v == "on"


def _store_spill_slots(options: dict, store: int, slot: int, entries: int, codec: str) -> int:
    """Expert slots the prefix store's cap holds while it is spilled (one-row decode), or 0. Only
    compact (pack16) slots take an annex: their reads land in staging, which is settled before an
    array set goes; bf16 slots are read into directly."""
    if not store_spill_wanted(options) or entries <= 0 or store <= 0 or slot <= 0:
        return 0
    if codec != "pack16":
        return 0
    return int(store) // int(slot)


def _embed_terms(scan, options: dict) -> tuple[int, int, int]:
    """(table bytes not resident, row-cache bytes charged, cache rows): zeros when the table is
    resident (option embed_stream "off", a tied checkpoint, a scan without the shard headers)."""
    from jitengine_qwen3_8_flash_next_m4pro.model import embed_store as _es

    if getattr(scan, "ckpt", None) is None or getattr(scan, "weight_map", None) is None:
        return 0, 0, 0
    streamed = _es.embed_streamed_bytes(scan, options)
    if not streamed:
        return 0, 0, 0
    return streamed, _es.embed_cache_bytes(scan, options), _es.cache_rows_for(options)


def _pool_codec(scan, cfg, options: dict) -> tuple[str, str, int, int]:
    """(codec, why, bytes per pool slot, staging bytes): pack16 compact slots plus the full-size
    staging set their whole-record reads land in, or the bf16 record as the slot and no staging."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels.pack16 import Geometry
    from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache import (
        landing_slots_for,
        reserve_slots_for,
    )
    from jitengine_qwen3_8_flash_next_m4pro.moe.offload_cache.codec import codec_for, staging_slots

    R = int(scan.record_bytes)
    ident = getattr(getattr(scan, "ckpt", None), "index_sha256", None)
    if not callable(getattr(scan, "fields", None)) or ident is None:
        return "off", "the scan carries no record field table", R, 0
    bits, gs = cfg.quant_of("model.layers.0.mlp.switch_mlp.gate_proj")
    codec, why = codec_for(
        scan.fields(),
        gs,
        bits,
        ident,
        int(scan.num_records),
        moe_kernel_for(scan, cfg, options),
        str(options.get("moe_kernel_variant") or "split"),
        str(options.get("pool_codec") or "auto"),
    )
    if codec != "pack16":
        return codec, why, R, 0
    n = staging_slots(landing_slots_for(options), reserve_slots_for(options))
    return codec, why, int(Geometry(scan.fields(), gs, bits).slot_bytes), n * R


def _apply_mb_overrides(options: dict, store: int, kv: int) -> tuple[int, int]:
    """The prefix store and KV sizes, replaced by the MB options when they are given."""
    if options.get("prefix_store_mb") is not None:
        store = int(float(options["prefix_store_mb"]) * 1e6)
    if options.get("kv_budget_mb") is not None:  # the --kv-budget-mb flag, in MB
        kv = int(float(options["kv_budget_mb"]) * 1e6)
    return store, kv


def _moe_kernel_and_expert_need(
    scan, cfg, options: dict, dtype: str, experts_per_step: int, R: int
) -> tuple:
    """The MoE kernel, whether float32 upcasts 16-bit pool metadata, and one step's expert bytes."""
    moe_kernel = moe_kernel_for(scan, cfg, options)
    upcast = dtype == "float32" and moe_kernel == "stock"
    meta_bf16_need = sum(
        int(b) for b, d in zip(scan.field_bytes, scan.field_dtype) if d in ("BF16", "F16")
    )
    e_need = experts_per_step * (R + (2 * meta_bf16_need if upcast else 0))
    return moe_kernel, upcast, meta_bf16_need, e_need


def _iterate_need(fixed_without_scratch, min_scratch, e_need) -> float:
    """The budget this plan needs: the fixed point of its budget-scaled caps after eight rounds."""
    need = fixed_without_scratch(0.0) + min_scratch + e_need / POOL_HAIRCUT
    for _ in range(8):
        need = fixed_without_scratch(need) + min_scratch + e_need / POOL_HAIRCUT
    return need


def _pick_score_tokens(
    score_rungs, geom, dtype: str, vocab: int, act: int, remainder: float
) -> int:
    """The longest score rung whose scratch fits in `remainder`, else the shortest rung."""
    # Reserve the longest score rung that fits. `score` always walks the full grid; if the
    # reservation is shorter, `stats()` says so and MLX's memory limit raises a catchable error.
    score_tokens = score_rungs[-1]
    for st in score_rungs:
        if score_scratch_bytes(st, geom, dtype, vocab, 1, act) <= remainder:
            score_tokens = st
            break
    return score_tokens


def _pick_readout_rows(
    options: dict, allow: float, score_tokens: int, geom, act: int, dtype: str, vocab: int
) -> int:
    """The score readout group: the option, else the largest ladder size that fits the allowance."""
    # the readout group: the largest ladder size whose score path fits the wave's allowance
    readout_rows = 1
    opt_rows = options.get("score_readout_rows")
    if opt_rows is not None:
        readout_rows = max(1, min(SCORE_READOUT_ROWS_MAX, int(opt_rows)))
    else:
        # the grid forward is charged at any group size, so size the group against what is left
        left = allow - wave_scratch_bytes(score_tokens, geom, score_tokens, act, dtype)
        for r in SCORE_READOUT_ROW_LADDER:
            if readout_scratch_bytes(r, vocab) <= left:
                readout_rows = r
                break
    return readout_rows


def _pick_wave_tokens(
    options: dict,
    slots_for,
    score_scratch: float,
    geom,
    grid: int,
    act: int,
    dtype: str,
    allow: float,
    experts_per_step: int,
) -> int:
    """The wave size: wave_tokens stepped down until a step fits, else the largest that fits."""
    opt_wave = options.get("wave_tokens")
    if opt_wave is not None:
        # An explicit wave_tokens overrides the WAVE_SCRATCH_SHARE policy but not the budget: step
        # down the 256-grid to the largest value whose pool still holds one decode step's experts,
        # instead of refusing a valid budget.
        want = max(WAVE_TOKENS_STEP, int(opt_wave))
        wave_tokens = WAVE_TOKENS_STEP
        for t in range(want, WAVE_TOKENS_STEP - 1, -WAVE_TOKENS_STEP):
            sc_t = max(wave_scratch_bytes(t, geom, grid, act, dtype), score_scratch)
            if slots_for(sc_t) >= experts_per_step:
                wave_tokens = t
                break
    else:
        wave_tokens = WAVE_TOKENS_STEP
        for t in range(WAVE_TOKENS_MAX, WAVE_TOKENS_STEP - 1, -WAVE_TOKENS_STEP):
            if max(wave_scratch_bytes(t, geom, grid, act, dtype), score_scratch) <= allow:
                wave_tokens = t
                break
    return wave_tokens


def _pool_slots(scan, upcast: bool, pool_bytes: float, R: int) -> tuple[int, int]:
    """The float32 upcast transient per slot and the pool's records, capped at the record count;
    `R` is the bytes one slot holds (the record, or its pack16 compact form)."""
    # With float32 activations, stock gather_qmm casts the whole pool's bf16 scales and biases to
    # float32 on every call, a transient of 2 x their bytes per slot. The grouped kernel casts per
    # group inside the kernel, so only "stock" with float32 is charged.
    meta_bf16 = sum(
        int(b) for b, d in zip(scan.field_bytes, scan.field_dtype) if d in ("BF16", "F16")
    )
    pool_transient_per_slot = 2 * meta_bf16 if upcast else 0
    slots = int(POOL_HAIRCUT * pool_bytes // (R + pool_transient_per_slot)) if pool_bytes > 0 else 0
    slots = min(slots, int(scan.num_records))
    return pool_transient_per_slot, slots
