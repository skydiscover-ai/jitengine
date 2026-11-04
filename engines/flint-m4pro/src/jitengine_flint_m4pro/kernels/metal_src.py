"""Metal source of the decode megakernel (one persistent dispatch per decode pass over all 49 layers + head).

Rendered in two modes from ONE source:
  CHAIN 0  persistent: G threadgroups x 256 threads loop over every phase of the pass, separated by a global
           barrier (monotonic counter, the fence idiom of rec, Recursive's own inference engine for Flint: plain
           data + device-scope seq_cst fences around the arrival and the release; bounded spin -> error word, every threadgroup exits, never a hang);
  CHAIN 1  chain: one dispatch per phase (phase id and layer from the `ph` input), no barrier: no cross-
           threadgroup traffic inside a dispatch, so it is the race-free reference of the persistent kernel.

Arithmetic = the reference implementation's declared vendor arithmetic (bf(x) = round to bf16 RNE):
GEMV y = bf(x @ W^T) (fp32 accumulate, int4 g64 affine), RMSNorm bf(w * bf(x * rsqrt(mean(x^2) + eps))), RoPE
(sliding layers, cos/sin from the host in fp64), attention fp32 online softmax with the per-head sink, router
logits rounded to bf16 then the mirrored fp32 sigmoid, top-6 (p desc, LOWER id on exact ties), w = bf(p / S),
SwiGLU bf(bf(g * bf(sig g)) * u), combine bf16 after every product and add in rank order, then the shared expert,
then the residual. Head logits bf16; log-probabilities fp32 log_softmax of the bf16 logits.

Every reduction has a fixed order that does not depend on which simdgroup / threadgroup computes it: GEMV rows
are lane-sequential over 512-wide blocks then simd_sum; attention chunks sit on an absolute key grid; merges run
the sink then the chunks in ascending order; head statistics are merged in simdgroup-index order. Repeats are
bit-identical, and the persistent kernel equals the chain kernel bit for bit.

The int4 dot (x pre-scaled by 1, 1/16, 1/256, 1/4096 against masked nibbles) and the GQA attention tile (8 q
heads = one 8x8 simdgroup-matrix tile, layout-2 operands) are this engine's own code. The short-context strided
attention (lane = 4 head dims, keys strided over request-fixed parts, score = simd_sum) follows the decomposition of
rec's attention_partial_path (read for technique; own code).

Phase structure (bit-identical to the plainer structure each item replaces):
  ld_vec       the O / DOWN / DDOWN input staging issues every uint4 load before any threadgroup store
  O, DOWN      1-row units
  MERGE        each lane's <= 2 partial rows loaded up front when there are <= 64 rows (same order and arithmetic)
  grid_sync    every threadgroup polls the arrival counter itself (arrive >= gen * G); no generation word is
               published after a second fence, and one fence pair surrounds arrive and release
DIAG 1 (benchmarks/kernels/megakernel.py only; production renders 0) adds the cold/warm diagnostic: prm[P_DIAG] bit ph
runs phase ph twice in its slot (bit 16: the second run on the next layer of the same kind).

Verify pass: render(mr = M > 1) appends HEADER_V / BODY_V, the verify pass of M rows (mode `code` speculation). Every
row repeats the M = 1 pass's arithmetic at its own position.

MoE block and attention dataflow (bit-identical; both extend to the verify pass: union in first-appearance order,
per-row rank-order combine; flags after all M rows, counters counting every row's tasks):
  MoE block    router fused into the team phase behind a router counter, so ONE grid barrier per sparse layer; a
               10-threadgroup shared-expert team starts before routing, 5-threadgroup routed teams follow;
               gate/up in 2-row units, DOWN in 4-row units, each DOWN piece gated on its own slot's counter; expert
               outputs to Y[rank] (bf16) and the vendor combine (exact arithmetic, rank order) in the next phase's
               prologue
  dataflow     per-head QKV flags and per-kv-head attention task counters replace the QKV -> ATT and ATT -> MERGE grid
               barriers; MERGE -> O keeps its barrier; the chain reference keeps plain dispatches
  150 grid barriers per pass (plan.n_barriers), against 342 for one after every phase but SELECT.

Full-layer attention: one family for raw and code, layout 6d (`att6`) when prm[P_6T] != 0 (plan policy "l6d", the
default), M = 1 and verify rows alike; sliding layers keep the strided path. It runs on a per-position grid: a row
whose last key is q uses chunks of pos_grid(q, T) = 8 ceil((q + 1) / span) keys (span 512, kv8 1,024 above
q + 1 = 32,768); prm[P_6T] carries T. Each row's grid is its own (att_grid, vflow_layer_tasks, vph_att, vph_merge), so
no request quantity reaches the reduction order (prefix-consistent streams) and verify rows stay exact across grid
changes. The layout-2 loops (attend, attend_stream) rescale their accumulators by a diagonal MMA (bit-identical) and
stay selectable through the legacy partition policies ("strided": the strided full path + long stream).
Items are dealt threadgroup-first (gsg / nsg) in ph_att and vph_att (bit-identical), so a second wave of items
(M >= 2 rows, or > 320 items) spreads over every threadgroup instead of landing on the first ones. A partial depends
only on its item, never on the simdgroup that computes it.

kv8 profile: render(kv8=True) is the kv8 VARIANT profile (kv8_src.py): the 14 full layers' K / V at 8 bits in groups of
64, quantised on write in the QKV epilogue and dequantised inside layout 6d (att6q8); main renders are unchanged byte
for byte.

M = 1 kernel prologues (bit-identical) do not wait on chains of cold loads behind a grid barrier. PASS_M1 1 (every
M = 1 render; verify renders 0, their text unchanged):
  descriptors  the table is copied into threadgroup memory once per dispatch (offset >> 4 as uint32; the expert bank
               word raw; plan.pack_descriptors asserts the encoding is lossless), so no phase's first weight address
               waits on a DRAM miss of the table
  LN weights   LN1 / LN2 / final-norm weights are loaded as one uint4 per thread before the prologue (norm2048_q)
  QKV tail     the q / k-norm weight and the RoPE cos / sin of the thread are loaded before the head's GEMV
  combine      the next prologue's combine reuses bf(h + o) the MoE block already formed in the same thread (4 KB of
               threadgroup memory tagged by layer; a stale tag, e.g. chain mode, falls back to the row loads)

Pass entry and failure handling:
  entry        pass_entry: a pass whose previous pass failed records E_PREV and returns in every threadgroup (one wait
               limit per stall, not one per queued pass); the kernel_stall fault seam (prm[P_STALL]: threadgroup G - 1
               returns before any work); the residency probe's flag ring (prm[P_PROBE]). Every error word is recorded
               with set_err, which also marks R_ERRC for the next pass's entry check
Spin waits (bit-identical): grid_sync and team_wait poll their counter with ONE relaxed load per spin (spin_until),
the error word every 64th spin, and the last value seen decides (no reload after the loop); grid_sync's first poll
is its own RMW's result, so the last arriver never polls.
  flow waits   in the M = 1 pass, flow_wait_heads / flow_wait_tasks issue ONE load instruction per spin (per-lane
               address: flag, counter or error word) instead of two divergent loads; verify passes keep the two
               loads (the one-load poll measured +0.04..+0.19% on verify passes on the M4 Pro)
  PK           the M = 1 pass reads prm[P_MASK .. P_6T] from a threadgroup copy made once per dispatch
Phase-mask specialization (PM1; bit-identical): the persistent M = 1 render (pmask 0, its default) has no per-phase
mask test. A production pass runs every phase (prm[P_MASK] = MASK_ALL) or, mask 0 (the residency probe's
barrier-only pass), only the pass's grid barriers in a loop of their own; there is no timing ablation. pmask 1 is
the general kernel (the phase-mask arms of benchmarks/kernels/megakernel.py), and every chain, verify, diag and sabl
render is the general text (specializing the verify pass measured slower at M = 4 on the M4 Pro).
Verify MoE block (bit-identical in production passes): each verify render's MoE block is HEADER_V with vmoe_src.py's
anchored edits, applied at render time: the snake DOWN deal (piece p on threadgroup G - 1 - p mod G) at every M, and
S1s (one union sweep from router-published x) at M >= 6. M = 1 renders and the chain pieces are unchanged.
"""

from __future__ import annotations

from importlib import resources

from jitengine_flint_m4pro.kernels import plan as P
from jitengine_flint_m4pro.kernels import vmoe_src


def read_metal(name: str) -> str:
    """The Metal text of kernels/<name>, byte for byte."""
    return resources.files(__package__).joinpath(name).read_bytes().decode("utf-8")


INPUTS = [
    "b0",
    "b1",
    "b2",
    "desc",
    "kf0",
    "kf1",
    "vf0",
    "vf1",
    "kr",
    "vr",
    "scr",
    "prm",
    "ph",
    "prev",
    "dep",
]

HEADER = read_metal("megakernel_prelude.metal")

HEADER2 = read_metal("megakernel_layers.metal")

BODY = read_metal("megakernel_body.metal")

HEADER_V = read_metal("megakernel_verify_layers.metal")

BODY_V = read_metal("megakernel_verify_body.metal")

# HEADER_V is the single source of every verify render: render() derives each verify M's MoE block from HEADER_V's
# current text with vmoe_src's anchored edits (the snake DOWN deal at every M, S1s from vmoe_src.S1S_MIN_ROWS rows;
# cached per text). Building them here makes an anchor that does not occur exactly once fail this import.
vmoe_src.variants(HEADER_V)


# ------------------------------------------------------------------------------------------- PM1
# The persistent M = 1 pass without per-phase mask tests: anchored edits of HEADER + HEADER2 (after the kv8 edits) and
# of BODY, applied by render(pmask = 0) only, so every other render keeps the general text byte for byte. Each anchor
# must match exactly `count` times, or rendering fails. (anchor, replacement, count)
PM1_HEADER = (
    # moe_fused: every part of the block runs (no RSG / EGU / DOWN arm)
    (
        "  const uint msk = PK(c, P_MASK);\n"
        "  const bool dr = ((msk >> PH_RSG) & 1u) != 0u, dg = ((msk >> PH_EGU) & 1u) != 0u, dd = ((msk >> PH_DOWN) & 1u) != 0u;\n",
        "  const bool dr = true, dg = true, dd = true;   // PM1: every part of the block runs\n",
        1,
    ),
    # flow_layer_tasks: every layer's attention tasks count (no ATT arm, no full-layer ablation)
    (
        "  if (((PK(c, P_MASK) >> PH_ATT) & 1u) == 0u) return 0u;\n"
        "  uint abl = PK(c, P_MASK) >> 16;\n"
        "  if ((abl & 16u) && FIDX[l] < 0) abl = 0u;\n"
        "  if ((abl & 8u) && FIDX[l] >= 0) return 0u;\n",
        "",
        1,
    ),
    # flow_wait_heads: QKV always publishes its heads
    ("  if (((PK(c, P_MASK) >> PH_QKV) & 1u) == 0u) return;\n", "", 1),
    # ph_att: no timing ablation (attend_strided / att_item / attend take abl = 0, a constant)
    (
        "  uint abl = PK(c, P_MASK) >> 16;           // timing ablations (bench_mk only): 1 no q load, 2 no keys, 4 no store,\n"
        "                                            // 8 no full-layer attention at all, 16 = ablations on full layers only\n"
        "  if ((abl & 16u) && FIDX[l] < 0) abl = 0u;\n"
        "  if ((abl & 8u) && FIDX[l] >= 0) return;\n",
        "  const uint abl = 0u;                      // PM1: no timing ablation\n",
        1,
    ),
    # run_phase: every phase runs
    ("  if (((PK(c, P_MASK) >> ph) & 1u) == 0u) return;\n", "", 1),
)
PM1_BODY = (
    (
        "      if ((PK(c, P_MASK) & MOE_BITS) != 0u && !moe_fused(c, l)) return;\n",
        "      if (!moe_fused(c, l)) return;\n",
        1,
    ),
    # the residency probe's barrier-only pass (mask 0; runner.probe): the pass's grid barriers and nothing else
    (
        "  // per layer: QKV -> (flags) ATT -> (task counters) MERGE | O | MoE block (or DGU | DDOWN): 150 grid barriers\n",
        "  if (PK(c, P_MASK) == 0u) {                // PM1: the barrier-only pass (mask 0; runner.probe): every grid barrier\n"
        "    for (uint l = 0; l < NL; ++l) {        // of the pass (plan.n_barriers: 3 per sparse layer, 4 per dense layer,\n"
        "      for (uint b = DENSE[l] ? 4u : 3u; b > 0u; --b) if (!grid_sync(c)) return;   // 1 for the head), no phase\n"
        "    }\n"
        "    grid_sync(c);\n"
        "    return;\n"
        "  }\n"
        "  // per layer: QKV -> (flags) ATT -> (task counters) MERGE | O | MoE block (or DGU | DDOWN): 150 grid barriers\n",
        1,
    ),
)


def _edit(src: str, edits, what: str) -> str:
    for anchor, new, count in edits:
        got = src.count(anchor)
        if got != count:
            raise AssertionError(
                f"{what} source edit: anchor {anchor[:60]!r} found {got} times, expected {count}"
            )
        src = src.replace(anchor, new)
    return src


def pm1(h12: str, body: str) -> tuple:
    """HEADER + HEADER2 (main or kv8) and BODY of the general M = 1 pass -> PM1's: no per-phase mask test is left."""
    h, b = _edit(h12, PM1_HEADER, "PM1 header"), _edit(body, PM1_BODY, "PM1 body")
    assert "PK(c, P_MASK) >>" not in h + b and "MOE_BITS) !=" not in b, (
        "a per-phase mask test is left"
    )
    return h, b


def render(
    cfg,
    expert_offsets: dict,
    chain: bool,
    kb: int = 1,
    hg: int = 4,
    sabl: int = 0,
    diag: int = 0,
    mr: int = 1,
    G: int = P.G_DEFAULT,
    xs: int = P.VXS,
    kv8: bool = False,
    pmask: int | None = None,
) -> tuple:
    """(header, body) for mx.fast.metal_kernel. diag = 1 compiles the bench-only cold/warm diagnostic (P_DIAG).
    mr > 1 renders the verify pass of mr rows: HEADER_V with that M's MoE block (vmoe_src.verify_header);
    mr = 1 the M = 1 pass (raw and forced requests).
    kv8 = True renders the kv8 profile (kv8_src.py: anchored edits of this source); False is the main profile.
    PASS_M1 = (mr == 1) selects the M = 1 pass's threadgroup descriptor copy; the verify pass keeps its
    device-table reads.
    pmask: 0 = PM1, the persistent M = 1 pass without per-phase mask tests (that render's default; a pass runs every
    phase, or with mask 0 only its grid barriers); 1 = the general kernel (phase-mask arms and timing ablations:
    benchmarks/kernels/megakernel.py). Chain, verify, diag and sabl renders are general."""
    general = bool(chain) or mr != 1 or bool(diag) or bool(sabl)
    pmask = int(general) if pmask is None else int(pmask)
    assert pmask in (0, 1) and (pmask or not general), (
        "pmask = 0 specializes the persistent M = 1 pass only"
    )
    fidx = [
        cfg.full_layers.index(layer) if layer in cfg.full_layers else -1
        for layer in range(cfg.n_layers)
    ]
    sidx = [
        cfg.sliding_layers.index(layer) if layer in cfg.sliding_layers else -1
        for layer in range(cfg.n_layers)
    ]
    dense = [1 if layer in cfg.dense_layers else 0 for layer in range(cfg.n_layers)]
    assert cfg.n_layers == P.NL and len(cfg.full_layers) == 14 and cfg.sliding_window == P.WINDOW
    assert (cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.top_k) == (
        2048,
        32,
        4,
        128,
        6,
    )
    assert (
        cfg.moe_inter == 512
        and cfg.shared_inter == 1024
        and cfg.dense_inter == 6144
        and cfg.vocab == P.VOCAB
    )
    assert kb >= 1 and hg in (1, 2, 4, 8) and sabl in (0, 1, 2) and diag in (0, 1)
    assert mr == 1 or (sabl == 0 and diag == 0), "the verify pass has no timing ablations"
    assert G > 10, "the MoE block's router threadgroups are w >= MQS = 10"
    vals = {
        "CHAIN": 1 if chain else 0,
        "DIAG": int(diag),
        "PASS_M1": 1 if mr == 1 else 0,
        "KB": int(kb),
        "HG": int(hg),
        "SABL": int(sabl),
        "VOCAB": P.VOCAB,
        "RING": P.RING,
        "WINDOW": P.WINDOW,
        "EPS": repr(float(cfg.rms_eps)),
        "NF": P.NF,
        "DG": P.D_GLOBAL,
        "FIDX": ",".join(map(str, fidx)),
        "SIDX": ",".join(map(str, sidx)),
        "DENSE": ",".join(map(str, dense)),
        "E_STRIDE": expert_offsets["stride"],
        "EG_W": expert_offsets["gate_proj.weight"],
        "EG_S": expert_offsets["gate_proj.scales"],
        "EG_B": expert_offsets["gate_proj.biases"],
        "EU_W": expert_offsets["up_proj.weight"],
        "EU_S": expert_offsets["up_proj.scales"],
        "EU_B": expert_offsets["up_proj.biases"],
        "ED_W": expert_offsets["down_proj.weight"],
        "ED_S": expert_offsets["down_proj.scales"],
        "ED_B": expert_offsets["down_proj.biases"],
    }
    vals.update({"OFF_" + k: v for k, v in P.OFF.items()})
    assert not kv8 or (sabl == 0 and diag == 0), "the kv8 profile has no timing ablations"
    h12 = HEADER + HEADER2
    if kv8:
        from jitengine_flint_m4pro.kernels import kv8_src

        h12 = kv8_src.apply_m1(h12)
    if mr == 1:
        vals.update(
            R_ARRIVE=P.R_ARRIVE,
            R_ERR=P.R_ERR,
            R_ATTQ=P.R_ATTQ,
            R_QFLAG=P.R_QFLAG,
            R_FCNT=P.R_FCNT,
            S_CTR=P.ctr_offset(1),
        )
        h, body = (h12, BODY) if pmask else pm1(h12, BODY)
    else:
        vals.update(verify_values(mr, G, xs))
        hv = vmoe_src.verify_header(HEADER_V, mr)  # this verify render's MoE block (vmoe_src.py)
        h, body = h12 + (kv8_src.apply_v(hv) if kv8 else hv), BODY_V
    for k, v in vals.items():
        h = h.replace("__" + k + "__", str(v))
    assert "__" not in h.replace("__metal", ""), [s for s in h.split() if "__" in s][:5]
    # the descriptor field numbering above must match plan.D
    assert (
        P.D["self_attn.q_proj.weight"],
        P.D["self_attn.k_proj.weight"],
        P.D["self_attn.v_proj.weight"],
        P.D["self_attn.o_proj.weight"],
        P.D["self_attn.q_norm.weight"],
        P.D["self_attn.k_norm.weight"],
        P.D["self_attn.softmax_offset"],
        P.D["mlp.gate.weight"],
        P.D["mlp.shared_expert.gate_proj.weight"],
        P.D["mlp.shared_expert.up_proj.weight"],
        P.D["mlp.shared_expert.down_proj.weight"],
        P.D["mlp.gate_proj.weight"],
        P.D["mlp.up_proj.weight"],
        P.D["mlp.down_proj.weight"],
        P.D_EBANK,
        P.D_EBASE,
    ) == (2, 5, 8, 11, 14, 15, 16, 17, 18, 21, 24, 27, 30, 33, 36, 37)
    return h, body


def verify_values(mr: int, G: int, xs: int = P.VXS) -> dict:
    """Compile-time constants of the mr-row verify pass: staging groups that fit xs bf16 of x per threadgroup."""
    assert 2 <= mr <= P.VMAX and G % 4 == 0 and 4 <= G and xs % 2048 == 0 and xs >= 6144
    og = P.vgroups(mr, 4)
    kco = max(k for k in (512, 1024, 2048, 4096) if og * k <= xs)
    vals = {
        "MR": mr,
        "XSV": xs,
        "RG2K": P.vgroups(mr, xs // 2048),
        "RG4K": P.vgroups(mr, xs // 4096),
        "RG6K": P.vgroups(mr, xs // 6144),
        "RGE": P.vgroups(mr, 2),
        "OG": og,
        "KCO": kco,
        "UO": -(-2048 // (8 * G)),
        "SCR_ROW": P.SCRATCH_BYTES,
        "R_ARRIVE": P.VR_ARRIVE,
        "R_ERR": P.VR_ERR,
        "R_ATTQ": P.VR_ATTQ,
        "R_QFLAG": P.VR_QFLAG,
        "R_FCNT": P.VR_FCNT,
        "S_CTR": P.ctr_offset(P.VMAX),
    }
    assert (
        P.VMAX * P.VROW_WORDS <= P.VR_ARRIVE
        and P.VR_FCNT + 4 <= P.VRES_WORDS
        and vals["RG2K"] * 2048 <= xs
        and vals["RG4K"] * 4096 <= xs
    )
    return vals
