"""Host-side plan of the decode megakernel: buffer layouts, the attention chunk grid, descriptors, parameters.

Everything here is pure numpy (no GPU work), so it is unit-testable without weights.

Scratch (one persistent uint8 input the kernel writes in place; offsets in bytes):
  HS      h trace [50][2048] bf16 (h_0 = embedding row, h_{l+1} = layer l's output)
  Q       [49][4096] bf16    post-norm (and RoPE) queries
  ATTN    [49][4096] bf16    merged attention output
  O       [49][2048] bf16    o_proj output
  SC      [49][256]  fp32    router probabilities (sigmoid of the bf16 router logits)
  ACT     [49][6144] bf16    SwiGLU activations (MoE: 6 x 512 routed | 1024 shared at 3072; dense: 6144)
  PART    [1024][8][132] fp32 attention partials (m, l, -, -, o[128]) per (kv head, part or chunk) x 8 q heads
  LOGITS  [150144] bf16      bf16 logits of the pass
  SELP    [512][48] u32      per-simdgroup head statistics (m, sum exp, finite, top-20 values and ids)
  Y       [7][2048] bf16     the MoE block's expert outputs of the pass's last sparse layer, rank order,
                             6 = shared (combined in the next phase's prologue)
After the row region(s): the MoE team counters [49][64] uint32 on 64-byte lines (synchronisation state, zeroed
by every pass; never compared): one row region at M = 1 (MK_SCRATCH_BYTES), VMAX in the verify scratch.

Result (the kernel's only output, zero-initialised per pass): RES_WORDS uint32 words, see R_* below; the grid
barrier's counter / generation / error words and the attention dataflow's flags and task counters live in
the same buffer on their own 128-byte lines.
"""

from __future__ import annotations

import math
import re

import numpy as np

NL = 49
HID = 2048
NKV = 4
VOCAB = 150144
RING = 528  # sliding-window ring slots (513 window + 15 slack), kv_cache.py
WINDOW = 513
TOPN = 20
G_DEFAULT = 40  # threadgroups (2 per core on the 20-core M4 Pro GPU)
TPG = 256  # threads per threadgroup
MAX_SG = 512  # SELP capacity (supports G <= 64)
MAX_CHUNKS = 1024  # partials per kv head (strided parts <= 1024; layout-2 grid <= 80)
PROW = 132  # floats per partial row (m, l, pad, pad, o[128]; 16-byte aligned o)
PARTS_SLIDE = 28  # strided parts of the 528-slot sliding rings (<= 19 keys per task; 224 tasks)
PARTS_FULL = 40  # strided parts of a full layer per 2,048 of N_max: 8 * 40 = 320 tasks = one task
# per simdgroup at G = 40, so every simdgroup walks the same N_max / 40 keys
# (measured on the M4 Pro: 40 / 28 beat 64 / 64 by 0.10 ms at 1.5K; power-of-two part counts camp on
# DRAM channels: key stride = parts x 256 B)
LONG_FULL = 24576  # N_max above this: full layers use the layout-2 MMA chunk grid (crossover
# measured between 16K (strided 11.90 vs 12.29 ms) and 33K (14.85 vs 15.66 ms))
# Layout 6d on a PER-POSITION grid: a full-layer row whose last key is q (keys 0..q) is cut into chunks of
# S(q) = 8 * ceil((q + 1) / span) keys, span L6_SPAN while q + 1 <= T, else L6_SPAN_FINE (pos_grid). The grid is a pure
# function of the row's own position, never of N_max / max_tokens / context_limit / capacity, so a greedy or seeded
# stream is the same whatever the request asks for, and a verify row equals the M = 1 pass at its position across
# every grid change (rows straddling one simply have different S).
# <= 64 chunks per kv head = <= 256 items on the 320 simdgroups at G = 40 (an M4 Pro sweep found 64-68
# partials per kv head best at 1.5K-258K)
L6_SPAN = 512
L6_SPAN_FINE = 1024  # the twice-finer span, <= 128 chunks per kv head (the merge's > 64 path)
T_INF = 1 << 30  # a span switch no position reaches: always span 512 (main)
# kv8: span 512 while q + 1 <= 32,768, then 1,024. Measured on the M4 Pro: a request-fixed switch has its
# crossover between N_max 33,793 (l6d2 / l6d 1.001) and 48K (0.993; 0.978 at 128K), and this position rule
# decodes kv8 M = 1 0.2-3.5% faster than the request-fixed grid it replaced, at 1K-128K, rows exact
KV8_FINE_FROM = 32768
SIXD_T = {
    "l6d": T_INF,
    "l6d2": 1,
    "l6dk": KV8_FINE_FROM,
}  # policy -> T (prm[P_6T]); "l6d2": T = 1, i.e. the fine span at
# every q (q = 0 has S = 8 on either span)
DEFAULT_POLICY = "l6d"  # main
KV8_POLICY = "l6dk"  # the kv8 profile

# ---------------------------------------------------------------- scratch layout
_SIZES = [
    ("HS", 50 * HID * 2),
    ("Q", NL * 4096 * 2),
    ("ATTN", NL * 4096 * 2),
    ("O", NL * HID * 2),
    ("SC", NL * 256 * 4),
    ("ACT", NL * 6144 * 2),
    ("PART", NKV * MAX_CHUNKS * 8 * PROW * 4),
    ("LOGITS", VOCAB * 2),
    ("SELP", MAX_SG * 48 * 4),
    ("Y", 7 * HID * 2),
]
OFF: dict = {}
_o = 0
for _n, _s in _SIZES:
    OFF[_n] = _o
    _o += (_s + 255) // 256 * 256
SCRATCH_BYTES = _o  # one row region (the M = 1 layout; verify row m at m * SCRATCH_BYTES)
CTR_BYTES = (
    NL * 64 * 64
)  # MoE team counters: [layer][slot] on 64-byte lines (slots 0..47 routed, 61 router, 63 shared)
MK_SCRATCH_BYTES = SCRATCH_BYTES + CTR_BYTES


def ctr_offset(rows: int) -> int:
    """Byte offset of the team counters in a scratch of `rows` row regions (1: M = 1 passes, VMAX: verify passes)."""
    return rows * SCRATCH_BYTES


# ---------------------------------------------------------------- result layout (uint32 words)
R_TOKEN, R_FINITE, R_ERR_COPY, R_ARGMAX, R_LSE, R_CHOSEN_LP, R_FORCED_LP, R_DONE = range(8)
R_TOPI = 8  # 20 ids
R_TOPV = 28  # 20 log-probabilities (fp32 bits)
R_ARRIVE, R_GEN, R_ERR = (
    128,
    160,
    192,
)  # R_GEN: unused (the barrier polls R_ARRIVE itself)
R_ATTQ = 224  # 14 work counters of the dynamic full-layer attention queue (one per full layer)
R_QFLAG = 256  # attention dataflow: 40 per-head "published layer + 1" flags ...
R_FCNT = 320  # ... and 4 per-kv-head attention task counters (persistent kernel only)
RES_WORDS = 384

# ---------------------------------------------------------------- per-pass parameters (uint32 words)
(
    P_POS,
    P_TOKMODE,
    P_TOKEN,
    P_FORCED,
    P_SELMODE,
    P_TOPK,
    P_TOPP,
    P_INVT,
    P_U,
    P_POISON,
    P_MASK,
    P_SPIN,
    P_SF,
    P_SS,
    P_CAP,
    P_PF,
    P_PS,
    P_FSUB,
    P_FASG,
    P_FFLAGS,
) = range(20)
P_DIAG = 20  # benchmarks/kernels/megakernel.py's cold/warm diagnostic (kernels rendered with diag=1 only)
P_6T = (
    21  # layout 6d: the span switch T of pos_grid (plan.SIXD_T; 0 = the strided / layout-2 paths)
)
P_STALL = (
    22  # fault seam kernel_stall (allow_fault_injection only): nonzero -> threadgroup G - 1 returns
)
# before any work of the pass, so its waiters time out in the kernel (faults.py)
P_PROBE = (
    23  # residency probe (runner.probe): nonzero -> a flag ring of that many laps through every
)
# threadgroup instead of the pass (rendered into every persistent kernel, never set in a pass)
P_COS, P_SIN = 24, 88
PRM_WORDS = 152
NO_TOKEN = 0xFFFFFFFF

# phases (bit i of P_MASK enables phase i; the grid barriers run regardless)
PHASES = ("QKV", "ATT", "MERGE", "O", "RSG", "EGU", "DOWN", "DGU", "DDOWN", "HEAD", "SELECT")
PH = {n: i for i, n in enumerate(PHASES)}
MASK_ALL = (1 << len(PHASES)) - 1
SPIN_DEFAULT = (
    1 << 23
)  # bounded spin per wait (barrier, team, flow): a stall is an error, never a hang.
# Measured with the kernel_stall fault (kernel_stall@5 in the engine, 1K and 32K alike): ~0.38 us
# per spin with the stalled head's tasks all spinning, so 2^26 took 23-26 s to the error
# (fault_loud bound 30 s), 2^24 6.4 s, 2^23 3.0-3.2 s; the longest legitimate wait is one
# phase (a few ms), ~1000x below
SPIN_PROBE = 1 << 17

# in-kernel error words (res[R_ERR]; the first failing wait writes it, every other wait then aborts) -> describe_err
E_TEAM = 0x40000000  # | barrier generation: a MoE team counter wait (moe_fused)
E_FLOW_HEADS = 0x20000000  # | layer + 1: an attention task's wait for its QKV head flags
E_FLOW_TASKS = (
    0x10000000  # | layer + 1: a merge task's wait for its kv head's attention task counter
)
E_PREV = 0x08000000  # the pass did not run: its previous pass (the `prev` input) had failed
E_RING = 0x04000000  # | threadgroup << 8 | lap: the residency probe's flag ring (runner.probe)
# (no prefix: the generation of a grid barrier that timed out)


def describe_err(word: int) -> str:
    """Human-readable name of an in-kernel error word (the barrier, team, flow or ring wait that timed out)."""
    w = int(word)
    if w & E_TEAM:
        return f"MoE team counter wait timed out after grid barrier {w & 0xFFFF} [0x{w:08x}]"
    if w & E_FLOW_HEADS:
        return f"attention head-flag wait (QKV -> ATT) timed out at layer {(w & 0xFFFF) - 1} [0x{w:08x}]"
    if w & E_FLOW_TASKS:
        return f"attention task-counter wait (ATT -> MERGE) timed out at layer {(w & 0xFFFF) - 1} [0x{w:08x}]"
    if w & E_PREV:
        return f"not run: the previous pass had failed [0x{w:08x}]"
    if w & E_RING:
        return f"residency flag ring timed out at threadgroup {(w >> 8) & 0xFF}, lap {w & 0xFF} [0x{w:08x}]"
    return f"grid barrier {w} timed out [0x{w:08x}]"


def n_barriers(dense_layers) -> int:
    """Grid barriers of one persistent pass: MERGE -> O and O -> MLP per layer, then DGU -> DDOWN and
    DDOWN -> next on a dense layer or ONE after the fused MoE block, and HEAD -> SELECT (QKV -> ATT -> MERGE run on
    flags and task counters). 150 for Flint (one after every phase but SELECT would be 342)."""
    return sum(4 if layer in dense_layers else 3 for layer in range(NL)) + 1


def phase_schedule(dense_layers) -> list:
    """[(phase id, layer)] of one decode pass, in execution order (the chain mode dispatches exactly these; on a
    sparse layer RSG = router + shared gate/up, EGU = routed gate/up, DOWN = every DOWN piece of the MoE block)."""
    s = []
    for layer in range(NL):
        s += [(PH["QKV"], layer), (PH["ATT"], layer), (PH["MERGE"], layer), (PH["O"], layer)]
        if layer in dense_layers:
            s += [(PH["DGU"], layer), (PH["DDOWN"], layer)]
        else:
            s += [(PH["RSG"], layer), (PH["EGU"], layer), (PH["DOWN"], layer)]
    s += [(PH["HEAD"], NL), (PH["SELECT"], NL)]
    return s


# ---------------------------------------------------------------- attention chunk grid
# Long full-layer stream (layout 2, N_max > LONG_FULL): (chunks per kv head C, sub-streams per chunk NSUB,
# streaming simdgroups per threadgroup ASG, flags). Items are (kv head, chunk, sub); sub-stream s of a chunk takes
# its 8-key blocks s, s + NSUB, s + 2*NSUB, ..., so the NSUB sub-streams of a chunk read adjacent blocks together.
FL_SGFIRST = 1  # deal items simdgroup-first inside a threadgroup (default: threadgroup-first)
FL_DYNAMIC = 4  # items taken from a per-layer atomic queue instead of a static deal
# Measured on the M4 Pro: 240 streaming simdgroups (6 of 8 per threadgroup), ONE item each, beat 320 (8 per
# threadgroup, C = 80) by 0.7 / 1.3 / 1.7 ms per pass at 33K / 66K / 131K; 4 or 5 per threadgroup
# starve DRAM, 7-8 oversubscribe it, and more items per stream (static or dynamic) pay a per-item cost each.
LONG_DEFAULT = (60, 1, 6, FL_SGFIRST)
_LONG_RE = re.compile(r"L(\d+)x(\d+)a(\d+)f(\d+)")


def parse_long(policy: str):
    """'L<C>x<NSUB>a<ASG>f<FLAGS>' -> (C, NSUB, ASG, FLAGS): the long full-layer stream at ANY N_max (sweeps)."""
    m = _LONG_RE.fullmatch(policy)
    if m is None:
        return None
    C, nsub, asg, fl = (int(x) for x in m.groups())
    if not (
        1 <= C
        and 1 <= nsub
        and C * nsub <= MAX_CHUNKS
        and 1 <= asg <= 8
        and fl & ~(FL_SGFIRST | FL_DYNAMIC) == 0
    ):
        raise ValueError(f"bad long-attention policy {policy!r}")
    return C, nsub, asg, fl


def long_chunk(n_max: int, C: int) -> int:
    """S_f of the long stream: the smallest multiple of 8 (>= 32) whose C chunks cover N_max."""
    return max(32, 8 * math.ceil(n_max / (8 * C)))


def pos_grid(q: int, T: int) -> int:
    """Keys per full-layer chunk of a layout-6d row whose last key is q (keys 0..q): 8 * ceil((q + 1) / span), span 512
    while q + 1 <= T, else 1,024 (<= 64 chunks per kv head on span 512, <= 128 on 1,024). The kernel's pos_grid
    (metal_src.py) is the same integer formula; tests/metal/kernels/test_pos_grid.py checks every q < 262,144."""
    n = int(q) + 1
    return 8 * -(-n // (L6_SPAN if n <= T else L6_SPAN_FINE))


def pos_chunks(q: int, T: int) -> int:
    """Chunks per kv head of the row whose last key is q."""
    return -(-(int(q) + 1) // pos_grid(q, T))


def six_text(policy: str) -> str:
    T = SIXD_T[policy]
    if T >= T_INF:
        return f"S(q) = 8*ceil((q+1)/{L6_SPAN})"
    if T <= 1:
        return f"S(q) = 8*ceil((q+1)/{L6_SPAN_FINE})"
    return f"S(q) = 8*ceil((q+1)/{L6_SPAN}) while q+1 <= {T}, 8*ceil((q+1)/{L6_SPAN_FINE}) above"


def is_6d(policy: str) -> bool:
    """True for the layout-6d partitions (every full layer 6d on the per-position grid, strided rings): the only
    partitions the engine serves."""
    return policy in SIXD_T


def default_policy(kv_bits: int = 16) -> str:
    """The partition a profile uses unless one is named: main "l6d" (span 512 at every q); kv8 "l6dk" (span 1,024
    above q + 1 = KV8_FINE_FROM)."""
    return KV8_POLICY if int(kv_bits) == 8 else DEFAULT_POLICY


def check_policy(policy: str) -> None:
    """ValueError unless `policy` names a partition (layout 6d, or a legacy one of chunk_sizes)."""
    if policy not in SIXD_T:
        chunk_sizes(1024, policy)


def chunk_sizes(n_max: int, policy: str) -> tuple:
    """Request-fixed partitions (timing sweeps of benchmarks/kernels/megakernel.py and
    tests/metal/kernels/test_megakernel.py only; the engine serves layout 6d, whose grid is per position: pos_grid /
    attn_params). (S_f, S_s, parts_f, parts_s), fixed from N_max.

    Strided (scalar) path, used where parts_* > 0: task = (kv head, part, group of HG = 4 q heads), the HG heads
    share each K/V load, and part k owns keys k, k + parts, k + 2*parts, ... (lane = 4 head dims, score = simd_sum).
    Layout-2 (MMA) path where parts_* == 0: chunk c owns keys [c*S, (c+1)*S) (S_f / S_s).
      strided : sliding parts 28; full parts 40 * ceil(N_max / 2048) while N_max <= LONG_FULL, else the
                long layout-2 stream LONG_DEFAULT (S_f = long_chunk(N_max, C))
      strided_all : strided on full layers at any N_max (sweeps)
      L<C>x<NSUB>a<ASG>f<FLAGS> : the long layout-2 stream with these parameters at any N_max (sweeps)
      grid    : layout 2 everywhere: S_s = 16, S_f = max(32, 8 * ceil(N_max / 640))
      c4      : layout 2 with 4 chunks per kv head (slower in this structure)
      sNN     : layout 2 with S_s = S_f lower bound NN (sweeps)
    """
    n_max = max(1, int(n_max))
    grid_f = max(32, 8 * math.ceil(n_max / 640))
    p_f = p_s = 0
    lg = parse_long(policy)
    if policy in SIXD_T:
        raise ValueError(
            f"{policy!r} is a per-position layout-6d partition (pos_grid), not a request-fixed one"
        )
    if lg is not None:
        s_f, s_s, p_s = long_chunk(n_max, lg[0]), 16, PARTS_SLIDE
    elif policy in ("strided", "strided_all"):
        s_f, s_s = long_chunk(n_max, LONG_DEFAULT[0]), 16
        p_s = PARTS_SLIDE
        if n_max <= LONG_FULL or policy == "strided_all":
            p_f = min(MAX_CHUNKS, PARTS_FULL * math.ceil(n_max / 2048))
    elif policy == "grid":
        s_f, s_s = grid_f, 16
    elif policy == "c4":
        s_f = max(8, 8 * math.ceil(n_max / 32))
        s_s = 8 * math.ceil(RING / 32)
    elif policy.startswith("s") and policy[1:].isdigit():
        k = int(policy[1:])
        s_f = max(k, 8 * math.ceil(n_max / 640))
        s_s = k
    else:
        raise ValueError(f"unknown attention partition policy {policy!r}")
    assert (
        -(-n_max // s_f) <= MAX_CHUNKS
        and -(-RING // s_s) <= MAX_CHUNKS
        and p_f <= MAX_CHUNKS
        and p_s <= MAX_CHUNKS
    )
    return s_f, s_s, p_f, p_s


def attn_params(policy: str = DEFAULT_POLICY, n_max: int | None = None) -> dict:
    """The attention words of make_params / make_vparams.
    Layout 6d (the engine's partitions): full layers on the per-position grid, whose only parameter is the span switch
    T = SIXD_T[policy] (six_t -> prm[P_6T]; the kernel derives every row's S and C from the row's own position, so
    nothing depends on the request and n_max is ignored); sliding rings strided over PARTS_SLIDE parts of ring slots.
    Request-fixed partitions (timing sweeps): fixed from n_max (chunk_sizes + the long full-layer stream)."""
    if policy in SIXD_T:
        return {
            "s_f": 0,
            "s_s": 16,
            "p_f": 0,
            "p_s": PARTS_SLIDE,
            "f_sub": 1,
            "f_asg": 8,
            "f_flags": 0,
            "six_t": SIXD_T[policy],
        }
    if n_max is None:
        raise ValueError(f"legacy partition {policy!r} is request-fixed: it needs n_max")
    s_f, s_s, p_f, p_s = chunk_sizes(n_max, policy)
    _C, nsub, asg, fl = parse_long(policy) or LONG_DEFAULT
    if p_f:  # strided full layers: the long-stream fields are unused
        nsub, asg, fl = 1, 8, 0
    assert -(-max(1, int(n_max)) // s_f) * nsub <= MAX_CHUNKS
    return {
        "s_f": s_f,
        "s_s": s_s,
        "p_f": p_f,
        "p_s": p_s,
        "f_sub": nsub,
        "f_asg": asg,
        "f_flags": fl,
        "six_t": 0,
    }


def verify_capable(att: dict) -> bool:
    """Verify passes exist for partitions whose full layers are 6d or strided and whose rings are strided (the
    legacy long layout-2 stream has no multi-row form)."""
    return att.get("p_s", 0) > 0 and (att.get("six_t", 0) > 0 or att.get("p_f", 0) > 0)


# ---------------------------------------------------------------- weight descriptors
LAYER_FIELDS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.scales",
    "self_attn.q_proj.biases",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.scales",
    "self_attn.k_proj.biases",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.scales",
    "self_attn.v_proj.biases",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.scales",
    "self_attn.o_proj.biases",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.softmax_offset",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.gate_proj.scales",
    "mlp.shared_expert.gate_proj.biases",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.up_proj.scales",
    "mlp.shared_expert.up_proj.biases",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert.down_proj.scales",
    "mlp.shared_expert.down_proj.biases",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.scales",
    "mlp.gate_proj.biases",
    "mlp.up_proj.weight",
    "mlp.up_proj.scales",
    "mlp.up_proj.biases",
    "mlp.down_proj.weight",
    "mlp.down_proj.scales",
    "mlp.down_proj.biases",
)
D = {f: i for i, f in enumerate(LAYER_FIELDS)}
D_EBANK = len(LAYER_FIELDS)  # 36: expert bank index (1 or 2)
D_EBASE = D_EBANK + 1  # 37: byte offset of expert 0 of this layer in that bank
NF = 40
GLOBAL_FIELDS = (
    "model.norm.weight",
    "lm_head.weight",
    "lm_head.scales",
    "lm_head.biases",
    "model.embed_tokens.weight",
    "model.embed_tokens.scales",
    "model.embed_tokens.biases",
)
D_GLOBAL = NL * NF
DESC_WORDS = D_GLOBAL + 8
EXPERT_RECORD = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)


def pack_descriptors(plan) -> np.ndarray:
    """uint64 [DESC_WORDS]: bank-0 byte offsets of every layer field (0 where absent), the expert bank and base
    of every routed layer, then the global tensors."""
    tab = np.zeros(DESC_WORDS, dtype=np.uint64)
    cfg = plan.cfg
    for layer in range(cfg.n_layers):
        for f, k in D.items():
            s = plan.slots.get(f"model.layers.{layer}.{f}")
            if s is not None:
                assert s.bank == 0 and s.offset % 16 == 0
                tab[layer * NF + k] = s.offset
        if layer in plan.expert_home:
            bank, off = plan.expert_offset(layer, "gate_proj", "weight")
            tab[layer * NF + D_EBANK] = bank
            tab[layer * NF + D_EBASE] = off
    for k, n in enumerate(GLOBAL_FIELDS):
        tab[D_GLOBAL + k] = plan.slots[n].offset
    check_tg_descriptors(tab)
    return tab


def check_tg_descriptors(tab: np.ndarray) -> None:
    """The M = 1 kernel keeps the table in threadgroup memory as uint32 words: offset >> 4 for
    every byte offset, the expert bank word raw. That encoding must be lossless: every offset 16-byte aligned and
    < 2^36, every bank word < 2^32."""
    t = np.asarray(tab, dtype=np.uint64)
    assert t.shape == (DESC_WORDS,)
    bank = np.zeros(DESC_WORDS, dtype=bool)
    bank[D_EBANK:D_GLOBAL:NF] = True
    off, bw = t[~bank], t[bank]
    if np.any(off % np.uint64(16)) or np.any(off >> np.uint64(36)) or np.any(bw >> np.uint64(32)):
        raise ValueError(
            "descriptor table not encodable as the M = 1 kernel's threadgroup copy (offset >> 4 in "
            "uint32 needs 16-byte aligned offsets < 2^36)"
        )


def expert_record_offsets(plan) -> dict:
    out = {}
    for name in EXPERT_RECORD:
        proj, suf = name.split(".")
        out[name] = plan.expert_fields[(proj, suf)].offset
    out["stride"] = plan.expert_stride
    return out


# ---------------------------------------------------------------- per-pass parameters
def rope_table(pos: int, theta: float, half: int = 64) -> tuple:
    """cos/sin of position * theta^(-i/64), angle in fp64 (R's exact mode), rounded to fp32."""
    i = np.arange(half, dtype=np.float64)
    ang = float(pos) * (theta ** (-i / half))
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def f32_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def make_params(
    *,
    pos: int,
    token: int | None,
    forced_next: int | None,
    sel_mode: int,
    top_k: int,
    top_p: float,
    temperature: float,
    u: float,
    poison: bool,
    s_f: int,
    s_s: int,
    cap: int,
    theta: float,
    mask: int = MASK_ALL,
    spin: int = SPIN_DEFAULT,
    p_f: int = 0,
    p_s: int = 0,
    f_sub: int = 1,
    f_asg: int = 8,
    f_flags: int = 0,
    six_t: int = 0,
    stall: bool = False,
    probe: int = 0,
) -> np.ndarray:
    """uint32 [PRM_WORDS] for one pass. token None = take the token from the previous pass's result.
    stall: the kernel_stall fault seam (faults.py); probe: residency-probe laps (runner.probe only)."""
    p = np.zeros(PRM_WORDS, dtype=np.uint32)
    p[P_POS] = pos
    p[P_TOKMODE] = 1 if token is None else 0
    p[P_TOKEN] = 0 if token is None else int(token)
    p[P_FORCED] = NO_TOKEN if forced_next is None else int(forced_next)
    p[P_SELMODE] = sel_mode
    p[P_TOPK] = max(1, min(TOPN, int(top_k)))
    p[P_TOPP] = f32_bits(top_p)
    p[P_INVT] = f32_bits(1.0 / temperature if temperature > 0 else 1.0)
    p[P_U] = f32_bits(u)
    p[P_POISON] = 1 if poison else 0
    p[P_MASK] = mask
    p[P_SPIN] = spin
    p[P_SF] = s_f
    p[P_SS] = s_s
    p[P_CAP] = cap
    p[P_PF] = p_f
    p[P_PS] = p_s
    p[P_FSUB] = f_sub
    p[P_FASG] = f_asg
    p[P_FFLAGS] = f_flags
    p[P_6T] = six_t
    p[P_STALL] = 1 if stall else 0
    p[P_PROBE] = probe
    c, s = rope_table(pos, theta)
    p[P_COS : P_COS + 64] = c.view(np.uint32)
    p[P_SIN : P_SIN + 64] = s.view(np.uint32)
    return p


def parse_result(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=np.uint32)
    f = r.view(np.float32)
    return {
        "token": int(r[R_TOKEN]),
        "finite": bool(r[R_FINITE]),
        "err": int(r[R_ERR]),
        "argmax": int(r[R_ARGMAX]),
        "lse": float(f[R_LSE]),
        "chosen_lp": float(f[R_CHOSEN_LP]),
        "forced_lp": float(f[R_FORCED_LP]),
        "done": int(r[R_DONE]),
        "top_ids": [int(x) for x in r[R_TOPI : R_TOPI + TOPN]],
        "top_lps": [float(x) for x in f[R_TOPV : R_TOPV + TOPN]],
    }


# ---------------------------------------------------------------- verify pass (mode `code` speculation)
# One persistent dispatch evaluates M = 2..VMAX rows: the fed token at position pos and M - 1 host drafts at
# pos + 1 .. pos + M - 1. Row m's arithmetic is the M = 1 pass's at position pos + m (same lane blocks, simd_sums,
# roundings, per-position attention partition, head deal and select), so every row's result is bit-identical to
# that position decoded alone. Rendered per M (compile-time rows); raw / forced requests never use it.
VMAX = 8  # <= RING_SLACK + 1: a row never evicts a live window entry of another row
VXS = 10240  # bf16 elements of x staged per threadgroup (20 KB + 1.25 KB of block sums; 40 x 32 KB co-resident)
VROW_WORDS = 64  # result words per row (token .. top-20 values), rows at m * 64
VR_ARRIVE, VR_ERR, VR_ATTQ = 512, 576, 608
VR_QFLAG, VR_FCNT = 640, 704
VRES_WORDS = 768
VSCRATCH_BYTES = VMAX * SCRATCH_BYTES + CTR_BYTES
P_ROWS, P_RTOK, P_RU, P_RCS = 152, 160, 168, 176
VPRM_WORDS = P_RCS + VMAX * 128  # row m: cos at P_RCS + 128 m, sin at P_RCS + 128 m + 64


def vgroups(M: int, per: int) -> int:
    """Rows per staging group: the fewest groups of <= per rows, balanced (8 rows at 6 per group -> 4 + 4)."""
    ng = -(-M // per)
    return -(-M // ng)


def make_vparams(
    *,
    pos: int,
    tokens: list,
    sel_mode: int,
    top_k: int,
    top_p: float,
    temperature: float,
    us: list,
    poison_rows: int,
    cap: int,
    theta: float,
    mask: int = MASK_ALL,
    spin: int = SPIN_DEFAULT,
    s_f: int,
    s_s: int,
    p_f: int,
    p_s: int,
    f_sub: int = 1,
    f_asg: int = 8,
    f_flags: int = 0,
    six_t: int = 0,
    stall: bool = False,
    probe: int = 0,
) -> np.ndarray:
    """uint32 [VPRM_WORDS] for one verify pass of M = len(tokens) rows at positions pos .. pos + M - 1."""
    M = len(tokens)
    assert 2 <= M <= VMAX and len(us) == M and p_s > 0 and (p_f > 0 or six_t > 0), (
        M,
        p_f,
        p_s,
        six_t,
    )
    p = np.zeros(VPRM_WORDS, dtype=np.uint32)
    p[P_POS] = pos
    p[P_FORCED] = NO_TOKEN
    p[P_SELMODE] = sel_mode
    p[P_TOPK] = max(1, min(TOPN, int(top_k)))
    p[P_TOPP] = f32_bits(top_p)
    p[P_INVT] = f32_bits(1.0 / temperature if temperature > 0 else 1.0)
    p[P_POISON] = poison_rows
    p[P_MASK] = mask
    p[P_SPIN] = spin
    p[P_SF], p[P_SS], p[P_CAP], p[P_PF], p[P_PS] = s_f, s_s, cap, p_f, p_s
    p[P_FSUB], p[P_FASG], p[P_FFLAGS] = f_sub, f_asg, f_flags
    p[P_6T] = six_t
    p[P_STALL] = 1 if stall else 0
    p[P_PROBE] = probe
    p[P_ROWS] = M
    p[P_RTOK : P_RTOK + M] = np.asarray(tokens, dtype=np.uint32)
    p[P_RU : P_RU + M] = np.asarray(us, dtype=np.float32).view(np.uint32)
    i = np.arange(64, dtype=np.float64)
    ang = (pos + np.arange(M, dtype=np.float64))[:, None] * (theta ** (-i / 64))[None, :]
    cs = np.empty((M, 128), dtype=np.float32)
    cs[:, :64], cs[:, 64:] = np.cos(ang), np.sin(ang)
    p[P_RCS : P_RCS + 128 * M] = cs.reshape(-1).view(np.uint32)
    return p


def parse_vresult(r: np.ndarray, M: int) -> list:
    """Per-row results of a verify pass (the barrier error word is shared by every row)."""
    r = np.asarray(r, dtype=np.uint32)
    out = []
    for m in range(M):
        w = np.zeros(RES_WORDS, dtype=np.uint32)
        w[:VROW_WORDS] = r[m * VROW_WORDS : (m + 1) * VROW_WORDS]
        w[R_ERR] = r[VR_ERR]
        out.append(parse_result(w))
    return out
