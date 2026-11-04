"""The verify pass's MoE block per row count: the snake DOWN deal at every M, one union sweep (S1s) from S1S_MIN_ROWS.

metal_src.HEADER_V (megakernel_verify_layers.metal) is the single source of every verify render. It holds ONE verify
MoE block (vmoe_fused over M rows), and the verify render of M rows is HEADER_V with the anchored textual edits below
(verify_header). They are applied to HEADER_V's current text at render time, as kv8_src's are, and cached per text, so
any edit of HEADER_V reaches every verify render (tests/unit/kernels/test_vmoe_src.py). metal_src also builds them once
at import, so an anchor that does not occur exactly once fails the import. M = 1 renders are untouched, and so is the
chain reference's plain-dispatch form of the block (the #else branch); the kv8 profile applies its own edits on top of
the text built here.

  snake  every verify render, M = 2 .. VMAX. The persistent block deals gate/up piece p and DOWN piece p over the G
         threadgroups. When both deals put piece p on threadgroup p mod G, the threadgroups that take one of the last
         gate/up wave's pieces also take one of the last DOWN wave's (and the low ones also run the shared team).
         DOWN piece p runs on threadgroup G - 1 - (p mod G) instead.
  S1s    verify renders with M >= S1S_MIN_ROWS, the row counts whose x needs two staging groups at the production xs
         (plan.VXS). One union sweep from router-published x (vmoe_s1s.metal): the router threadgroups publish each
         row's vprep values (load_x16's prescaled x as bf16 + block sums) into the row's LOGITS region, which is dead
         from the pass's start until the head rewrites every logit after the last MoE block, behind a grid barrier. A
         routed gate/up piece copies only its slot's routed rows (<= XR = XSV / 2048 per sub-batch, one row at a time,
         so one uint4 and one float are live per thread) and walks the union once. The shared team keeps its
         per-group staging (it starts before routing). vmoe_fused becomes a call of vmoe_s1, whose DOWN deal is the
         snake.

Exact by construction in every production pass (the phase mask is MASK_ALL): every row's router scores, top-6, union
order, gate/up and DOWN unit arithmetic, stores and combine order are the base block's. Only which threadgroup stages,
copies and computes which unit changes, and a unit's arithmetic does not depend on the threadgroup that runs it.
Not under every phase mask: with gate/up on and the router off (timing arms only: benchmarks/kernels/megakernel.py
--verify-phases, only_EGU), S1s's routed gate/up reads x that no router threadgroup published in that pass, so at
M >= S1S_MIN_ROWS such an arm's timing stands but its values do not. The Metal comments in vmoe_s1s.metal that say
"the base block's phase-mask semantics" are rendered text and stay as they are.

Measured on the M4 Pro against the verify pass without them: the snake -0.25 / -1.00 / -2.35 ms per pass at
M = 2 / 4 / 8, S1s -1.19 ms at M = 8; both together, code decode x1.015 (geomean), with identical streams.

The selection is keyed on M alone (not on the xs a render is given), and the rendered text, comments included, is the
measured build's byte for byte (tests/unit/kernels/test_render_measured.py): the runner names each kernel by a hash of
its text. The two note lines are the measured build's wording (the snake note says "M = 4" although the snake is on at
every verify M); rewording them changes every verify render's bytes, so do it only together with new pins.
"""

from __future__ import annotations

import functools
from importlib import resources

from jitengine_flint_m4pro.kernels import plan as P

# ------------------------------------------------------------------------------------------------ anchors in HEADER_V
NOTE = "// scan, and every verify row is bit-identical to it). M = 1 untouched.\n"
FUSED_START = "// the persistent verify MoE block of sparse layer l (moe_fused over M rows; same phase-mask semantics)\n"
FUSED_HEAD = (
    "inline bool vmoe_fused(thread Ctx& c, thread VX& vx, uint l) {\n"
    "  const uint msk = c.prm[P_MASK];\n"
    "  const bool dr = ((msk >> PH_RSG) & 1u) != 0u, dg = ((msk >> PH_EGU) & 1u) != 0u, dd = ((msk >> PH_DOWN) & 1u) != 0u;\n"
)
FUSED_END = "#else\n// chain mode: the verify block's pieces as plain dispatches"
DOWN_DEAL = "    for (uint p = c.w; p < np; p += c.G) {\n      const bool sh = p < MQS;\n"

# ------------------------------------------------------------------------------------------------ snake
SNAKE_NOTE = "// PS: M = 4 piece deal, snake on, split off.\n"
SNAKE_DEAL = (
    "    for (uint p = c.G - 1u - c.w; p < np; p += c.G) {                "
    "// PS: DOWN pieces in reverse threadgroup order\n      const bool sh = p < MQS;\n"
)

# ------------------------------------------------------------------------------------------------ S1s
S1S_MIN_ROWS = 6
S1S_NOTE = "// S1s: one union sweep from router-published x, this render (two staging groups).\n"
# read as metal_src.read_metal reads a kernel file (this module cannot import metal_src, which imports it)
S1S_BLOCK = resources.files(__package__).joinpath("vmoe_s1s.metal").read_bytes().decode("utf-8")
S1S_CALL = "  return vmoe_s1(c, vx, l, dr, dg, dd);\n}\n"

# ------------------------------------------------------------------------------------------------ anchored edits
# (anchor, replacement); an insertion keeps its anchor next to the inserted text
_SNAKE = (
    (NOTE, NOTE + SNAKE_NOTE),
    (DOWN_DEAL, SNAKE_DEAL),
)
_S1S = (
    (NOTE, NOTE + S1S_NOTE),
    (FUSED_START, S1S_BLOCK + FUSED_START),
)


def _once(src: str, anchor: str, what: str) -> int:
    """Index of `anchor` in src; AssertionError unless it occurs exactly once."""
    got = src.count(anchor)
    if got != 1:
        raise AssertionError(
            f"verify MoE source edit ({what}): anchor {anchor[:60]!r} found {got} times, expected 1"
        )
    return src.index(anchor)


def _edit(src: str, edits, what: str) -> str:
    for anchor, new in edits:
        _once(src, anchor, what)
        src = src.replace(anchor, new)
    return src


def _span(src: str, start: str, end: str, new: str, what: str) -> str:
    """Replace the text strictly between the unique anchors `start` and `end` (start before end) with `new`."""
    i, j = _once(src, start, what) + len(start), _once(src, end, what)
    if i > j:
        raise AssertionError(
            f"verify MoE source edit ({what}): anchor {end[:60]!r} precedes {start[:60]!r}"
        )
    return src[:i] + new + src[j:]


def snake(header_v: str) -> str:
    """HEADER_V -> its snake form: the persistent block's DOWN pieces dealt in reverse threadgroup order."""
    return _edit(header_v, _SNAKE, "snake")


def s1s(header_v: str) -> str:
    """The snake form of HEADER_V -> its S1s form: one union sweep from router-published x (vmoe_s1, which carries its
    own snake DOWN deal), and vmoe_fused calls it."""
    return _span(_edit(header_v, _S1S, "S1s"), FUSED_HEAD, FUSED_END, S1S_CALL, "S1s")


def _two_staging_groups(m: int) -> bool:
    """m rows of x need two staging groups at the production xs (plan.VXS)."""
    return P.vgroups(m, P.VXS // 2048) < m


if [m for m in range(2, P.VMAX + 1) if _two_staging_groups(m)] != list(
    range(S1S_MIN_ROWS, P.VMAX + 1)
):
    raise AssertionError(
        f"S1S_MIN_ROWS = {S1S_MIN_ROWS} is no longer the first verify M with two staging groups at "
        f"plan.VXS = {P.VXS}: measure S1s again before moving it"
    )


@functools.lru_cache(maxsize=4)
def variants(header_v: str) -> tuple:
    """(snake form, S1s form) of this HEADER_V text; AssertionError if an anchor does not occur exactly once."""
    sn = snake(header_v)
    return sn, s1s(sn)


def verify_header(header_v: str, m: int) -> str:
    """The header of the verify pass of m rows: header_v with the snake, and with S1s from S1S_MIN_ROWS rows."""
    sn, sv = variants(header_v)
    return sv if m >= S1S_MIN_ROWS else sn
