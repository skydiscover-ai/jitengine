"""pack16: one exact 16-bit word per (scale, bias) group of a 4-bit affine expert record.

MLX's affine quantizer leaves every group's bias at `q * scale` (q in [-15, -7]) before bf16
rounding, so a group's bf16 pair is recovered from one word (one group of the checkpoint's 3.8 billion
is not, and the escape table below covers it):

    bits 15..5  scale: sign (1) | exponent - EXP_BASE (3) | mantissa (7)
    bits  4..0  c = (q + 15) * 3 + (res + 1), c in [0, 26]      bias bits = bf16_rne(q * scale) + res
    c == ESC    escape: (scale, bias) = exc[bits 15..5], the raw bf16 bits as scale | bias << 16

Decoding returns the ORIGINAL bf16 bits as floats, so every float operation after the per-group
load sees the inputs the bf16 kernels see; the grouped kernels' p16 forms
(`grouped_expert_qmv.grouped_qmv_p16`, `grouped_qmv_x_p16`) differ from theirs only in that load.

The escape table is global (one resident u32 array) and content-addressed: the encoder looks an
unrepresentable pair up by value, so a record may be re-encoded any number of times. A pair that
neither fits a code nor is in the table raises the encoder's error count; the pool fails on it
(`OffloadMoeCache._check_encodes`), it never stores a lossy word.

`encode_kernel` copies one whole staged record (the pack's nine-field layout) into a compact pool
slot: the three weight fields verbatim and one word per group. It writes the pool arrays it takes as
inputs, which MLX does not track, so its only declared output is the error count, and every reader
of what it wrote must take that output (or a join of it) as an input: that is what makes MLX put a
barrier, or a fence across command encoders, between the write and the read.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

EXP_BASE = 116
ESC = 31
ECAP = 2048  # 11 index bits; MLX passes a 1-element input as a `constant` scalar, so it is padded
PROJ = ("gate_proj", "up_proj", "down_proj")

# ----------------------------------------------------------------------------- the Metal decode
P16_DECODE = f"""
// pack16 word -> the ORIGINAL bf16 bits of (scale, bias), as exact floats (global escape table).
inline void p16_decode(const ushort w, const device uint* exc, thread float& s, thread float& b) {{
  const uint c = uint(w) & 31u;
  const uint hi = uint(w) >> 5;
  if (c == {ESC}u) {{
    const uint e = exc[hi];
    s = as_type<float>((e & 0xFFFFu) << 16);
    b = as_type<float>((e >> 16) << 16);
    return;
  }}
  const uint sbits = (((hi >> 10) & 1u) << 15) | ((((hi >> 7) & 7u) + {EXP_BASE}u) << 7) | (hi & 0x7Fu);
  const float sf = as_type<float>(sbits << 16);
  const uint qi = c / 3u;
  const uint ri = c - qi * 3u;
  const float p = (float(int(qi)) - 15.0f) * sf;
  const uint u = as_type<uint>(p);
  const uint pb = (u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16;
  const uint bb = (pb + ri - 1u) & 0xFFFFu;
  s = sf;
  b = as_type<float>(bb << 16);
}}
"""


# ----------------------------------------------------------------------------- host reference
def bf16_f32(u: np.ndarray) -> np.ndarray:
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def rne_bits(x) -> np.ndarray:
    """float32 -> bf16 bits, round to nearest even (the Metal and MLX conversion for finite x)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


def encode_np(s: np.ndarray, b: np.ndarray, table: list) -> tuple[np.ndarray, np.ndarray]:
    """(words, unfit) for bf16 bit arrays `s`, `b` (u16, one shape), the encoder's rule on the host.

    Codes are searched exhaustively, q from -15 up, the first fit wins (as the kernel does). A pair
    no code fits is looked up in `table` (u32 values, scale | bias << 16) and, if absent, appended to
    it; `unfit` marks those pairs. Raises past `ECAP` table entries.
    """
    shp = s.shape
    s = np.ascontiguousarray(s, dtype=np.uint16).ravel()
    b = np.ascontiguousarray(b, dtype=np.uint16).ravel()
    sf = bf16_f32(s)
    exp = ((s >> 7) & 0xFF).astype(np.int32)
    ok_exp = (exp >= EXP_BASE) & (exp <= EXP_BASE + 7)
    code = np.full(s.shape, -1, np.int32)
    bi = b.astype(np.int32)
    for q in range(-15, -6):
        r = bi - rne_bits(np.float32(q) * sf).astype(np.int32)
        hit = ok_exp & (code < 0) & (np.abs(r) <= 1)
        code[hit] = (q + 15) * 3 + (r[hit] + 1)
    fit = code >= 0
    hi = (
        ((s >> 15).astype(np.uint32) << 10)
        | ((exp - EXP_BASE).clip(0, 7).astype(np.uint32) << 7)
        | (s & 0x7F).astype(np.uint32)
    )
    w = np.where(fit, (hi << 5) | code.clip(0).astype(np.uint32), 0).astype(np.uint32)
    unfit = np.flatnonzero(~fit)
    if unfit.size:
        index = {int(v): i for i, v in enumerate(table)}
        for i in unfit.tolist():
            key = int(s[i]) | (int(b[i]) << 16)
            t = index.get(key)
            if t is None:
                if len(table) >= ECAP:
                    raise ValueError(
                        f"pack16: more than {ECAP} distinct escaped (scale, bias) pairs"
                    )
                t = index[key] = len(table)
                table.append(key)
            w[i] = (t << 5) | ESC
    return w.astype(np.uint16).reshape(shp), (~fit).reshape(shp)


def decode_np(words: np.ndarray, table) -> tuple[np.ndarray, np.ndarray]:
    """The kernel's decode on the host: words u16 -> (scale bits, bias bits) u16, one shape."""
    w = np.asarray(words).astype(np.uint32)
    c = w & 31
    hi = w >> 5
    s = (((hi >> 10) & 1) << 15) | ((((hi >> 7) & 7) + EXP_BASE) << 7) | (hi & 0x7F)
    qi = c // 3
    ri = c - qi * 3
    p = (qi.astype(np.int32).astype(np.float32) - 15.0) * bf16_f32(s)
    b = (rne_bits(p).astype(np.uint32) + ri - 1) & 0xFFFF
    esc = c == ESC
    if esc.any():
        e = np.asarray(table, dtype=np.uint32)[hi[esc]]
        s, b = s.copy(), b.copy()
        s[esc] = e & 0xFFFF
        b[esc] = e >> 16
    return s.astype(np.uint16), b.astype(np.uint16)


def exc_array(table) -> mx.array:
    """The global table as the kernels' u32 input, padded to its `ECAP` capacity."""
    if len(table) > ECAP:
        raise ValueError(f"pack16 escape table of {len(table)} entries exceeds {ECAP}")
    t = np.zeros(ECAP, np.uint32)
    t[: len(table)] = np.asarray(table, dtype=np.uint32)
    a = mx.array(t)
    mx.eval(a)
    return a


# ----------------------------------------------------------------------------- record geometry
class Geometry:
    """Where a record's fields sit (the pack's layout) and what a compact slot holds.

    `fields` are the pack header's nine field specs. Per projection p: groups G[p] (N x K/gs),
    weight u32 per group WPG (gs * bits / 32), and the u32 offsets of its weight, scales and biases
    in the record.
    """

    def __init__(self, fields: list, group_size: int, bits: int):
        by = {f["name"]: f for f in fields}
        self.group_size, self.bits = int(group_size), int(bits)
        if self.bits != 4 or (self.group_size * self.bits) % 32:
            raise ValueError(
                f"pack16 needs 4-bit groups of whole u32 words, not {bits}-bit/{group_size}"
            )
        self.wpg = self.group_size * self.bits // 32
        self.record_bytes = sum(int(f["bytes"]) for f in fields)
        if self.record_bytes % 4:
            raise ValueError("pack16 needs records of whole u32 words")
        self.groups, self.wshape, self.gshape, self.offsets = [], [], [], []
        for p in PROJ:
            w, s, b = by[f"{p}.weight"], by[f"{p}.scales"], by[f"{p}.biases"]
            if w["dtype"] != "U32" or s["dtype"] != "BF16" or b["dtype"] != "BF16":
                raise ValueError(f"pack16 needs U32 weights and BF16 scales/biases ({p})")
            if list(s["shape"]) != list(b["shape"]) or len(s["shape"]) != 2:
                raise ValueError(f"pack16: {p} scales/biases shapes differ")
            n, kg = (int(d) for d in s["shape"])
            if [int(d) for d in w["shape"]] != [n, kg * self.wpg]:
                raise ValueError(
                    f"pack16: {p} weight shape {w['shape']} is not [N, K/gs * {self.wpg}]"
                )
            for f in (w, s, b):
                if int(f["offset"]) % 4:
                    raise ValueError(f"pack16 needs u32-aligned fields ({f['name']})")
            self.groups.append(n * kg)
            self.wshape.append((n, kg * self.wpg))
            self.gshape.append((n, kg))
            self.offsets.append(
                (int(w["offset"]) // 4, int(s["offset"]) // 4, int(b["offset"]) // 4)
            )
        self.total_groups = sum(self.groups)
        self.slot_bytes = sum(4 * g * self.wpg + 2 * g for g in self.groups)

    def compact_fields(self) -> list[dict]:
        """The compact slot's six arrays, in the pool's field-spec form (offset = order only)."""
        out, off = [], 0
        for p, (n, kw), (_, kg) in zip(PROJ, self.wshape, self.gshape, strict=True):
            for name, dt, shape, nb in (
                (f"{p}.weight", "U32", [n, kw], 4 * n * kw),
                (f"{p}.words", "U16", [n, kg], 2 * n * kg),
            ):
                out.append({"name": name, "dtype": dt, "shape": shape, "bytes": nb, "offset": off})
                off += nb
        return out

    def template(self) -> list:
        g, o = self.groups, self.offsets
        return [
            ("G0", g[0]),
            ("G1", g[1]),
            ("G2", g[2]),
            ("WPG", self.wpg),
            ("RW", self.record_bytes // 4),
            *[(f"O{t}{p}", o[p][i]) for p in range(3) for i, t in enumerate("WSB")],
        ]


# ----------------------------------------------------------------------------- the encode kernel
# An error count of `STALE_UNIT` or more names encodes that ran before their read was published.
STALE_UNIT = 1 << 24


def stale_and_lossy(count: int) -> tuple[int, int]:
    """(stale encodes, lossy groups) of an encode error count."""
    return int(count) >> 24, int(count) & (STALE_UNIT - 1)


_ENCODE_SOURCE = f"""
    const uint i = thread_position_in_grid.x;            // group index within the record
    const uint job = thread_position_in_grid.y;
    if (i >= uint(G0 + G1 + G2)) return;
    const uint st = jobs[JW * job];
    const int slot_raw = int(jobs[JW * job + 1]);
    if (STAMPED && stamp[st] != jobs[JW * job + 2]) {{
      // the staging record does not hold this job's read (an encode evaluated before its read was
      // published): write nothing, and count it so the next check raises
      if (i == 0) atomic_fetch_add_explicit(&err[0], {STALE_UNIT}u, memory_order_relaxed);
      return;
    }}
    const bool in_ann = slot_raw >= MAIN2;
    const bool in_ext = !in_ann && slot_raw >= MAIN;
    const ulong slot = in_ann ? (ulong)(slot_raw - MAIN2)
                     : in_ext ? (ulong)(slot_raw - MAIN) : (ulong)slot_raw;
    uint p, j, gp, ow, os, ob;
    if (i < uint(G0)) {{ p = 0; j = i; gp = G0; ow = OW0; os = OS0; ob = OB0; }}
    else if (i < uint(G0 + G1)) {{ p = 1; j = i - G0; gp = G1; ow = OW1; os = OS1; ob = OB1; }}
    else {{ p = 2; j = i - G0 - G1; gp = G2; ow = OW2; os = OS2; ob = OB2; }}
    const device uint* rec = stg + (ulong)st * (ulong)RW;
    device uint* wdst;
    device ushort* sdst;
    if (p == 0) {{
      wdst = const_cast<device uint*>(in_ann ? wga : in_ext ? wgx : wg);
      sdst = const_cast<device ushort*>(in_ann ? sga : in_ext ? sgx : sg);
    }} else if (p == 1) {{
      wdst = const_cast<device uint*>(in_ann ? wua : in_ext ? wux : wu);
      sdst = const_cast<device ushort*>(in_ann ? sua : in_ext ? sux : su);
    }} else {{
      wdst = const_cast<device uint*>(in_ann ? wda : in_ext ? wdx : wd);
      sdst = const_cast<device ushort*>(in_ann ? sda : in_ext ? sdx : sd);
    }}
    const ulong wbase = slot * (ulong)gp * WPG + (ulong)j * WPG;
    const device uint* wsrc = rec + ow + (ulong)j * WPG;
    #pragma unroll
    for (int q = 0; q < WPG; q++) wdst[wbase + q] = wsrc[q];
    const device ushort* r16 = (const device ushort*)rec;
    const uint s = uint(r16[2 * (ulong)os + j]);
    const uint b = uint(r16[2 * (ulong)ob + j]);
    const uint e = (s >> 7) & 0xFFu;
    bool done = false;
    uint word = 0;
    if (e >= {EXP_BASE}u && e <= {EXP_BASE + 7}u) {{
      const float sf = as_type<float>(s << 16);
      for (int q = -15; q <= -7 && !done; q++) {{
        const uint u = as_type<uint>(float(q) * sf);
        const uint pb = ((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16) & 0xFFFFu;
        const int r = int(b) - int(pb);
        if (r >= -1 && r <= 1) {{
          const uint hi = ((s >> 15) << 10) | ((e - {EXP_BASE}u) << 7) | (s & 0x7Fu);
          word = (hi << 5) | uint((q + 15) * 3 + (r + 1));
          done = true;
        }}
      }}
    }}
    if (!done) {{
      const uint key = s | (b << 16);
      for (uint t = 0; t < uint(NEXC); t++) {{
        if (exc[t] == key) {{ word = (t << 5) | {ESC}u; done = true; break; }}
      }}
    }}
    if (!done) atomic_fetch_add_explicit(&err[0], 1u, memory_order_relaxed);
    sdst[slot * (ulong)gp + j] = ushort(word);
"""


@functools.cache
def encode_kernel():
    """The compiled `moe_engine_pack16_encode` kernel, built once. With STAMPED, each job carries the
    sequence number of the read it encodes and the kernel checks it against `stamp[staging]`, which
    the pool writes only once that read is published (`codec.py`)."""
    return mx.fast.metal_kernel(
        name="moe_engine_pack16_encode",
        input_names=[
            "stg",
            "wg",
            "sg",
            "wu",
            "su",
            "wd",
            "sd",
            "wgx",
            "sgx",
            "wux",
            "sux",
            "wdx",
            "sdx",
            "wga",
            "sga",
            "wua",
            "sua",
            "wda",
            "sda",
            "exc",
            "jobs",
            "dep",
            "stamp",
        ],
        output_names=["err"],
        header="#include <metal_atomic>\nusing namespace metal;\n",
        source=_ENCODE_SOURCE,
        ensure_row_contiguous=False,  # a copy of a pool array would take the writes
        atomic_outputs=True,
    )


ENCODE_TG = 256


def encode_records(
    geom: Geometry,
    staging,
    main: list,
    ext: list | None,
    main_slots: int,
    exc,
    n_exc: int,
    jobs: np.ndarray,
    dep,
    stamp=None,
    ann: list | None = None,
    ann_slots: int = 1 << 30,
) -> mx.array:
    """Encode staged records into compact slots; returns the error count, u32 [1] (lazy).

    `main` / `ext` are the six compact arrays (gate w, words, up w, words, down w, words) of each
    set, `jobs` (stage index, destination slot) pairs, a slot >= `main_slots` addressing `ext` and,
    with `ann` (the annex set's six arrays), a slot >= `ann_slots` addressing `ann`.
    `dep` orders this write after every earlier one (a join of their error counts). With `stamp`
    (u32 [staging records]), `jobs` rows are (stage index, slot, read sequence number) and a job
    whose staging record is not stamped with its number writes nothing and adds `STALE_UNIT` to the
    count (`stale_and_lossy` splits it).
    """
    n = int(jobs.shape[0])
    sets = main + (ext if ext else main)
    sets = sets + (ann if ann else sets[6:])
    stamped = stamp is not None
    jw = 3 if stamped else 2
    if int(jobs.shape[1]) != jw:
        raise ValueError(
            f"encode jobs need {jw} columns {'with' if stamped else 'without'} a stamp"
        )
    return encode_kernel()(
        inputs=[
            staging,
            *sets,
            exc,
            mx.array(np.ascontiguousarray(jobs, dtype=np.uint32).ravel()),
            dep,
            stamp if stamped else dep,
        ],
        template=[
            *geom.template(),
            ("MAIN", int(main_slots) if ext else (1 << 30)),
            ("MAIN2", int(ann_slots) if ann else (1 << 30)),
            ("NEXC", int(n_exc)),
            ("JW", jw),
            ("STAMPED", bool(stamped)),
        ],
        grid=(-(-geom.total_groups // ENCODE_TG) * ENCODE_TG, n, 1),
        threadgroup=(ENCODE_TG, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )[0]


# ----------------------------------------------------------------------------- the decode kernel
@functools.cache
def decode_kernel():
    """`p16_decode` as a kernel (tests and the codec proof): words -> (scale bits, bias bits)."""
    return mx.fast.metal_kernel(
        name="moe_engine_pack16_decode",
        input_names=["words", "exc"],
        output_names=["sb"],
        header="#include <metal_stdlib>\nusing namespace metal;\n" + P16_DECODE,
        source="""
    const uint i = thread_position_in_grid.x;
    if (i >= uint(words_shape[0])) return;
    float s, b;
    p16_decode(words[i], exc, s, b);
    sb[2 * i] = ushort(as_type<uint>(s) >> 16);
    sb[2 * i + 1] = ushort(as_type<uint>(b) >> 16);
""",
    )


def decode_words(words: mx.array, exc) -> mx.array:
    """(scale bits, bias bits) u16 [n, 2] of flat words u16 [n], through the Metal decode."""
    n = int(words.size)
    return decode_kernel()(
        inputs=[words.reshape(-1), exc],
        grid=(-(-n // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, 2)],
        output_dtypes=[mx.uint16],
    )[0]
