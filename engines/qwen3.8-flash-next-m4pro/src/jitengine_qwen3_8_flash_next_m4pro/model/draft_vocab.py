"""The MTP draft's readout over a vocabulary subset (option `draft_vocab`, `subset` by default).

`MTPHead.draft` ends in the argmax of lm_head over all 248,320 rows: 1.55 ms at M = 1, on the verify
pass's critical path (the next pass needs the draft). With `subset` that argmax runs over S, the K ids
of `draft_vocab_ids.npy` UNION the request's own prompt ids (at most `P_MAX` beyond the base):
- the base is a QLinear copy of lm_head's K rows, made once per head load and held with the head;
- the tail is a `P_MAX`-row QLinear gathered from lm_head per request, its unused rows masked to -inf.
  It is a separate launch, so installing it never copies the base.

The id file ships with the package and is never re-ranked here: the top-K token ids by count over a
neutral corpus (DETAILS.md has the rule and its manifest). No benchmark text is involved.

Exactness. A draft only decides whether a pass verifies one position or two. The verify pass draws its
second row only when the first row's token equals the draft, and that row is computed on the true
token (`scheduler.decode.DecodingMixin._mtp_round`), so a subset changes acceptance, never an emitted
token or a logged row. Its draft is the full draft whenever that lies in S: the copies hold lm_head's
row bytes and run on lm_head's kernel and config (rows are independent, in a k-order set by KL alone),
and ties go to the lowest id, as `mx.argmax` breaks them over the full row.

`full` is the head's own readout, `mx.argmax(lm_head(mixed))`: no copy, no charge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from jitengine_qwen3_8_flash_next_m4pro.linear import QLinear

MODES = ("subset", "full")
MODE_DEFAULT = "subset"
# Prompt ids beyond the base one request may add; the rest are dropped and counted on /stats.
P_MAX = 1024
IDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab_ids.npy")


def mode_of(options) -> str:
    """The `draft_vocab` option, checked: "subset" (the default) or "full"."""
    mode = str((options or {}).get("draft_vocab") or MODE_DEFAULT).strip().lower()
    if mode not in MODES:
        raise ValueError(f"draft_vocab {mode!r} not in {MODES}")
    return mode


def load_ids(n_rows: int, path: str = IDS_FILE) -> np.ndarray:
    """The base ids, int32 and ascending; refused unless 1-D, integral, unique and in [0, n_rows)."""
    ids = np.load(path, allow_pickle=False)
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError(f"{path}: want a non-empty 1-D integer array, got {ids.dtype} {ids.shape}")
    ids = np.sort(ids.astype(np.int64))
    if ids[0] < 0 or ids[-1] >= int(n_rows):
        raise ValueError(f"{path}: ids outside [0, {n_rows})")
    if (np.diff(ids) == 0).any():
        raise ValueError(f"{path}: repeated ids")
    return ids.astype(np.int32)


def readout_row_bytes(scan, cfg, compute_dtype: str, dense_kernel: str) -> int:
    """One lm_head row as the engine holds it, from the shard headers: weight, scales and biases, with
    16-bit metadata at 4 bytes where `linear.apply_dense_policy` stores it as float32 (float32
    activations on a shape the row-blocked kernel does not serve; never the tied embedding), as
    `budget.dense_fp32_meta_bytes` prices it."""
    from jitengine_qwen3_8_flash_next_m4pro.kernels import qmv_row_blocked as _dk

    prefix = scan.prefix or ""
    tied = (prefix + "lm_head.weight") not in scan.weight_map
    base = "model.embed_tokens" if tied else "lm_head"
    total = meta16 = 0
    rows = cols = None
    for part in ("weight", "scales", "biases"):
        name = f"{prefix}{base}.{part}"
        shard = scan.weight_map.get(name)
        if shard is None:
            continue
        hdr, _off = scan.headers[shard]
        meta = hdr[name]
        lo, hi = meta["data_offsets"]
        n = int(meta["shape"][0])
        if rows is not None and n != rows:
            raise ValueError(f"{name}: {n} rows, the weight has {rows}")
        rows = n
        total += int(hi) - int(lo)
        if part == "scales":
            cols = int(meta["shape"][1])
        if part != "weight" and meta["dtype"] in ("BF16", "F16"):
            meta16 += int(hi) - int(lo)
    if not rows or cols is None:
        raise ValueError(f"no {prefix}{base} weight and scales in the checkpoint headers")
    per_row = total // rows
    if not tied and str(compute_dtype) == "float32":
        bits, gs = cfg.quant_of(base)
        served = (
            str(dense_kernel) == "rb"
            and int(bits) == 4
            and _dk.config_for(rows, cols * int(gs), "float32") is not None
        )
        if not served:
            per_row += meta16 // rows  # 2 bytes -> 4 bytes
    return per_row


def buffer_bytes(k: int, row_bytes: int, n_rows: int, p_max: int = P_MAX) -> int:
    """What a subset readout holds: K + p_max rows, the int32 id map and the float32 tail mask in
    MLX, and the host's membership table (one byte per lm_head row)."""
    rows = int(k) + int(p_max)
    return rows * int(row_bytes) + rows * 4 + int(p_max) * 4 + int(n_rows)


@dataclass(frozen=True)
class DraftVocabPlan:
    """The subset readout an engine holds with its head: the base ids and its priced bytes."""

    ids: np.ndarray
    row_bytes: int
    bytes: int

    @property
    def rows(self) -> int:
        return int(self.ids.size) + P_MAX


def plan_for(eng) -> DraftVocabPlan | None:
    """The subset readout `eng`'s head would hold, priced from the checkpoint headers; None for
    `full` or without a checkpoint scan to price it (the head then reads out in full)."""
    if mode_of(getattr(eng, "options", None)) != "subset":
        return None
    scan, cfg = getattr(eng, "ckpt_scan", None), getattr(eng, "cfg", None)
    if scan is None or cfg is None:
        return None
    n_rows = int(cfg.vocab_size)
    ids = load_ids(n_rows)
    rb = readout_row_bytes(scan, cfg, str(eng.compute_dtype_name), str(eng.dense_kernel))
    return DraftVocabPlan(ids, rb, buffer_bytes(ids.size, rb, n_rows))


def rows_of(lin: QLinear, ids: mx.array) -> QLinear:
    """A QLinear over copies of rows `ids` of `lin`, on `lin`'s own kernel and config."""
    q = QLinear(
        mx.take(lin.weight, ids, axis=0),
        mx.take(lin.scales, ids, axis=0),
        None if lin.biases is None else mx.take(lin.biases, ids, axis=0),
        lin.group_size,
        lin.bits,
    )
    q.kernel_cfg, q.kernel_dtype = lin.kernel_cfg, lin.kernel_dtype
    return q


def _arrays(q: QLinear) -> list:
    return [a for a in (q.weight, q.scales, q.biases) if a is not None]


class SubsetReadout:
    """lm_head's argmax over the base ids and one request's prompt ids (see the module docstring)."""

    def __init__(self, lm_head: QLinear, base_ids, p_max: int = P_MAX):
        self.lm_head = lm_head
        self.n_rows = int(lm_head.out_dims)
        self.p_max = int(p_max)
        base = np.asarray(base_ids, dtype=np.int64)
        if (
            base.ndim != 1
            or base.size == 0
            or base[0] < 0
            or base[-1] >= self.n_rows
            or (np.diff(base) <= 0).any()
        ):
            raise ValueError("base ids must be ascending, unique and inside lm_head's rows")
        self.base_ids = base.astype(np.int32)
        self.in_base = np.zeros(self.n_rows, dtype=bool)
        self.in_base[self.base_ids] = True
        self._base_ids = mx.array(self.base_ids)
        self.base = rows_of(lm_head, self._base_ids)
        mx.eval(*_arrays(self.base), self._base_ids)
        self.prompt_rows = 0
        self.prepare([])

    @property
    def rows(self) -> int:
        """Rows each draft reads: the base and the whole tail."""
        return int(self.base_ids.size) + self.p_max

    @property
    def nbytes(self) -> int:
        return buffer_bytes(
            self.base_ids.size, self.lm_head.nbytes // self.n_rows, self.n_rows, self.p_max
        )

    def prompt_ids(self, prompt) -> tuple[np.ndarray, bool]:
        """The prompt's ids outside the base: unique, in first-occurrence order, at most `p_max`;
        and whether any were dropped. Ids outside lm_head's rows are never drafted."""
        p = np.asarray(prompt, dtype=np.int64).reshape(-1)
        p = p[(p >= 0) & (p < self.n_rows)]
        p = p[~self.in_base[p]]
        uniq, first = np.unique(p, return_index=True)
        new = uniq[np.argsort(first, kind="stable")]
        return new[: self.p_max].astype(np.int32), bool(new.size > self.p_max)

    def prepare(self, prompt) -> tuple[int, bool]:
        """Install one request's prompt ids into the tail, every other tail row masked; returns the
        rows installed and whether some were dropped. The tail is gathered from lm_head afresh
        (`p_max` rows), so nothing of an earlier request survives and the base is never copied."""
        new, cut = self.prompt_ids(prompt)
        n = int(new.size)
        ids = np.full(self.p_max, self.base_ids[0], dtype=np.int32)  # masked rows: any real row
        ids[:n] = new
        mask = np.full(self.p_max, -np.inf, dtype=np.float32)
        mask[:n] = 0.0
        tail_ids = mx.array(ids)
        self.tail = rows_of(self.lm_head, tail_ids)
        self.mask = mx.array(mask)
        self.id_map = mx.concatenate([self._base_ids, tail_ids])
        self.tail_ids, self.prompt_rows = ids, n
        return n, cut

    def __call__(self, mixed: mx.array) -> mx.array:
        """The draft (lazy int32 [1]): the lowest id among S's maxima."""
        x = mixed.reshape(-1, self.lm_head.in_dims)
        y = mx.concatenate([self.base(x).reshape(-1), self.tail(x).reshape(-1) + self.mask])
        # a row with no maximum (NaN) still drafts a real row: the last
        best = mx.where(y == mx.max(y), self.id_map, self.n_rows - 1)
        return mx.min(best).reshape(1)


def draft_id(lm_head, readout, mixed: mx.array) -> mx.array:
    """The draft read out of `mixed`: over the subset with a `SubsetReadout`, else over every row."""
    if readout is not None:
        return readout(mixed)
    return mx.argmax(lm_head(mixed).reshape(-1)).reshape(1)
