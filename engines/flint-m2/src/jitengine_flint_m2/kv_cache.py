"""Attention history: what decode reads and prefill writes.

Full-attention layers (14): separate K and V arrays per layer, [1, kv_heads=4, capacity, 128] bf16,
position p at index p. Each array stays below 2^31 elements at 262,144 positions, so no offset into one
of them can overflow a 32-bit signed index.

Sliding layers (35): a ring of RING = 513 + 15 slots per K and V, [1, 4, 528, 128] bf16, position p
at slot p % 528. The window is the current token plus the 512 before it; the 15 slack slots let a
future multi-position pass write ahead without overwriting a live window slot. A decode query at
position p reads every written slot whose (p - slot) mod 528 < 513.

Rollback is by length, never by array reference (MLX `a[i] = v` rebinds the same Python object, so a
reference "snapshot" would see later writes). The post-prefill snapshot copies the rings (37.8 MB)
and records the length; full-layer positions below that length are never rewritten by decode.

MKHistory8 is the kv8 VARIANT profile's layout: full-layer K / V as 8-bit g64 codes + bf16 scale / bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jitengine_flint_m2.kernels.mathmode import METAL_COMPILE_OPTIONS
from jitengine_flint_m2.model.config import FlintConfig

RING_SLACK = 15
CAPACITY_QUANTUM = 4096


def _copy(a):
    import mlx.core as mx

    return mx.multiply(
        a, mx.array(1, dtype=a.dtype)
    )  # a real copy (x * 1 keeps -0.0 and NaN payloads)


@dataclass
class Snapshot:
    prompt: tuple
    length: int
    rings_k: dict
    rings_v: dict
    logits: object  # [vocab] bf16 logits of the last prompt position


class History:
    def __init__(self, cfg: FlintConfig, context_limit: int):
        self.cfg = cfg
        self.limit = context_limit
        self.window = cfg.sliding_window
        self.ring = cfg.sliding_window + RING_SLACK
        self.capacity = 0
        self.k: dict = {}
        self.v: dict = {}
        self.rk: dict = {}
        self.rv: dict = {}
        self.length = 0
        self.valid = False

    # ------------------------------------------------------------------ allocation
    def reserve(self, need: int) -> None:
        """Make room for `need` positions (prompt + every position decode will write)."""
        import mlx.core as mx

        if need > self.limit:
            raise ValueError(f"{need} positions exceed the context limit {self.limit}")
        cfg = self.cfg
        shape_r = (1, cfg.n_kv_heads, self.ring, cfg.head_dim)
        if not self.rk:
            for i in cfg.sliding_layers:
                self.rk[i] = mx.zeros(shape_r, dtype=mx.bfloat16)
                self.rv[i] = mx.zeros(shape_r, dtype=mx.bfloat16)
            mx.eval(list(self.rk.values()) + list(self.rv.values()))
        if need <= self.capacity:
            return
        cap = min(self.limit, -(-need // CAPACITY_QUANTUM) * CAPACITY_QUANTUM)
        keep = self.length if self.valid else 0
        for i in cfg.full_layers:
            for store in (self.k, self.v):
                new = mx.zeros((1, cfg.n_kv_heads, cap, cfg.head_dim), dtype=mx.bfloat16)
                old = store.get(i)
                if old is not None and keep:
                    new[:, :, :keep, :] = old[:, :, :keep, :]
                store[i] = None
                del old
                mx.eval(new)
                store[i] = new
        self.capacity = cap
        # the replaced arrays would sit in MLX's buffer cache, which never reuses a buffer of another size
        # (+2.9 GB idle after a 131K -> 258K growth); release them now (once per new capacity)
        mx.clear_cache()

    def release(self) -> None:
        self.k.clear()
        self.v.clear()
        self.rk.clear()
        self.rv.clear()
        self.capacity = 0
        self.length = 0
        self.valid = False

    def invalidate(self) -> None:
        self.length = 0
        self.valid = False

    def state_arrays(self) -> list:
        return (
            list(self.rk.values())
            + list(self.rv.values())
            + list(self.k.values())
            + list(self.v.values())
        )

    # ------------------------------------------------------------------ writes
    def write_full(self, layer: int, start: int, k, v) -> None:
        n = k.shape[2]
        self.k[layer][:, :, start : start + n, :] = k
        self.v[layer][:, :, start : start + n, :] = v

    def write_ring(self, layer: int, start: int, k, v) -> None:
        """Write positions start .. start+n-1 (only the last `ring` of them land)."""
        n = k.shape[2]
        if n > self.ring:
            k, v = k[:, :, n - self.ring :, :], v[:, :, n - self.ring :, :]
            start += n - self.ring
            n = self.ring
        s0 = start % self.ring
        first = min(n, self.ring - s0)
        self.rk[layer][:, :, s0 : s0 + first, :] = k[:, :, :first, :]
        self.rv[layer][:, :, s0 : s0 + first, :] = v[:, :, :first, :]
        if first < n:
            self.rk[layer][:, :, : n - first, :] = k[:, :, first:, :]
            self.rv[layer][:, :, : n - first, :] = v[:, :, first:, :]

    # ------------------------------------------------------------------ reads
    def full_kv(self, layer: int, length: int):
        return self.k[layer][:, :, :length, :], self.v[layer][:, :, :length, :]

    def ring_mask(self, p: int):
        """(n_slots, mask) for a decode query at position p: mask is None when slots 0..p are exactly
        the window, else a bool [1, 1, 1, n_slots] selecting slots with (p - slot) mod ring < window."""
        import mlx.core as mx

        if p + 1 <= self.window:
            return p + 1, None
        n = min(p + 1, self.ring)
        s = np.arange(n)
        m = ((p - s) % self.ring) < self.window
        return n, mx.array(m.reshape(1, 1, 1, n))

    def ring_window_ordered(self, layer: int, start: int):
        """K, V of positions max(0, start-512) .. start-1 in position order (prefill context)."""
        import mlx.core as mx

        w = min(start, self.window - 1)
        if w == 0:
            return None, None, 0
        idx = mx.array([(start - w + j) % self.ring for j in range(w)], dtype=mx.int32)
        return (mx.take(self.rk[layer], idx, axis=2), mx.take(self.rv[layer], idx, axis=2), w)

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, prompt, logits) -> Snapshot:
        import mlx.core as mx

        rk = {i: _copy(a) for i, a in self.rk.items()}
        rv = {i: _copy(a) for i, a in self.rv.items()}
        mx.eval(list(rk.values()) + list(rv.values()), logits)
        return Snapshot(tuple(prompt), self.length, rk, rv, logits)

    def restore(self, snap: Snapshot) -> None:
        import mlx.core as mx

        for i in self.rk:
            self.rk[i] = _copy(snap.rings_k[i])
            self.rv[i] = _copy(snap.rings_v[i])
        mx.eval(list(self.rk.values()) + list(self.rv.values()))
        self.length = snap.length
        self.valid = True

    @property
    def nbytes(self) -> int:
        cfg = self.cfg
        full = len(cfg.full_layers) * 2 * cfg.n_kv_heads * self.capacity * cfg.head_dim * 2
        ring = len(self.rk) * 2 * cfg.n_kv_heads * self.ring * cfg.head_dim * 2
        return full + ring


# ====================================================================== consolidated history (megakernel)
_WRITER_SRC = r"""
  const uint t = thread_position_in_grid.x, i = thread_position_in_grid.y, h = thread_position_in_grid.z;
  const uint ls = prm[0], rows = prm[1], start = prm[2], n = prm[3], ring = prm[4];
  if (i >= n || t >= 64u) return;
  const size_t r = ring != 0u ? size_t(ls * 4u + h) * rows + (start + i) % rows
                              : size_t(ls * 4u + h) * rows + start + i;
  ((device uint*)dst)[r * 64u + t] = ((const device uint*)src)[(size_t(h) * n + i) * 64u + t];
  if (t == 0u && i == 0u && h == 0u) done[0] = 1u;
"""
_writer_kernel = None


def _writer():
    """In-place row writer: copies src [1, 4, n, 128] bf16 into a consolidated history array.

    MLX cannot express an in-place update of a 1.9 GB array through a view without copying it (a slice
    update on a view copies and misses the base array), so prefill writes go through this kernel: it takes
    the base array as an input and stores into it; the caller evaluates it before anything reads the rows."""
    global _writer_kernel
    if _writer_kernel is None:
        import mlx.core as mx

        _writer_kernel = mx.fast.metal_kernel(
            name="flint_hist_writer_v1",
            input_names=["dst", "src", "prm"],
            output_names=["done"],
            source=_WRITER_SRC,
            ensure_row_contiguous=True,
            compile_options=METAL_COMPILE_OPTIONS,
        )
    return _writer_kernel


class MKHistory:
    """History in the layout the decode megakernel binds (Metal's 31-entry argument table cannot hold 98
    per-layer arrays):

      full layers  K_lo, K_hi, V_lo, V_hi = [7][4][capacity][128] bf16 (full layer f lives in array f // 7 at
                   index f % 7; each array < 2^31 bytes at 262,144 positions), position p at row p;
      rings        K_ring, V_ring = [35][4][528][128] bf16, position p at slot p % 528.

    Prefill (the MLX path) reads per-layer slice views and writes through the in-place writer kernel, which
    is evaluated before the next read. Rollback is by length, as in History; the prefix-cache snapshot copies
    the two ring arrays (37.8 MB)."""

    def __init__(self, cfg: FlintConfig, context_limit: int):
        self.cfg = cfg
        self.limit = context_limit
        self.window = cfg.sliding_window
        self.ring = cfg.sliding_window + RING_SLACK
        self.capacity = 0
        self.full = [None, None, None, None]  # K_lo, K_hi, V_lo, V_hi
        self.kr = self.vr = None
        self.length = 0
        self.valid = False
        self._fidx = {layer: i for i, layer in enumerate(cfg.full_layers)}
        self._sidx = {layer: i for i, layer in enumerate(cfg.sliding_layers)}
        assert len(cfg.full_layers) == 14

    @property
    def arrays(self) -> list:
        return [self.full[0], self.full[1], self.full[2], self.full[3], self.kr, self.vr]

    def state_arrays(self) -> list:
        return [a for a in self.arrays if a is not None]

    # ------------------------------------------------------------------ allocation
    def reserve(self, need: int) -> None:
        import mlx.core as mx

        if need > self.limit:
            raise ValueError(f"{need} positions exceed the context limit {self.limit}")
        cfg = self.cfg
        if self.kr is None:
            shape_r = (len(cfg.sliding_layers), cfg.n_kv_heads, self.ring, cfg.head_dim)
            self.kr = mx.zeros(shape_r, dtype=mx.bfloat16)
            self.vr = mx.zeros(shape_r, dtype=mx.bfloat16)
            mx.eval(self.kr, self.vr)
        if need <= self.capacity:
            return
        # a multiple of 8 even when the context limit is not: the decode megakernel's layout-6d attention loads whole
        # 8-key blocks (keys past the live length are masked, so their rows only need to exist and be finite: zeros
        # or stale finite values)
        cap = min(-(-self.limit // 8) * 8, -(-need // CAPACITY_QUANTUM) * CAPACITY_QUANTUM)
        keep = self.length if self.valid else 0
        for a in range(4):
            new = mx.zeros((7, cfg.n_kv_heads, cap, cfg.head_dim), dtype=mx.bfloat16)
            old = self.full[a]
            if old is not None and keep:
                new[:, :, :keep, :] = old[:, :, :keep, :]
            self.full[a] = None
            del old
            mx.eval(new)
            self.full[a] = new
        self.capacity = cap
        # the replaced arrays would sit in MLX's buffer cache, which never reuses a buffer of another size
        # (+2.9 GB idle after a 131K -> 258K growth); release them now (once per new capacity)
        mx.clear_cache()

    def release(self) -> None:
        self.full = [None, None, None, None]
        self.kr = self.vr = None
        self.capacity = 0
        self.length = 0
        self.valid = False

    def invalidate(self) -> None:
        self.length = 0
        self.valid = False

    # ------------------------------------------------------------------ writes (in place, evaluated)
    def _write(self, dst, src, ls: int, rows: int, start: int, ring: bool) -> None:
        import mlx.core as mx

        n = int(src.shape[2])
        prm = mx.array([ls, rows, start, n, 1 if ring else 0], dtype=mx.uint32)
        done = _writer()(
            inputs=[dst, src.astype(mx.bfloat16), prm],
            grid=(64, n, self.cfg.n_kv_heads),
            threadgroup=(64, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
        )[0]
        mx.eval(done)

    def write_full(self, layer: int, start: int, k, v) -> None:
        f = self._fidx[layer]
        self._write(self.full[f // 7], k, f % 7, self.capacity, start, False)
        self._write(self.full[2 + f // 7], v, f % 7, self.capacity, start, False)

    def write_ring(self, layer: int, start: int, k, v) -> None:
        n = k.shape[2]
        if n > self.ring:
            k, v = k[:, :, n - self.ring :, :], v[:, :, n - self.ring :, :]
            start += n - self.ring
        s = self._sidx[layer]
        self._write(self.kr, k, s, self.ring, start, True)
        self._write(self.vr, v, s, self.ring, start, True)

    # ------------------------------------------------------------------ reads (views)
    def full_kv(self, layer: int, length: int):
        f = self._fidx[layer]
        i = f % 7
        return (
            self.full[f // 7][i : i + 1, :, :length, :],
            self.full[2 + f // 7][i : i + 1, :, :length, :],
        )

    def ring_window_ordered(self, layer: int, start: int):
        """K, V of positions max(0, start-512) .. start-1 in position order, evaluated now (the in-place ring
        write that follows in the same prefill layer must not be visible to it)."""
        import mlx.core as mx

        w = min(start, self.window - 1)
        if w == 0:
            return None, None, 0
        s = self._sidx[layer]
        idx = mx.array([(start - w + j) % self.ring for j in range(w)], dtype=mx.int32)
        k = mx.take(self.kr[s : s + 1], idx, axis=2)
        v = mx.take(self.vr[s : s + 1], idx, axis=2)
        mx.eval(k, v)
        return k, v, w

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, prompt, logits) -> Snapshot:
        import mlx.core as mx

        rk, rv = _copy(self.kr), _copy(self.vr)
        mx.eval(rk, rv, logits)
        return Snapshot(tuple(prompt), self.length, {"all": rk}, {"all": rv}, logits)

    def restore(self, snap: Snapshot) -> None:
        import mlx.core as mx

        self.kr = _copy(snap.rings_k["all"])
        self.vr = _copy(snap.rings_v["all"])
        mx.eval(self.kr, self.vr)
        self.length = snap.length
        self.valid = True

    @property
    def rk(self):
        """Truthy once the rings exist (engine prefix-cache check, shared with History)."""
        return self.kr is not None

    @property
    def nbytes(self) -> int:
        cfg = self.cfg
        full = len(cfg.full_layers) * 2 * cfg.n_kv_heads * self.capacity * cfg.head_dim * 2
        ring = len(cfg.sliding_layers) * 2 * cfg.n_kv_heads * self.ring * cfg.head_dim * 2
        return full + ring


# ====================================================================== kv8 profile history
_WRITER_W_SRC = r"""
  const uint t = thread_position_in_grid.x, i = thread_position_in_grid.y, h = thread_position_in_grid.z;
  const uint ls = prm[0], rows = prm[1], start = prm[2], n = prm[3], words = prm[4];
  if (i >= n || t >= words) return;
  const size_t r = size_t(ls * 4u + h) * rows + start + i;
  ((device uint*)dst)[r * words + t] = ((const device uint*)src)[(size_t(h) * n + i) * words + t];
  if (t == 0u && i == 0u && h == 0u) done[0] = 1u;
"""
_writer_w_kernel = None


def _writer_w():
    """In-place writer of `words` uint32 per row: src [1, 4, n, words] (any 4-byte packing) into a consolidated
    [layers][4][rows][words] array at rows start .. start + n - 1 (the kv8 history's codes and scale / bias words)."""
    global _writer_w_kernel
    if _writer_w_kernel is None:
        import mlx.core as mx

        _writer_w_kernel = mx.fast.metal_kernel(
            name="flint_hist_writer_w_v1",
            input_names=["dst", "src", "prm"],
            output_names=["done"],
            source=_WRITER_W_SRC,
            ensure_row_contiguous=True,
            compile_options=METAL_COMPILE_OPTIONS,
        )
    return _writer_w_kernel


KV8_GROUP = 64
KV8_ROW_BYTES = 128 + 2 * 4  # 128 code bytes + 2 (scale | bias) words per 128-wide row


class MKHistory8(MKHistory):
    """The kv8 profile's history (kernels/kv8_src.py): the 14 full layers' K and V at 8 bits in groups of 64,
    as config.json declares (kv_key_bits = kv_value_bits = 8, kv_group_size = 64); the sliding rings stay bf16.

      Kq, Vq    [14][4][capacity][32] uint32: row p = mx.quantize(row, group_size=64, bits=8)'s packed codes (byte d =
                the code of head dim d), 1.88 GB each at 262,144 positions (< 2^31 bytes)
      Ksb, Vsb  [14][4][capacity][4] bf16: scale_0, bias_0, scale_1, bias_1 (the megakernel reads one uint32 = scale |
                bias << 16 per 64-dim group)

    arrays = [Kq, Ksb, Vq, Vsb, K_ring, V_ring], the megakernel's kf0, kf1, vf0, vf1, kr, vr slots. Prefill writes
    mx.quantize's bytes of its K / V (the decode megakernel's quantise-on-write reproduces them bit for bit) and attends
    over mx.dequantize of the stored rows, so both paths read the same 8-bit history (the decode megakernel folds the
    dequantisation into its products, kv8_fold.py: within 1 bf16 ulp of attention over the dequantised rows)."""

    kv_bits = 8

    def __init__(self, cfg: FlintConfig, context_limit: int):
        super().__init__(cfg, context_limit)
        assert cfg.head_dim == 2 * KV8_GROUP

    @property
    def arrays(self) -> list:
        return [self.full[0], self.full[1], self.full[2], self.full[3], self.kr, self.vr]

    def reserve(self, need: int) -> None:
        import mlx.core as mx

        if need > self.limit:
            raise ValueError(f"{need} positions exceed the context limit {self.limit}")
        cfg = self.cfg
        if self.kr is None:
            shape_r = (len(cfg.sliding_layers), cfg.n_kv_heads, self.ring, cfg.head_dim)
            self.kr = mx.zeros(shape_r, dtype=mx.bfloat16)
            self.vr = mx.zeros(shape_r, dtype=mx.bfloat16)
            mx.eval(self.kr, self.vr)
        if need <= self.capacity:
            return
        # a multiple of 8 (layout 6d loads whole 8-key blocks; rows past the live length only need finite contents)
        cap = min(-(-self.limit // 8) * 8, -(-need // CAPACITY_QUANTUM) * CAPACITY_QUANTUM)
        keep = self.length if self.valid else 0
        nf = len(cfg.full_layers)
        for a in range(4):
            shape, dt = (
                ((nf, cfg.n_kv_heads, cap, cfg.head_dim // 4), mx.uint32)
                if a % 2 == 0
                else ((nf, cfg.n_kv_heads, cap, 4), mx.bfloat16)
            )
            new = mx.zeros(shape, dtype=dt)
            old = self.full[a]
            if old is not None and keep:
                new[:, :, :keep, :] = old[:, :, :keep, :]
            self.full[a] = None
            del old
            mx.eval(new)
            self.full[a] = new
        self.capacity = cap
        # the replaced arrays would sit in MLX's buffer cache, which never reuses a buffer of another size
        # (+2.9 GB idle after a 131K -> 258K growth); release them now (once per new capacity)
        mx.clear_cache()

    def _write_words(self, dst, src, ls: int, start: int) -> None:
        import mlx.core as mx

        n, words = int(src.shape[2]), int(src.shape[3]) * src.dtype.size // 4
        prm = mx.array([ls, self.capacity, start, n, words], dtype=mx.uint32)
        done = _writer_w()(
            inputs=[dst, src, prm],
            grid=(words, n, self.cfg.n_kv_heads),
            threadgroup=(words, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
        )[0]
        mx.eval(done)

    def write_full(self, layer: int, start: int, k, v) -> None:
        import mlx.core as mx

        f = self._fidx[layer]
        for a, x in ((0, k), (2, v)):
            wq, s, b = mx.quantize(
                x.astype(mx.bfloat16), group_size=KV8_GROUP, bits=8
            )  # [1,4,n,32] / [1,4,n,2]
            sb = mx.stack([s, b], axis=-1).reshape(*s.shape[:-1], 4)  # s0, b0, s1, b1
            self._write_words(self.full[a], wq, f, start)
            self._write_words(self.full[a + 1], sb, f, start)

    def full_kv(self, layer: int, length: int):
        """bf16 K, V [1, 4, length, 128] of a full layer: mx.dequantize of the stored 8-bit rows."""
        import mlx.core as mx

        f = self._fidx[layer]
        out = []
        for a in (0, 2):
            wq = self.full[a][f : f + 1, :, :length, :]
            sb = self.full[a + 1][f : f + 1, :, :length, :]
            out.append(
                mx.dequantize(wq, sb[..., 0::2], sb[..., 1::2], group_size=KV8_GROUP, bits=8)
            )
        return out[0], out[1]

    @property
    def nbytes(self) -> int:
        cfg = self.cfg
        full = len(cfg.full_layers) * 2 * cfg.n_kv_heads * self.capacity * KV8_ROW_BYTES
        ring = len(cfg.sliding_layers) * 2 * cfg.n_kv_heads * self.ring * cfg.head_dim * 2
        return full + ring
