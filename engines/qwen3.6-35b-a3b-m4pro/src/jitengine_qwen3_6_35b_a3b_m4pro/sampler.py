"""Token sampling in mlx_lm's order (top_p, then top_k, then categorical), or argmax when greedy.

Each request draws under its own key, never the global RNG, so seeded requests are reproducible and
can share a batch.
"""

from __future__ import annotations

import os

import mlx.core as mx

KEY_BLOCK = 256
# Sampler._sample_exact draws what mlx_lm's filters + categorical draw for the same key, from one
# argsort. Relies on mlx 0.32.2 (see tests/metal/test_sampler.py): argpartition is the
# stable merge sort argsort uses, so apply_top_k keeps the K largest by (value desc, index asc), and one-sample
# categorical is an fp32 exclusive cumsum of exp(x - max) in vocab order, then
# searchsorted(cdf, u * sum, "right") - 1.
# minimum width of _sample_exact's top window: a performance knob, not a correctness one
WINDOW_MIN = 256
# integer counts over sorted / monotone rows by binary search (exact by definition)
SORTED_COUNTS = True

_BOUNDS_SRC = """
  // one thread per row: on a non-decreasing row a[b, :], the number of entries < x[b] and <= x[b]
  const uint b = thread_position_in_grid.x;
  if ((int)b >= NB) return;
  const device T* r = a + b * NN;
  const float xv = float(x[b]);
  int lo = 0, hi = NN;                      // first index with r[i] >= xv
  while (lo < hi) { const int mid = (lo + hi) >> 1; if (float(r[mid]) < xv) lo = mid + 1; else hi = mid; }
  const int lt = lo;
  lo = lt; hi = NN;                         // first index with r[i] > xv
  while (lo < hi) { const int mid = (lo + hi) >> 1; if (float(r[mid]) <= xv) lo = mid + 1; else hi = mid; }
  n_lt[b] = lt;
  n_le[b] = lo;
"""
_KB = None


def sorted_bounds(a: mx.array, x: mx.array):
    """a [B, n] non-decreasing per row, x [B, 1] (same dtype) -> (count(a < x), count(a <= x)) as
    int32 [B, 1]: the integers mx.sum(a < x) / mx.sum(a <= x) give, without two full-row passes."""
    global _KB
    if _KB is None:
        _KB = mx.fast.metal_kernel(
            name="sampler_sorted_bounds",
            input_names=["a", "x"],
            output_names=["n_lt", "n_le"],
            source=_BOUNDS_SRC,
        )
    B, n = a.shape
    lt, le = _KB(
        inputs=[a, x.reshape(B)],
        template=[("T", a.dtype), ("NN", n), ("NB", B)],
        grid=(max(32, B), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 1), (B, 1)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return lt, le


# The windowed draw after its sort as two kernels (draw_fused) instead of ~30 small ops per row.
# Set by the engine after selfcheck.check_sampler_tail. Nothing in it is new arithmetic:
#   _TAIL_W: sv where cum > 1 - top_p (the bf16 compare, on the fly), v, the two counts by the same
#     binary search as sorted_bounds, both windows, keep, the bf16 temperature product, max, the
#     precise exp weights, scattered into a zero-initialised w [B, n] (overlap writes equal values);
#   _TAIL_C: sum(w) with mlx 0.32.2's all_reduce for one float32 row replayed (two passes: 128
#     partial rows of ceil(n/128), threadgroup ceil(rows/4) rounded to 32, four sequential reads per
#     thread, simd_sum, simd_sum of the simdgroup sums; then 32 lanes x 4 reads, simd_sum), the
#     uniform as mx.random.uniform(shape=(1,), float32, key) computes it (rbitsc's single output word
#     threefry2x32(key, (0, 0)).x / float(UINT32_MAX), clamped below 1.0), and count(cdf <= u) - 1 by
#     binary search on the non-decreasing exclusive cumsum.
# The two cumsums stay stock ops.
_TAIL_W_SRC = """
  const uint b = threadgroup_position_in_grid.y;
  const uint t = thread_position_in_threadgroup.x;
  const uint nt = threads_per_threadgroup.x;
  const device T* r = sv + (size_t)b * NN;
  const device T* cr = cum + (size_t)b * NN;
  const device uint* ir = idx + (size_t)b * NN;
  device float* wr = w + (size_t)b * NN;
  threadgroup int sh[2];
  threadgroup float red[32];
  const float th = float(thr[0]);
  #define SV2(p) (TOPP ? (float(cr[p]) > th ? float(r[p]) : -INFINITY) : float(r[p]))
  const float vf = SV2(NN - KK);
  if (t == 0) {
    int lo = 0, hi = NN;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (SV2(mid) < vf) lo = mid + 1; else hi = mid; }
    const int lt = lo;
    lo = lt; hi = NN;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (SV2(mid) <= vf) lo = mid + 1; else hi = mid; }
    sh[0] = lt; sh[1] = lo;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int first = sh[0], above = NN - sh[1];
  const int cut = first + (KK - above);
  const int start = min(first, NN - KK);
  const float it = float(inv_t[0]);
  constexpr int W = WT + KK;
  float xv[(W + 255) / 256];
  uint iv[(W + 255) / 256];
  float mx_l = -INFINITY;
  for (int j = t, c = 0; j < W; j += nt, c++) {
    const int pos = j < WT ? NN - WT + j : start + (j - WT);
    const float sf = SV2(pos);
    const bool keep = (sf > vf) || ((sf == vf) && (pos < cut));
    const float x = float(T((keep ? sf : -INFINITY) * it));
    xv[c] = x;
    iv[c] = ir[pos];
    mx_l = max(mx_l, x);
  }
  mx_l = simd_max(mx_l);
  if (t % 32 == 0) red[t / 32] = mx_l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = red[0];
  for (uint q = 1; q < nt / 32; q++) m = max(m, red[q]);
  const bool minf = isinf(m);
  for (int j = t, c = 0; j < W; j += nt, c++) {
    const float x = xv[c];
    wr[iv[c]] = minf ? (x == m ? 1.0f : 0.0f) : metal::precise::exp(x - m);
  }
"""
_TAIL_C_SRC = """
  const uint b = threadgroup_position_in_grid.y;
  const uint lid = thread_position_in_threadgroup.x;
  const uint sg = lid / 32, ln = lid % 32;
  constexpr int NSG = TG / 32;
  const device float* in0 = w + (size_t)b * NN;
  threadgroup float shv[32];
  threadgroup float inter[NR];
  for (int g = 0; g < NR; g++) {
    long start_idx = (long)g * RS;
    long actual = (start_idx + RS <= NN) ? RS : NN - start_idx;
    long blocks = actual / (TG * 4);
    int extra = actual - blocks * (TG * 4);
    extra -= lid * 4;
    start_idx += lid * 4;
    const device float* in = in0 + start_idx;
    if (extra >= 4) { blocks++; extra = 0; }
    float total = 0.0f;
    for (long q = 0; q < blocks; q++) {
      for (int i = 0; i < 4; i++) total = in[i] + total;
      in += TG * 4;
    }
    if (extra > 0) for (int i = 0; i < extra; i++) total = in[i] + total;
    total = simd_sum(total);
    if (NSG > 1) {
      if (ln == 0) shv[sg] = total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      total = lid < NSG ? shv[lid] : 0.0f;
      total = simd_sum(total);
    }
    if (lid == 0) inter[g] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg == 0) {
    float unif;
    {
      const uint2 key = uint2(keys[2 * b], keys[2 * b + 1]);
      const uint ks[3] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};
      uint2 v = uint2(ks[0], ks[1]);
      const uint rot[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
      for (int i = 0; i < 5; ++i) {
        for (int q = 0; q < 4; q++) {
          const uint rr = rot[i % 2][q];
          v.x += v.y;
          v.y = (v.y << rr) | (v.y >> (32 - rr));
          v.y ^= v.x;
        }
        v.x += ks[(i + 1) % 3];
        v.y += ks[(i + 2) % 3] + i + 1;
      }
      unif = min(float(v.x) / float(4294967295u), as_type<float>(0x3f7fffffu));
      unif = 1.0f * unif + 0.0f;
    }
    long blocks = NR / (32 * 4);
    int extra = NR - blocks * (32 * 4);
    extra -= ln * 4;
    int o = ln * 4;
    if (extra >= 4) { blocks++; extra = 0; }
    float total = 0.0f;
    for (long q = 0; q < blocks; q++) {
      for (int i = 0; i < 4; i++) total = inter[o + i] + total;
      o += 32 * 4;
    }
    if (extra > 0) for (int i = 0; i < extra; i++) total = inter[o + i] + total;
    total = simd_sum(total);
    if (ln == 0) {
      const float u = unif * total;
      const device float* c = cdf + (size_t)b * NN;
      int lo = 0, hi = NN;
      while (lo < hi) { const int mid = (lo + hi) >> 1; if (c[mid] <= u) lo = mid + 1; else hi = mid; }
      tok[b] = uint(lo - 1);
      dbg[b] = DBG ? unif : total;
    }
  }
"""
_KTW = _KTC = None


def tail_supported(lp: mx.array, k: int) -> bool:
    """Shapes the fused tail reproduces: bf16 rows, top-k windowed draw, and a vocabulary inside the
    all_reduce branch replayed above (more than 4 x 1024 entries, at most 2^26 bytes of float32)."""
    n = lp.shape[-1]
    return (
        lp.dtype == mx.bfloat16
        and lp.ndim == 2
        and 0 < k
        and 4 * k <= n
        and 4 * 1024 < n
        and 4 * n <= (1 << 26)
    )


def _reduce_plan(n: int):
    nr = 32 * 4
    rs = (n + nr - 1) // nr
    tg = min((rs + 3) // 4, 1024)
    return nr, rs, ((tg + 31) // 32) * 32


def draw_fused(lp: mx.array, samplers: list, steps: list, debug: bool = False):
    """The windowed draw for rows sharing (temperature, top_p, top_k): uint32 [B] tokens, the same
    as sample_rows / Sampler._sample_exact. debug=True also returns (w, sum or uniform per row)."""
    global _KTW, _KTC
    if _KTW is None:
        _KTW = mx.fast.metal_kernel(
            name="sampler_tail_w",
            input_names=["sv", "cum", "idx", "thr", "inv_t"],
            output_names=["w"],
            source=_TAIL_W_SRC,
        )
        _KTC = mx.fast.metal_kernel(
            name="sampler_tail_c",
            input_names=["w", "cdf", "keys"],
            output_names=["tok", "dbg"],
            source=_TAIL_C_SRC,
        )
    B, n = lp.shape
    s0 = samplers[0]
    k, top_p = s0.top_k, s0.top_p
    w_top = s0._window(n)
    topp = 0.0 < top_p < 1.0
    if s0.radix:
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import radix_argsort_values

        if topp:
            # e == mx.exp(sv), from the same scatter
            idx, sv, e = radix_argsort_values(lp, with_exp=True)
        else:
            idx, sv = radix_argsort_values(lp)
    else:
        idx = mx.argsort(lp, axis=-1)
        sv = mx.take_along_axis(lp, idx, axis=-1)
        e = mx.exp(sv) if topp else None
    cum = mx.cumsum(e, axis=-1) if topp else sv
    thr = mx.array(1 - top_p, dtype=lp.dtype).reshape(1)
    inv_t = mx.array(1.0 / s0.temperature, dtype=lp.dtype).reshape(1)
    w = _KTW(
        inputs=[sv, cum, idx.astype(mx.uint32), thr, inv_t],
        template=[("T", lp.dtype), ("NN", n), ("KK", k), ("WT", w_top), ("TOPP", topp)],
        grid=(256, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, n)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]
    cdf = mx.cumsum(w, axis=-1, inclusive=False)
    keys = mx.stack([s.key(st) for s, st in zip(samplers, steps)])
    nr, rs, tg = _reduce_plan(n)
    tok, dbg = _KTC(
        inputs=[w, cdf, keys],
        template=[("NN", n), ("NR", nr), ("RS", rs), ("TG", tg), ("DBG", debug == "u")],
        grid=(tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B,), (B,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return (tok, w, dbg) if debug else tok


def _argsort(lp: mx.array, radix: bool) -> mx.array:
    """mx.argsort over the last axis; `radix` runs radix_argsort, which equals it (a stable sort's
    permutation is unique) and which the engine enables once its check passes on this device."""
    if radix and lp.dtype == mx.bfloat16 and lp.ndim == 2:
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import radix_argsort

        return radix_argsort(lp)
    return mx.argsort(lp, axis=-1)


def logprobs_from_logits(logits: mx.array) -> mx.array:
    """mlx_lm.generate: logprobs = logits - logsumexp(logits) (same dtype as the logits)."""
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    inverse = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative = mx.take_along_axis(cumulative, inverse, axis=-1)
    return mx.where(cumulative > 1 - top_p, logprobs, -float("inf"))


def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1)


class Sampler:
    """One per request. `__call__(logits [1, V], step)` -> token [1] (lazy)."""

    def __init__(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        max_steps: int,
        vocab: int,
        radix: bool = False,
        fused_tail: bool = False,
    ):
        self.temperature = float(temperature or 0.0)
        self.radix = radix
        self.fused_tail = fused_tail
        self.top_p = float(top_p if top_p is not None else 1.0)
        self.top_k = int(top_k or 0)
        self.greedy = self.temperature <= 0.0
        self._root = None
        self._chain = None  # chain key of the next block to derive
        self._block = -1  # index of the block held in self._keys
        self._keys = None
        if not self.greedy:
            if self.top_k >= vocab:
                self.top_k = 0
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "little")
            self._root = self._chain = mx.random.key(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def key(self, step: int) -> mx.array:
        """Return the key for `step`, deriving keys lazily one KEY_BLOCK block at a time.

        Block b has (block key, chain_{b+1}) = split(chain_b, 2) with chain_0 = root, and its step
        keys are split(block key, KEY_BLOCK); an earlier block is re-derived from the root.
        """
        b = step // KEY_BLOCK
        if b != self._block:
            if b < self._block:
                self._chain, self._block = self._root, -1
            while self._block < b:
                blk, self._chain = mx.random.split(self._chain, 2)
                self._block += 1
            self._keys = mx.random.split(blk, KEY_BLOCK)
        return self._keys[step % KEY_BLOCK]

    def uniform(self, step: int) -> mx.array:
        """The [1] float32 uniform the draw at `step` multiplies by sum(w)."""
        return mx.random.uniform(shape=(1,), dtype=mx.float32, key=self.key(step))

    def __call__(self, logits: mx.array, step: int) -> mx.array:
        lp = logprobs_from_logits(logits)
        if self.greedy:
            return mx.argmax(lp, axis=-1)
        if lp.ndim == 2 and lp.shape[0] == 1:
            return self._sample_exact(lp, step)
        if 0.0 < self.top_p < 1.0:
            lp = apply_top_p(lp, self.top_p)
        if self.top_k > 0:
            lp = apply_top_k(lp, self.top_k)
        return mx.random.categorical(lp * (1.0 / self.temperature), axis=-1, key=self.key(step))

    def _sample_exact(self, lp: mx.array, step: int) -> mx.array:
        """`_sample_exact_full` computed on two short slices of the sorted order: the same draw.

        After the top_p cut the sorted values are non-decreasing, so with v = sv[n-k] the kept set
        is [first, first + (k - above)) union [n - above, n): at most k ties at v starting at
        `first`, and a strictly-larger suffix that fits the top window because above < k. Every
        other position has weight exactly +0.0 in the full path too, so the weight vector, cumsum
        and draw are identical.
        """
        n = lp.shape[-1]
        k = self.top_k
        if k <= 0 or 4 * k > n:
            return self._sample_exact_full(lp, step)
        if self.fused_tail and SORTED_COUNTS and tail_supported(lp, k):
            return draw_fused(lp, [self], [step])
        w_top = self._window(n)
        idx = _argsort(lp, self.radix)
        sv = mx.take_along_axis(lp, idx, axis=-1)
        if 0.0 < self.top_p < 1.0:
            # == exp(lp) gathered by idx (elementwise): one gather fewer
            cum = mx.cumsum(mx.exp(sv), axis=-1)
            sv = mx.where(cum > 1 - self.top_p, sv, -float("inf"))  # stays non-decreasing
        v = sv[..., n - k : n - k + 1]
        if SORTED_COUNTS and sv.ndim == 2:
            first, le = sorted_bounds(sv, v)  # sv non-decreasing: the same integers
            above = n - le
        else:
            above = mx.sum(sv > v, axis=-1, keepdims=True)
            first = mx.sum(sv < v, axis=-1, keepdims=True)
        cut = first + (k - above)
        inv_t = 1.0 / self.temperature

        # window A: the top w_top sorted positions, which always hold the strictly-larger survivors
        pos_a = mx.arange(n - w_top, n, dtype=mx.int32)
        sv_a = sv[..., n - w_top :]
        idx_a = idx[..., n - w_top :]
        # window B: k positions anchored at `first`, which always hold the kept ties at v
        start = mx.minimum(first.reshape(-1).astype(mx.int32), mx.array(n - k, dtype=mx.int32))
        pos_b = start + mx.arange(k, dtype=mx.int32)
        sv_b = mx.slice(sv, start, (1,), (1, k))
        idx_b = mx.slice(idx, start, (1,), (1, k))

        # where the windows overlap, both copies carry the same value and the scatter writes it
        # twice; the max must span both windows (with all-equal logits the top one can be all -inf)
        svw = mx.concatenate([sv_a.reshape(-1), sv_b.reshape(-1)])
        posw = mx.concatenate([pos_a, pos_b])
        keep = (svw > v.reshape(-1)) | ((svw == v.reshape(-1)) & (posw < cut.reshape(-1)))
        xw = (mx.where(keep, svw, -float("inf")) * inv_t).astype(mx.float32)
        m = mx.max(xw)
        ww = mx.where(mx.isinf(m), (xw == m).astype(mx.float32), mx.exp(xw - m))
        w = mx.put_along_axis(
            mx.zeros((n,), dtype=mx.float32),
            mx.concatenate([idx_a.reshape(-1), idx_b.reshape(-1)]),
            ww,
            axis=-1,
        )
        cdf = mx.cumsum(w, axis=0, inclusive=False)
        u = mx.random.uniform(shape=(1,), dtype=mx.float32, key=self.key(step)) * mx.sum(w)
        if SORTED_COUNTS:
            return (sorted_bounds(cdf[None, :], u.reshape(1, 1))[1].reshape(1) - 1).astype(
                mx.uint32
            )
        return (mx.sum(cdf[None, :] <= u[:, None], axis=-1) - 1).astype(mx.uint32)

    def _window(self, n: int) -> int:
        """Width of the static top window; it must be at least top_k to hold the kept suffix."""
        return min(n, max(WINDOW_MIN, self.top_k))

    def _sample_exact_full(self, lp: mx.array, step: int) -> mx.array:
        """Bit-for-bit the stock draw (apply_top_p, apply_top_k, categorical) for one row [1, V].

        Uses one stable ascending argsort: top_k keeps values above the K-th largest v plus the
        first (lowest-index) positions holding v, and the draw counts cdf <= u instead of a search.
        """
        n = lp.shape[-1]
        idx = mx.argsort(lp, axis=-1)
        sv = mx.take_along_axis(lp, idx, axis=-1)
        if 0.0 < self.top_p < 1.0:
            cum = mx.cumsum(mx.take_along_axis(mx.exp(lp), idx, axis=-1), axis=-1)
            sv = mx.where(cum > 1 - self.top_p, sv, -float("inf"))  # stays non-decreasing
        if self.top_k > 0:
            v = sv[..., n - self.top_k : n - self.top_k + 1]
            above = mx.sum(sv > v, axis=-1, keepdims=True)
            first = mx.sum(sv < v, axis=-1, keepdims=True)
            pos = mx.arange(n, dtype=mx.int32)
            keep = (sv > v) | ((sv == v) & (pos < first + (self.top_k - above)))
            sv = mx.where(keep, sv, -float("inf"))
        x = mx.put_along_axis(mx.zeros_like(lp), idx, sv, axis=-1) * (1.0 / self.temperature)
        x = x.reshape(-1).astype(mx.float32)
        m = mx.max(x)
        w = mx.where(mx.isinf(m), (x == m).astype(mx.float32), mx.exp(x - m))
        cdf = mx.cumsum(w, axis=0, inclusive=False)
        u = mx.random.uniform(shape=(1,), dtype=mx.float32, key=self.key(step)) * mx.sum(w)
        return (mx.sum(cdf[None, :] <= u[:, None], axis=-1) - 1).astype(mx.uint32)


def rows_key(s: Sampler, n: int):
    """Samplers with equal keys can draw as one batch (sample_rows); None = draw alone."""
    if s.greedy or s.top_k <= 0 or 4 * s.top_k > n:
        return None
    return (s.temperature, s.top_p, s.top_k)


def sample_rows(lp: mx.array, samplers: list, steps: list) -> mx.array:
    """`Sampler._sample_exact` for B rows sharing one rows_key (lp [B, V]); the same draw per row.

    sum(w) is the one float reduction whose order could depend on the shape, so it stays a 1-D sum
    per row, as in the single-row path.
    """
    B, n = lp.shape
    s0 = samplers[0]
    if s0.fused_tail and SORTED_COUNTS and tail_supported(lp, s0.top_k):
        return draw_fused(lp, samplers, steps)
    k, top_p, inv_t = s0.top_k, s0.top_p, 1.0 / s0.temperature
    w_top = s0._window(n)
    idx = _argsort(lp, s0.radix)
    sv = mx.take_along_axis(lp, idx, axis=-1)
    if 0.0 < top_p < 1.0:
        # == exp(lp) gathered by idx (elementwise): one gather fewer
        cum = mx.cumsum(mx.exp(sv), axis=-1)
        sv = mx.where(cum > 1 - top_p, sv, -float("inf"))
    v = sv[:, n - k : n - k + 1]
    if SORTED_COUNTS:
        # sv is non-decreasing (sorted, top_p's -inf only at the front): the counts by binary search
        first, le = sorted_bounds(sv, v)
        above = n - le
    else:
        above = mx.sum(sv > v, axis=-1, keepdims=True)
        first = mx.sum(sv < v, axis=-1, keepdims=True)
    cut = first + (k - above)
    pos_a = mx.arange(n - w_top, n, dtype=mx.int32)
    sv_a = sv[:, n - w_top :]
    idx_a = idx[:, n - w_top :]
    start = mx.minimum(first.astype(mx.int32), mx.array(n - k, dtype=mx.int32))
    pos_b = start + mx.arange(k, dtype=mx.int32)[None, :]
    sv_b = mx.take_along_axis(sv, pos_b, axis=-1)
    idx_b = mx.take_along_axis(idx, pos_b, axis=-1)
    svw = mx.concatenate([sv_a, sv_b], axis=-1)
    posw = mx.concatenate([mx.broadcast_to(pos_a[None, :], (B, w_top)), pos_b], axis=-1)
    keep = (svw > v) | ((svw == v) & (posw < cut))
    xw = (mx.where(keep, svw, -float("inf")) * inv_t).astype(mx.float32)
    m = mx.max(xw, axis=-1, keepdims=True)
    ww = mx.where(mx.isinf(m), (xw == m).astype(mx.float32), mx.exp(xw - m))
    w = mx.put_along_axis(
        mx.zeros((B, n), dtype=mx.float32), mx.concatenate([idx_a, idx_b], axis=-1), ww, axis=-1
    )
    cdf = mx.cumsum(w, axis=-1, inclusive=False)
    tot = mx.stack([mx.sum(w[b]) for b in range(B)]).reshape(B, 1)
    u = (
        mx.concatenate(
            [
                mx.random.uniform(shape=(1,), dtype=mx.float32, key=s.key(st))
                for s, st in zip(samplers, steps)
            ]
        ).reshape(B, 1)
        * tot
    )
    if SORTED_COUNTS:
        # cdf is a cumsum of non-negative weights, so non-decreasing: count(cdf <= u) by binary search
        return (sorted_bounds(cdf, u)[1].reshape(B) - 1).astype(mx.uint32)
    return (mx.sum(cdf <= u, axis=-1) - 1).astype(mx.uint32)


def check_sampler_tail(radix: bool) -> bool:
    """sampler.draw_fused vs the op-chain draw (sample_rows with the tail off): tokens on tie-heavy,
    all-equal, two-valued, confident and spread rows under four (T, top_p, top_k) settings; the
    replayed sum against mx.sum of each row; the in-kernel uniform against mx.random.uniform; and,
    with the radix sort, its sorted values against take_along_axis."""
    V = 248320
    if radix:
        from jitengine_qwen3_6_35b_a3b_m4pro.kernels.radix_argsort import radix_argsort_values

        x = (mx.random.normal((3, V), key=mx.random.key(3100)) * 3).astype(mx.bfloat16)
        x = mx.where(mx.arange(V) % 7 == 0, mx.array(-0.0, dtype=mx.bfloat16), x)
        i, sv, e = radix_argsort_values(x, with_exp=True)
        r = mx.argsort(x, axis=-1)
        if not (
            mx.array_equal(i, r).item()
            and mx.array_equal(
                sv.view(mx.uint16), mx.take_along_axis(x, r, axis=-1).view(mx.uint16)
            ).item()
            and mx.array_equal(e.view(mx.uint16), mx.exp(sv).view(mx.uint16)).item()
        ):
            return False
    settings = ((0.7, 0.95, 40), (1.0, 0.9, 20), (0.6, 0.8, 64), (1.3, 1.0, 5))
    for t in range(12):
        key = mx.random.key(3200 + t)
        kind = t % 6
        if kind == 0:
            lg = mx.random.normal((4, V), key=key) * 3
        elif kind == 1:
            lg = mx.random.randint(-3, 3, (4, V), key=key).astype(mx.float32)
        elif kind == 2:
            lg = mx.zeros((4, V))
        elif kind == 3:
            lg = (mx.random.uniform(shape=(4, V), key=key) < 0.001).astype(mx.float32) * 4
        elif kind == 4:
            lg = mx.random.normal((4, V), key=key) + 30 * (mx.arange(V) < 3)
        else:
            lg = mx.random.normal((4, V), key=key) * 0.3
        lg = lg.astype(mx.bfloat16)
        lp = mx.concatenate([logprobs_from_logits(lg[i : i + 1]) for i in range(4)], axis=0)
        T_, P_, K_ = settings[t % 4]
        smp = [Sampler(T_, P_, K_, 3300 + 4 * t + i, 512, V, radix=radix) for i in range(4)]
        steps = [5 * t + 17 * i for i in range(4)]
        ref = sample_rows(lp, smp, steps)
        tok, w, tot = draw_fused(lp, smp, steps, debug=True)
        _, _, u = draw_fused(lp, smp, steps, debug="u")
        ru = mx.concatenate(
            [
                mx.random.uniform(shape=(1,), dtype=mx.float32, key=s.key(st))
                for s, st in zip(smp, steps)
            ]
        )
        sums = mx.stack([mx.sum(w[i]) for i in range(4)])
        if not (
            mx.array_equal(ref, tok).item()
            and mx.array_equal(sums, tot).item()
            and mx.array_equal(u, ru).item()
        ):
            return False
    return True
