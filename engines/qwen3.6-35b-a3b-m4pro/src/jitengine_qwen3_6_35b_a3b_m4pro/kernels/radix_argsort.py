"""Exact stable argsort of bfloat16 rows for the sampler: LSD radix sort, two 8-bit passes.

A stable sort's permutation is unique, so this equals mx.argsort given MLX's key order: bf16
ascending, -0 == +0 (kept in index order), no NaN. `selfcheck` re-verifies it on the device.
"""

from __future__ import annotations

import functools

import mlx.core as mx

TILE = 2048  # elements per threadgroup tile (256 threads x 8 rounds)

KEYS = """
  const uint i = thread_position_in_grid.x;
  const uint r = thread_position_in_grid.y;
  const int n = NN;
  if ((int)i >= n) return;
  uint16_t b = as_type<uint16_t>(x[r * n + i]);
  if ((b & 0x7fff) == 0) b = 0;                     // -0 == +0
  keys[r * n + i] = (b & 0x8000) ? (uint16_t)(~b) : (uint16_t)(b | 0x8000);
  idx[r * n + i] = i;
"""

HIST = """
  // one threadgroup per (tile, row): digit counts of the tile -> hist[row][digit][tile]
  const uint tile = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  const uint t = thread_position_in_threadgroup.x;   // 256
  const int n = NN;
  constexpr int NT = (NN + TILE - 1) / TILE;
  threadgroup atomic_uint cnt[256];
  atomic_store_explicit(&cnt[t], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = t; j < TILE; j += 256) {
    const int i = tile * TILE + j;
    if (i < n) atomic_fetch_add_explicit(&cnt[(keys[r * n + i] >> SHIFT) & 0xff], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  hist[(r * 256 + t) * NT + tile] = atomic_load_explicit(&cnt[t], memory_order_relaxed);
"""

SCATTER = """
  // stable scatter of one tile: element order = (round, simdgroup, lane)
  const uint tile = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  const uint t = thread_position_in_threadgroup.x;   // 256 = 8 simdgroups
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int n = NN;
  constexpr int NT = (NN + TILE - 1) / TILE;
  threadgroup uint run[256];                // elements of each digit already placed from this tile
  threadgroup uint cnt[8][256];             // this round: per simdgroup, per digit
  run[t] = offs[(r * 256 + t) * NT + tile];
  for (int round = 0; round < TILE / 256; round++) {
    for (int d = 0; d < 8; d++) cnt[d][t] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int i = tile * TILE + round * 256 + t;
    const bool valid = i < n;
    const uint k = valid ? keys[r * n + i] : 0;
    const uint v = valid ? vals[r * n + i] : 0;
    const uint dg = (k >> SHIFT) & 0xff;
    // lanes of this simdgroup holding the same digit (8 ballots on the digit bits)
    uint64_t same = (uint64_t)simd_ballot(valid);
    for (int b = 0; b < 8; b++) {
      const bool bit = (dg >> b) & 1;
      const uint64_t m = (uint64_t)simd_ballot(bit);
      same &= bit ? m : ~m;
    }
    const uint64_t below = (lane == 0) ? 0 : (same & ((((uint64_t)1) << lane) - 1));
    const uint rank = popcount(below);
    const uint total = popcount(same);
    if (valid && rank == 0) cnt[sg][dg] = total;        // the lowest lane of the digit group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
      uint pos = run[dg] + rank;
      for (uint s = 0; s < sg; s++) pos += cnt[s][dg];
      okeys[r * n + pos] = (uint16_t)k;
      ovals[r * n + pos] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint add = 0;
    for (int s = 0; s < 8; s++) add += cnt[s][t];
    run[t] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@functools.cache
def _kern(name, src, inputs, outputs):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src)


# radix_argsort_values: the same sort, plus the sorted values (== take_along_axis(x, idx)) written by
# the last scatter. Sorted keys decode back to bf16 bit patterns; the one pattern the key map merges
# (-0 into +0) rides as bit 31 of the index payload (indices < 2^31).
SCATTER_V = SCATTER.replace(
    """      okeys[r * n + pos] = (uint16_t)k;
      ovals[r * n + pos] = v;""",
    """      if (LAST) {
        const uint16_t kk = (uint16_t)k;
        const uint16_t bb = (kk & 0x8000) ? (uint16_t)(kk & 0x7fff) : (uint16_t)(~kk);
        const bfloat16_t sval = as_type<bfloat16_t>((v >> 31) ? (uint16_t)0x8000 : bb);
        osv[r * n + pos] = sval;
        if (EXPV) oexp[r * n + pos] = metal::precise::exp(sval);   // == mx.exp, all 65,536 bf16 inputs
        ovals[r * n + pos] = v & 0x7fffffffu;
      } else {
        okeys[r * n + pos] = (uint16_t)k;
        ovals[r * n + pos] = v;
      }""",
)


# The values sort skips the key pass: its first histogram and first scatter compute each key (and the
# index payload with the -0 flag) from x on the fly.
_KEY_OF = """
  #define KEY_OF(xv) ([&]() { uint16_t bq = as_type<uint16_t>(xv); if ((bq & 0x7fff) == 0) bq = 0; \\
      return (bq & 0x8000) ? (uint16_t)(~bq) : (uint16_t)(bq | 0x8000); }())
"""
HIST_X = _KEY_OF + HIST.replace("(keys[r * n + i] >> SHIFT)", "(KEY_OF(x[r * n + i]) >> SHIFT)")
SCATTER_X = _KEY_OF + SCATTER_V.replace(
    "const uint k = valid ? keys[r * n + i] : 0;",
    "const uint k = valid ? KEY_OF(x[r * n + i]) : 0;",
).replace(
    "const uint v = valid ? vals[r * n + i] : 0;",
    "const uint v = valid ? (uint(i) | ((as_type<uint16_t>(x[r * n + i]) == 0x8000) ? 0x80000000u : 0u)) : 0;",
)


def _radix(x, values: bool, with_exp: bool = False):
    R, n = x.shape
    nt = (n + TILE - 1) // TILE
    tmpl = [("NN", n), ("TILE", TILE)]
    if values:
        hist = _kern("rxx_hist", HIST_X, ("x",), ("hist",))(
            inputs=[x],
            template=tmpl + [("SHIFT", 0)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R * 256 * nt,)],
            output_dtypes=[mx.uint32],
        )[0]
        offs = (
            mx.cumsum(hist.reshape(R, 256 * nt), axis=-1, inclusive=False)
            .astype(mx.uint32)
            .reshape(-1)
        )
        keys, vals, _, _ = _kern(
            "rxx_scatter", SCATTER_X, ("x", "offs"), ("okeys", "ovals", "osv", "oexp")
        )(
            inputs=[x, offs],
            template=tmpl + [("SHIFT", 0), ("LAST", False), ("EXPV", False)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R * n,), (R * n,), (1,), (1,)],
            output_dtypes=[mx.uint16, mx.uint32, x.dtype, x.dtype],
        )
        hist = _kern("rx2_hist", HIST, ("keys",), ("hist",))(
            inputs=[keys],
            template=tmpl + [("SHIFT", 8)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R * 256 * nt,)],
            output_dtypes=[mx.uint32],
        )[0]
        offs = (
            mx.cumsum(hist.reshape(R, 256 * nt), axis=-1, inclusive=False)
            .astype(mx.uint32)
            .reshape(-1)
        )
        _, vals, sv, e = _kern(
            "rxv_scatter_e", SCATTER_V, ("keys", "vals", "offs"), ("okeys", "ovals", "osv", "oexp")
        )(
            inputs=[keys, vals, offs],
            template=tmpl + [("SHIFT", 8), ("LAST", True), ("EXPV", with_exp)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,), (R * n,), (R * n,), (R * n if with_exp else 1,)],
            output_dtypes=[mx.uint16, mx.uint32, x.dtype, x.dtype],
        )
        if with_exp:
            return vals.reshape(R, n), sv.reshape(R, n), e.reshape(R, n)
        return vals.reshape(R, n), sv.reshape(R, n)
    keys, vals = _kern("rx2_keys", KEYS, ("x",), ("keys", "idx"))(
        inputs=[x],
        template=tmpl,
        grid=(((n + 255) // 256) * 256, R, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(R * n,), (R * n,)],
        output_dtypes=[mx.uint16, mx.uint32],
    )
    for shift in (0, 8):
        hist = _kern("rx2_hist", HIST, ("keys",), ("hist",))(
            inputs=[keys],
            template=tmpl + [("SHIFT", shift)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R * 256 * nt,)],
            output_dtypes=[mx.uint32],
        )[0]
        offs = (
            mx.cumsum(hist.reshape(R, 256 * nt), axis=-1, inclusive=False)
            .astype(mx.uint32)
            .reshape(-1)
        )
        keys, vals = _kern("rx2_scatter", SCATTER, ("keys", "vals", "offs"), ("okeys", "ovals"))(
            inputs=[keys, vals, offs],
            template=tmpl + [("SHIFT", shift)],
            grid=(256 * nt, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R * n,), (R * n,)],
            output_dtypes=[mx.uint16, mx.uint32],
        )
    return vals.reshape(R, n), None


def radix_argsort(x):
    """x [R, N] bfloat16 -> uint32 [R, N]: stable ascending order per row (== mx.argsort)."""
    return _radix(x, False)[0]


def radix_argsort_values(x, with_exp: bool = False):
    """x [R, N] bfloat16 -> (radix_argsort(x), take_along_axis(x, that, -1)[, mx.exp of those]), bitwise."""
    return _radix(x, True, with_exp)


def check_sort() -> bool:
    """radix_argsort == mx.argsort (the permutation of a stable sort is unique)."""
    V = 248320
    for t in range(8):
        key = mx.random.key(700 + t)
        R = 1 + t % 3
        lg = (
            mx.random.randint(-3, 3, (R, V), key=key).astype(mx.bfloat16)
            if t % 2
            else (mx.random.normal((R, V), key=key) * 4).astype(mx.bfloat16)
        )
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        for a in (lp, lg):
            if not mx.array_equal(
                radix_argsort(a), mx.argsort(a, axis=-1).astype(mx.uint32)
            ).item():
                return False
    return True
