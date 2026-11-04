"""Single-query attention for decode, bit-identical to mx.fast.scaled_dot_product_attention.

MLX's `sdpa_vector` arithmetic line for line, reading the slot's KV buffer directly with the valid
length as a runtime value.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m4pro.kernels import constants
from jitengine_qwen3_6_35b_a3b_m4pro.kernels.source import substitute

_SRC = """
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  typedef float U;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int N = n_keys[0];
  const int head = threadgroup_position_in_grid.x;
  const int kv_head = head / G;
  const size_t khs = (size_t)cap[0] * D;
  const device T* kp = keys + kv_head * khs + simd_gid * D + simd_lid * qk_per_thread;
  const device T* vp = values + kv_head * khs + simd_gid * D + simd_lid * v_per_thread;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale[0]) * queries[head * D + simd_lid * qk_per_thread + i];
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  for (int i = simd_gid; i < N; i += BN) {
    U k[qk_per_thread], v[v_per_thread];
    for (int j = 0; j < qk_per_thread; j++) k[j] = kp[j];
    for (int j = 0; j < v_per_thread; j++) v[j] = vp[j];
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * v[j];
    kp += BN * D;
    vp += BN * D;
  }
  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) out[head * D + simd_gid * v_per_thread + i] = static_cast<T>(o[i]);
  }
"""


def _validate(q: mx.array, keys: mx.array, values: mx.array, n: int) -> None:
    """Raise ValueError unless q is [1, H, 1, D] against [1, Hkv, cap, D] key and value buffers of
    its dtype, with H a multiple of Hkv and 1 <= n <= cap: the op's contract, which neither this
    kernel nor stock's would honour, so a caller that breaks it has a bug to see."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError(f"sdpa_decode takes one query, [1, H, 1, D]; got {tuple(q.shape)}")
    if (
        keys.ndim != 4
        or keys.shape != values.shape
        or keys.shape[0] != 1
        or keys.shape[3] != q.shape[3]
    ):
        raise ValueError(
            f"sdpa_decode takes [1, Hkv, cap, {q.shape[3]}] key and value buffers; got "
            f"{tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if q.shape[1] % keys.shape[1]:
        raise ValueError(f"{q.shape[1]} query heads do not share {keys.shape[1]} KV heads evenly")
    if not q.dtype == keys.dtype == values.dtype:
        raise ValueError(
            f"sdpa_decode takes one dtype; got {q.dtype}, {keys.dtype} and {values.dtype}"
        )
    if not 1 <= n <= keys.shape[2]:
        raise ValueError(f"sdpa_decode reads 1 to {keys.shape[2]} keys of these buffers, not {n}")


def supported(q: mx.array, keys: mx.array) -> bool:
    """Whether this kernel covers operands that keep the op's contract: a head dim stock attention's
    vector kernels take and at most 32 query heads per KV head."""
    return q.shape[3] in (64, 96, 128, 192, 256) and q.shape[1] // keys.shape[1] <= 32


@functools.cache
def _sdpa_decode_1pass_kernel():
    return mx.fast.metal_kernel(
        name="sdpa_decode_exact",
        input_names=["queries", "keys", "values", "n_keys", "cap", "scale"],
        output_names=["out"],
        source=_SRC,
    )


def sdpa_decode_1pass(
    q: mx.array, keys_buf: mx.array, values_buf: mx.array, n: int, scale: float
) -> mx.array:
    """q [1, H, 1, D] over the first n positions of [1, Hkv, cap, D] buffers; bitwise stock SDPA."""
    H, D = q.shape[1], q.shape[3]
    Hkv, cap = keys_buf.shape[1], keys_buf.shape[2]
    return _sdpa_decode_1pass_kernel()(
        inputs=[
            q,
            keys_buf,
            values_buf,
            mx.array([n], dtype=mx.int32),
            constants.i32(cap),
            constants.f32(scale),
        ],
        template=[("T", q.dtype), ("D", D), ("G", H // Hkv)],
        grid=(32 * H, 32, 1),
        threadgroup=(32, 32, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


# B rows in one launch, grid (heads, rows): each threadgroup runs the single-row kernel unchanged on
# its row's own KV buffer and length
MAXB = 8
_ROWS_SRC = substitute(
    _SRC,
    (
        "  const int N = n_keys[0];\n  const int head = threadgroup_position_in_grid.x;",
        "  const int row = threadgroup_position_in_grid.y;\n"
        "  const int N = n_keys[row];\n  const int head = threadgroup_position_in_grid.x;",
    ),
    (
        "  const size_t khs = (size_t)cap[0] * D;\n"
        "  const device T* kp = keys + kv_head * khs + simd_gid * D + simd_lid * qk_per_thread;\n"
        "  const device T* vp = values + kv_head * khs + simd_gid * D + simd_lid * v_per_thread;",
        "  const size_t khs = (size_t)cap[row] * D;\n"
        "  const device T* kb = SELECT(keys);\n  const device T* vb = SELECT(values);\n"
        "  const device T* kp = kb + kv_head * khs + simd_gid * D + simd_lid * qk_per_thread;\n"
        "  const device T* vp = vb + kv_head * khs + simd_gid * D + simd_lid * v_per_thread;",
    ),
    (
        "queries[head * D + simd_lid * qk_per_thread + i]",
        "queries[(row * NH + head) * D + simd_lid * qk_per_thread + i]",
    ),
    (
        "out[head * D + simd_gid * v_per_thread + i]",
        "out[(row * NH + head) * D + simd_gid * v_per_thread + i]",
    ),
)
_SELECT = (
    "#define SELECT(a) (row == 0 ? a##0 : row == 1 ? a##1 : row == 2 ? a##2 : row == 3 ? a##3 : "
    "row == 4 ? a##4 : row == 5 ? a##5 : row == 6 ? a##6 : a##7)\n"
)


@functools.cache
def _rows_kernel():
    names = [f"keys{i}" for i in range(MAXB)] + [f"values{i}" for i in range(MAXB)]
    return mx.fast.metal_kernel(
        name="sdpa_decode_exact_rows",
        input_names=["queries", *names, "n_keys", "cap", "scale"],
        output_names=["out"],
        header=_SELECT,
        source=_ROWS_SRC,
    )


def sdpa_decode_rows_1pass(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float
) -> mx.array:
    """`sdpa_decode_1pass` for B <= MAXB rows in one launch; row b uses its own buffers and ns[b]."""
    B = qs.shape[0]
    if not 1 <= B <= MAXB:
        raise ValueError(f"sdpa_decode_rows_1pass takes 1 to {MAXB} rows, got {B}")
    H, D = qs.shape[1], qs.shape[3]
    Hkv = keys_bufs[0].shape[1]
    kb = list(keys_bufs) + [keys_bufs[0]] * (MAXB - B)
    vb = list(values_bufs) + [values_bufs[0]] * (MAXB - B)
    caps = [k.shape[2] for k in keys_bufs] + [0] * (MAXB - B)
    return _rows_kernel()(
        inputs=[
            qs,
            *kb,
            *vb,
            mx.array(list(ns) + [0] * (MAXB - B), dtype=mx.int32),
            mx.array(caps, dtype=mx.int32),
            constants.f32(scale),
        ],
        template=[("T", qs.dtype), ("D", D), ("G", H // Hkv), ("NH", H)],
        grid=(32 * H, 32 * B, 1),
        threadgroup=(32, 32, 1),
        output_shapes=[qs.shape],
        output_dtypes=[qs.dtype],
    )[0]


# MLX's two-pass decode attention, transcribed from its sdpa_vector_2pass_1 / _2 kernels (no mask, no
# sinks, non-causal) and reading each row's own slot buffer. On 's' and 'd' architectures MLX sends a
# single query with N >= 1024 keys (or GQA with N >= 4096) there instead of to sdpa_vector, so the
# one-pass kernels are bit-identical to the stock op only below that length. The block count follows
# MLX's rule with n_simds = gqa_factor.
_P1_SRC = """
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};
  const uint simd_lid = thread_index_in_simdgroup;
  const uint3 tid = threadgroup_position_in_grid;
  const uint3 tidtg = thread_position_in_threadgroup;
  const int kv_head_idx = tid.x;
  const int row = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = G;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_q_heads = NH;
  const int q_batch_head_idx = (row * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx;
  const int N = n_keys[row];
  const size_t k_head_stride = (size_t)cap[row] * D;
  const size_t k_seq_stride = D;
  const device T* kb = SELECT(keys);
  const device T* vb = SELECT(values);
  const device T* qp = queries + o_offset * D + simd_lid * qk_per_thread;
  const device T* kp = kb + kv_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = vb + kv_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * v_per_thread;
  device T* op = partials + o_offset * blocks * D + block_idx * D + simd_lid * v_per_thread;
  device float* sp = sums + o_offset * blocks + block_idx;
  device float* mp = maxs + o_offset * blocks + block_idx;
  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale[0]) * qp[i];
  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  for (int i = block_idx; i < N; i += blocks) {
    U score = 0;
    for (int i = 0; i < qk_per_thread; i++) score += q[i] * kp[i];
    score = simd_sum(score);
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int i = 0; i < v_per_thread; i++) o[i] = o[i] * factor + exp_score * vp[i];
    kp += blocks * int(k_seq_stride);
    vp += blocks * int(k_seq_stride);
  }
  if (simd_lid == 0) {
    sp[0] = sum_exp_score;
    mp[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) op[i] = static_cast<T>(o[i]);
"""
# The second pass, MLX's `sdpa_vector_2pass_2` with the same arithmetic in fewer threads. Stock runs
# 32 simdgroups of 32 lanes (1024 threads): simdgroup j, lane l sums block group j (blocks j, j + 32,
# ...) for D-slice l, a threadgroup transpose hands slice l's 32 partial sums to simdgroup l, and
# simd_sum adds them. Here lane l sums block group l for one D-slice at a time, so the same 32 values
# reach the same lanes of the same simd_sum, and SIMDGROUPS simdgroups share the 32 slices: the bits
# are stock's, and the threadgroup stays far under every device's limit (a custom pipeline may get
# fewer than 1024 threads, as on an M1 Pro).
_P2_SRC = """
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = BLOCKS;
  typedef float U;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int q_offset = threadgroup_position_in_grid.x;
  const device float* sp = sums + q_offset * blocks;
  const device float* mp = maxs + q_offset * blocks;
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < blocks / BN; ++b) max_score = max(max_score, mp[simd_lid + BN * b]);
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int g = simd_gid; g < BD; g += SIMDGROUPS) {
    thread U o[elem_per_thread] = {0};
    const device T* pp = partials + q_offset * blocks * D + simd_lid * D + g * elem_per_thread;
    for (int b = 0; b < blocks / BN; ++b) {
      U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
      for (int i = 0; i < elem_per_thread; i++) o[i] += factor * static_cast<U>(pp[i]);
      pp += BN * D;
    }
    for (int i = 0; i < elem_per_thread; i++) {
      U v = simd_sum(o[i]);
      v = sum_exp_score == 0 ? v : (v / sum_exp_score);
      if (simd_lid == 0) out[q_offset * D + g * elem_per_thread + i] = static_cast<T>(v);
    }
  }
"""


@functools.cache
def _devc() -> str:
    """The last character of the Metal architecture name, MLX's routing key."""
    try:
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:  # noqa: BLE001
        return ""


def mlx_uses_2pass(n: int, h: int, hkv: int) -> bool:
    """Whether MLX sends a one-query SDPA over n keys to its two-pass kernels."""
    return (_devc() in ("d", "s") and n >= 1024) or (hkv < h and n >= 4096)


def mlx_2pass_blocks(n: int, gqa: int) -> int:
    """MLX's two-pass block count for n keys at GQA factor `gqa`."""
    d, n_simds = _devc(), gqa
    if d == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            blocks = 128 if n <= 8192 else 256 if n <= 32768 else 512 if n <= 65536 else 1024
        return blocks
    if d == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
        return blocks
    return 64 if n_simds >= 4 else 32


P2_SIMDGROUPS = 8  # the second pass's simdgroups per threadgroup (256 threads)


@functools.cache
def _two_pass_kernels(blocks: int):
    names = [f"keys{i}" for i in range(MAXB)] + [f"values{i}" for i in range(MAXB)]
    return (
        mx.fast.metal_kernel(
            name=f"sdpa_2pass1_rows_{blocks}",
            input_names=["queries", *names, "n_keys", "cap", "scale"],
            output_names=["partials", "sums", "maxs"],
            header=_SELECT,
            source=_P1_SRC,
        ),
        mx.fast.metal_kernel(
            name=f"sdpa_2pass2_sg_{blocks}",
            input_names=["partials", "sums", "maxs"],
            output_names=["out"],
            source=_P2_SRC,
        ),
    )


def sdpa_decode_rows_2pass(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float, blocks: int
) -> mx.array:
    """Two-pass attention for B <= MAXB rows (each its own buffers), bitwise MLX's two-pass output."""
    B = qs.shape[0]
    if not 1 <= B <= MAXB:
        raise ValueError(f"sdpa_decode_rows_2pass takes 1 to {MAXB} rows, got {B}")
    H, D = qs.shape[1], qs.shape[3]
    Hkv = keys_bufs[0].shape[1]
    G = H // Hkv
    k1, k2 = _two_pass_kernels(blocks)
    kb = list(keys_bufs) + [keys_bufs[0]] * (MAXB - B)
    vb = list(values_bufs) + [values_bufs[0]] * (MAXB - B)
    caps = [k.shape[2] for k in keys_bufs] + [0] * (MAXB - B)
    part, sums, maxs = k1(
        inputs=[
            qs,
            *kb,
            *vb,
            mx.array(list(ns) + [0] * (MAXB - B), dtype=mx.int32),
            mx.array(caps, dtype=mx.int32),
            constants.f32(scale),
        ],
        template=[("T", qs.dtype), ("D", D), ("G", G), ("NH", H), ("BLOCKS", blocks)],
        grid=(32 * Hkv, G * B, blocks),
        threadgroup=(32, G, 1),
        output_shapes=[(B * H, blocks, D), (B * H, blocks), (B * H, blocks)],
        output_dtypes=[qs.dtype, mx.float32, mx.float32],
    )
    return k2(
        inputs=[part, sums, maxs],
        template=[("T", qs.dtype), ("D", D), ("BLOCKS", blocks), ("SIMDGROUPS", P2_SIMDGROUPS)],
        grid=(32 * P2_SIMDGROUPS * B * H, 1, 1),
        threadgroup=(32 * P2_SIMDGROUPS, 1, 1),
        output_shapes=[qs.shape],
        output_dtypes=[qs.dtype],
    )[0]


def sdpa_decode_rows_stock(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float
) -> mx.array:
    """Per row, the kernel MLX would dispatch (two-pass at its block count, else one-pass), grouped so
    rows with the same routing share one launch. Bitwise the stock op per row."""
    B, H = qs.shape[0], qs.shape[1]
    Hkv = keys_bufs[0].shape[1]
    keyf = [
        ("2", mlx_2pass_blocks(n, H // Hkv)) if mlx_uses_2pass(n, H, Hkv) else ("1", 0) for n in ns
    ]
    if all(k == keyf[0] for k in keyf):
        if keyf[0][0] == "2":
            return sdpa_decode_rows_2pass(qs, keys_bufs, values_bufs, ns, scale, keyf[0][1])
        return sdpa_decode_rows_1pass(qs, keys_bufs, values_bufs, ns, scale)
    outs = [None] * B
    for kf in set(keyf):
        idx = [i for i in range(B) if keyf[i] == kf]
        sub = qs[mx.array(idx)]
        kbs, vbs, nss = (
            [keys_bufs[i] for i in idx],
            [values_bufs[i] for i in idx],
            [ns[i] for i in idx],
        )
        y = (
            sdpa_decode_rows_2pass(sub, kbs, vbs, nss, scale, kf[1])
            if kf[0] == "2"
            else sdpa_decode_rows_1pass(sub, kbs, vbs, nss, scale)
        )
        for j, i in enumerate(idx):
            outs[i] = y[j : j + 1]
    return mx.concatenate(outs, axis=0)


def sdpa_decode(
    q: mx.array, keys_buf: mx.array, values_buf: mx.array, n: int, scale: float
) -> mx.array:
    """Attention for one query over the first n positions, bitwise stock at every length; operands
    outside the op's contract raise ValueError (`_validate`)."""
    _validate(q, keys_buf, values_buf, n)
    H, Hkv = q.shape[1], keys_buf.shape[1]
    if mlx_uses_2pass(n, H, Hkv):
        return sdpa_decode_rows_2pass(
            q, [keys_buf], [values_buf], [n], scale, mlx_2pass_blocks(n, H // Hkv)
        )
    return sdpa_decode_1pass(q, keys_buf, values_buf, n, scale)


def sdpa_decode_rows(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float
) -> mx.array:
    """`sdpa_decode` for each of B <= MAXB rows, each on its own buffers and ns[b]."""
    B = qs.shape[0]
    if not 1 <= B <= MAXB:
        raise ValueError(f"sdpa_decode_rows takes 1 to {MAXB} rows, got {B}")
    if not len(keys_bufs) == len(values_bufs) == len(ns) == B:
        raise ValueError(
            f"sdpa_decode_rows takes a key buffer, a value buffer and a length per row: {B} rows, "
            f"{len(keys_bufs)}, {len(values_bufs)} and {len(ns)}"
        )
    for b in range(B):
        _validate(qs[b : b + 1], keys_bufs[b], values_bufs[b], ns[b])
    return sdpa_decode_rows_stock(qs, keys_bufs, values_bufs, ns, scale)


# key counts on each side of stock attention's switches to its two-pass kernel (1024 keys on Pro,
# Max and Ultra GPUs, 4096 with grouped-query attention): below them sdpa_decode runs its own kernel,
# from them on stock, and both must give stock's bits
SDPA_LENGTHS = (37, 700, 1023, 1024, 4095, 4096)


def _kv(shape: tuple, seed: int, offset: bool) -> mx.array:
    """A random [1, Hkv, cap, D] KV buffer, or the same shape as the second half of one twice as
    large: a view whose data starts past its allocation, which the kernel must read from there."""
    if not offset:
        return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)
    big = mx.random.normal((2, *shape[1:]), key=mx.random.key(seed)).astype(mx.bfloat16)
    return big[1:2]


def check_sdpa(heads: int, kv_heads: int, head_dim: int) -> bool:
    """`sdpa_decode` at the served geometry, bitwise stock on each side of stock's two-pass
    switches, on fresh buffers and on offset views."""
    scale = head_dim**-0.5
    for n, cap in ((37, 64), (700, 1024), (1024, 1100), (3001, 4096)):
        q = mx.random.normal((1, heads, 1, head_dim), key=mx.random.key(n)).astype(mx.bfloat16)
        for offset in (False, True):
            kb = _kv((1, kv_heads, cap, head_dim), n + 1, offset)
            vb = _kv((1, kv_heads, cap, head_dim), n + 2, offset)
            want = mx.fast.scaled_dot_product_attention(
                q, kb[:, :, :n], vb[:, :, :n], scale=scale, mask=None
            )
            if not mx.array_equal(sdpa_decode(q, kb, vb, n, scale), want).item():
                return False
    return True


def check_sdpa_rows(heads: int, kv_heads: int, head_dim: int) -> bool:
    """`sdpa_decode_rows` at the served geometry: row b bitwise stock on its own KV prefix; every
    other row's buffers an offset view."""
    scale = head_dim**-0.5
    for B, caps in ((2, (64, 1024)), (4, (512, 64, 1024, 300)), (3, (3100, 1500, 700))):
        ks = [_kv((1, kv_heads, c, head_dim), 800 + c + b, b % 2 == 1) for b, c in enumerate(caps)]
        vs = [_kv((1, kv_heads, c, head_dim), 900 + c + b, b % 2 == 1) for b, c in enumerate(caps)]
        ns = [max(1, c - 7 * (b + 1)) for b, c in enumerate(caps)]
        q = mx.random.normal((B, heads, 1, head_dim), key=mx.random.key(B)).astype(mx.bfloat16)
        want = mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    q[b : b + 1], ks[b][:, :, : ns[b]], vs[b][:, :, : ns[b]], scale=scale, mask=None
                )
                for b in range(B)
            ]
        )
        if not mx.array_equal(sdpa_decode_rows(q, ks, vs, ns, scale), want).item():
            return False
    return True
