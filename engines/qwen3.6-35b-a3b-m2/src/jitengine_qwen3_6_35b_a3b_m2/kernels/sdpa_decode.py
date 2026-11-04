"""Single-query attention for decode, bit-identical to mx.fast.scaled_dot_product_attention.

MLX's `sdpa_vector` arithmetic line for line, reading the slot's KV buffer directly with the valid
length as a runtime value. Stock decode attention switches to a two-pass kernel at longer contexts,
so `decode` and `decode_rows` use this kernel only where stock runs `sdpa_vector`, and stock beyond.
"""

from __future__ import annotations

import functools

import mlx.core as mx

from jitengine_qwen3_6_35b_a3b_m2.kernels import constants
from jitengine_qwen3_6_35b_a3b_m2.kernels.source import substitute

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


@functools.cache
def _gpu_family() -> str:
    """The last letter of the Metal architecture name ("s" when it cannot be read), read once."""
    try:
        return str(mx.device_info().get("architecture", ""))[-1:] or "s"
    except Exception:  # noqa: BLE001
        return "s"


def single_pass(n: int, heads: int, kv_heads: int) -> bool:
    """Whether stock decode attention over n keys runs `sdpa_vector`, the kernel this file copies.

    MLX 0.32.2 switches to `sdpa_vector_2pass` from 1024 keys on Pro, Max and Ultra GPUs (Metal
    architecture names ending in "s" or "d") and from 4096 keys with grouped-query attention on every
    GPU. An architecture that cannot be read counts as Pro, the earlier switch.
    """
    return not ((_gpu_family() in ("s", "d") and n >= 1024) or (kv_heads < heads and n >= 4096))


def _validate(q: mx.array, keys: mx.array, values: mx.array, n: int) -> None:
    """Raise ValueError unless q is [1, H, 1, D] against [1, Hkv, cap, D] key and value buffers of
    its dtype, with H a multiple of Hkv and 1 <= n <= cap: the op's contract, which neither this
    kernel nor stock's would honour, so a caller that breaks it has a bug to see."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError(f"decode takes one query, [1, H, 1, D]; got {tuple(q.shape)}")
    if (
        keys.ndim != 4
        or keys.shape != values.shape
        or keys.shape[0] != 1
        or keys.shape[3] != q.shape[3]
    ):
        raise ValueError(
            f"decode takes [1, Hkv, cap, {q.shape[3]}] key and value buffers; got "
            f"{tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if q.shape[1] % keys.shape[1]:
        raise ValueError(f"{q.shape[1]} query heads do not share {keys.shape[1]} KV heads evenly")
    if not q.dtype == keys.dtype == values.dtype:
        raise ValueError(f"decode takes one dtype; got {q.dtype}, {keys.dtype} and {values.dtype}")
    if not 1 <= n <= keys.shape[2]:
        raise ValueError(f"decode reads 1 to {keys.shape[2]} keys of these buffers, not {n}")


def supported(q: mx.array, keys: mx.array, n: int) -> bool:
    """Whether this kernel covers operands that keep the op's contract: stock attention runs its
    single-pass kernel there, with a head dim its vector kernel takes, at most 32 query heads per
    KV head and n below the two-pass switch."""
    heads, kv_heads = q.shape[1], keys.shape[1]
    return (
        q.shape[3] in (64, 96, 128, 192, 256)
        and heads // kv_heads <= 32
        and single_pass(n, heads, kv_heads)
    )


def decode(q: mx.array, keys_buf: mx.array, values_buf: mx.array, n: int, scale: float) -> mx.array:
    """Attention for one query over the first n positions, bitwise stock at every length; operands
    outside the op's contract raise ValueError (`_validate`)."""
    _validate(q, keys_buf, values_buf, n)
    if supported(q, keys_buf, n):
        return sdpa_decode(q, keys_buf, values_buf, n, scale)
    return mx.fast.scaled_dot_product_attention(
        q, keys_buf[:, :, :n], values_buf[:, :, :n], scale=scale, mask=None
    )


@functools.cache
def _sdpa_decode_kernel():
    return mx.fast.metal_kernel(
        name="sdpa_decode_exact",
        input_names=["queries", "keys", "values", "n_keys", "cap", "scale"],
        output_names=["out"],
        source=_SRC,
    )


def sdpa_decode(
    q: mx.array, keys_buf: mx.array, values_buf: mx.array, n: int, scale: float
) -> mx.array:
    """q [1, H, 1, D] over the first n positions of [1, Hkv, cap, D] buffers; bitwise stock SDPA."""
    H, D = q.shape[1], q.shape[3]
    Hkv, cap = keys_buf.shape[1], keys_buf.shape[2]
    return _sdpa_decode_kernel()(
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


def sdpa_decode_rows(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float
) -> mx.array:
    """`sdpa_decode` for B <= MAXB rows in one launch; row b uses its own buffers and ns[b]."""
    B = qs.shape[0]
    if not 1 <= B <= MAXB:
        raise ValueError(f"sdpa_decode_rows takes 1 to {MAXB} rows, got {B}")
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


def decode_rows(
    qs: mx.array, keys_bufs: list, values_bufs: list, ns: list, scale: float
) -> mx.array:
    """`decode` for each row: one `sdpa_decode_rows` launch when every row takes the exact kernel,
    row by row otherwise. Each row keeps `decode`'s contract, or ValueError."""
    B = qs.shape[0]
    if not len(keys_bufs) == len(values_bufs) == len(ns) == B:
        raise ValueError(
            f"decode_rows takes a key buffer, a value buffer and a length per row: {B} rows, "
            f"{len(keys_bufs)}, {len(values_bufs)} and {len(ns)}"
        )
    for b in range(B):
        _validate(qs[b : b + 1], keys_bufs[b], values_bufs[b], ns[b])
    if B <= MAXB and all(supported(qs[b : b + 1], keys_bufs[b], ns[b]) for b in range(B)):
        return sdpa_decode_rows(qs, keys_bufs, values_bufs, ns, scale)
    return mx.concatenate(
        [decode(qs[b : b + 1], keys_bufs[b], values_bufs[b], ns[b], scale) for b in range(B)]
    )


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
    """`decode` at the served geometry, bitwise stock on each side of stock's two-pass switches,
    on fresh buffers and on offset views."""
    scale = head_dim**-0.5
    for n in SDPA_LENGTHS:
        q = mx.random.normal((1, heads, 1, head_dim), key=mx.random.key(n)).astype(mx.bfloat16)
        for offset in (False, True):
            kb = _kv((1, kv_heads, n + 64, head_dim), n + 1, offset)
            vb = _kv((1, kv_heads, n + 64, head_dim), n + 2, offset)
            want = mx.fast.scaled_dot_product_attention(
                q, kb[:, :, :n], vb[:, :, :n], scale=scale, mask=None
            )
            if not mx.array_equal(decode(q, kb, vb, n, scale), want).item():
                return False
    return True


def check_sdpa_rows(heads: int, kv_heads: int, head_dim: int) -> bool:
    """`decode_rows` at the served geometry: row b bitwise stock on its own KV prefix, in one launch
    below the two-pass switches and row by row across them; every other row's buffers an offset
    view."""
    scale = head_dim**-0.5
    for ns in ((57, 1010), (505, 50, 1003, 272), (1023, 1024, 4095), (4095, 4096)):
        B = len(ns)
        shapes = [(1, kv_heads, n + 64, head_dim) for n in ns]
        ks = [_kv(shapes[b], 800 + n + b, b % 2 == 1) for b, n in enumerate(ns)]
        vs = [_kv(shapes[b], 900 + n + b, b % 2 == 1) for b, n in enumerate(ns)]
        q = mx.random.normal((B, heads, 1, head_dim), key=mx.random.key(B)).astype(mx.bfloat16)
        want = mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(
                    q[b : b + 1], ks[b][:, :, : ns[b]], vs[b][:, :, : ns[b]], scale=scale, mask=None
                )
                for b in range(B)
            ]
        )
        if not mx.array_equal(decode_rows(q, ks, vs, list(ns), scale), want).item():
            return False
    return True
