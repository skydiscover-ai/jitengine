"""Load-time repack of the checkpoint into the weight banks, plus typed MLX views into them.

Every tensor is read from its safetensors file (F_NOCACHE reads into one reusable buffer, one tensor at
a time) and copied into its bank slot with an in-place slice update, so the peak is the banks plus the
largest single tensor (twice: the read buffer and its MLX copy).
The MLX graph path then addresses the banks through `mx.as_strided` views (no copies): attention and
dense projections are row-contiguous views into bank 0, and each routed-expert tensor is a
[256, rows, cols] view whose expert stride is the 1,769,472-byte record of weight_banks.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from jitengine_flint_m2.model.config import FlintConfig
from jitengine_flint_m2.weight_banks import EXPERT_PROJ, QSUFFIX, BankPlan

FLUSH_BYTES = 256 * 2**20  # evaluate the pending slice updates about every 256 MB


@dataclass
class QLinear:
    w: object  # uint32 [..., out, in * bits / 32]
    s: object  # bf16 [..., out, in / group]
    b: object  # bf16 [..., out, in / group]
    bits: int
    group: int


@dataclass
class LayerWeights:
    index: int
    sliding: bool
    ln1: object
    ln2: object
    q: QLinear
    k: QLinear
    v: QLinear
    o: QLinear
    q_norm: object
    k_norm: object
    sinks: object  # bf16 [n_heads]
    dense: tuple | None = None  # (gate, up, down) QLinear for mlp_only layers
    router: object = None  # bf16 [n_experts, hidden]
    shared: tuple | None = None  # (gate, up, down) QLinear
    experts: tuple | None = None  # (gate, up, down) QLinear with a leading expert axis


@dataclass
class Weights:
    cfg: FlintConfig
    plan: BankPlan
    banks: list
    layers: list
    embed: QLinear
    lm_head: QLinear
    norm: object
    load_s: float = 0.0
    pool: object = None  # offload mode: the expert pool (expert_pool.py) the MoE layers read in place of the banks


F_NOCACHE = 48  # <sys/fcntl.h>: bypass the unified buffer cache for these reads


class _Reader:
    """Reads tensor bytes with F_NOCACHE into one reusable buffer (no page-cache copy of 22.6 GB).

    The buffer (the largest tensor, the 293 MiB embedding) is an anonymous mapping, unmapped once the reader and the
    views it returned are gone: a malloc'd one stays dirty in libmalloc's large cache after the free, 587 MiB of
    phys_footprint for the process's life (two readers), which the decode-time annex could not use."""

    def __init__(self, index: dict):
        import fcntl
        import mmap
        import os

        self.index = index
        self.fds = {}
        self.buf = np.frombuffer(mmap.mmap(-1, max(v[4] for v in index.values())), dtype=np.uint8)
        self._fcntl, self._os = fcntl, os

    def read(self, name: str) -> np.ndarray:
        path, _, _, off, nbytes = self.index[name]
        fd = self.fds.get(path)
        if fd is None:
            fd = self._os.open(path, self._os.O_RDONLY)
            try:
                self._fcntl.fcntl(fd, F_NOCACHE, 1)
            except OSError:
                pass
            self.fds[path] = fd
        view = memoryview(self.buf)[:nbytes]
        got = 0
        while got < nbytes:
            k = self._os.preadv(fd, [view[got:]], off + got)
            if k <= 0:
                raise OSError(f"short read of {name} from {path}")
            got += k
        return self.buf[:nbytes]

    def close(self) -> None:
        for fd in self.fds.values():
            self._os.close(fd)
        self.fds.clear()
        self.buf = None  # the mapping goes with the last view of it


def load_weights(cfg: FlintConfig, index: dict, log=print, experts: bool = True) -> Weights:  # noqa: C901, PLR0915
    """The weight banks and their views. experts=False (offload mode) loads bank 0 only, the non-expert weights:
    the layers' `experts` stay None and the routed experts come from an expert pool."""
    import mlx.core as mx

    t0 = time.perf_counter()
    plan = BankPlan(cfg, index)
    banks = [mx.zeros((plan.bank_sizes[0],), dtype=mx.uint8)]
    for _b, nl in enumerate(plan.expert_layers_per_bank if experts else ()):
        banks.append(mx.zeros((nl, cfg.n_experts, plan.expert_stride), dtype=mx.uint8))
    mx.eval(*banks)

    pending = 0

    def flush(force=False):
        nonlocal pending
        if force or pending >= FLUSH_BYTES:
            mx.eval(*banks)
            pending = 0

    reader = _Reader(index)
    for name, slot in plan.slots.items():
        src = mx.array(reader.read(name))
        banks[0][slot.offset : slot.offset + slot.nbytes] = src
        pending += slot.nbytes
        flush()
    flush(True)
    for layer in cfg.moe_layers if experts else ():
        bank, j = plan.expert_home[layer]
        for proj in EXPERT_PROJ:
            for s in QSUFFIX:
                f = plan.expert_fields[(proj, s)]
                src = mx.array(reader.read(f"model.layers.{layer}.mlp.switch_mlp.{proj}.{s}"))
                banks[bank][j, :, f.offset : f.offset + f.nbytes] = src.reshape(
                    cfg.n_experts, f.nbytes
                )
                pending += f.nbytes * cfg.n_experts
                flush()
        log(
            f"[load] experts of layer {layer} placed in bank {bank} ({time.perf_counter() - t0:.1f}s)"
        )
    flush(True)
    reader.close()

    u32 = [banks[0].view(mx.uint32)]
    bf16 = [banks[0].view(mx.bfloat16)]

    def contiguous_strides(shape):
        st, acc = [], 1
        for d in reversed(shape):
            st.append(acc)
            acc *= d
        return list(reversed(st))

    def view(name: str):
        slot = plan.slots[name]
        if slot.dtype == "U8":
            shape = (*slot.shape[:-1], slot.shape[-1] // 4)
            assert slot.offset % 4 == 0
            return mx.as_strided(u32[0], shape, contiguous_strides(shape), slot.offset // 4)
        assert slot.dtype == "BF16" and slot.offset % 2 == 0
        return mx.as_strided(bf16[0], slot.shape, contiguous_strides(slot.shape), slot.offset // 2)

    def qlin(prefix: str, bits: int = cfg.bits) -> QLinear:
        return QLinear(
            view(prefix + ".weight"),
            view(prefix + ".scales"),
            view(prefix + ".biases"),
            bits,
            cfg.group_size,
        )

    # MLX shape dimensions are int32 and as_strided flattens its input, so routed-expert views are taken
    # over each layer's [1, 256, EXPERT_STRIDE] slice of its bank (a view), never over a whole 10.9 GB bank
    # (2.7e9 uint32 elements would overflow). The load ends with an active-memory check (no hidden copy).
    layer_views: dict = {}

    def expert_view(layer: int, proj: str, s: str):
        bank, j = plan.expert_home[layer]
        if layer not in layer_views:
            sub = banks[bank][
                j : j + 1
            ]  # a slice is a view; integer indexing (banks[b][j]) would copy
            layer_views[layer] = (sub.view(mx.uint32), sub.view(mx.bfloat16))
        lu32, lbf16 = layer_views[layer]
        f = plan.expert_fields[(proj, s)]
        if f.dtype == "U8":
            shape = (cfg.n_experts, f.shape[0], f.shape[1] // 4)
            st = (plan.expert_stride // 4, f.shape[1] // 4, 1)
            return mx.as_strided(lu32, shape, st, f.offset // 4)
        shape = (cfg.n_experts, *f.shape)
        st = (plan.expert_stride // 2, f.shape[1], 1)
        return mx.as_strided(lbf16, shape, st, f.offset // 2)

    layers = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}"
        a = p + ".self_attn"
        lw = LayerWeights(
            index=i,
            sliding=i in cfg.sliding_layers,
            ln1=view(p + ".input_layernorm.weight"),
            ln2=view(p + ".post_attention_layernorm.weight"),
            q=qlin(a + ".q_proj"),
            k=qlin(a + ".k_proj"),
            v=qlin(a + ".v_proj"),
            o=qlin(a + ".o_proj"),
            q_norm=view(a + ".q_norm.weight"),
            k_norm=view(a + ".k_norm.weight"),
            sinks=view(a + ".softmax_offset"),
        )
        if i in cfg.dense_layers:
            lw.dense = tuple(qlin(f"{p}.mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj"))
        else:
            lw.router = view(p + ".mlp.gate.weight")
            lw.shared = tuple(
                qlin(f"{p}.mlp.shared_expert.{n}") for n in ("gate_proj", "up_proj", "down_proj")
            )
        if i not in cfg.dense_layers and experts:
            lw.experts = tuple(
                QLinear(
                    expert_view(i, n, "weight"),
                    expert_view(i, n, "scales"),
                    expert_view(i, n, "biases"),
                    cfg.bits,
                    cfg.group_size,
                )
                for n in EXPERT_PROJ
            )
        layers.append(lw)
    embed = qlin("model.embed_tokens", bits=cfg.embed_bits)
    lm_head = qlin("lm_head")
    norm = view("model.norm.weight")

    def arrays():
        for lw in layers:
            for v in (lw.ln1, lw.ln2, lw.q_norm, lw.k_norm, lw.sinks, lw.router):
                if v is not None:
                    yield v
            for q in (
                lw.q,
                lw.k,
                lw.v,
                lw.o,
                *(lw.dense or ()),
                *(lw.shared or ()),
                *(lw.experts or ()),
            ):
                yield from (q.w, q.s, q.b)
        for q in (embed, lm_head):
            yield from (q.w, q.s, q.b)
        yield norm

    mx.eval(*list(arrays()))
    mx.clear_cache()
    active = mx.get_active_memory()
    loaded = plan.total_bytes if experts else plan.bank_sizes[0]
    if active > loaded + 512 * 2**20:
        raise RuntimeError(
            f"weight views hold {active / 1e9:.2f} GB, more than the {loaded / 1e9:.2f} GB "
            f"banks: a view materialised a copy"
        )
    _selfcheck(cfg, index, layers, embed, lm_head)
    w = Weights(cfg, plan, banks, layers, embed, lm_head, norm, load_s=time.perf_counter() - t0)
    log(f"[load] {loaded / 1e9:.2f} GB in {len(banks)} banks, {w.load_s:.1f}s")
    return w


def _bytes_of(a) -> bytes:
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a).tobytes()


def _selfcheck(
    cfg: FlintConfig, index: dict, layers: list, embed: QLinear, lm_head: QLinear
) -> None:
    """Byte-compare sampled bank views with the safetensors source (layout/offset errors fail loudly)."""
    reader = _Reader(index)
    try:
        moe = cfg.moe_layers
        for layer in (moe[0], moe[len(moe) // 2], moe[-1]):
            lw = layers[layer]
            for proj, q in zip(("gate_proj", "up_proj", "down_proj"), lw.experts or ()):
                for suffix, view in (("weight", q.w), ("scales", q.s), ("biases", q.b)):
                    src = reader.read(f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}")
                    per = src.shape[0] // cfg.n_experts
                    for e in (0, 137, cfg.n_experts - 1):
                        if _bytes_of(view[e]) != src[e * per : (e + 1) * per].tobytes():
                            raise RuntimeError(
                                f"bank self-check: layer {layer} expert {e} {proj}.{suffix} differs from the checkpoint"
                            )
        for layer in (0, cfg.n_layers - 1):
            lw = layers[layer]
            for name, q in (("q_proj", lw.q), ("o_proj", lw.o)):
                src = reader.read(f"model.layers.{layer}.self_attn.{name}.weight")
                if _bytes_of(q.w) != src.tobytes():
                    raise RuntimeError(
                        f"bank self-check: layer {layer} {name} differs from the checkpoint"
                    )
        src = reader.read("lm_head.scales")
        if _bytes_of(lm_head.s) != src.tobytes():
            raise RuntimeError("bank self-check: lm_head.scales differs from the checkpoint")
        src = reader.read("model.embed_tokens.weight")
        if _bytes_of(embed.w[-1]) != src[-embed.w.shape[1] * 4 :].tobytes():
            raise RuntimeError("bank self-check: embedding last row differs from the checkpoint")
    finally:
        reader.close()
