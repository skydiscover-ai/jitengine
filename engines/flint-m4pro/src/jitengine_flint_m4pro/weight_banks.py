"""Weight-bank layout: where every checkpoint tensor lives after the load-time repack.

Three banks (the megakernel can bind them as three buffers; each is below the 30 GB Metal buffer cap):

  bank 0  non-expert weights, 1-D bytes: per-layer blocks in layer order (norms, q/k/v/o projections,
          q/k norms, sinks, then the dense MLP or the router + shared expert), then the final norm,
          the lm_head and the 8-bit embedding. Every tensor starts on a 256-byte boundary.
  bank 1, bank 2  routed experts, [layers_in_bank][256 experts][EXPERT_STRIDE] bytes. One expert is one
          contiguous 1,769,472-byte record: gate (w | scales | biases), up (...), down (...), so a
          routed expert is a single contiguous read.

`descriptor_table()` exports the byte offsets as integer arrays for kernels that address the banks
directly; the MLX graph path takes typed strided views into the same bytes (loader.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jitengine_flint_m4pro.model.config import FlintConfig

ALIGN = 256
N_EXPERT_BANKS = 2
EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")
QSUFFIX = ("weight", "scales", "biases")


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class Slot:
    bank: int
    offset: int  # bytes from the start of the bank
    nbytes: int
    dtype: str  # safetensors dtype of the source tensor
    shape: tuple


@dataclass(frozen=True)
class ExpertField:
    offset: int  # bytes from the start of one expert record
    nbytes: int
    dtype: str
    shape: tuple  # per-expert shape


class BankPlan:
    def __init__(self, cfg: FlintConfig, index: dict):
        self.cfg = cfg
        self.slots: dict[str, Slot] = {}
        off = 0

        def put(name: str) -> None:
            nonlocal off
            _, dt, shape, _, nbytes = index[name]
            self.slots[name] = Slot(0, off, nbytes, dt, tuple(shape))
            off = _align(off + nbytes)

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}"
            names = [p + ".input_layernorm.weight", p + ".post_attention_layernorm.weight"]
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names += [f"{p}.self_attn.{proj}.{s}" for s in QSUFFIX]
            names += [
                p + ".self_attn.q_norm.weight",
                p + ".self_attn.k_norm.weight",
                p + ".self_attn.softmax_offset",
            ]
            if i in cfg.dense_layers:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    names += [f"{p}.mlp.{proj}.{s}" for s in QSUFFIX]
            else:
                names.append(p + ".mlp.gate.weight")
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    names += [f"{p}.mlp.shared_expert.{proj}.{s}" for s in QSUFFIX]
            for n in names:
                put(n)
        put("model.norm.weight")
        for s in QSUFFIX:
            put(f"lm_head.{s}")
        for s in QSUFFIX:
            put(f"model.embed_tokens.{s}")
        self.bank_sizes = [off]

        # one expert record: gate | up | down, each weight | scales | biases
        first = cfg.moe_layers[0]
        self.expert_fields: dict[tuple, ExpertField] = {}
        eoff = 0
        for proj in EXPERT_PROJ:
            for s in QSUFFIX:
                _, dt, shape, _, nbytes = index[f"model.layers.{first}.mlp.switch_mlp.{proj}.{s}"]
                per = nbytes // cfg.n_experts
                self.expert_fields[(proj, s)] = ExpertField(eoff, per, dt, tuple(shape[1:]))
                eoff += per
        self.expert_stride = _align(eoff)
        moe = list(cfg.moe_layers)
        per_bank = -(-len(moe) // N_EXPERT_BANKS)
        self.expert_home: dict[int, tuple] = {}
        for b in range(N_EXPERT_BANKS):
            layers = moe[b * per_bank : (b + 1) * per_bank]
            for j, layer in enumerate(layers):
                self.expert_home[layer] = (1 + b, j)
            self.bank_sizes.append(len(layers) * cfg.n_experts * self.expert_stride)
        self.expert_layers_per_bank = [
            sum(1 for v in self.expert_home.values() if v[0] == 1 + b)
            for b in range(N_EXPERT_BANKS)
        ]

    @property
    def total_bytes(self) -> int:
        return sum(self.bank_sizes)

    def expert_offset(self, layer: int, proj: str, suffix: str) -> tuple:
        """(bank, byte offset of expert 0's field) for a routed-expert tensor."""
        bank, j = self.expert_home[layer]
        f = self.expert_fields[(proj, suffix)]
        return bank, j * self.cfg.n_experts * self.expert_stride + f.offset

    def descriptor_table(self) -> dict:
        """Integer offset tables for direct-addressing kernels.

        layer_offsets [n_layers, n_fields] int64: byte offset in bank 0 of every non-expert field,
        -1 where the layer has no such field; expert_base [n_layers, 2] int64: (bank, byte offset of
        expert 0) for routed layers, (-1, -1) for dense layers; plus the per-record field offsets.
        """
        cfg = self.cfg
        fields = sorted(
            {
                n.split(f"model.layers.{i}.", 1)[1]
                for i in range(cfg.n_layers)
                for n in self.slots
                if n.startswith(f"model.layers.{i}.")
            }
        )
        tab = np.full((cfg.n_layers, len(fields)), -1, dtype=np.int64)
        for i in range(cfg.n_layers):
            for k, f in enumerate(fields):
                s = self.slots.get(f"model.layers.{i}.{f}")
                if s is not None:
                    tab[i, k] = s.offset
        base = np.full((cfg.n_layers, 2), -1, dtype=np.int64)
        for layer in cfg.moe_layers:
            bank, off = self.expert_offset(layer, "gate_proj", "weight")
            base[layer] = (bank, off)
        return {
            "layer_fields": fields,
            "layer_offsets": tab,
            "expert_base": base,
            "expert_stride": self.expert_stride,
            "expert_fields": {f"{p}.{s}": v.offset for (p, s), v in self.expert_fields.items()},
            "globals": {
                n: self.slots[n].offset for n in self.slots if not n.startswith("model.layers.")
            },
            "bank_sizes": list(self.bank_sizes),
        }
