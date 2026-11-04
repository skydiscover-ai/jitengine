"""The checkpoint's multi-token-prediction head, used as a draft for greedy decoding.

One full-attention decoder layer with its own routed experts, fed by
    x = flatten(fc_embedding(rmsnorm(embed(token))) + fc_hidden(rmsnorm_fullwidth(h_pre)) per stream)
where h_pre is the main stack's output before the terminal mixer at the previous position, and read
out through its own mixer and the main model's lm_head, over a vocabulary subset unless `draft_vocab`
is "full" (`model.draft_vocab`). It predicts the token after next.

It changes nothing the engine emits: a draft only decides whether `Qwen4ExpForCausalLM.verify2`
computes one position or two in a pass, and every emitted token is the argmax of logits that are
bit-identical to the solo decode step's. So the head's own arithmetic is free: the bf16 tensors the
converter left unquantized are quantized 8-bit g64 here, and its cache starts at the request's first
decode position.

Weights come from `mtp.safetensors` (centered norms carry the +1 already, so it is taken off here
for the engine's classes, which add it). The head (~1.5 GB) is loaded only while the single-row
extension is live and lives in that memory: the engine charges its file plus `MTP_TRANSIENT_BYTES`
against the extension's slots and releases it before the extension shrinks.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from jitengine_qwen3_8_flash_next_m4pro import envs
from jitengine_qwen3_8_flash_next_m4pro.kvcache.kv_cache import KVSlot
from jitengine_qwen3_8_flash_next_m4pro.linear import qlinear_from
from jitengine_qwen3_8_flash_next_m4pro.model import model as L
from jitengine_qwen3_8_flash_next_m4pro.model.draft_vocab import draft_id

MTP_PATH = envs.MTP_PATH.get() or ""


def resolve(model_path=None) -> str:
    """Where the head lives: the MTP_PATH env var if set, else <model dir>/mtp.safetensors if it
    exists (where the `mtp` subcommand writes it), else nowhere, which leaves `present()` False."""
    global MTP_PATH
    env = envs.MTP_PATH.get()
    beside = (
        os.path.join(os.path.expanduser(str(model_path)), "mtp.safetensors") if model_path else ""
    )
    if env:
        MTP_PATH = env
    elif beside and os.path.exists(beside):
        MTP_PATH = beside
    else:
        MTP_PATH = ""
    return MTP_PATH


# The verify pass's transients beyond the head's file (its KV slot, two-position activations),
# charged with the file against the single-row extension.
MTP_TRANSIENT_BYTES = 256 << 20
# The head is on only when the single-row operating point holds at least this many expert slots
# (main pool plus single-row extension, before the head's share). A replay of the benchmark
# prompts' routing with the head in the extension puts break-even at about 9,400-10,100 slots;
# below it the head's slots cost more misses than the second position saves. On at 40 GB, off at
# 24 GB (6,139 slots, 6,820 with 16-bit pool metadata).
MTP_MIN_SLOTS = 10_500
_FOLDED = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
)
_BF16 = (
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
)


def resident_bytes() -> int:
    """What the plan charges for the head: its file."""
    n = os.path.getsize(MTP_PATH)
    return n


def present(cfg=None) -> bool:
    """The head file exists and (given `cfg`) is THIS model's: its fused projections, router and
    hyper-connection widths match the config (read from the safetensors header, no tensor loaded)."""
    if not MTP_PATH or not os.path.exists(MTP_PATH):
        return False
    if cfg is None:
        return True
    try:
        import json
        import struct

        with open(MTP_PATH, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hid, hc = int(cfg.hidden_size), int(cfg.hc_count)
        return (
            hdr["mtp.fc_embedding.scales"]["shape"][0] == hid
            and hdr["mtp.pre_fc_norm_hidden.weight"]["shape"][0] == hid * hc
            and hdr["mtp.layers.0.mlp.gate.weight"]["shape"] == [int(cfg.num_experts), hid]
        )
    except Exception:  # noqa: BLE001 -- an unreadable or foreign file is simply not this head
        return False


class MTPHead:
    def __init__(self, cfg, compute_dtype, lm_head, embed):
        w = mx.load(MTP_PATH)
        for k in list(w):
            if k.endswith(_FOLDED):
                w[k] = (w[k].astype(mx.float32) - 1.0).astype(w[k].dtype)
        for name in _BF16:
            q, s, b = mx.quantize(w[name + ".weight"], group_size=64, bits=8)
            w[name + ".weight"], w[name + ".scales"], w[name + ".biases"] = q, s, b
            cfg.q_overrides[name] = (8, 64)
        for k in list(w):
            if k.endswith(".scales"):
                cfg.q_overrides.setdefault(k[: -len(".scales")], (4, 64))
        mx.eval(w)
        self.cfg, self.dt = cfg, compute_dtype
        p = "mtp.layers.0."
        self.attn = L.Qwen4ExpTextAttention(cfg, w, p + "self_attn.", compute_dtype)
        self.hc_a = L.Qwen4ExpTextGatedResidual(
            cfg, w, p + "attn_hyper_connection.", compute_dtype=compute_dtype
        )
        self.hc_m = L.Qwen4ExpTextGatedResidual(
            cfg, w, p + "mlp_hyper_connection.", compute_dtype=compute_dtype
        )
        self.mixer = L.Qwen4ExpTextGatedResidual(
            cfg, w, "mtp.hyper_connection_mixer.", with_inject=False, compute_dtype=compute_dtype
        )
        self.fc_e = qlinear_from(w, "mtp.fc_embedding", cfg)
        self.fc_h = qlinear_from(w, "mtp.fc_hidden", cfg)
        self.gate = qlinear_from(w, p + "mlp.gate", cfg)
        self.shg = qlinear_from(w, p + "mlp.shared_expert_gate", cfg)
        self.sgp = qlinear_from(w, p + "mlp.shared_expert.gate_proj", cfg)
        self.sup = qlinear_from(w, p + "mlp.shared_expert.up_proj", cfg)
        self.sdp = qlinear_from(w, p + "mlp.shared_expert.down_proj", cfg)
        self.sw = {
            n: (
                w[f"{p}mlp.switch_mlp.{n}.weight"],
                w[f"{p}mlp.switch_mlp.{n}.scales"],
                w[f"{p}mlp.switch_mlp.{n}.biases"],
            )
            for n in ("gate_proj", "up_proj", "down_proj")
        }
        self.nw_e = w["mtp.pre_fc_norm_embedding.weight"]
        self.nw_h = w["mtp.pre_fc_norm_hidden.weight"]
        self.eps = float(cfg.rms_norm_eps)
        self.lm_head, self.embed = lm_head, embed
        # the draft's readout: a draft_vocab.SubsetReadout the engine installs (`_mtp_load`), or None
        # for the argmax over every lm_head row
        self.readout = None
        self.nbytes = sum(int(a.nbytes) for a in w.values())
        self._w = w

    def linears(self) -> list:
        return (
            self.attn.linears()
            + self.hc_a.linears()
            + self.hc_m.linears()
            + self.mixer.linears()
            + [self.fc_e, self.fc_h, self.sgp, self.sup, self.sdp]
        )

    def new_state(self, capacity: int):
        return {"slot": KVSlot(self.cfg, max(16, capacity), dtype=self.dt), "n": 0}

    def _moe(self, x):
        cfg = self.cfg
        if self.sw is None:
            return mx.sigmoid(self.shg(x)) * self.sdp(nn.silu(self.sgp(x)) * self.sup(x))
        k = cfg.num_experts_per_tok
        g = mx.softmax(self.gate(x).astype(mx.float32), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
        sc = mx.take_along_axis(g, inds, axis=-1)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        # bf16 activations: the experts' scales/biases are bf16, and fp32 activations would make the
        # stock gather_qmm upcast every expert's metadata on each call (draft arithmetic is free)
        xe = mx.expand_dims(x.astype(mx.bfloat16), (-2, -3))

        def gq(z, n):
            wq, s, b = self.sw[n]
            return mx.gather_qmm(
                z, wq, s, b, rhs_indices=inds, transpose=True, group_size=64, bits=4
            )

        y = gq(nn.silu(gq(xe, "gate_proj")) * gq(xe, "up_proj"), "down_proj").squeeze(-2)
        routed = (y * sc[..., None]).sum(axis=-2).astype(x.dtype)
        return routed + mx.sigmoid(self.shg(x)) * self.sdp(nn.silu(self.sgp(x)) * self.sup(x))

    def draft(self, st: dict, h_pre: mx.array, tokens: list) -> mx.array:
        """Consume entries (h_pre[j], tokens[j]) at the head's next cache positions and return the
        argmax draft of the last one (lazy int32 [1]). h_pre is [n, hc*hid]."""
        cfg = self.cfg
        n = len(tokens)
        hc, hid = cfg.hc_count, cfg.hidden_size
        tok = (
            mx.concatenate([mx.array(t, dtype=mx.int32).reshape(1) for t in tokens])
            if any(isinstance(t, mx.array) for t in tokens)
            else mx.array(tokens, dtype=mx.int32)
        )
        e = self.embed.lookup(tok.reshape(1, n)).astype(self.dt)
        ef = self.fc_e(mx.fast.rms_norm(e, self.nw_e, self.eps))
        hh = mx.fast.rms_norm(h_pre.reshape(1, n, hc * hid).astype(self.dt), self.nw_h, self.eps)
        x = (mx.expand_dims(ef, -2) + self.fc_h(hh.reshape(1, n, hc, hid))).reshape(1, n, hc * hid)
        m1, inj1 = self.hc_a(x)
        ao = self.attn(m1, st["slot"], st["n"])
        st["n"] += n
        x = x + (mx.expand_dims(ao, -2) * mx.expand_dims(inj1, -1)).reshape(x.shape)
        m2, inj2 = self.hc_m(x)
        x = x + (mx.expand_dims(self._moe(m2), -2) * mx.expand_dims(inj2, -1)).reshape(x.shape)
        mixed, _ = self.mixer(x[:, -1:, :])
        return draft_id(self.lm_head, self.readout, mixed)
