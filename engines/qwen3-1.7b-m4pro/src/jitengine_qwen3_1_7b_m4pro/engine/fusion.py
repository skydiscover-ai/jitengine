"""Load-time fusion of Qwen3's q|k|v and gate|up quantized projections.

The packed weights, scales and biases of sibling projections are joined along the output axis, so
each output row is computed from the same bytes as before. Decode of 1 to 16 rows runs the fused
matmul (the exact kernel from 13 rows, stock below that). Other shapes slice the fused arrays back
into the original projections and run stock `quantized_matmul`: stock splits K differently for a
joined N, so a fused call above 16 rows would differ from the unfused one by about 1 ulp.
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import mlx.nn as nn

FUSED_MAX_ROWS = 16
SEGMENTS = "_jit_segments"
# sha256 of the stock `qwen3.Attention.__call__` and `qwen3.MLP.__call__` source at the pinned
# mlx-lm commit: the replacements below are copies of those bodies and are used only if they match
ATTENTION_SHA256 = "789af9d8868081c15ebcd8afbb603a57739aefb6e1a17d4817d660a08b34033e"
MLP_SHA256 = "dc08d40bfdf18cf60619caad43c76a1094f3de1b14d2c53b9e2228fa698bc40a"


def _fuse(linears):
    f = linears[0]
    for lin in linears:
        if not isinstance(lin, nn.QuantizedLinear) or "bias" in lin:
            return None
        if (lin.group_size, lin.bits, lin.mode) != (f.group_size, f.bits, f.mode):
            return None
        if lin.weight.shape[1] != f.weight.shape[1]:
            return None
    n = sum(lin.weight.shape[0] for lin in linears)
    in_dims = f.weight.shape[1] * 32 // f.bits
    out = nn.QuantizedLinear(
        in_dims, n, bias=False, group_size=f.group_size, bits=f.bits, mode=f.mode
    )
    out.weight = mx.concatenate([lin.weight for lin in linears], axis=0)
    out.scales = mx.concatenate([lin.scales for lin in linears], axis=0)
    if "biases" in f:
        out.biases = mx.concatenate([lin.biases for lin in linears], axis=0)
    mx.eval(out.parameters())
    object.__setattr__(out, SEGMENTS, tuple(lin.weight.shape[0] for lin in linears))
    return out


def _rows(lin, x, a, b):
    """Output rows [a:b) of a fused linear, computed as the original projection."""
    return mx.quantized_matmul(
        x,
        lin.weight[a:b],
        lin.scales[a:b],
        lin.biases[a:b] if "biases" in lin else None,
        transpose=True,
        group_size=lin.group_size,
        bits=lin.bits,
        mode=lin.mode,
    )


def _split(lin, x, bounds):
    return [_rows(lin, x, a, b) for a, b in itertools.pairwise(bounds)]


def _fused_ok(x) -> bool:
    return x.shape[-2] == 1 and x.shape[0] <= FUSED_MAX_ROWS


_orig = {}


def _attn_call(self, x, mask=None, cache=None):
    if "qkv_proj" not in self:
        return _orig["attn"](self, x, mask, cache)
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    nq = self.n_heads * self.head_dim_
    nk = self.n_kv_heads * self.head_dim_
    if _fused_ok(x):
        qkv = self.qkv_proj(x)
        queries, keys, values = qkv[..., :nq], qkv[..., nq : nq + nk], qkv[..., nq + nk :]
    else:
        queries, keys, values = _split(self.qkv_proj, x, (0, nq, nq + nk, nq + 2 * nk))
    queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def _mlp_call(self, x):
    if "gate_up_proj" not in self:
        return _orig["mlp"](self, x)
    h = self.hidden_dim_
    if _fused_ok(x):
        gu = self.gate_up_proj(x)
        gate, up = gu[..., :h], gu[..., h:]
    else:
        gate, up = _split(self.gate_up_proj, x, (0, h, 2 * h))
    return self.down_proj(_orig["swiglu"](gate, up))


def guard() -> str | None:
    """None when the two methods fusion replaces are the pinned source, else why it must not run."""
    from mlx_lm.models import qwen3

    from jitengine_qwen3_1_7b_m4pro.engine.source_patch import source_sha256

    for label, fn, expected in (
        ("qwen3.Attention.__call__", qwen3.Attention.__call__, ATTENTION_SHA256),
        ("qwen3.MLP.__call__", qwen3.MLP.__call__, MLP_SHA256),
    ):
        try:
            found = source_sha256(fn)
        except (OSError, TypeError) as e:
            return f"no source for {label}: {e}"
        if found != expected:
            return f"{label} is not the pinned mlx-lm source (sha256 {found[:12]}...)"
    return None


def fuse_model(model) -> bool:
    """Fuse in place; True when any layer was fused. A model that is not a quantized Qwen3 is left alone."""
    if getattr(model, "model_type", None) != "qwen3":
        return False
    from mlx_lm.models import qwen3
    from mlx_lm.models.activations import swiglu

    if not _orig and guard() is not None:
        return False
    if not _orig:
        _orig.update(attn=qwen3.Attention.__call__, mlp=qwen3.MLP.__call__, swiglu=swiglu)
    patched = False
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        qkv = _fuse([a.q_proj, a.k_proj, a.v_proj])
        if qkv is not None:
            a.head_dim_ = a.q_proj.weight.shape[0] // a.n_heads
            a.qkv_proj = qkv
            del a["q_proj"], a["k_proj"], a["v_proj"]
            patched = True
        gu = _fuse([m.gate_proj, m.up_proj])
        if gu is not None:
            m.hidden_dim_ = m.gate_proj.weight.shape[0]
            m.gate_up_proj = gu
            del m["gate_proj"], m["up_proj"]
            patched = True
    if patched:
        qwen3.Attention.__call__ = _attn_call
        qwen3.MLP.__call__ = _mlp_call
    return patched
