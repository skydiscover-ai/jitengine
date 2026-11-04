"""Header-only checkpoint inventory and validation; imports no MLX and reads no tensor data.

Supports affine quantization with one group size and 2/3/4/5/6/8 bits per module. Anything else,
or a missing required tensor, raises `CheckpointError` instead of yielding a partial model.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field

LANG_PREFIX = "language_model."
VISION_PREFIXES = ("vision_tower.", "model.visual.", "visual.")
EXPERT_MARK = ".mlp.switch_mlp."
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
SUPPORTED_MODES = ("affine",)
HEADER_LIMIT = 100 << 20  # a safetensors header above 100 MiB is malformed, not large

_DTYPE_BYTES = {
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "U16": 2,
    "U8": 1,
    "I8": 1,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
}


class CheckpointError(ValueError):
    """The checkpoint is not one this engine can serve (the message says why)."""


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int
    mode: str = "affine"

    def __post_init__(self):
        if self.mode not in SUPPORTED_MODES:
            raise CheckpointError(f"quantization mode {self.mode!r} is not supported (only affine)")
        if int(self.bits) not in SUPPORTED_BITS:
            raise CheckpointError(
                f"{self.bits}-bit quantization is not supported ({SUPPORTED_BITS})"
            )
        if int(self.group_size) <= 0:
            raise CheckpointError(f"group_size must be positive, got {self.group_size}")

    def packed_cols(self, in_dims: int) -> int:
        """uint32 columns an affine-packed row of `in_dims` values occupies (exact or error)."""
        bits_total = int(in_dims) * int(self.bits)
        if bits_total % 32:
            raise CheckpointError(
                f"{in_dims} inputs x {self.bits} bits is not a whole number of uint32 words"
            )
        return bits_total // 32


def canonical(name: str) -> str:
    """Strip the vision-language conversion's `language_model.` prefix."""
    return name[len(LANG_PREFIX) :] if name.startswith(LANG_PREFIX) else name


def family(name: str) -> str:
    """`vision`, `mtp`, `expert`, `dense` (text) for a raw checkpoint tensor name."""
    if name.startswith(VISION_PREFIXES):
        return "vision"
    c = canonical(name)
    if c.startswith("mtp.") or ".mtp." in c or c.startswith("model.mtp"):
        return "mtp"
    if EXPERT_MARK in c:
        return "expert"
    return "dense"


def parse_quantization(raw_config: dict) -> tuple[QuantSpec | None, dict[str, QuantSpec]]:
    """(default spec or None for an unquantized checkpoint, {canonical module path: override}).

    A `False` override (MLX's "leave this module unquantized") is left out of the map; a
    group-size override that differs from the default raises.
    """
    q = raw_config.get("quantization")
    if q is None:
        q = raw_config.get("quantization_config")
    if not q:
        return None, {}
    if not isinstance(q, dict):
        raise CheckpointError("quantization block is not an object")
    try:
        default = QuantSpec(int(q["bits"]), int(q["group_size"]), str(q.get("mode", "affine")))
    except KeyError as exc:
        raise CheckpointError(f"quantization block lacks {exc}") from None
    overrides: dict[str, QuantSpec] = {}
    for k, v in q.items():
        if k in ("bits", "group_size", "mode", "quant_method", "linear_class", "quantization_mode"):
            continue
        if isinstance(v, dict):
            spec = QuantSpec(
                int(v.get("bits", default.bits)),
                int(v.get("group_size", default.group_size)),
                str(v.get("mode", default.mode)),
            )
            if spec.group_size != default.group_size:
                raise CheckpointError(
                    f"{k}: group_size override {spec.group_size} differs from the "
                    f"checkpoint default {default.group_size}; mixed group sizes "
                    f"are not supported"
                )
            overrides[canonical(k)] = spec
        elif v is False:
            continue
        elif not isinstance(v, (int, float, str)):
            raise CheckpointError(f"unrecognised quantization entry {k!r}: {v!r}")
    return default, overrides


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """(header dict without __metadata__, data offset). Bounded read; malformed headers raise."""
    with open(path, "rb") as f:
        raw_n = f.read(8)
        if len(raw_n) != 8:
            raise CheckpointError(f"{path}: truncated safetensors header")
        n = struct.unpack("<Q", raw_n)[0]
        if n <= 0 or n > HEADER_LIMIT:
            raise CheckpointError(f"{path}: implausible safetensors header length {n}")
        raw = f.read(n)
    if len(raw) != n:
        raise CheckpointError(f"{path}: truncated safetensors header")
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


@dataclass
class TensorInfo:
    name: str  # raw checkpoint name
    shard: str
    dtype: str
    shape: tuple
    nbytes: int
    offset: int  # absolute byte offset of the data in the shard


@dataclass
class Inventory:
    model_path: str
    model_type: str
    default_quant: QuantSpec | None
    overrides: dict
    tensors: dict = field(default_factory=dict)  # raw name -> TensorInfo
    family_bytes: dict = field(default_factory=dict)  # family -> bytes
    family_count: dict = field(default_factory=dict)
    module_quant: dict = field(default_factory=dict)  # canonical module -> QuantSpec (verified)
    bits_histogram: dict = field(default_factory=dict)  # bits -> bytes of quantized weights
    index_sha256: str = ""
    config_sha256: str = ""
    shards: list = field(default_factory=list)

    @property
    def dense_bytes(self) -> int:
        return int(self.family_bytes.get("dense", 0))

    @property
    def expert_bytes(self) -> int:
        return int(self.family_bytes.get("expert", 0))

    @property
    def text_bytes(self) -> int:
        return self.dense_bytes + self.expert_bytes

    @property
    def dropped_bytes(self) -> int:
        return sum(v for k, v in self.family_bytes.items() if k not in ("dense", "expert"))

    def quant_of(self, module: str) -> QuantSpec | None:
        return self.module_quant.get(canonical(module))

    def summary(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "quantization": None
            if self.default_quant is None
            else {
                "bits": self.default_quant.bits,
                "group_size": self.default_quant.group_size,
                "mode": self.default_quant.mode,
            },
            "overrides": {k: v.bits for k, v in sorted(self.overrides.items())},
            "override_bits": sorted({v.bits for v in self.overrides.values()}),
            "family_bytes": dict(sorted(self.family_bytes.items())),
            "family_tensors": dict(sorted(self.family_count.items())),
            "dense_bytes": self.dense_bytes,
            "expert_bytes": self.expert_bytes,
            "text_bytes": self.text_bytes,
            "dropped_bytes": self.dropped_bytes,
            "quantized_weight_bytes_by_bits": {
                str(k): v for k, v in sorted(self.bits_histogram.items())
            },
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "shards": self.shards,
        }


# Layer-local tensors the forward pass requires (canonical names relative to `model.layers.{i}.`).
_QLIN_GDN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
_QLIN_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_QLIN_MOE = (
    "gate",
    "shared_expert_gate",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
)
_EXPERT_QLIN = ("gate_proj", "up_proj", "down_proj")


def _required_layer_tensors(cfg, i: int) -> list[str]:
    p = f"model.layers.{i}."
    out = [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
    if cfg.is_attention(i):
        out += [p + "self_attn." + q for q in _QLIN_ATTN]
        out += [p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"]
    else:
        out += [p + "linear_attn." + q for q in _QLIN_GDN]
        out += [
            p + "linear_attn." + n for n in ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
        ]
    out += [p + "mlp." + q for q in _QLIN_MOE]
    out += [p + "mlp.switch_mlp." + q for q in _EXPERT_QLIN]
    return out


def _is_qmodule(module: str, names: set) -> bool:
    return (module + ".weight") in names and (module + ".scales") in names


def inventory(model_path: str, cfg=None, require_text: bool = True) -> Inventory:
    """Scan `model_path` (config + index + shard headers) and validate it. Raises CheckpointError."""
    model_path = os.path.abspath(os.path.expanduser(model_path))
    cfg_raw_bytes, raw_cfg, default, overrides, idx_raw, weight_map, shard_names = (
        _read_checkpoint_meta(model_path)
    )

    inv = Inventory(
        model_path=model_path,
        model_type=str(
            raw_cfg.get("model_type", raw_cfg.get("text_config", {}).get("model_type", ""))
        ),
        default_quant=default,
        overrides=overrides,
        index_sha256=hashlib.sha256(idx_raw).hexdigest() if idx_raw else "",
        config_sha256=hashlib.sha256(cfg_raw_bytes).hexdigest(),
    )
    for sn in shard_names:
        sp = os.path.join(model_path, sn)
        if not os.path.isfile(sp):
            raise CheckpointError(f"{model_path}: shard {sn} named by the index is missing")
        st = os.stat(sp)
        hdr, data_off = read_safetensors_header(sp)
        inv.shards.append({"name": sn, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        for name, meta in hdr.items():
            _register_tensor(name, meta, sn, st, data_off, weight_map, inv)
    _check_index_complete(weight_map, inv)

    for name, t in inv.tensors.items():
        fam = family(name)
        inv.family_bytes[fam] = inv.family_bytes.get(fam, 0) + t.nbytes
        inv.family_count[fam] = inv.family_count.get(fam, 0) + 1

    # every text module with .scales must agree with its declared spec
    text_names = {canonical(n) for n in inv.tensors if family(n) in ("dense", "expert")}
    by_canon = {canonical(n): t for n, t in inv.tensors.items() if family(n) in ("dense", "expert")}
    for cname in sorted(text_names):
        if not cname.endswith(".scales"):
            continue
        _check_quantized_module(cname, by_canon, default, overrides, inv)
    _check_overrides_used(overrides, inv, text_names)

    if require_text and cfg is not None:
        _check_required(inv, cfg, text_names)
    return inv


def _read_checkpoint_meta(model_path: str) -> tuple:
    """Read config.json and the shard index; return their bytes, the quantization spec and the shard list."""
    cfg_path = os.path.join(model_path, "config.json")
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(cfg_path):
        raise CheckpointError(f"{model_path}: no config.json")
    with open(cfg_path, "rb") as f:
        cfg_raw_bytes = f.read()
    raw_cfg = json.loads(cfg_raw_bytes)
    default, overrides = parse_quantization(raw_cfg)
    if os.path.isfile(idx_path):
        with open(idx_path, "rb") as f:
            idx_raw = f.read()
        weight_map = json.loads(idx_raw)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        idx_raw = b""
        shard_names = sorted(n for n in os.listdir(model_path) if n.endswith(".safetensors"))
        weight_map = None
    if not shard_names:
        raise CheckpointError(f"{model_path}: no safetensors shards")
    return cfg_raw_bytes, raw_cfg, default, overrides, idx_raw, weight_map, shard_names


def _register_tensor(
    name: str,
    meta: dict,
    sn: str,
    st: os.stat_result,
    data_off: int,
    weight_map: dict | None,
    inv: Inventory,
) -> None:
    """Validate one shard-header entry and record it in `inv.tensors`."""
    lo, hi = (int(x) for x in meta["data_offsets"])
    shape = tuple(int(d) for d in meta["shape"])
    dt = meta["dtype"]
    if dt not in _DTYPE_BYTES:
        raise CheckpointError(f"{name}: unsupported dtype {dt}")
    n_el = 1
    for d in shape:
        n_el *= d
    if n_el * _DTYPE_BYTES[dt] != hi - lo:
        raise CheckpointError(f"{name}: {shape} {dt} does not match its {hi - lo}-byte span")
    if data_off + hi > st.st_size:
        raise CheckpointError(f"{name}: data extends past the end of {sn} (truncated shard)")
    if weight_map is not None and weight_map.get(name) not in (None, sn):
        raise CheckpointError(f"{name}: index names {weight_map.get(name)}, found in {sn}")
    inv.tensors[name] = TensorInfo(name, sn, dt, shape, hi - lo, data_off + lo)


def _check_index_complete(weight_map: dict | None, inv: Inventory) -> None:
    """Raise if the index names a tensor that no shard holds."""
    if weight_map is not None:
        missing = [k for k in weight_map if k not in inv.tensors]
        if missing:
            raise CheckpointError(
                f"{len(missing)} tensors in the index are absent from the shards, "
                f"e.g. {missing[:3]}"
            )


def _check_quantized_module(
    cname: str, by_canon: dict, default: QuantSpec | None, overrides: dict, inv: Inventory
) -> None:
    """Check one `.scales` tensor against its declared spec and record the module in `inv`."""
    module = cname[: -len(".scales")]
    w = by_canon.get(module + ".weight")
    s = by_canon[cname]
    if w is None:
        raise CheckpointError(f"{module}: has scales but no weight")
    if default is None:
        raise CheckpointError(
            f"{module}: quantized tensor in a checkpoint without a quantization block"
        )
    if w.dtype != "U32":
        raise CheckpointError(f"{module}.weight: packed weight must be U32, got {w.dtype}")
    spec = overrides.get(module, default)
    in_dims = s.shape[-1] * spec.group_size
    want_cols = spec.packed_cols(in_dims)
    if w.shape[-1] != want_cols:
        implied = w.shape[-1] * 32 / in_dims
        raise CheckpointError(
            f"{module}: declared {spec.bits}-bit/g{spec.group_size} needs {want_cols} packed "
            f"columns for {in_dims} inputs, found {w.shape[-1]} (implies {implied:g} bits)"
        )
    if w.shape[:-1] != s.shape[:-1]:
        raise CheckpointError(f"{module}: weight rows {w.shape[:-1]} != scales rows {s.shape[:-1]}")
    b = by_canon.get(module + ".biases")
    if b is None:
        raise CheckpointError(f"{module}: affine quantization needs .biases")
    if b.shape != s.shape:
        raise CheckpointError(f"{module}: biases {b.shape} != scales {s.shape}")
    inv.module_quant[module] = spec
    inv.bits_histogram[spec.bits] = (
        inv.bits_histogram.get(spec.bits, 0) + w.nbytes + s.nbytes + b.nbytes
    )


def _check_overrides_used(overrides: dict, inv: Inventory, text_names: set) -> None:
    """Raise if a quantization override names a module that is not quantized in the checkpoint."""
    stale = sorted(
        m
        for m in overrides
        if m not in inv.module_quant and any(n.startswith(m + ".") for n in text_names)
    )
    if stale:
        raise CheckpointError(f"quantization overrides name unquantized modules: {stale[:3]}")


def _check_required(inv: Inventory, cfg, text_names: set) -> None:
    by_canon = _by_canonical_name(inv)
    _check_text_layers(cfg, text_names)
    _check_expert_geometry(inv, cfg, by_canon)


def _by_canonical_name(inv: Inventory) -> dict:
    """The inventory's tensors keyed by canonical name."""
    by_canon = {canonical(n): t for n, t in inv.tensors.items()}
    return by_canon


def _check_text_layers(cfg, text_names: set) -> None:
    """Raise unless every tensor the config's layers need is present and no extra layer exists."""
    need = ["model.embed_tokens.weight", "model.norm.weight"]
    for i in range(cfg.num_hidden_layers):
        need += _required_layer_tensors(cfg, i)
    missing = []
    for n in need:
        if n in text_names:
            continue
        if _is_qmodule(n, text_names):
            continue
        missing.append(n)
    if missing:
        raise CheckpointError(
            f"{len(missing)} required text tensors are missing, e.g. {missing[:4]}"
        )
    if cfg.num_hidden_layers and not any(
        n.startswith(f"model.layers.{cfg.num_hidden_layers - 1}.") for n in text_names
    ):
        raise CheckpointError("last decoder layer absent")
    extra_layers = [n for n in text_names if n.startswith(f"model.layers.{cfg.num_hidden_layers}.")]
    if extra_layers:
        raise CheckpointError(
            f"checkpoint has more decoder layers than config ({extra_layers[:2]})"
        )


def _check_expert_geometry(inv: Inventory, cfg, by_canon: dict) -> None:
    """Every layer's three stacked expert projections are quantized and have `num_experts` leading rows."""
    # expert geometry: every layer's three stacked projections have num_experts leading rows
    for i in range(cfg.num_hidden_layers):
        for q in _EXPERT_QLIN:
            m = f"model.layers.{i}.mlp.switch_mlp.{q}"
            spec = inv.module_quant.get(m)
            t = by_canon.get(m + ".weight")
            if t is None or spec is None:
                raise CheckpointError(f"{m}: routed experts must be quantized stacks")
            if t.shape[0] != cfg.num_experts:
                raise CheckpointError(f"{m}: {t.shape[0]} experts, config says {cfg.num_experts}")
