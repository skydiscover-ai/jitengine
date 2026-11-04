"""The engine's arguments: the option defaults and their validation, and the versioned JSON
configuration with its named profiles, resolved into options with the origin of every key.

Precedence: built-in defaults < one packaged profile < one `--config` file < CLI flags; nothing is
read from the environment. Config-file paths resolve relative to the file, CLI paths to the cwd.
MLX-free, so the CLI works without Metal.
"""

from __future__ import annotations

import json
import math
import os
from importlib import resources

from jitengine_qwen3_6_35b_a3b_m2.memory.budget import (
    DEFAULT_ALLOCATOR_CACHE_BYTES,
    DEFAULT_HOST_BYTES,
    DEFAULT_PREFIX_BYTES,
    DEFAULT_SAFETY_BYTES,
    DEFAULT_SCRATCH_BYTES,
    GIB,
)

# The 24gb-static profile's values, used when `create_engine` is called with no options.
DEFAULTS = {
    "process_budget_bytes": 18 * GIB,
    "system_reserve_bytes": 6 * GIB,
    "mode": "auto",
    "memory_policy": "static",
    # KV memory is bounded by kv_budget_rows, so slots beyond it add only recurrent state.
    "max_slots": 8,
    # while rows decode, a prompt head up to this long still takes the layer-major pass
    "merge_decoding_max_tokens": 512,
    # several pending prompts in one layer-major pass, up to this many tokens in total
    "group_prefill": True,
    "group_prefill_max_tokens": 4096,
    # KV pool of min(max_slots, kv_budget_rows) rows of max_context_tokens (0 = max_slots rows).
    # Each admitted request is charged prompt + max_tokens; what does not fit waits in the queue.
    "kv_budget_rows": 4,
    # With nothing decoding, each layer's routed experts run as one gather over all prompt chunks;
    # attention, DeltaNet, router and shared expert stay per chunk. Same prefill state and logits.
    "merge_prefill_experts": True,
    "max_queued_requests": 16,
    "max_context_tokens": 8192,
    "prefill_chunk": 256,
    "prefix_cache_bytes": DEFAULT_PREFIX_BYTES,
    "allocator_cache_bytes": DEFAULT_ALLOCATOR_CACHE_BYTES,
    "scratch_bytes": DEFAULT_SCRATCH_BYTES,
    "host_bytes": DEFAULT_HOST_BYTES,
    "safety_bytes": DEFAULT_SAFETY_BYTES,
}
_BYTE_OPTIONS = (
    "process_budget_bytes",
    "system_reserve_bytes",
    "prefix_cache_bytes",
    "allocator_cache_bytes",
    "scratch_bytes",
    "host_bytes",
    "safety_bytes",
)
_INT_OPTIONS = (
    "kv_budget_rows",
    "max_slots",
    "max_queued_requests",
    "max_context_tokens",
    "prefill_chunk",
    "kv_reserve",
    "expert_io_workers",
    "expert_pool_seed_per_layer",
)
_OTHER_OPTIONS = (
    "mode",
    "memory_policy",
    "memory_budget_gb",
    "model_name",
    "log_requests",
    "expert_pack_path",
    "build_expert_pack",
    "expert_io_nocache",
    "expert_store_factory",
    "physical_bytes",
    "metal_bytes",
    "monitor",
    "monitor_interval_s",
    "policy_overrides",
    "speculative_gather",
    "layer_major_prefill",
    "fuse_projections",
    "merge_decoding_max_tokens",
    "group_prefill",
    "group_prefill_max_tokens",
    "merge_prefill_experts",
)


def resolve_options(options: dict) -> dict:
    """Validate and complete engine options (bytes as integers); raises ValueError."""
    unknown = set(options) - set(DEFAULTS) - set(_INT_OPTIONS) - set(_OTHER_OPTIONS)
    if unknown:
        raise ValueError(f"unknown engine options {sorted(unknown)}")
    out = _with_defaults(options)
    _apply_memory_budget_gb(options, out)
    out.pop("memory_budget_gb", None)
    _check_byte_options(out)
    _check_positive_ints(out)
    _check_modes(out)
    return out


def _with_defaults(options: dict) -> dict:
    """The defaults overlaid with every option that is not None."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in options.items() if v is not None})
    return out


def _apply_memory_budget_gb(options: dict, out: dict) -> None:
    """Convert the decimal `memory_budget_gb` into `process_budget_bytes`, refusing both at once."""
    if options.get("memory_budget_gb") is not None:
        if options.get("process_budget_bytes") is not None:
            raise ValueError("pass memory_budget_gb (decimal GB) or process_budget_bytes, not both")
        from decimal import Decimal

        gb = Decimal(str(options["memory_budget_gb"]))
        if not gb.is_finite() or gb <= 0:
            raise ValueError(f"memory_budget_gb must be a positive finite number, got {gb}")
        out["process_budget_bytes"] = int(gb * 10**9)


def _check_byte_options(out: dict) -> None:
    """Every byte option is a non-negative integer."""
    for k in _BYTE_OPTIONS:
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{k} must be a non-negative integer number of bytes, got {v!r}")


def _check_positive_ints(out: dict) -> None:
    """Slots, context tokens and prefill chunk are positive integers."""
    for k in ("max_slots", "max_context_tokens", "prefill_chunk"):
        if isinstance(out[k], bool) or not isinstance(out[k], int) or out[k] <= 0:
            raise ValueError(f"{k} must be a positive integer, got {out[k]!r}")


def _check_modes(out: dict) -> None:
    """The queue bound, the mode and the memory policy are valid together."""
    if not isinstance(out["max_queued_requests"], int) or out["max_queued_requests"] < 0:
        raise ValueError("max_queued_requests must be a non-negative integer")
    if out["mode"] not in ("auto", "resident", "offload"):
        raise ValueError(f"mode must be auto, resident or offload, got {out['mode']!r}")
    if out["memory_policy"] not in ("static", "adaptive"):
        raise ValueError(f"memory_policy must be static or adaptive, got {out['memory_policy']!r}")
    if out["memory_policy"] == "adaptive" and out["mode"] == "resident":
        raise ValueError(
            "memory_policy=adaptive needs mode=offload or auto (a resident model has "
            "no resizable expert pool)"
        )


def resolve_engine_options(cfg: dict) -> dict:
    """A resolved `profiles` configuration -> validated engine options."""
    return resolve_options(engine_options(cfg))


SCHEMA_VERSION = 1


# key -> (type, engine option name or None for server-only keys)
KEYS = {
    "memory_budget_bytes": (int, "process_budget_bytes"),
    "system_reserve_bytes": (int, "system_reserve_bytes"),
    "mode": (str, "mode"),
    "memory_policy": (str, "memory_policy"),
    "max_slots": (int, "max_slots"),
    "max_queued_requests": (int, "max_queued_requests"),
    "max_context_tokens": (int, "max_context_tokens"),
    "prefill_chunk_tokens": (int, "prefill_chunk"),
    "prefix_cache_bytes": (int, "prefix_cache_bytes"),
    "allocator_cache_bytes": (int, "allocator_cache_bytes"),
    "scratch_bytes": (int, "scratch_bytes"),
    "host_bytes": (int, "host_bytes"),
    "safety_bytes": (int, "safety_bytes"),
    "expert_pack_path": (str, "expert_pack_path"),
    "build_expert_pack": (bool, "build_expert_pack"),
    "kv_budget_rows": (int, "kv_budget_rows"),
    "model_path": (str, None),
    "host": (str, None),
    "port": (int, None),
}


PATH_KEYS = ("expert_pack_path", "model_path")


CHOICES = {"mode": ("auto", "resident", "offload"), "memory_policy": ("static", "adaptive")}


BUILTIN = {
    "memory_budget_bytes": 18 << 30,
    "system_reserve_bytes": 6 << 30,
    "mode": "auto",
    "memory_policy": "static",
    # What `serve` uses with no profile or config file; must agree with engine_options.DEFAULTS.
    "max_slots": 8,
    "max_queued_requests": 16,
    "max_context_tokens": 8192,
    "prefill_chunk_tokens": 256,
    "prefix_cache_bytes": 128 << 20,
    "allocator_cache_bytes": 128 << 20,
    "scratch_bytes": 1 << 30,
    "host_bytes": 768 << 20,
    "safety_bytes": 512 << 20,
    "build_expert_pack": False,
    "host": "127.0.0.1",
    "port": 8080,
}


class ConfigError(ValueError):
    """A profile, `--config` file or flag value the engine cannot use; the CLI prints its message
    and exits."""

    pass


def profile_names() -> list[str]:
    """The names of the packaged profiles (the `profiles/*.json` files), sorted."""
    files = resources.files("jitengine_qwen3_6_35b_a3b_m2").joinpath("profiles")
    return sorted(p.name[:-5] for p in files.iterdir() if p.name.endswith(".json"))


def load_profile(name: str) -> dict:
    """The packaged profile `name`, validated; a name that is not one raises `ConfigError`."""
    if name not in profile_names():
        raise ConfigError(f"unknown profile {name!r}; available: {', '.join(profile_names())}")
    raw = (
        resources.files("jitengine_qwen3_6_35b_a3b_m2")
        .joinpath("profiles", f"{name}.json")
        .read_text()
    )
    return validate(json.loads(raw), f"profile {name}", base_dir=None)


def load_config_file(path: str) -> dict:
    """The `--config` file at `path`, validated; relative paths in it resolve against its
    directory. An unreadable file, invalid JSON or a non-finite number raises `ConfigError`."""
    path = os.path.abspath(os.path.expanduser(path))
    try:
        with open(path) as f:
            raw = json.load(f, parse_constant=_reject_constant)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from None
    return validate(raw, path, base_dir=os.path.dirname(path))


def _reject_constant(name):
    raise ConfigError(f"non-finite number {name} is not allowed in a configuration")


def validate(raw: dict, origin: str, base_dir: str | None) -> dict:
    """Check a parsed configuration against `SCHEMA_VERSION` and `KEYS`, raising `ConfigError`
    that names `origin`; returns the keys without `schema_version`, path values made absolute
    against `base_dir` (the cwd when it is None)."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{origin}: configuration must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"{origin}: schema_version must be {SCHEMA_VERSION}, got {raw.get('schema_version')!r}"
        )
    out = {}
    for k, v in raw.items():
        if k == "schema_version":
            continue
        _check_value(k, v, origin)
        v = _absolute_path(k, v, base_dir)
        out[k] = v
    return out


def _check_value(k: str, v: object, origin: str) -> None:
    """Check one configuration value's key, type, sign and choice, raising `ConfigError` that names `origin`."""
    if k not in KEYS:
        raise ConfigError(f"{origin}: unknown key {k!r}")
    typ = KEYS[k][0]
    if typ is int and (isinstance(v, bool) or not isinstance(v, int)):
        if isinstance(v, float) and not math.isfinite(v):
            raise ConfigError(f"{origin}: {k} is not finite")
        raise ConfigError(f"{origin}: {k} must be an integer (bytes/tokens), got {v!r}")
    if typ is int and v < 0:
        raise ConfigError(f"{origin}: {k} must be >= 0")
    if typ is str and not isinstance(v, str):
        raise ConfigError(f"{origin}: {k} must be a string")
    if typ is bool and not isinstance(v, bool):
        raise ConfigError(f"{origin}: {k} must be true or false")
    if k in CHOICES and v not in CHOICES[k]:
        raise ConfigError(f"{origin}: {k} must be one of {CHOICES[k]}, got {v!r}")


def _absolute_path(k: str, v: object, base_dir: str | None) -> object:
    """Make a path value absolute against `base_dir` (the cwd when it is None)."""
    if k in PATH_KEYS:
        v = os.path.abspath(
            os.path.join(base_dir, os.path.expanduser(v)) if base_dir else os.path.expanduser(v)
        )
    return v


def resolve(
    profile: str | None = None, config_path: str | None = None, flags: dict | None = None
) -> tuple[dict, dict]:
    """(resolved config, {key: origin}). `flags` are already-converted CLI values (None = unset)."""
    cfg, origin = _builtin_layer()
    layers = _config_layers(profile, config_path, flags)
    _merge_layers(cfg, origin, layers)
    if cfg["memory_policy"] == "adaptive" and cfg["mode"] == "resident":
        raise ConfigError("memory_policy=adaptive needs mode offload or auto")
    return cfg, origin


def _builtin_layer() -> tuple:
    """The built-in defaults, each key's origin being `default`."""
    cfg, origin = dict(BUILTIN), {k: "default" for k in BUILTIN}
    return cfg, origin


def _config_layers(profile: str | None, config_path: str | None, flags: dict | None) -> list:
    """The configuration layers in precedence order: profile, config file, then the command-line flags."""
    layers = []
    if profile:
        layers.append((load_profile(profile), f"profile:{profile}"))
    if config_path:
        layers.append((load_config_file(config_path), f"config:{os.path.abspath(config_path)}"))
    cli = {k: v for k, v in (flags or {}).items() if v is not None}
    for k in cli:
        if k not in KEYS:
            raise ConfigError(f"unknown flag key {k!r}")
    if "model_path" in cli:
        cli["model_path"] = os.path.abspath(os.path.expanduser(cli["model_path"]))
    if "expert_pack_path" in cli:
        cli["expert_pack_path"] = os.path.abspath(os.path.expanduser(cli["expert_pack_path"]))
    layers.append(
        (
            validate({"schema_version": SCHEMA_VERSION, **cli}, "command line", None)
            if cli
            else {},
            "flag",
        )
    )
    return layers


def _merge_layers(cfg: dict, origin: dict, layers: list) -> None:
    """Overlay each layer on `cfg`, recording the layer that set each key."""
    for layer, name in layers:
        for k, v in layer.items():
            cfg[k] = v
            origin[k] = name


def engine_options(cfg: dict) -> dict:
    """The resolved configuration as `create_engine(**options)`."""
    out = {}
    for k, (_, opt) in KEYS.items():
        if opt is not None and k in cfg and cfg[k] is not None:
            out[opt] = cfg[k]
    return out
