"""What this Mac can run: device facts, admission by capability, and the megakernel grid per device.

The engine is tuned on one device, the 20-core M4 Pro (applegpu_g16s), and admitted on any Apple silicon GPU that has
what it needs: Metal, an applegpu architecture, enough wired memory for the weights and a 4,096-token history
(config.resolve_context_limit), and a largest buffer that holds the largest weight bank. Whether the decode megakernel
itself runs on a GPU is decided at load by building it and proving its grid co-resident (loadcheck.py), never by a
device list: a GPU that fails a load check decodes with stock MLX ops (profile main) or is refused (kv8, or a pinned
grid / decode_backend "mk").

Facts. MLX's device_info of the GPU (architecture, device name, memory size, max recommended working set, max buffer
length); the GPU core count from the IORegistry (`ioreg -rc AGXAccelerator -d1` "gpu-core-count"; MLX reports none,
and it needs no root); the GPU wired limit (sysctl iogpu.wired_limit_mb; 0 = the OS default, the max recommended
working set).

Grid. G is the persistent kernel's threadgroup count (256 threads each). The tuned device runs G = 40 (two per core)
with no search. Any other GPU starts from G0 = clamp(4 * floor(2 * cores / 4), 16, 64): two threadgroups per core, a
multiple of 4 (the MoE block's shared team), inside the layout's domain (plan.MAX_SG = 512 SELP lists, 8 per
threadgroup, caps G at 64); 40 when the core count is unknown, so the engine never goes above 40 blind. The load
check probes G0, then G0 - 4, ... down to 16, and takes the first grid whose residency probe passes. A pinned grid
(mk_grid) is probed at that value only and is never searched or replaced: G changes the head's log-sum-exp reduction
(the per-simdgroup vocab partition), so logprob bits differ across grids while tokens, logits and top-20 ids do not,
and an engine must run the one grid it was asked for.

Tuning (opt-in). `doctor --tune` (tune.py) times every co-resident grid on this Mac and stores the fastest in a cache
file keyed by the device (architecture, name, GPU cores), the kernel sources (kernel_source_sha256, which covers every
file that renders the megakernel), the profile's history width, MLX and macOS. plan_grid tries a cached grid first,
before the core-count rule, on a device that is not in the tuned table; the tuned table and a pinned grid always win.
Without a cache entry (the default: nothing writes one unless --tune ran) the plan is the rule's. G +- 4 costs more than
1% on the M4 Pro, and nothing measured elsewhere says the rule's 2-per-core grid is the fastest on other GPUs.

Simulation (tests only). `sim` replaces the device facts and the wired limit with injected values and can fail the
compile stage or the probe at chosen grids, or inject one defect into the passes of the later load-check stages
(check_mutant, loadcheck.py). It is honoured only on an engine built with allow_fault_injection (refused otherwise),
and /health echoes it as `simulated`. The real GPU still runs every kernel: a simulated 32-core device has its G0 = 64
refused by the M4 Pro's real probe.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass

from jitengine_flint_m2.model.config import StartupError

# (architecture, device name, GPU cores) -> G, measured fastest there; no search
TUNED = {("applegpu_g16s", "Apple M4 Pro", 20): 40}
G_MIN, G_MAX, G_STEP = 16, 64, 4  # the megakernel's grid domain (runner.MegaKernel; plan.MAX_SG)
G_BLIND = 40  # G0 when the core count is unknown
ARCH_RE = re.compile(r"applegpu_g(\d+)([a-z]+)")
KNOWN_GENERATIONS = (13, 14, 15, 16, 17)  # g13 (M1) .. g17; others are admitted with a note
VERIFIED_GENERATIONS = (14, 16)  # bf16 + simdgroup_matrix custom kernels ran on g14g and g16s
SIM_KEYS = ("device", "wired_limit_mb", "fail_compile", "fail_probe_G", "check_mutant")
# loadcheck.py's test-only defects
CHECK_MUTANTS = ("wrong_expert", "pos_shift", "stale_history", "chain_bitflip", "verify_row")
GRID_CACHE_ENV = "FLINT_GRID_CACHE"  # the cache file (default ~/.cache/flint-engine/grid-v1.json)
GRID_CACHE_VERSION = 1
SIM_DEVICE_KEYS = (
    "architecture",
    "device_name",
    "gpu_cores",
    "memory_size",
    "max_recommended_working_set_size",
    "max_buffer_length",
)


# ------------------------------------------------------------------------------------------------------------ facts
@dataclass(frozen=True)
class Facts:
    architecture: str
    device_name: str
    gpu_cores: int | None
    memory_size: int
    max_recommended_working_set_size: int
    max_buffer_length: int
    wired_limit_mb: int  # sysctl iogpu.wired_limit_mb (0: not set, the OS default applies)
    metal: bool = True

    @property
    def wired_bytes(self) -> int:
        """The GPU wired limit the engine budgets against (iogpu.wired_limit_mb if set, else the working set)."""
        if self.wired_limit_mb > 0:
            return self.wired_limit_mb * 2**20
        return int(self.max_recommended_working_set_size)

    @property
    def key(self) -> tuple:
        return (self.architecture, self.device_name, self.gpu_cores)

    @property
    def tuned_grid(self) -> int | None:
        return TUNED.get(self.key)

    def summary(self) -> dict:
        d = asdict(self)
        d["wired_bytes"] = self.wired_bytes
        d["tuned"] = self.tuned_grid is not None
        return d


def gpu_core_count() -> int | None:
    """The GPU core count from the IORegistry (no root): `ioreg -rc AGXAccelerator -d1` "gpu-core-count"."""
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator", "-d1"], capture_output=True, text=True, timeout=10
        ).stdout
    except OSError, subprocess.SubprocessError:
        return None
    m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return int(m.group(1)) if m else None


class _RUsageV2(ctypes.Structure):  # struct rusage_info_v2 (<libproc.h>)
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_LIBPROC: list = []  # [libproc handle or None], loaded on first use


def phys_footprint(pid: int) -> int | None:
    """A process's physical footprint (proc_pid_rusage RUSAGE_INFO_V2 ri_phys_footprint, what footprint(1) and the
    benchmark's memory sampler count), or None when it cannot be read (the process is gone, or not macOS)."""
    if not _LIBPROC:
        try:
            _LIBPROC.append(ctypes.CDLL("/usr/lib/libproc.dylib"))
        except OSError:
            _LIBPROC.append(None)
    lib = _LIBPROC[0]
    info = _RUsageV2()
    if (
        lib is None
        or lib.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2), ctypes.byref(info)) != 0
    ):
        return None
    return int(info.ri_phys_footprint)


def sysctl_wired_limit_mb() -> int:
    try:
        v = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return max(0, int(v))
    except OSError, subprocess.SubprocessError, ValueError:
        return 0


def mlx_device_info() -> dict:
    """MLX's facts about the real GPU, whatever the default device is ({} when MLX has no Metal device)."""
    import mlx.core as mx

    try:
        if not mx.metal.is_available():
            return {}
        return dict(mx.device_info(mx.gpu))
    except Exception:  # noqa: BLE001  (no Metal backend in this MLX build)
        return {}


def read_facts(sim: Sim | None = None) -> Facts:
    """This Mac's facts; with a simulation, its injected device and wired limit replace the real ones."""
    info = mlx_device_info()
    real = dict(
        architecture=info.get("architecture", ""),
        device_name=info.get("device_name", ""),
        gpu_cores=gpu_core_count() if info else None,
        memory_size=int(info.get("memory_size", 0)),
        max_recommended_working_set_size=int(info.get("max_recommended_working_set_size", 0)),
        max_buffer_length=int(info.get("max_buffer_length", 0)),
    )
    wired = sysctl_wired_limit_mb()
    if sim is not None and sim.device is not None:
        real.update(sim.device)
    if sim is not None and sim.wired_limit_mb is not None:
        wired = sim.wired_limit_mb
    return Facts(**real, wired_limit_mb=wired, metal=bool(info))


# -------------------------------------------------------------------------------------------------------- admission
def admit(facts: Facts, bank_sizes) -> list:
    """Refuse (StartupError, before any weight) only for: no Metal, a non-applegpu architecture, or a weight bank larger
    than the GPU's largest buffer (memory is config.resolve_context_limit's). Returns the notes to log."""
    if not facts.metal:
        raise StartupError("MLX reports no Metal GPU: Flint decodes on an Apple silicon GPU only")
    m = ARCH_RE.fullmatch(facts.architecture or "")
    if m is None:
        raise StartupError(
            f"GPU architecture {facts.architecture!r} ({facts.device_name}) is not Apple silicon "
            f"(applegpu_g<generation><tier>); Flint runs on Apple silicon Macs only"
        )
    big = max(bank_sizes)
    if big > facts.max_buffer_length:
        raise StartupError(
            f"the largest weight bank is {big / 1e9:.2f} GB but this GPU's largest buffer "
            f"(max_buffer_length) is {facts.max_buffer_length / 1e9:.2f} GB"
        )
    cores = facts.gpu_cores if facts.gpu_cores is not None else "an unknown number of"
    if facts.tuned_grid is not None:
        return [
            f"device {facts.device_name} ({facts.architecture}, {cores} GPU cores): the tuned device"
        ]
    (arch, name, n), _ = next(iter(TUNED.items()))
    notes = [
        f"device {facts.device_name} ({facts.architecture}, {cores} GPU cores) is not the tuned device ({name}, "
        f"{arch}, {n} cores): the megakernel grid is sized from the core count and checked at load; if a check "
        f"fails, profile main decodes with stock MLX ops"
    ]
    gen = int(m.group(1))
    if gen not in KNOWN_GENERATIONS:
        notes.append(
            f"GPU generation {gen} is outside those this engine was built against "
            f"(g{KNOWN_GENERATIONS[0]}..g{KNOWN_GENERATIONS[-1]}): the compile and residency checks decide"
        )
    elif gen not in VERIFIED_GENERATIONS:
        notes.append(
            f"GPU generation {gen}: the megakernel's Metal features (bfloat, simdgroup matrices, device "
            f"atomics and fences) are expected but not yet verified on it; the load check decides"
        )
    return notes


# ------------------------------------------------------------------------------------------------------------- grid
@dataclass(frozen=True)
class GridPlan:
    candidates: tuple  # the grids the load check probes, in order (the first that passes is used)
    source: str  # tuned | rule | pinned | cache
    G0: int
    reason: str = ""  # why these grids (logged, and reported in /health)


def g0_rule(cores: int | None) -> int:
    """Two threadgroups per core, a multiple of 4 in 16..64; 40 when the core count is unknown."""
    if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
        return G_BLIND
    return min(G_MAX, max(G_MIN, G_STEP * (2 * cores // G_STEP)))


def check_grid(G) -> int:
    if isinstance(G, bool) or not isinstance(G, int) or G % G_STEP or not G_MIN <= G <= G_MAX:
        raise StartupError(
            f"mk_grid must be a multiple of {G_STEP} in {G_MIN}..{G_MAX} threadgroups, got {G!r}"
        )
    return G


def plan_grid(facts: Facts, pinned: int | None = None, cached: dict | None = None) -> GridPlan:
    """The one place the megakernel grid is chosen (the load check probes the candidates in order). Precedence: a pinned
    grid, the tuned table, a `doctor --tune` cache entry for this device and kernel source (read_grid_cache; its grid
    first, then the rule's search without it), the core-count rule."""
    if pinned is not None:
        G = check_grid(pinned)
        return GridPlan(
            (G,),
            "pinned",
            G,
            f"pinned: mk_grid={G}, probed at {G} only (refused if not co-resident)",
        )
    if facts.tuned_grid is not None:
        G = facts.tuned_grid
        return GridPlan(
            (G,),
            "tuned",
            G,
            f"tuned: {facts.device_name} ({facts.architecture}, {facts.gpu_cores} GPU cores) runs G {G}, "
            f"measured fastest there; no search",
        )
    cores = facts.gpu_cores
    G0 = g0_rule(cores)
    why = (
        f"2 x {cores} GPU cores -> G0 {G0}"
        if isinstance(cores, int) and cores >= 1
        else f"GPU core count unknown -> G0 {G0} (never above {G_BLIND} blind)"
    )
    rule = tuple(range(G0, G_MIN - 1, -G_STEP))
    if cached is not None:
        Gc = cached["G"]
        return GridPlan(
            (Gc, *(g for g in rule if g != Gc)),
            "cache",
            Gc,
            f"cache: `doctor --tune` timed the co-resident grids on this Mac ({cached.get('when', '?')}) "
            f"and G {Gc} was fastest; if it is refused, the rule's search ({why}, down to {G_MIN})",
        )
    return GridPlan(
        rule,
        "rule",
        G0,
        f"rule: {why} (a multiple of {G_STEP} in {G_MIN}..{G_MAX}); the first co-resident grid from G0 "
        f"down to {G_MIN} in steps of {G_STEP}",
    )


# ------------------------------------------------------------------------------------------------------- grid cache
def grid_cache_path() -> str:
    return os.environ.get(GRID_CACHE_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "flint-engine", "grid-v1.json"
    )


def grid_cache_key(facts: Facts, versions: dict, kv_bits: int) -> str:
    """What a timed grid depends on: the device, the kernel sources, the profile's history width, MLX and macOS."""
    return "|".join(
        [
            facts.architecture,
            facts.device_name,
            str(facts.gpu_cores),
            f"ksrc {versions.get('kernel_source_sha256')}",
            f"kv{kv_bits}",
            f"mlx {versions.get('mlx')}",
            f"macOS {versions.get('macos')}",
        ]
    )


def _read_cache_file(path: str) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
    except OSError, ValueError:
        return {}
    if (
        not isinstance(d, dict)
        or d.get("version") != GRID_CACHE_VERSION
        or not isinstance(d.get("entries"), dict)
    ):
        return {}
    return d


def read_grid_cache(
    facts: Facts, versions: dict, kv_bits: int, path: str | None = None
) -> dict | None:
    """The cached grid for this device and kernel source ({"G", "ms", "when", ...}), or None (no file, no entry, a
    malformed file or an out-of-domain grid: the cache is advisory and never refuses a start)."""
    entries = _read_cache_file(path or grid_cache_path()).get("entries", {})
    e = entries.get(grid_cache_key(facts, versions, kv_bits))
    if not isinstance(e, dict):
        return None
    G = e.get("G")
    if isinstance(G, bool) or not isinstance(G, int) or G % G_STEP or not G_MIN <= G <= G_MAX:
        return None
    return e


def write_grid_cache(
    facts: Facts, versions: dict, kv_bits: int, entry: dict, path: str | None = None
) -> str:
    """Store a timed grid under this device's key (other entries kept); returns the file written."""
    path = path or grid_cache_path()
    check_grid(entry["G"])
    d = _read_cache_file(path) or {"version": GRID_CACHE_VERSION, "entries": {}}
    key = grid_cache_key(facts, versions, kv_bits)
    d["entries"][key] = {**entry, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------------- simulation
@dataclass(frozen=True)
class Sim:
    device: dict | None = None
    wired_limit_mb: int | None = None
    fail_compile: bool = False
    fail_probe_G: object = ()  # a tuple of grids, or "all"
    # one of CHECK_MUTANTS: a defect injected into the later load-check stages
    check_mutant: str | None = None

    def probe_fails(self, G: int) -> bool:
        return self.fail_probe_G == "all" or G in self.fail_probe_G

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "wired_limit_mb": self.wired_limit_mb,
            "fail_compile": self.fail_compile,
            "fail_probe_G": self.fail_probe_G
            if self.fail_probe_G == "all"
            else list(self.fail_probe_G),
            "check_mutant": self.check_mutant,
        }


def _int(v, what: str, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise StartupError(f"sim {what} must be an integer >= {lo}, got {v!r}")
    return v


def _sim_device(dev) -> dict | None:
    if dev is None:
        return None
    if not isinstance(dev, dict):
        raise StartupError("sim device must be a dict")
    bad = sorted(set(dev) - set(SIM_DEVICE_KEYS))
    if bad:
        raise StartupError(f"unknown sim device keys {bad}; known: {', '.join(SIM_DEVICE_KEYS)}")
    dev = dict(dev)
    for k in ("architecture", "device_name"):
        if k in dev and not isinstance(dev[k], str):
            raise StartupError(f"sim device {k} must be a string")
    if dev.get("gpu_cores") is not None:
        _int(dev["gpu_cores"], "device gpu_cores", 1)
    for k in ("memory_size", "max_recommended_working_set_size", "max_buffer_length"):
        if k in dev:
            _int(dev[k], f"device {k}", 1)
    return dev


def parse_sim(sim, allow_fault_injection: bool) -> Sim | None:
    """Validate a simulation request; None when absent. Unknown keys are refused, never ignored."""
    if sim is None:
        return None
    if not allow_fault_injection:
        raise StartupError(
            "sim (device / wired-limit / load-check simulation) is a test hook: it needs "
            "allow_fault_injection"
        )
    if not isinstance(sim, dict):
        raise StartupError(f"sim must be a dict, got {type(sim).__name__}")
    bad = sorted(set(sim) - set(SIM_KEYS))
    if bad:
        raise StartupError(f"unknown sim keys {bad}; known: {', '.join(SIM_KEYS)}")
    wired = sim.get("wired_limit_mb")
    if wired is not None:
        _int(wired, "wired_limit_mb")
    fc = sim.get("fail_compile", False)
    if not isinstance(fc, bool):
        raise StartupError("sim fail_compile must be a boolean")
    fp = sim.get("fail_probe_G", ())
    if fp != "all":
        if not isinstance(fp, (list, tuple)):
            raise StartupError('sim fail_probe_G must be a list of grids or "all"')
        fp = tuple(_int(g, "fail_probe_G entry", 1) for g in fp)
    cm = sim.get("check_mutant")
    if cm is not None and cm not in CHECK_MUTANTS:
        raise StartupError(
            f"sim check_mutant must be one of {', '.join(CHECK_MUTANTS)}, got {cm!r}"
        )
    return Sim(_sim_device(sim.get("device")), wired, fc, fp, cm)
