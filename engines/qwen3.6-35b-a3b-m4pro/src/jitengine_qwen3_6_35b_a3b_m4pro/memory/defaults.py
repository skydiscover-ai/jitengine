"""This build's memory defaults: the scratch and prefix-cache allowances it was measured with and its per-host default budget.
The planner itself (memory/budget.py) is shared with the other Qwen3.6 build."""

from __future__ import annotations

from jitengine_qwen3_6_35b_a3b.memory.budget import GIB, MIB

DEFAULT_SCRATCH_BYTES = 384 * MIB
DEFAULT_PREFIX_BYTES = 768 * MIB


def host_default_prefix_bytes() -> int:
    """Prefix-store budget for this host: 1152 MiB where unified memory is >= 40 GiB, else 768 MiB."""
    import os

    try:
        mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except ValueError, OSError:
        mem = 0
    # Sized so the store fills within about 18 short-prompt entries: memory must stop growing early in a run of
    # distinct prompts, and a larger store keeps growing past that point.
    return 1152 * MIB if mem >= 40 * GIB else DEFAULT_PREFIX_BYTES


def host_default_budget_bytes() -> int:
    """Default process budget for this host: 34 GiB where unified memory is >= 40 GiB (so the 4-bit checkpoint plans
    resident), else 18 GiB. MLX-free (sysconf)."""
    import os

    try:
        mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except ValueError, OSError:
        mem = 0
    return 34 * GIB if mem >= 40 * GIB else 18 * GIB
