"""This build's memory defaults: the scratch and prefix-cache allowances it was measured with.
The planner itself (memory/budget.py) is shared with the other Qwen3.6 build."""

from __future__ import annotations

from jitengine_qwen3_6_35b_a3b.memory.budget import GIB, MIB

DEFAULT_SCRATCH_BYTES = 1 * GIB
DEFAULT_PREFIX_BYTES = 128 * MIB
