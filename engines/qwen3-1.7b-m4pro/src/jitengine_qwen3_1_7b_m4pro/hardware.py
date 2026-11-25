"""Facts about this Mac for `doctor` and `/stats`. Nothing here imports MLX at module level."""

from __future__ import annotations

import platform
import re
import subprocess
import sys

TESTED = {"chip": "Apple M4 Pro", "memory_gib": 24}


def parse_pmset_batt(text: str) -> dict:
    """Power source and battery from `pmset -g batt` output."""
    m = re.search(r"Now drawing from '([^']+)'", text)
    pct = re.search(r"(\d+)%", text)
    return {
        "source": m.group(1) if m else None,
        "battery_percent": int(pct.group(1)) if pct else None,
    }


def parse_pmset_powermode(text: str) -> int | None:
    m = re.search(r"^\s*(?:powermode|lowpowermode)\s+(\d+)", text, re.M)
    return int(m.group(1)) if m else None


def power() -> dict:
    out = {}
    try:
        out.update(
            parse_pmset_batt(
                subprocess.run(
                    ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
                ).stdout
            )
        )
        out["power_mode"] = parse_pmset_powermode(
            subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
        )
    except OSError, subprocess.SubprocessError:
        pass
    return out


def tested_status(chip: str | None, memory_gib: float | None) -> str:
    """'tested' on the Mac the numbers were measured on, else 'untested'."""
    if (
        chip == TESTED["chip"]
        and memory_gib is not None
        and round(memory_gib) == TESTED["memory_gib"]
    ):
        return "tested"
    return "untested"


def base_report() -> dict:
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0] or None,
    }
