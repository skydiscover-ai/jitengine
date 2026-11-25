"""An engine's environment variables, as typed values. Each engine's own `envs.py` defines its variables, named after
the engine, with these: `EnvPath` for a path (where a file is), `EnvSwitch` for an on/off default a launcher that cannot
add flags sets (the flag, when given, wins)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvPath:
    """A path from the environment, with `~` expanded; None when unset or empty."""

    name: str

    def get(self) -> str | None:
        value = os.environ.get(self.name)
        return os.path.expanduser(value) if value else None


ON_OFF = {"on": True, "off": False}


@dataclass(frozen=True)
class EnvSwitch:
    """An on/off setting from the environment ("on" or "off"); None when unset or empty."""

    name: str

    def get(self) -> bool | None:
        value = os.environ.get(self.name)
        if not value:
            return None
        if value not in ON_OFF:
            raise ValueError(f"{self.name}={value!r}: expected on or off")
        return ON_OFF[value]
