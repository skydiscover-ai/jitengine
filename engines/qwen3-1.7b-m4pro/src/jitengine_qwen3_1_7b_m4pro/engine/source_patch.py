"""Replace one function of an installed module with an edited copy, in this process only.

The installed package is never written. The patch applies only when the function's source hashes
to the value recorded for the pinned mlx-lm commit and every old text appears exactly once. If not,
`apply` returns a reason and the stock function stays in place.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import textwrap


def source_sha256(fn) -> str:
    """Hash of the function's stock source; a patched function reports the hash it was patched from."""
    recorded = getattr(fn, "_jit_source_sha256", None)
    return recorded or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def apply(owner, name: str, expected_sha256: str, edits: list[tuple[str, str]]) -> str | None:
    """Patch `owner.<name>`; return None on success or the reason it was left as stock."""
    original = getattr(owner, name)
    if getattr(original, "_jit_source_sha256", None) == expected_sha256:
        return None  # already applied
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as e:
        return f"no source for {name}: {e}"
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != expected_sha256:
        return f"{name} source is {digest[:12]}, expected {expected_sha256[:12]}"
    for old, _ in edits:
        if source.count(old) != 1:
            return f"{name}: an edit's old text appears {source.count(old)} times"
    for old, new in edits:
        source = source.replace(old, new)
    module = sys.modules[original.__module__]
    namespace = module.__dict__
    code = compile(textwrap.dedent(source), f"<jitengine-patched {original.__qualname__}>", "exec")
    previous = namespace.get(name, _MISSING)
    exec(code, namespace)  # defines `name` in the module's globals so global lookups resolve
    patched = namespace[name]
    if previous is _MISSING:
        del namespace[name]
    else:
        namespace[name] = previous
    patched.__qualname__ = original.__qualname__
    patched._jit_source_sha256 = expected_sha256
    setattr(owner, name, patched)
    return None


_MISSING = object()
