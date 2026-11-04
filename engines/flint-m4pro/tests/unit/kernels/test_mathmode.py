"""Every Metal kernel this package compiles is pinned to math mode "safe" (kernels/mathmode.py), the mode its
bit-identity record was measured under, instead of MLX's default; an MLX that cannot take the pin is refused."""

from __future__ import annotations

import ast
import os

import pytest

from jitengine_flint_m4pro.model.config import StartupError

PIN = "METAL_COMPILE_OPTIONS"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HOME = os.path.join("src", "jitengine_flint_m4pro", "kernels", "mathmode.py")


def scan(root: str) -> list:
    """Every mx.fast.metal_kernel call under `root` (.py files, AST): [(file, line, problem or None)], plus any
    rebinding of the pinned constant outside mathmode.py."""
    out = []
    for dp, dns, fns in os.walk(root):
        # the package's sources only: not caches, dot-directories, or the build/ and *.egg-info a pip install
        # of this directory leaves beside them (CI installs the engine from here)
        dns[:] = sorted(
            d
            for d in dns
            if d not in ("__pycache__", "build", "dist")
            and not d.startswith(".")
            and not d.endswith(".egg-info")
        )
        for fn in sorted(f for f in fns if f.endswith(".py")):
            rel = os.path.relpath(os.path.join(dp, fn), root)
            with open(os.path.join(dp, fn), encoding="utf-8") as f:
                mod = ast.parse(f.read(), filename=rel)
            imports_pin = any(
                isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("kernels.mathmode")
                and any(a.name == PIN and a.asname is None for a in n.names)
                for n in ast.walk(mod)
            )
            imports_mod = any(
                isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("kernels")
                and any(a.name == "mathmode" and a.asname is None for a in n.names)
                for n in ast.walk(mod)
            )
            for n in ast.walk(mod):
                if rel != HOME and isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    if any(isinstance(t, ast.Name) and t.id == PIN for t in targets):
                        out.append((rel, n.lineno, f"{PIN} rebound outside mathmode.py"))
                if not (
                    isinstance(n, ast.Call)
                    and (
                        (isinstance(n.func, ast.Attribute) and n.func.attr == "metal_kernel")
                        or (isinstance(n.func, ast.Name) and n.func.id == "metal_kernel")
                    )
                ):
                    continue
                kw = {k.arg: k.value for k in n.keywords}
                v = kw.get("compile_options")
                if None in kw:
                    why = "a ** splat could override compile_options"
                elif v is None:
                    why = "no compile_options (MLX's default math mode)"
                elif isinstance(v, ast.Name) and v.id == PIN and (imports_pin or rel == HOME):
                    why = None
                elif (
                    isinstance(v, ast.Attribute)
                    and v.attr == PIN
                    and isinstance(v.value, ast.Name)
                    and v.value.id == "mathmode"
                    and imports_mod
                ):
                    why = None
                else:
                    why = f"compile_options={ast.unparse(v)} is not the pinned {PIN}"
                out.append((rel, n.lineno, why))
    return out


def test_every_metal_kernel_call_is_pinned():
    found = scan(ROOT)
    bad = [f"{f}:{ln}: {why}" for f, ln, why in found if why]
    assert not bad, "metal_kernel calls without the pinned math mode:\n  " + "\n  ".join(bad)
    sites = {f for f, _, _ in found}
    src = os.path.join("src", "jitengine_flint_m4pro")
    assert {os.path.join(src, "kernels", "runner.py"), os.path.join(src, "kv_cache.py")} <= sites, (
        sorted(sites)
    )


def test_the_scan_flags_an_unpinned_call(tmp_path):
    (tmp_path / "k.py").write_text(
        "import mlx.core as mx\n"
        "a = mx.fast.metal_kernel(name='a', input_names=[], output_names=[], source='')\n"
        "b = mx.fast.metal_kernel(name='b', input_names=[], output_names=[], source='',\n"
        "                         compile_options={'math_mode': 'fast'})\n"
    )
    whys = [why for _, _, why in scan(str(tmp_path))]
    assert len(whys) == 2 and all(whys), whys


def test_the_pin_and_this_mlx():
    from jitengine_flint_m4pro.engine.engine import toolchain_versions
    from jitengine_flint_m4pro.kernels import mathmode

    assert type(mathmode.METAL_COMPILE_OPTIONS) is dict and mathmode.METAL_COMPILE_OPTIONS == {
        "math_mode": "safe"
    }
    assert toolchain_versions()["metal_math_mode"] == "safe"
    mathmode.require()  # this MLX takes the pin


def test_an_mlx_that_cannot_take_the_pin_is_refused(fake_gpu, monkeypatch):
    """An MLX from before compile_options, or one that rejects "safe", is refused at construction, never decoded
    with the default math mode or sent to the stock-MLX fallback."""
    import mlx.core as mx

    from jitengine_flint_m4pro.engine.engine import Engine
    from jitengine_flint_m4pro.kernels import mathmode

    real = mx.fast.metal_kernel

    def no_options(name, input_names, output_names, source, header="", ensure_row_contiguous=True):
        return real(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
        )

    def rejects_safe(**kw):
        raise ValueError("[metal_kernel] Expected math_mode to be 'precise' or 'fast'.")

    for fake in (no_options, rejects_safe):
        monkeypatch.setattr(mx.fast, "metal_kernel", fake)
        with pytest.raises(RuntimeError, match="compile_options"):
            mathmode.require()
        with pytest.raises(StartupError, match="math mode"):
            Engine(fake_gpu.model, start=False, context_limit=4096)
