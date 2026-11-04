"""engine/engine.py's table of the state its mixins share matches the code."""

import ast
import importlib.util
import os
import re

PKG = "jitengine_qwen3_6_35b_a3b_m4pro"


def _parse(module: str) -> ast.Module:
    root = os.path.dirname(os.path.dirname(importlib.util.find_spec(PKG).origin))
    return ast.parse(open(os.path.join(root, *module.split(".")) + ".py").read())


def _classes() -> list:
    """[(name, ClassDef)]: `Engine` and each base it lists, found through engine.py's imports."""
    tree = _parse(f"{PKG}.engine.engine")
    imported = {
        al.asname or al.name: node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for al in node.names
    }
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine")
    out = [("Engine", engine)]
    for base in engine.bases:
        body = _parse(imported[base.id]).body
        out.append(
            (base.id, next(n for n in body if isinstance(n, ast.ClassDef) and n.name == base.id))
        )
    return out


def _writers() -> dict:
    """{attribute: {class: {method}}}: every `self.<attribute>` a class assigns or deletes."""
    out: dict = {}
    for name, cls in _classes():
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    out.setdefault(node.attr, {}).setdefault(name, set()).add(fn.name)
    return out


def _table() -> dict:
    """{attribute: (owner, other writers)} from the docstring's table."""
    doc = ast.get_docstring(_parse(f"{PKG}.engine.engine"))
    rows = {}
    for line in doc.splitlines():
        m = re.fullmatch(r"\| (`.+`) \| (\w+Mixin) \|(.*)\|", line)
        if m:
            for attr in re.findall(r"`(\w+)`", m.group(1)):
                rows[attr] = (m.group(2), set(re.findall(r"\w+Mixin", m.group(3))))
    return rows


def test_the_shared_state_table_names_every_writer():
    """Every attribute two classes write is in the table, with its owner and every other writer;
    `Engine` itself writes it only in an `init_*` step."""
    shared = {a: w for a, w in _writers().items() if len(w) > 1}
    table = _table()
    assert set(table) == set(shared), sorted(set(table) ^ set(shared))
    for attr, (owner, others) in table.items():
        written = shared[attr]
        assert all(m.startswith("init_") for m in written.get("Engine", ())), (attr, written)
        assert {owner, *others} == set(written) - {"Engine"}, (attr, written)
