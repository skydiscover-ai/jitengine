import sys
import types

from jitengine_qwen3_1_7b_m4pro.engine import source_patch


def _owner():
    mod = types.ModuleType("fake_mod")
    sys.modules["fake_mod"] = mod
    src = "def f(x):\n    y = x + 1\n    return y\n"
    exec(compile(src, "fake_mod.py", "exec"), mod.__dict__)

    class Owner:
        pass

    return mod, Owner


def test_hash_mismatch_leaves_stock(tmp_path):
    f = tmp_path / "m1.py"
    f.write_text("class C:\n    def f(self, x):\n        return x + 1\n")
    sys.path.insert(0, str(tmp_path))
    import m1

    assert "expected" in source_patch.apply(m1.C, "f", "0" * 64, [("x + 1", "x + 2")])
    assert m1.C().f(1) == 2


def test_edit_applies_when_hash_and_text_match(tmp_path):
    f = tmp_path / "m2.py"
    f.write_text("class C:\n    def f(self, x):\n        return x + 1\n")
    sys.path.insert(0, str(tmp_path))
    import m2

    digest = source_patch.source_sha256(m2.C.f)
    assert source_patch.apply(m2.C, "f", digest, [("x + 1", "x + 100")]) is None
    assert m2.C().f(1) == 101
    assert not hasattr(m2, "f")  # the module namespace is left as it was


def test_ambiguous_or_missing_edit_text_leaves_stock(tmp_path):
    f = tmp_path / "m3.py"
    f.write_text("class C:\n    def f(self, x):\n        return x + 1 + 1\n")
    sys.path.insert(0, str(tmp_path))
    import m3

    digest = source_patch.source_sha256(m3.C.f)
    assert "2 times" in source_patch.apply(m3.C, "f", digest, [("+ 1", "+ 5")])
    assert "0 times" in source_patch.apply(m3.C, "f", digest, [("nowhere", "x")])
    assert m3.C().f(1) == 3
