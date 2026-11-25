"""A derived kernel's source: each edit applies exactly once, or the derivation fails loudly."""

import pytest

from jitengine_runtime.kernel_source import substitute


def test_each_edit_applies_once_in_turn():
    assert substitute("a + b; c", ("a", "x"), ("x + b", "y")) == "y; c"


@pytest.mark.parametrize("source", ["xyz", "a a"])
def test_an_edit_that_is_not_exactly_once_fails(source):
    with pytest.raises(ValueError, match="expects one occurrence"):
        substitute(source, ("a", "b"))
