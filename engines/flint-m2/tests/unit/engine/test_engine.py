"""The kernel-source hash /health reports, off a Mac."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jitengine_flint_m2.engine.engine import kernel_source_sha256
from jitengine_flint_m2.kernels import kv8_fold, kv8_src, metal_src, vmoe_src


def test_kernel_source_hash_reads_every_source():
    """A wrong path to the sources would hash too little; the empty input gives e3b0c44298fc1c14."""
    kernels = Path(metal_src.__file__).parent
    modules = (metal_src, vmoe_src, kv8_src, kv8_fold)
    sources = [Path(m.__file__) for m in modules] + sorted(kernels.glob("*.metal"))
    assert len(sources) == 11
    assert all(p.stat().st_size > 0 for p in sources)
    want = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:16]
    assert kernel_source_sha256() == want != hashlib.sha256(b"").hexdigest()[:16]
