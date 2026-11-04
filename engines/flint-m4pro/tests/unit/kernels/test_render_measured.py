"""The megakernel's Metal code is the measured build's, for every variant the engine compiles.

The hashes are sha256 of kernels.metal_src.render's header + source for the main and kv8 profiles, the
persistent pass (at M = 1 its phase-mask specialization, as the engine compiles it) and its
one-dispatch-per-phase chain, at M = 1 and at every verify row count M = 2..8, rendered
with the engine's own arguments (kernels/runner.py) from the build the published numbers were measured on.
Each hash is taken over the code the Metal compiler reads: comments removed and whitespace collapsed. A change
to any code token fails here, off a Mac; a comment may be reworded.
"""

import hashlib
import re

import pytest

from jitengine_flint_m4pro.kernels import metal_src as MS
from jitengine_flint_m4pro.kernels import plan as P
from jitengine_flint_m4pro.model.config import load_config, tensor_index
from jitengine_flint_m4pro.weight_banks import BankPlan

MEASURED = {
    "kv8/chain/m1": "04530d0c817ba2f8aebe26ecee5b262ab758bc3157f6231a375924bb2ad02a4b",
    "kv8/chain/m2": "4a8dbccbfddfbbadda8bb8ddfcd6777f7905aea1ae6be29f9036f9582d85dd2f",
    "kv8/chain/m3": "c2856334407e389047705d8741e08f898af67ba9236e2d79d0d85c208442c423",
    "kv8/chain/m4": "5f88257218d03f7b0267f149324101ce10c367428ef5b409889775e21c43fe20",
    "kv8/chain/m5": "0913e905ae7339d49b9f2e777e1f0f83b2d14966712202e3c236c5dcaed74e79",
    "kv8/chain/m6": "0e7ad61fcc168e37e73426e054e2817f70d437092a8bd3f85cbb66b3750bf51a",
    "kv8/chain/m7": "bde8ce77757854c15a58554096181b2027dab38ea551196c083fab7f6b0eaeca",
    "kv8/chain/m8": "6d040439effdae7c93026d05a90dee9e29edd6b18da9b51b6b96ed7a9dd2d0be",
    "kv8/persist/m1": "9f4a689623579237c399dbe047375e735a468df66981e24f9596ea922fb01ba4",
    "kv8/persist/m2": "82c697b39d3fcdbdd7ad8d9b50013a8bc9a69e10e64f8f8ac4f71909ad73dd67",
    "kv8/persist/m3": "3acefbca03c723f0facc12d44655642089d0d174f4b04ecd8627b652c5a0dffc",
    "kv8/persist/m4": "393c7a9dae3f234449522da03a998615730c113891bb50207a4b937976407f80",
    "kv8/persist/m5": "4a7f204a1ce54dbc07d54d31c37f572fd38044cfc6d198926d74f5e90dab4a10",
    "kv8/persist/m6": "b1fab26e6655b631861ac990564d2da21cbad86da7c8e17c781a25dac7c70d5d",
    "kv8/persist/m7": "5dff707e0bde91371433ec1d45203779a5c5e4d7ed3adf9510c8cd6bf95e6543",
    "kv8/persist/m8": "18b082492d0d71d89ce13ed77f6f96e9cb93c7052690015781cf2d252dda6d07",
    "main/chain/m1": "de1d2054c05d4e6766135eda1b8be15763a48a1959dffe1fa107cff1f84ae833",
    "main/chain/m2": "f1d5bfd39146e797ea4ea371f73cebd53dacda413def3a71fd8ebe785b92268f",
    "main/chain/m3": "94deed1048171d2c24ad468bda8f4d63c8c9ce26593757deba4f971e3caf3fd9",
    "main/chain/m4": "fead347ad26f3e72667dc3006a3189d193db67eddde388433dd5808ba791235b",
    "main/chain/m5": "a1261ad297192443c5e286569bd83f2b41f22266cdbbefaf72f5ebc05f1d61ed",
    "main/chain/m6": "5df1fe3bec9d33d1134b63cc0ef12d8dbea9e9eb71a9146016272902c3fed6ae",
    "main/chain/m7": "301f1c535605c4349a2ff1f9bb9c8d73e66c94dcc01b22263b8b7eb4aef37e4c",
    "main/chain/m8": "5a1765bccee7071482120485fe435d5ac0fb802e256a613ad098a47dc2cb2a7a",
    "main/persist/m1": "64b3bf19756864801417e94cbb3c5e55d6642e1601f88b8153397e25baffb9eb",
    "main/persist/m2": "260cc5d6446a0556cc9024415d431cc157da75d4a9fd221ee7f52569fd31ba88",
    "main/persist/m3": "89268e568e447c6e53a173f78f78f2da086eca88e21bd53a92b2bee6e6142d43",
    "main/persist/m4": "7384e321d0a2ba4f6926ff27f89f64750b08d15b1e912170e42b95fa199cec29",
    "main/persist/m5": "446fca8cf61ef482d60a68c525af0e51f4f0b241a4628c4b81d907e67fad9105",
    "main/persist/m6": "bbb14d47a3b118c02d57fddefb8afff029e46079b324529f47a558634ba97d1a",
    "main/persist/m7": "b1a77597d40b3f155e30d461647b8bd1510f3cf84e7c35b206beb22b9c6c2ca7",
    "main/persist/m8": "201dd32997e1d4a49004b2cdbe84dc0657137d0bc0c0a6896e4701504474a35f",
}

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _code(text: str) -> str:
    """The Metal text without its comments, whitespace collapsed to single spaces."""
    return " ".join(_COMMENT.sub(" ", text).split())


@pytest.mark.parametrize("variant", sorted(MEASURED))
def test_render_is_the_measured_code(flint_headers, variant):
    profile, mode, rows = variant.split("/")
    kv8, chain, mr = profile == "kv8", mode == "chain", int(rows[1:])
    cfg = load_config(flint_headers)
    eo = P.expert_record_offsets(BankPlan(cfg, tensor_index(flint_headers, cfg)))
    if mr == 1:
        h, s = MS.render(cfg, eo, chain, 1, 4, 0, 0, kv8=kv8)
    else:
        h, s = MS.render(cfg, eo, chain, 1, 4, mr=mr, G=P.G_DEFAULT, xs=P.VXS, kv8=kv8)
    assert hashlib.sha256(_code(h + s).encode()).hexdigest() == MEASURED[variant]


def test_code_drops_comments_and_nothing_else():
    text = "kernel void k(device float* x) { // one\n  x[0] = 1.0f; /* two\n three */ x[1] = 2.0f;\n}\n"
    assert _code(text) == "kernel void k(device float* x) { x[0] = 1.0f; x[1] = 2.0f; }"
