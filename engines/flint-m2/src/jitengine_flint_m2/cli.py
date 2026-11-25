"""`jitengine-flint-m2 serve | doctor | inspect`: Flint's command line (jitengine_flint.cli) under this build's name."""

from jitengine_flint.cli import main as _main
from jitengine_flint_m2 import PROG


def main(argv=None) -> int:
    return _main(argv, prog=PROG)
