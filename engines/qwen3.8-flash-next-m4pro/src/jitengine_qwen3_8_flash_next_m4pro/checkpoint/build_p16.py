#!/usr/bin/env python3
"""Build the SKYP16X1 pack (p16_writer.py) at a GIVEN path, one verified layer chunk per step.

    build_p16 --model M --source S --out O --chunk next   # build the next unverified layer (exit 10: none left)
    build_p16 --model M --source S --out O --finalize     # marker, validation, hashes, manifest, rename
    build_p16 --model M --source S --out O --status       # what is verified so far
    build_p16 --model M --source S --out O --all          # every chunk, then finalize (one sitting)

`--source` defaults to the checkpoint's expert pack (`default_pack_path`); `--out` has no default.
The exit code is 0 only when the step completed and verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.p16_writer import (
    EXIT_DONE,
    MIN_FREE_BYTES,
    P16Build,
)
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    ExpertScan,
    PackError,
    default_pack_path,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_p16")
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--source", default=None, help="the FNXEPK01 expert pack (default: the engine's)"
    )
    ap.add_argument(
        "--out", required=True, help="the pack16 file to write (its .partial while building)"
    )
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--chunk", default=None, help="'next' or a layer index")
    step.add_argument("--finalize", action="store_true")
    step.add_argument("--status", action="store_true")
    step.add_argument("--all", action="store_true")
    ap.add_argument("--report", default=None, help="write the step's JSON report here")
    ap.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_FREE_BYTES / 2**30,
        help="free space the volume must keep after the whole pack (default 8)",
    )
    a = ap.parse_args(argv)

    scan = ExpertScan(a.model)  # the prefix derived from the index, as the engine's scan does
    source = a.source or default_pack_path(scan)
    report: dict = {"out": os.path.abspath(a.out), "source": os.path.abspath(source)}
    rc = 0
    try:
        b = P16Build(scan, source, a.out, min_free=int(a.min_free_gib * 2**30))
        if a.status:
            st = b.peek()
            report.update(
                finished=os.path.exists(b.out),
                started=st is not None,
                todo=[] if os.path.exists(b.out) else b.todo(st),
            )
        elif a.finalize:
            b.open()
            m = b.finalize()
            report.update(sha256=m["sha256"], size=m["size"], manifest=b.out + ".manifest.json")
        elif a.all:
            b.open()
            for layer in b.todo():
                b.build_layer(layer)
            m = b.finalize()
            report.update(sha256=m["sha256"], size=m["size"], manifest=b.out + ".manifest.json")
        else:
            b.open()
            todo = b.todo()
            layer = (todo[0] if todo else None) if a.chunk == "next" else int(a.chunk)
            if layer is None:
                report.update(todo=[])
                rc = EXIT_DONE
            else:
                report.update(layer=layer, chunk=b.build_layer(layer), todo=b.todo())
    except PackError as exc:
        report.update(error=str(exc))
        print(f"[pack16] FAILED: {exc}", file=sys.stderr)
        rc = 3
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "chunk"}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
