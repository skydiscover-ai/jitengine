#!/usr/bin/env python3
"""Build the PLE n-gram row pack offline and verify it byte for byte against the shards.

`--verify N` reads N rows back uncached, always including both ends of every segment, and refuses a
sample that misses a present shard (a wrong shard offset returns another row's plausible bytes).
The exit code is 0 only if the pack validates and every verified row matches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_format import (
    PLE_FIELDS,
    PLE_MAGIC,
    PLEScan,
    default_ple_pack_path,
    ple_pack_bytes,
    read_header_block,
    row_offset,
    rows_present,
    segments_for,
    validate_ple,
)
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.pack_io import Reader, ShardReader
from jitengine_qwen3_8_flash_next_m4pro.checkpoint.ple_writer import (
    build_pack,
    parse_rows,
    parse_shards,
    scan_checkpoint,
)


def verify(scan: PLEScan, pack_path: str, n: int, seed: int = 0) -> dict:
    """Read `n` rows uncached from the pack, one 100 B pread each, and compare to the shards."""
    meta = read_header_block(pack_path, PLE_MAGIC)
    segs = meta["segments"]
    mandatory, picks = _pick_rows(segs, n, seed)
    shards_covered = _shards_covered(meta, segs, picks)

    reader = Reader(pack_path)
    src = ShardReader()
    mismatches: list[dict] = []
    t0 = time.monotonic()
    try:
        for row in picks:
            _compare_row(scan, reader, src, meta, row, mismatches)
    finally:
        reader.close()
        src.close()
    dt = time.monotonic() - t0
    return {
        "rows_verified": len(picks),
        "shards_covered": shards_covered,
        "segment_endpoints": mandatory,
        "bytes_compared": len(picks) * meta["row_bytes"],
        "seconds": round(dt, 2),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def _pick_rows(segs: list[dict], n: int, seed: int) -> tuple[list[int], list[int]]:
    """The sample: both ends of every segment, then random distinct rows up to n."""
    mandatory: list[int] = []
    for seg in segs:
        mandatory += [seg["row_start"], seg["row_start"] + seg["rows"] - 1]
    picks: list[int] = []
    seen: set[int] = set()
    for r in mandatory:
        if r not in seen:
            seen.add(r)
            picks.append(r)
    rng = random.Random(seed)
    while len(picks) < n:
        seg = segs[rng.randrange(len(segs))]
        cand = seg["row_start"] + rng.randrange(seg["rows"])
        if cand not in seen:
            seen.add(cand)
            picks.append(cand)
    return mandatory, picks


def _shards_covered(meta: dict, segs: list[dict], picks: list[int]) -> list[int]:
    """The shards the sample touches; AssertionError if it misses one holding a segment end."""
    shards_covered = sorted({r // meta["rows_per_shard"] for r in picks})
    want_shards = sorted(
        {
            (seg["row_start"] + k) // meta["rows_per_shard"]
            for seg in segs
            for k in (0, seg["rows"] - 1)
        }
    )
    if not set(want_shards) <= set(shards_covered):
        raise AssertionError(f"sample covers shards {shards_covered}, not all of {want_shards}")
    return shards_covered


def _compare_row(
    scan: PLEScan, reader: Reader, src: ShardReader, meta: dict, row: int, mismatches: list[dict]
) -> None:
    """Read a row from the pack and compare its fields with the shards, recording differences."""
    got = reader.read(row_offset(meta, row), meta["row_bytes"])
    at = 0
    for field in PLE_FIELDS:
        path, start, length = scan.slice_of(row, field)
        want = src.read(path, start, length)
        have = got[at : at + length]
        if have != want:
            bad = next(k for k in range(length) if have[k] != want[k])
            mismatches.append(
                {
                    "row": row,
                    "shard": row // meta["rows_per_shard"],
                    "field": field,
                    "first_bad_byte": bad,
                }
            )
        at += length
    if at != meta["row_bytes"]:
        raise AssertionError(f"fields sum to {at}, not {meta['row_bytes']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--pack",
        default=None,
        help="the pack file (default: $JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PLE_PACK, else one named after the "
        "checkpoint in $JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_PACK_DIR)",
    )
    ap.add_argument("--shards", default=None, help="n-gram shards to build, e.g. 0,1,127 or 0-3")
    ap.add_argument(
        "--rows",
        default=None,
        help="shard-local rows: N (=0:N), lo:hi, or S=lo:hi per shard (default: all)",
    )
    ap.add_argument("--rows-per-chunk", type=int, default=262_144)
    ap.add_argument(
        "--verify", type=int, default=0, help="verify N rows byte-for-byte (>=64 required)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-build", action="store_true", help="never write; fail if the pack is absent/stale"
    )
    ap.add_argument("--report", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)

    scan = scan_checkpoint(a.model)
    shards = parse_shards(a.shards, scan.num_shards)
    rows = parse_rows(a.rows, shards, scan.rows_per_shard)
    segments = segments_for(scan, shards, rows)
    path = a.pack or default_ple_pack_path(scan, segments)
    want_size = ple_pack_bytes(scan, segments)
    n_rows = sum(s["rows"] for s in segments)
    print(
        f"[ple-pack] model={scan.model_path} ple_layer={scan.layer} shards={scan.num_shards} "
        f"rows_per_shard={scan.rows_per_shard} total_rows={scan.total_rows} "
        f"row_bytes={scan.row_bytes}"
    )
    print(
        f"[ple-pack] building {n_rows} rows in {len(segments)} segment(s) from shards {shards} "
        f"-> {path} expected_size={want_size}"
    )

    report = {
        "tool": "build_ple_pack",
        "pack": path,
        "model": scan.model_path,
        "shards": shards,
        "rows": {str(k): list(v) for k, v in rows.items()},
        "segments": segments,
        "rows_selected": n_rows,
        "row_bytes": scan.row_bytes,
        "expected_size": want_size,
        "full_pack_bytes": ple_pack_bytes(
            scan,
            segments_for(
                scan,
                list(range(scan.num_shards)),
                {s: (0, scan.rows_per_shard) for s in range(scan.num_shards)},
            ),
        ),
    }
    t0 = time.monotonic()
    built = False
    ok, why = validate_ple(path, scan, segments) if os.path.exists(path) else (False, "absent")
    if not ok:
        print(f"[ple-pack] pack not usable: {why}")
        if a.no_build:
            print("[ple-pack] --no-build: refusing to write")
            return 2
        built, ok, why = _rebuild_pack(scan, path, segments, a)
        if not ok:
            print(f"[ple-pack] FRESHLY BUILT PACK DOES NOT VALIDATE: {why}")
            return 3
    _report_validated_pack(report, built, t0, path, why)

    if a.verify:
        if a.verify < 64:
            print("[ple-pack] --verify must be >= 64")
            return 4
        _run_verify(scan, path, a, report)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if report.get("verify", {"ok": True})["ok"] else 5


def _report_validated_pack(report: dict, built: bool, t0: float, path: str, why: str) -> None:
    """Record the pack's size, build time and validation in the report and print the summary."""
    secs = time.monotonic() - t0
    size = os.path.getsize(path)
    report.update(
        {
            "built": built,
            "build_seconds": round(secs, 1),
            "size": size,
            "validate": why,
            "rows_present": rows_present(read_header_block(path, PLE_MAGIC)),
            "build_gb_per_s": round(size / 1e9 / secs, 3) if built else None,
        }
    )
    print(f"[ple-pack] validated: size={size} ({'built' if built else 'reused'} in {secs:.0f}s)")


def _run_verify(scan: PLEScan, path: str, a: argparse.Namespace, report: dict) -> None:
    """Verify a.verify rows against the shards, store the result in the report and print it."""
    rep = verify(scan, path, a.verify, a.seed)
    report["verify"] = rep
    print(
        f"[ple-pack] verify: {rep['rows_verified']} rows across shards {rep['shards_covered']}, "
        f"{rep['bytes_compared']} B compared uncached in {rep['seconds']}s -> "
        f"{'OK (byte-identical)' if rep['ok'] else 'MISMATCH'}"
    )
    if not rep["ok"]:
        print(json.dumps(rep["mismatches"][:10], indent=1))


def _rebuild_pack(
    scan: PLEScan, path: str, segments: list[dict], a: argparse.Namespace
) -> tuple[bool, bool, str]:
    """Replace the unusable pack: delete it, build it again and validate the result."""
    if os.path.exists(path):
        os.unlink(path)
    build_pack(scan, path, segments, rows_per_chunk=a.rows_per_chunk)
    built = True
    ok, why = validate_ple(path, scan, segments)
    return built, ok, why


if __name__ == "__main__":
    raise SystemExit(main())
