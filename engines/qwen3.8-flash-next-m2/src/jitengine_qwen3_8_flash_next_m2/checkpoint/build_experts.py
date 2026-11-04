#!/usr/bin/env python3
"""Build the expert pack offline and verify it byte for byte against the checkpoint shards.

`--verify N` reads N records back uncached, always including the four corner records; the exit
code is 0 only if the pack validates and every verified record matches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from jitengine_qwen3_8_flash_next_m2.checkpoint.expert_writer import build_pack, scan_checkpoint
from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import (
    FIELDS,
    MAGIC,
    ExpertScan,
    PackError,
    default_pack_path,
    pack_bytes,
    read_header_block,
    record_index,
    record_offset,
    validate,
)
from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_io import Reader, ShardReader


def parse_layers(spec: str | None, num_layers: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(num_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise PackError(f"duplicate layers in {spec!r}")
    for layer in out:
        if not 0 <= layer < num_layers:
            raise PackError(f"layer {layer} out of range 0..{num_layers - 1}")
    return out


def verify(scan: ExpertScan, pack_path: str, n: int, seed: int = 0) -> dict:
    """Read `n` records uncached from the pack and compare them to the shards, field by field."""
    meta = read_header_block(pack_path, MAGIC)
    layers = meta["layers"]
    E = meta["num_experts"]
    corners, picks = _pick_sample(layers, E, n, seed)
    layers_covered = len({layer for layer, _ in picks})
    if layers_covered < min(8, len(layers)):
        raise AssertionError(f"verification sample spans only {layers_covered} layers")

    reader = Reader(pack_path)
    shards = ShardReader()
    mismatches: list[dict] = []
    pad_checked = 0
    t0 = time.monotonic()
    try:
        for i, (layer, expert) in enumerate(picks):
            off = record_offset(meta, layer, expert)
            got = reader.read(off, meta["record_bytes"])
            if len(got) != meta["record_bytes"]:
                mismatches.append({"layer": layer, "expert": expert, "why": "short record"})
                continue
            _compare_record_fields(scan, shards, got, layer, expert, meta, mismatches)
            pad_checked = _check_stride_pad(
                i, meta, layer, expert, reader, off, mismatches, pad_checked
            )
    finally:
        reader.close()
        shards.close()
    dt = time.monotonic() - t0
    return {
        "records_verified": len(picks),
        "layers_covered": layers_covered,
        "corners": corners,
        "pads_checked": pad_checked,
        "bytes_compared": len(picks) * meta["record_bytes"],
        "seconds": round(dt, 2),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def _pick_sample(layers: list[int], E: int, n: int, seed: int) -> tuple:
    """The sample: four corner records, then random distinct ones up to n."""
    # corners catch record-index off-by-ones (a layer position in partial packs) and truncation
    corners = [(layers[0], 0), (layers[0], E - 1), (layers[-1], 0), (layers[-1], E - 1)]
    picks: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for c in corners:
        if c not in seen:
            seen.add(c)
            picks.append(c)
    rng = random.Random(seed)
    while len(picks) < n:
        cand = (rng.choice(layers), rng.randrange(E))
        if cand not in seen:
            seen.add(cand)
            picks.append(cand)
    return corners, picks


def _compare_record_fields(
    scan: ExpertScan,
    shards: ShardReader,
    got: bytes,
    layer: int,
    expert: int,
    meta: dict,
    mismatches: list[dict],
) -> None:
    """Compare a record's fields with the shards, recording each field that differs."""
    at = 0
    for field in FIELDS:
        src, start, length = scan.slice_of(layer, expert, field)
        want = shards.read(src, start, length)
        have = got[at : at + length]
        if have != want:
            bad = next(k for k in range(length) if have[k] != want[k])
            mismatches.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "field": field,
                    "record": record_index(meta, layer, expert),
                    "first_bad_byte": bad,
                }
            )
        at += length
    if at != meta["record_bytes"]:
        raise AssertionError(f"fields sum to {at}, not {meta['record_bytes']}")


def _check_stride_pad(
    i: int,
    meta: dict,
    layer: int,
    expert: int,
    reader: Reader,
    off: int,
    mismatches: list[dict],
    pad_checked: int,
) -> int:
    """For the first records, check the pad after the record is zero; returns the pads checked."""
    # non-zero pad bytes mean the stride arithmetic placed a neighbour's bytes there
    if i < 8 and record_index(meta, layer, expert) + 1 < meta["records"]:
        pad = reader.read(off + meta["record_bytes"], meta["record_stride"] - meta["record_bytes"])
        pad_checked += 1
        if any(pad):
            mismatches.append({"layer": layer, "expert": expert, "why": "non-zero stride pad"})
    return pad_checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--pack",
        default=None,
        help="the pack file (default: $JITENGINE_QWEN3_8_FLASH_NEXT_M2_EXPERT_PACK, else one named after the "
        "checkpoint in $JITENGINE_QWEN3_8_FLASH_NEXT_M2_PACK_DIR)",
    )
    ap.add_argument(
        "--layers", default=None, help="subset to build, e.g. 0,1 or 0-3 (default: all 48)"
    )
    ap.add_argument("--experts-per-chunk", type=int, default=64)
    ap.add_argument(
        "--verify", type=int, default=0, help="verify N records byte-for-byte (>=64 required)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-build", action="store_true", help="never write; fail if the pack is absent/stale"
    )
    ap.add_argument("--report", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)

    scan = scan_checkpoint(a.model)
    layers = parse_layers(a.layers, scan.num_layers)
    path = a.pack or default_pack_path(scan, layers)
    want_size = pack_bytes(scan, layers)
    print(
        f"[expert-pack] model={scan.model_path} layers={scan.num_layers} experts={scan.num_experts} "
        f"record_bytes={scan.record_bytes} stride={scan.record_stride} "
        f"records={scan.num_records} layer_bytes={scan.record_bytes * scan.num_experts}"
    )
    print(f"[expert-pack] building layers={layers} -> {path} expected_size={want_size}")

    report = {
        "tool": "build_expert_pack",
        "pack": path,
        "model": scan.model_path,
        "layers": layers,
        "expected_size": want_size,
        "records": len(layers) * scan.num_experts,
        "record_bytes": scan.record_bytes,
        "record_stride": scan.record_stride,
        "full_pack_bytes": pack_bytes(scan, list(range(scan.num_layers))),
    }
    t0 = time.monotonic()
    built = False
    ok, why = validate(path, scan, layers) if os.path.exists(path) else (False, "absent")
    if not ok:
        print(f"[expert-pack] pack not usable: {why}")
        if a.no_build:
            print("[expert-pack] --no-build: refusing to write")
            return 2
        if os.path.exists(path):
            os.unlink(path)
        build_pack(scan, path, layers, experts_per_chunk=a.experts_per_chunk)
        built = True
        ok, why = validate(path, scan, layers)
        if not ok:
            print(f"[expert-pack] FRESHLY BUILT PACK DOES NOT VALIDATE: {why}")
            return 3
    _report_validated_pack(report, built, t0, path, why)

    if a.verify:
        if a.verify < 64:
            print("[expert-pack] --verify must be >= 64")
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
            "build_gb_per_s": round(size / 1e9 / secs, 3) if built else None,
        }
    )
    print(f"[expert-pack] validated: size={size} ({'built' if built else 'reused'} in {secs:.0f}s)")


def _run_verify(scan: ExpertScan, path: str, a: argparse.Namespace, report: dict) -> None:
    """Verify a.verify records against the shards, store the result in the report and print it."""
    rep = verify(scan, path, a.verify, a.seed)
    report["verify"] = rep
    print(
        f"[expert-pack] verify: {rep['records_verified']} records, {rep['layers_covered']} "
        f"layer(s), {rep['bytes_compared']} B compared uncached in {rep['seconds']}s -> "
        f"{'OK (byte-identical)' if rep['ok'] else 'MISMATCH'}"
    )
    if not rep["ok"]:
        print(json.dumps(rep["mismatches"][:10], indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
