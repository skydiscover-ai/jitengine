"""Build the multi-token-prediction head the engine drafts with, `mtp.safetensors`, once.

The 4-bit checkpoint declares the head (`mtp_num_hidden_layers = 1`) but carries none of its tensors:
the MLX conversion drops every `mtp.*`. This reads exactly those 31 tensors out of the official bf16
release by HTTP range request (4.9 GB, not the whole checkpoint), converts them and writes the file
`model/mtp.py` loads, by default beside the checkpoint, where the engine looks for it:

- `experts.gate_up_proj` [E, 2I, H] becomes `switch_mlp.gate_proj` and `up_proj` (gate rows first),
  and `experts.down_proj` becomes `switch_mlp.down_proj`;
- the centered norms (stored as a deviation from one) get the +1 folded in once;
- the rest is quantized 4-bit, group 64, affine, except the router, the gates, the hyper-connection
  inject weights, the indexer's projection and the norms, which stay bf16.

Ported from slotstream's `Tools/mtp_convert.py` (https://github.com/carloslfu/slotstream, MIT License,
Copyright (c) 2026 Carlos Galarza), whose output this engine's published numbers were measured with:
the transforms, the recipe and the file's metadata are the same. The head only drafts, so it never
changes the output; without it the engine decodes one token per pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.request

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
GROUP, BITS = 64, 4

# the 31 tensors under `mtp.` in the official checkpoint (checked against the pinned revision's
# index before anything is downloaded)
EXPECTED_SRC = {
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}
# the norms stored as a deviation from one: the +1 is folded in here, once
CENTERED = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
# kept bf16, as the checkpoint keeps the same modules of the main layers
KEEP_BF16 = (
    "mlp.gate.weight",
    "shared_expert_gate.weight",
    "block_inject_weight.weight",
    "indexer.index_qk_proj.weight",
)
METADATA = {
    "source_repo": REPO,
    "source_revision": REVISION,
    "recipe": f"{BITS}-bit group {GROUP} affine, router/gates/norms bf16, "
    "centered norms +1-folded (incl. pre_fc norms)",
}


def hf_url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def fetch(url: str, rng: tuple[int, int] | None = None) -> bytes:
    """GET `url` (bytes `rng[0]`..`rng[1]` inclusive when given), with four tries."""
    err: Exception | None = None
    for attempt in range(4):
        req = urllib.request.Request(url)
        if rng is not None:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except OSError as exc:
            err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {err}")


def shard_header(shard: str) -> tuple[dict, int]:
    """A remote safetensors shard's tensor table and the offset its data starts at."""
    n = struct.unpack("<Q", fetch(hf_url(shard), (0, 7)))[0]
    if n > 50 << 20:
        raise RuntimeError(f"{shard}: implausible header length {n}")
    return json.loads(fetch(hf_url(shard), (8, 8 + n - 1))), 8 + n


def staged(staging: str, name: str) -> str:
    return os.path.join(staging, name.replace("/", "_") + ".bin")


def download_tensor(staging: str, name: str, shard: str, entry: dict, data_start: int) -> str:
    """Range-download one tensor's bytes into the staging directory; resumes a partial file."""
    lo, hi = entry["data_offsets"]
    size = hi - lo
    path = staged(staging, name)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == size:
        return path
    if have > size:
        os.remove(path)
        have = 0
    with open(path, "ab") as f:
        got = fetch(hf_url(shard), (data_start + lo + have, data_start + hi - 1))
        if len(got) != size - have:
            raise RuntimeError(f"{name}: got {len(got)} bytes, wanted {size - have}")
        f.write(got)
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(staging: str) -> list[tuple[str, dict]]:
    """Every head tensor's bytes in `staging`; returns (name, header entry) in name order."""
    index = json.loads(fetch(hf_url("model.safetensors.index.json")))
    mtp = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    if set(mtp) != EXPECTED_SRC:
        raise RuntimeError(
            f"the release's mtp tensors changed: missing {sorted(EXPECTED_SRC - set(mtp))}, "
            f"extra {sorted(set(mtp) - EXPECTED_SRC)}"
        )
    shards = sorted(set(mtp.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        headers = dict(zip(shards, ex.map(shard_header, shards), strict=True))
    plan = []
    for name, shard in sorted(mtp.items()):
        table, data_start = headers[shard]
        if table[name]["dtype"] != "BF16":
            raise RuntimeError(f"{name}: expected bf16, found {table[name]['dtype']}")
        plan.append((name, shard, table[name], data_start))
    total = sum(e["data_offsets"][1] - e["data_offsets"][0] for _, _, e, _ in plan)
    print(
        f"[mtp] {len(plan)} tensors, {total / 1e9:.2f} GB from {REPO}@{REVISION[:12]}", flush=True
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(download_tensor, staging, *p) for p in plan]
        for done, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            fut.result()
            print(f"[mtp] downloaded {done}/{len(plan)}", flush=True)
    return [(name, entry) for name, _, entry, _ in plan]


def convert(staging: str, plan: list[tuple[str, dict]]) -> dict:
    """The head's tensors in the engine's format."""
    import mlx.core as mx
    import numpy as np

    def load_bf16(name: str, shape: list[int]):
        a = mx.array(np.fromfile(staged(staging, name), dtype=np.uint16)).view(mx.bfloat16)
        a = a.reshape(shape)
        mx.eval(a)
        return a

    out: dict = {}

    def emit_quant(dst: str, w) -> None:
        wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
        out[dst + ".weight"] = wq
        out[dst + ".scales"] = scales.astype(mx.bfloat16)
        out[dst + ".biases"] = biases.astype(mx.bfloat16)
        mx.eval(wq, scales, biases)

    for name, entry in plan:
        shape = entry["shape"]
        v = load_bf16(name, shape)
        if name == "mtp.layers.0.mlp.experts.gate_up_proj":
            inter = shape[1] // 2
            emit_quant("mtp.layers.0.mlp.switch_mlp.gate_proj", v[:, :inter, :])
            emit_quant("mtp.layers.0.mlp.switch_mlp.up_proj", v[:, inter:, :])
        elif name == "mtp.layers.0.mlp.experts.down_proj":
            emit_quant("mtp.layers.0.mlp.switch_mlp.down_proj", v)
        elif name.endswith((*KEEP_BF16, "norm.weight", "layernorm.weight", *CENTERED)):
            if name.endswith(CENTERED):
                v = (v.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            mx.eval(v)
            out[name] = v
        else:
            emit_quant(name.removesuffix(".weight"), v)
        del v
    return out


def check(out: dict) -> None:
    """The shapes and types `model/mtp.py` expects, and every quantized weight's scales and biases."""
    import mlx.core as mx

    H, HC, E, inter, NH, ND, KV = 2560, 4, 512, 640, 24, 256, 2
    expect = {
        "mtp.fc_embedding.weight": (mx.uint32, [H, H // 8]),
        "mtp.fc_hidden.weight": (mx.uint32, [H, H // 8]),
        "mtp.pre_fc_norm_embedding.weight": (mx.bfloat16, [H]),
        "mtp.pre_fc_norm_hidden.weight": (mx.bfloat16, [HC * H]),
        "mtp.hyper_connection_mixer.hc_norm.weight": (mx.bfloat16, [HC * H]),
        "mtp.layers.0.self_attn.q_proj.weight": (mx.uint32, [NH * 2 * ND, H // 8]),
        "mtp.layers.0.self_attn.k_proj.weight": (mx.uint32, [KV * ND, H // 8]),
        "mtp.layers.0.self_attn.v_proj.weight": (mx.uint32, [KV * ND, H // 8]),
        "mtp.layers.0.self_attn.o_proj.weight": (mx.uint32, [H, NH * ND // 8]),
        "mtp.layers.0.self_attn.q_norm.weight": (mx.bfloat16, [ND]),
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (mx.bfloat16, [640, H]),
        "mtp.layers.0.mlp.gate.weight": (mx.bfloat16, [E, H]),
        "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": (mx.uint32, [E, inter, H // 8]),
        "mtp.layers.0.mlp.switch_mlp.down_proj.weight": (mx.uint32, [E, H, inter // 8]),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (mx.bfloat16, [1, H]),
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": (
            mx.bfloat16,
            [HC, HC * H],
        ),
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (
            mx.uint32,
            [320, HC * H // 8],
        ),
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (
            mx.uint32,
            [HC * H, 320 // 8],
        ),
    }
    for k, (dtype, shape) in expect.items():
        if k not in out:
            raise RuntimeError(f"missing output tensor {k}")
        if out[k].dtype != dtype or list(out[k].shape) != shape:
            raise RuntimeError(f"{k}: {out[k].dtype} {out[k].shape}, expected {dtype} {shape}")
    for k, v in out.items():
        if v.dtype == mx.uint32:
            base = k.removesuffix(".weight")
            for side in (".scales", ".biases"):
                s = out.get(base + side)
                if s is None or s.dtype != mx.bfloat16:
                    raise RuntimeError(f"{base}{side} missing or not bf16")
                if list(s.shape) != [*v.shape[:-1], v.shape[-1] * 8 // GROUP]:
                    raise RuntimeError(f"{base}{side}: shape {s.shape} for weight {v.shape}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_mtp", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--out", default=None, help="the file to write (default <model>/mtp.safetensors)"
    )
    ap.add_argument("--keep-staging", action="store_true", help="keep the downloaded bf16 tensors")
    a = ap.parse_args(argv)

    out_path = os.path.abspath(
        os.path.expanduser(a.out or os.path.join(os.path.expanduser(a.model), "mtp.safetensors"))
    )
    if os.path.exists(out_path):
        print(f"[mtp] {out_path} exists; remove it to rebuild", flush=True)
        return 0
    staging = os.path.join(os.path.dirname(out_path), ".mtp-staging")
    os.makedirs(staging, exist_ok=True)
    try:
        plan = download(staging)
        import mlx.core as mx

        out = convert(staging, plan)
        check(out)
        partial = out_path + ".partial.safetensors"
        mx.save_safetensors(partial, out, metadata=METADATA)
        os.replace(partial, out_path)
    except RuntimeError as exc:
        print(f"[mtp] FAILED: {exc}", file=sys.stderr)
        return 3
    size, digest = os.path.getsize(out_path), sha256_file(out_path)
    print(f"[mtp] wrote {out_path} ({size / 1e9:.2f} GB, sha256 {digest})", flush=True)
    if not a.keep_staging:
        shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
