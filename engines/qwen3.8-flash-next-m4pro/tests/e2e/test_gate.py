"""The Qwen3.8-Flash-Next end-to-end test on a Mac: the greedy-512 text hash, every kernel check true
and the measured paths on, decode.

The published numbers were measured by serving the engine at a 40 GB (37.3 GiB) budget,
`--memory-budget-gb 40`, and running benchmarks/benchmark.py with
`--mode interactive --repeats 3 --max-tokens 512 --temperature 0`: the chat endpoint, task "dedent"
(task seed 0), one warmup request, then three repeats, each `text_sha` being
`hashlib.sha256(out_text.encode()).hexdigest()[:16]` of the streamed text (`benchmark.run_stream`).
On the 48 GB M4 Pro at 40 GB every greedy run gave 07a676778e3a6720 (as did 16, 24 and 32 GB), and
the published decode is 32.9 tok/s there (32.94 when it was published; this test read 32.7). This
test runs both as subprocesses, the server as `python -m jitengine_qwen3_8_flash_next_m4pro serve --model-path M
--memory-budget-gb 40 --expert-pack-path E --ple-pack-path P`, with the packs `jitengine-qwen3.8-flash-next-m4pro
pack` names for the checkpoint, in the pack directory given.

The engine checks each custom kernel against stock MLX on this Mac at load, bit for bit where the
kernel claims bit-identity and elsewhere to a stated tolerance, and switches off any that fails, so
the hash could hold while measured kernels are off: /stats must show every load-time check true,
one for each kernel in `kernel_checks.NAMES`. Then the paths they serve are checked, so that the
measured path is the one under test: /stats must show the grouped expert kernel (split variant) on
every layer, the row-blocked dense kernel on every shape its config table lists, float32 compute
and the compiled hyper-connection path, and the startup log the fused decode paths
(hyper-connections, GDN glue, MoE, speculative gather) on every layer they apply to and the
lookahead decode on.

Run with `-s` to see the decode rate. Environment: JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_CHECKPOINT (required:
the Qwen3.8-Flash-Next-4bit checkpoint directory); JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PACKS (required with it:
the directory holding its expert and PLE packs); JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_BUDGET_GB (decimal GB, as the
budgets were measured; default 40, or 37.3 GiB); JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_SHA (default 07a676778e3a6720, the M4 Pro's);
JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_MIN_TOKS (a decode tok/s floor; unset, the rate is only reported). Needs
`pip install '.[benchmark]'`.
"""

import hashlib
import os
import re

import pytest

M4_PRO_SHA = "07a676778e3a6720"
CHECKPOINT = "JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_CHECKPOINT"
WHAT = "a Qwen3.8-Flash-Next-4bit checkpoint directory (and its packs, see the docstring)"


def _packs(model: str) -> list[str]:
    """The expert and PLE pack paths `jitengine-qwen3.8-flash-next-m4pro pack` gives `model`, in the pack directory
    JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PACKS: `<checkpoint name>-<index sha256[:12]>.experts` and `.ple`."""
    root = os.environ.get("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PACKS")
    assert root, (
        "set JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_PACKS to the directory holding the checkpoint's packs"
    )
    with open(os.path.join(model, "model.safetensors.index.json"), "rb") as f:
        tag = f"{os.path.basename(model.rstrip('/'))}-{hashlib.sha256(f.read()).hexdigest()[:12]}"
    paths = [os.path.join(os.path.expanduser(root), tag + ext) for ext in (".experts", ".ple")]
    missing = [p for p in paths if not os.path.isfile(p)]
    assert not missing, (
        f"no pack at {missing} (`jitengine-qwen3.8-flash-next-m4pro pack` builds them)"
    )
    return paths


@pytest.fixture(scope="module")
def flashnext(gate, gate_server):
    """(checkpoint, budget, server): the end-to-end server, started once for this module's tests."""
    model = gate.model(CHECKPOINT, WHAT)
    experts, ple = _packs(model)
    budget = os.environ.get("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_BUDGET_GB") or "40"
    server = gate_server(
        [
            "-m",
            "jitengine_qwen3_8_flash_next_m4pro",
            "serve",
            "--model-path",
            model,
            "--memory-budget-gb",
            budget,
            "--expert-pack-path",
            experts,
            "--ple-pack-path",
            ple,
        ]
    )
    return model, budget, server


def test_measured_command_buffer_setting(flashnext):
    """The package commits a Metal command buffer every 10 ops (its `__init__` sets
    MLX_MAX_OPS_PER_BUFFER=10 unless the caller set it), as measured; the server process must have
    that setting when it loads MLX, so nothing it imports first may set another."""
    _, _, server = flashnext
    got = server.mlx_env()
    assert got["MLX_MAX_OPS_PER_BUFFER"] == "10", f"the server process loaded MLX with {got}"


def test_greedy_512_hash_paths_and_decode(flashnext, gate, tmp_path, record_testsuite_property):
    model, budget, server = flashnext
    want = os.environ.get("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_SHA") or M4_PRO_SHA
    floor = gate.env_float("JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_MIN_TOKS")
    stats = server.get("/stats")
    result = gate.run(server, model, str(tmp_path / "greedy.json"))
    rate = float(result["decode_tok_per_sec_median"])
    shas = gate.shas(result)
    record_testsuite_property("flashnext_decode_tok_per_sec", rate)  # in a junit report
    record_testsuite_property("flashnext_text_sha", ",".join(shas))
    print(f"\n[e2e] Flash-Next greedy-512 at {budget} GB: text_sha {shas}, decode {rate:.2f} tok/s")

    problems = []
    if not result["valid"]:
        problems.append(f"the benchmark run is invalid: {result['problems']}")
    if len(shas) != 3 or set(shas) != {want}:
        problems.append(f"text_sha {shas}, want {want} on every repeat")
    problems += _measured_paths(stats, server.log(), model)
    if floor is not None and rate < floor:
        problems.append(
            f"decode {rate:.2f} tok/s is below JITENGINE_QWEN3_8_FLASH_NEXT_M4PRO_E2E_MIN_TOKS={floor}"
        )
    assert not problems, "\n".join(problems)


def _measured_paths(stats: dict, log: str, model: str) -> list[str]:
    from jitengine_qwen3_8_flash_next_m4pro.kernels.qmv_row_blocked import CONFIGS
    from jitengine_qwen3_8_flash_next_m4pro.model.config import load_config
    from jitengine_qwen3_8_flash_next_m4pro.model.kernel_checks import NAMES

    problems = []
    checks = stats["kernel_checks"]
    off = sorted(name for name in NAMES if checks.get(name) is not True)
    if off or set(checks) != set(NAMES):
        problems.append(
            f"kernel checks {checks}: not all true {off} {stats['kernel_check_errors']}"
        )
    moe = stats["moe_kernel"]
    if (moe["kernel"], moe["variant"], moe["fallback_layers"]) != ("grouped", "split", []):
        problems.append(
            f"routed experts: {moe['kernel']} {moe['variant']}, fallback layers {moe['fallback_layers']}"
        )
    dense = stats["dense_kernel"]
    listed = [
        key
        for key in dense["unserved_shapes"]
        if re.fullmatch(r"(\d+)x(\d+)/4b", key)
        and (*map(int, key.split("/")[0].split("x")), "float32") in CONFIGS
    ]
    if (
        dense["option"] != "rb"
        or dense["compute_dtype"] != "float32"
        or listed
        or not dense["served_linears"]
    ):
        problems.append(
            f"dense kernel {dense['option']} at {dense['compute_dtype']}, listed shapes on stock: {listed}"
        )
    if stats["hc_path"]["option"] != "compiled":
        problems.append(f"hyper-connection path {stats['hc_path']['option']}")
    cfg = load_config(model)
    gdn_layers = sum(not cfg.is_attention(i) for i in range(cfg.num_hidden_layers))
    m = re.search(
        r"fused decode: hc (\d+)/(\d+), gdn (\d+), moe (\d+), spec (\d+), lookahead (\w+)", log
    )
    if m is None:
        problems.append("the server logged no fused-decode line")
    else:
        hc, layers, gdn, moe_n, spec, look = m.groups()
        if not (hc == layers == moe_n == spec and int(gdn) == gdn_layers and look == "ON"):
            problems.append(
                f"fused decode: {m.group(0)} (want every layer, {gdn_layers} GDN, lookahead ON)"
            )
    return problems
