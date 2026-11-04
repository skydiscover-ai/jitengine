"""The Qwen3.6-35B-A3B end-to-end test on a Mac: the greedy-512 text hash, every kernel check on, decode.

As published (README.md) the server is `jitengine-qwen3.6-35b-a3b-m2 serve --model-path M --profile 24gb-static`
(the built-in defaults resolve to the same configuration) and the client is
`python benchmarks/benchmark.py --model-path M --mode interactive --temperature 0
--max-tokens 512 --output-json greedy.json`, run with `--repeats 3`: the chat
endpoint, task "dedent" (task seed 0), one warmup request, then three repeats.
`benchmark.run_stream` computes each repeat's `text_sha` as
`hashlib.sha256(out_text.encode()).hexdigest()[:16]` over the text the server streamed; with the
3-bit checkpoint (andrevp/Qwen3.6-35B-A3B-3bit-MLX at 794408c4) every repeat gave 44e24c591fc59187
on the 24 GB M2. This test runs both commands as subprocesses, the server as `python -m
jitengine_qwen3_6_35b_a3b_m2 serve`.

The engine checks its custom kernels against stock MLX on this Mac and switches off any that
differ, so the hash could hold while the measured kernels are off. /stats reports the checks, and
the test reads it after the run, since the dense kernels' prescaled and prefill shapes are checked
on first use: every `kernel_checks` entry (the load-time checks) must be true, and no dense kernel
in `dense_kernels` may differ from stock or have raised on any shape.

Run with `-s` to see the decode rate. Environment: JITENGINE_QWEN3_6_35B_A3B_M2_E2E_CHECKPOINT (required: the
checkpoint directory); JITENGINE_QWEN3_6_35B_A3B_M2_E2E_SHA (default 44e24c591fc59187, the M2's; on another Mac,
that Mac's reference); JITENGINE_QWEN3_6_35B_A3B_M2_E2E_MIN_TOKS (a decode tok/s floor; unset, the rate is only
reported; the README's 44.9 on the M2 is the single-user rate of `--mode interactive`, and this test measures
greedy-512, the median of 3 repeats).
Needs `pip install '.[benchmark]'`.
"""

import os

M2_SHA = "44e24c591fc59187"


def test_greedy_512_hash_kernel_checks_and_decode(
    gate, gate_server, tmp_path, record_testsuite_property
):
    model = gate.model(
        "JITENGINE_QWEN3_6_35B_A3B_M2_E2E_CHECKPOINT",
        "a Qwen3.6-35B-A3B 3-bit checkpoint directory",
    )
    want = os.environ.get("JITENGINE_QWEN3_6_35B_A3B_M2_E2E_SHA") or M2_SHA
    floor = gate.env_float("JITENGINE_QWEN3_6_35B_A3B_M2_E2E_MIN_TOKS")
    server = gate_server(
        [
            "-m",
            "jitengine_qwen3_6_35b_a3b_m2",
            "serve",
            "--profile",
            "24gb-static",
            "--model-path",
            model,
        ]
    )
    result = gate.run(server, model, str(tmp_path / "greedy.json"))
    stats = server.get("/stats")
    rate = float(result["decode_tok_per_sec_median"])
    shas = gate.shas(result)
    record_testsuite_property("qwen36_decode_tok_per_sec", rate)  # in a junit report
    record_testsuite_property("qwen36_text_sha", ",".join(shas))
    print(f"\n[e2e] Qwen3.6 greedy-512: text_sha {shas}, decode {rate:.2f} tok/s (median of 3)")

    problems = []
    if not result["valid"]:
        problems.append(f"the benchmark run is invalid: {result['problems']}")
    if len(shas) != 3 or set(shas) != {want}:
        problems.append(f"text_sha {shas}, want {want} on every repeat")
    checks = stats["kernel_checks"]
    off = sorted(k for k, ok in checks.items() if ok is not True)
    if off or not checks:
        problems.append(f"kernel checks not all true: {off} {stats['kernel_check_errors']}")
    dense = {
        k: v for k, v in stats["dense_kernels"].items() if isinstance(v, dict) and "differ" in v
    }
    bad = {k: {"differ": v["differ"], "raised": v["raised"]} for k, v in dense.items()}
    bad = {k: v for k, v in bad.items() if v["differ"] or v["raised"]}
    if bad or not dense:
        problems.append(f"dense kernels that differ from stock or raised: {bad or dense}")
    if floor is not None and rate < floor:
        problems.append(
            f"decode {rate:.2f} tok/s is below JITENGINE_QWEN3_6_35B_A3B_M2_E2E_MIN_TOKS={floor}"
        )
    assert not problems, "\n".join(problems)
