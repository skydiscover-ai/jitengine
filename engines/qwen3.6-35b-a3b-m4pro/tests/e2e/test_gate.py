"""The Qwen3.6-35B-A3B end-to-end test on an M4 Pro: the greedy-512 text hash, every kernel check on,
decode.

As published (README.md) the server is `jitengine-qwen3.6-35b-a3b-m4pro serve --model-path M` (the
exact default; `--profile relaxed` is not exact and has no hash) and the check is `python
benchmarks/benchmark.py --model-path M --mode interactive --temperature 0 --max-tokens 512
--repeats 3 --output-json greedy.json`: the chat endpoint, task "dedent" (task seed 0), one warmup
request, then three repeats. `benchmark.run_stream` computes each repeat's `text_sha` as
`hashlib.sha256(out_text.encode()).hexdigest()[:16]` over the text the server streamed; with the
4-bit checkpoint every repeat gave 44d9aee27f3f890e on the 48 GB M4 Pro, at 194.6 tok/s (the
figure in the README). This test runs both commands as subprocesses, the server as
`python -m jitengine_qwen3_6_35b_a3b_m4pro serve`.

The engine checks its custom kernels against stock MLX on this Mac and switches off any that
differ, so the hash could hold while the measured kernels are off. /stats reports the checks, and
the test reads it after the run, since the dense kernels' prescaled and prefill shapes are checked
on first use: every `kernel_checks` entry (the load-time checks) must be true, and no dense kernel
in `dense_kernels` may differ from stock or have raised on any shape.

Run with `-s` to see the decode rate. Environment: JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_CHECKPOINT
(required: the checkpoint directory; the MTP head is found beside it as the engine finds it for a
user); JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_SHA (default 44d9aee27f3f890e, the M4 Pro's; on another
Mac, that Mac's reference); JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_MIN_TOKS (a decode tok/s floor; unset,
the rate is only reported). Needs `pip install '.[benchmark]'`.
"""

import os

PREFIX = "JITENGINE_QWEN3_6_35B_A3B_M4PRO_E2E_"
M4_PRO_SHA = "44d9aee27f3f890e"


def test_greedy_512_hash_kernel_checks_and_decode(
    gate, gate_server, tmp_path, record_testsuite_property
):
    model = gate.model(PREFIX + "CHECKPOINT", "a Qwen3.6-35B-A3B 4-bit checkpoint directory")
    want = os.environ.get(PREFIX + "SHA") or M4_PRO_SHA
    floor = gate.env_float(PREFIX + "MIN_TOKS")
    server = gate_server(["-m", "jitengine_qwen3_6_35b_a3b_m4pro", "serve", "--model-path", model])
    result = gate.run(server, model, str(tmp_path / "greedy.json"))
    stats = server.get("/stats")
    rate = float(result["decode_tok_per_sec_median"])
    shas = gate.shas(result)
    record_testsuite_property("qwen36_m4pro_decode_tok_per_sec", rate)  # in a junit report
    record_testsuite_property("qwen36_m4pro_text_sha", ",".join(shas))
    print(
        f"\n[e2e] Qwen3.6 M4 Pro greedy-512: text_sha {shas}, decode {rate:.2f} tok/s (median of 3)"
    )

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
        problems.append(f"decode {rate:.2f} tok/s is below {PREFIX}MIN_TOKS={floor}")
    assert not problems, "\n".join(problems)
