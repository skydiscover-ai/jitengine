"""The benchmark files are the frozen ones: their hashes match the recorded values."""

import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"
WORKLOAD = json.loads((BENCH / "workload.json").read_text())


def _sha(name):
    return hashlib.sha256((BENCH / name).read_bytes()).hexdigest()


def test_hashes_match_workload_and_readme():
    readme = (BENCH / "README.md").read_text()
    for name, digest in WORKLOAD["sha256"].items():
        assert _sha(name) == digest, f"{name} changed"
        assert digest in readme


def test_the_client_is_the_one_that_measured_skydiscover():
    assert (
        WORKLOAD["sha256"]["benchmark.py"]
        == "5b6b3f679e54a4483f2adea0171330ffd0f2003dc5b680d7cb62f73b2175784c"
    )


def test_scored_prompts_are_the_clients_built_in_list():
    spec = importlib.util.spec_from_file_location("client", BENCH / "benchmark.py")
    pytest_client = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(
        "tqdm", type(sys)("tqdm")
    )  # the client imports tqdm; the list needs none
    sys.modules["tqdm"].tqdm = lambda *a, **k: None
    try:
        spec.loader.exec_module(pytest_client)
    except ModuleNotFoundError:  # aiohttp is a benchmark extra
        return
    scored = [line for line in (BENCH / "prompts_scored.txt").read_text().splitlines() if line]
    assert scored == pytest_client.DEFAULT_PROMPTS


def test_heldout_prompts_are_disjoint_from_the_scored_ones():
    scored = set((BENCH / "prompts_scored.txt").read_text().split("\n"))
    held = [line for line in (BENCH / "prompts_heldout.txt").read_text().splitlines() if line]
    assert len(held) == 15 and not scored & set(held)


def _runner():
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location("bench_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_reports_medians_ranges_and_the_ratio():
    run = _runner()
    runs = [
        {
            "arm": "engine",
            "ok": True,
            "tokens_per_sec": s,
            "total_tokens": t,
            "within_bounds": True,
            "ttft_avg": 0.3,
            "latency_mean": 2.0,
        }
        for s, t in ((700, 8090), (760, 8085), (730, 8091))
    ] + [
        {
            "arm": "stock",
            "ok": True,
            "tokens_per_sec": s,
            "total_tokens": 8090,
            "within_bounds": True,
            "ttft_avg": 0.4,
            "latency_mean": 4.0,
        }
        for s in (470, 490, 480)
    ]
    runs.append({"arm": "stock", "ok": False})
    summary = run.summarize(runs, ["engine", "stock"])
    assert (
        summary["engine"]["tokens_per_sec_median"] == 730
        and summary["engine"]["total_tokens_min"] == 8085
    )
    assert summary["stock"]["failed_runs"] == 1 and summary["stock"]["runs"] == 3
    assert abs(summary["ratio_of_medians"]["engine/stock"] - 730 / 480) < 1e-9
