"""The command line, run in process (`main([...])`) so coverage sees it: help text, the resolved
configuration and its precedence, `inspect`, `pack`, `doctor`, `status`, the `serve` refusals and
a `serve` whose engine is replaced by a recorder. Nothing here needs Metal, downloads anything or
starts a server."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import urllib.error
import urllib.request

import pytest

from jitengine_qwen3_6_35b_a3b_m4pro import cli

GIB = 1 << 30
PROG = "jitengine-qwen3.6-35b-a3b-m4pro"


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """`serve` configures the package's logger; put it back so other tests see the default."""
    pkg = logging.getLogger("jitengine_qwen3_6_35b_a3b_m4pro")
    state = (list(pkg.handlers), pkg.level, pkg.propagate)
    yield
    pkg.handlers[:] = state[0]
    pkg.setLevel(state[1])
    pkg.propagate = state[2]


@pytest.fixture(autouse=True)
def _a_24_gib_host(monkeypatch):
    """The plan caps memory at physical RAM less the system reserve, so pin RAM to 24 GiB, the M2's,
    which the plans expected below assume: the tests then read the same on any host, CI's small
    runners included. This build also sizes its default budget and prefix store by the host's RAM
    when engine.args is imported (34 GiB and 1152 MiB at 40 GiB or more), so those four defaults
    are set to their 24 GiB values too."""
    real = os.sysconf
    pages = 24 * GIB // real("SC_PAGE_SIZE")
    monkeypatch.setattr(
        os, "sysconf", lambda name: pages if name == "SC_PHYS_PAGES" else real(name)
    )
    from jitengine_qwen3_6_35b_a3b_m4pro.engine import args
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    for table, key, value in (
        (args.DEFAULTS, "process_budget_bytes", budget.host_default_budget_bytes()),
        (args.DEFAULTS, "prefix_cache_bytes", budget.host_default_prefix_bytes()),
        (args.BUILTIN, "memory_budget_bytes", budget.host_default_budget_bytes()),
        (args.BUILTIN, "prefix_cache_bytes", budget.host_default_prefix_bytes()),
    ):
        monkeypatch.setitem(table, key, value)


@pytest.fixture
def checkpoint(tmp_path, fake_checkpoint):
    return fake_checkpoint(str(tmp_path / "model"))


def _run(argv, capsys):
    rc = cli.main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _config(out: str) -> tuple[dict, dict]:
    doc = json.loads(out)
    assert doc["schema_version"] == 1
    return doc["resolved"], doc["origins"]


# ---------------------------------------------------------------- help and version
def test_version_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith(PROG + " ")


def test_no_command_prints_the_top_level_help_and_succeeds(capsys):
    rc, out, _ = _run([], capsys)
    assert rc == 0
    for cmd in ("doctor", "inspect", "pack", "serve", "status"):
        assert cmd in out


@pytest.mark.parametrize("cmd", ["doctor", "inspect", "pack", "serve", "status"])
def test_each_command_has_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([cmd, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert f"{PROG} {cmd}" in out


def test_serve_help_documents_the_precedence_and_both_budget_flags(capsys):
    with pytest.raises(SystemExit):
        cli.main(["serve", "--help"])
    out = capsys.readouterr().out
    assert "defaults < --profile < --config < flags" in out
    assert "--memory-budget-gib" in out and "--memory-budget-gb" in out
    assert "--log-level" in out and "--preflight-wait" in out


# ---------------------------------------------------------------- serve --print-config
def test_print_config_with_no_options_is_the_builtin_defaults(capsys):
    rc, out, _ = _run(["serve", "--print-config"], capsys)
    cfg, origin = _config(out)
    assert rc == 0
    assert cfg["memory_budget_bytes"] == 18 * GIB and cfg["system_reserve_bytes"] == 6 * GIB
    assert (cfg["host"], cfg["port"]) == ("127.0.0.1", 8080)
    assert cfg["mode"] == "auto" and cfg["memory_policy"] == "static"
    assert set(origin.values()) == {"default"}
    assert "model_path" not in cfg


def test_print_config_with_a_profile_names_the_profile_as_origin(capsys):
    _, out, _ = _run(["serve", "--print-config", "--profile", "24gb-adaptive"], capsys)
    cfg, origin = _config(out)
    assert (cfg["mode"], cfg["memory_policy"]) == ("offload", "adaptive")
    assert origin["mode"] == "profile:24gb-adaptive"


def test_flags_beat_the_config_file_which_beats_the_profile(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_slots": 2,
                "max_context_tokens": 4096,
                "prefill_chunk_tokens": 64,
                "port": 9001,
                "mode": "offload",
            }
        )
    )
    argv = ["serve", "--print-config", "--profile", "24gb-static", "--config", str(path)]
    _, out, _ = _run(argv + ["--max-slots", "3", "--port", "9002", "--prefill-chunk", "32"], capsys)
    cfg, origin = _config(out)
    assert cfg["max_slots"] == 3 and origin["max_slots"] == "flag"
    assert cfg["port"] == 9002 and origin["port"] == "flag"
    assert cfg["prefill_chunk_tokens"] == 32 and origin["prefill_chunk_tokens"] == "flag"
    assert cfg["max_context_tokens"] == 4096
    assert origin["max_context_tokens"] == f"config:{path}"
    assert cfg["mode"] == "offload" and origin["mode"] == f"config:{path}"
    assert cfg["memory_policy"] == "static" and origin["memory_policy"] == "profile:24gb-static"
    assert (
        cfg["max_queued_requests"] == 16 and origin["max_queued_requests"] == "profile:24gb-static"
    )


def test_default_host_and_port_do_not_override_a_config_file(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema_version": 1, "host": "0.0.0.0", "port": 9100}))
    _, out, _ = _run(["serve", "--print-config", "--config", str(path)], capsys)
    cfg, origin = _config(out)
    assert (cfg["host"], cfg["port"]) == ("0.0.0.0", 9100)
    assert origin["host"] == origin["port"] == f"config:{path}"
    _, out, _ = _run(["serve", "--print-config", "--config", str(path), "--host", "::1"], capsys)
    cfg, origin = _config(out)
    assert cfg["host"] == "::1" and origin["host"] == "flag" and cfg["port"] == 9100


def test_every_flag_lands_in_the_resolved_config(tmp_path, capsys):
    pack = tmp_path / "x.experts"
    argv = [
        "serve", "--print-config", "--model-path", str(tmp_path / "m"),
        "--memory-budget-gib", "12.5", "--reserve-system-gib", "4",
        "--mode", "offload", "--memory-policy", "adaptive",
        "--max-slots", "2", "--max-queued", "5", "--max-context", "1024",
        "--prefill-chunk", "128", "--prefix-cache-mib", "64",
        "--expert-pack-path", str(pack), "--build-expert-pack",
    ]  # fmt: skip
    rc, out, _ = _run(argv, capsys)
    cfg, origin = _config(out)
    assert rc == 0
    assert cfg["model_path"] == str(tmp_path / "m")
    assert cfg["memory_budget_bytes"] == int(12.5 * GIB)
    assert cfg["system_reserve_bytes"] == 4 * GIB
    assert (cfg["mode"], cfg["memory_policy"]) == ("offload", "adaptive")
    assert (cfg["max_slots"], cfg["max_queued_requests"], cfg["max_context_tokens"]) == (2, 5, 1024)
    assert cfg["prefill_chunk_tokens"] == 128 and cfg["prefix_cache_bytes"] == 64 << 20
    assert cfg["expert_pack_path"] == str(pack) and cfg["build_expert_pack"] is True
    assert origin["build_expert_pack"] == "flag"


def test_the_decimal_gb_flag_is_decimal_and_the_gib_flag_binary(capsys):
    _, out, _ = _run(["serve", "--print-config", "--memory-budget-gb", "18"], capsys)
    assert _config(out)[0]["memory_budget_bytes"] == 18 * 10**9
    _, out, _ = _run(["serve", "--print-config", "--memory-budget-gib", "18"], capsys)
    assert _config(out)[0]["memory_budget_bytes"] == 18 * GIB


# ---------------------------------------------------------------- usage errors
def test_both_budget_flags_are_refused_with_exit_2(capsys):
    argv = ["serve", "--print-config", "--memory-budget-gib", "1", "--memory-budget-gb", "1"]
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "not both" in err and err.startswith(f"{PROG}: error:")


@pytest.mark.parametrize(
    "flag, value, message",
    [
        ("--memory-budget-gib", "abc", "not a number"),
        ("--memory-budget-gib", "-1", "non-negative"),
        ("--memory-budget-gb", "nan", "finite"),
        ("--reserve-system-gib", "inf", "finite"),
        ("--prefix-cache-mib", "x", "not a number"),
        ("--max-slots", "1.5", "invalid int value"),
        ("--max-context", "many", "invalid int value"),
        ("--port", "http", "invalid int value"),
        ("--mode", "fast", "invalid choice"),
        ("--memory-policy", "lazy", "invalid choice"),
        ("--log-level", "loud", "invalid choice"),
    ],
)
def test_bad_flag_values_are_argparse_errors(flag, value, message, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--print-config", flag, value])
    assert e.value.code == 2
    assert message in capsys.readouterr().err


def test_unknown_command_and_missing_required_flag_exit_2(capsys):
    for argv in (["bogus"], ["inspect"], ["pack"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2
        capsys.readouterr()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": 1, "nope": 1}, "unknown key"),
        ({"schema_version": 1, "max_slots": 1.5}, "must be an integer"),
        ({"schema_version": 1, "mode": "fast"}, "must be one of"),
        ({"schema_version": 1, "mode": "resident", "memory_policy": "adaptive"}, "adaptive"),
    ],
)
def test_a_bad_config_file_is_reported_not_raised(tmp_path, payload, message, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--print-config", "--config", str(path)])
    assert e.value.code == 2
    assert message in capsys.readouterr().err


def test_a_missing_config_file_and_an_unknown_profile_are_reported(tmp_path, capsys):
    for argv, message in (
        (["--config", str(tmp_path / "absent.json")], "cannot read config"),
        (["--profile", "nope"], "unknown profile"),
    ):
        with pytest.raises(SystemExit) as e:
            cli.main(["serve", "--print-config", *argv])
        assert e.value.code == 2
        assert message in capsys.readouterr().err


def test_adaptive_policy_with_resident_mode_is_refused_on_the_command_line(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", "--print-config", "--mode", "resident", "--memory-policy", "adaptive"])
    assert e.value.code == 2
    assert "adaptive" in capsys.readouterr().err


# ---------------------------------------------------------------- inspect
def test_inspect_json_reports_the_inventory_and_a_resident_plan(checkpoint, capsys):
    rc, out, _ = _run(["inspect", "--model-path", checkpoint, "--json", "--no-metal"], capsys)
    rep = json.loads(out)
    assert rc == 0
    assert rep["model"]["model_type"] == "qwen3_5_moe"
    assert rep["model"]["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert rep["model"]["family_tensors"] == {"dense": 146, "expert": 36, "vision": 1}
    assert "overrides" not in rep["model"]
    assert rep["num_expert_records"] == 32 and rep["expert_record_bytes"] == 13824
    assert rep["expert_record_stride"] >= rep["expert_record_bytes"]
    assert rep["metal"] == {"error": "skipped (--no-metal)"}
    assert rep["plan"]["mode"] == "resident"
    assert rep["plan"]["process_cap_bytes"] == 18 * GIB
    assert rep["plan_text"].startswith("resident:")


def test_inspect_text_names_the_checkpoint_and_says_metal_was_skipped(checkpoint, capsys):
    rc, out, _ = _run(["inspect", "--model-path", checkpoint, "--no-metal"], capsys)
    assert rc == 0
    assert f"{checkpoint}: qwen3_5_moe, affine 4-bit g64" in out
    assert "32 records x 13824 B" in out and "plan: resident:" in out
    assert "Metal limits unknown: skipped (--no-metal)" in out


def test_inspect_uses_the_resolved_budget_and_can_force_offload(checkpoint, capsys):
    argv = ["inspect", "--model-path", checkpoint, "--json", "--no-metal"]
    _, out, _ = _run(argv + ["--memory-budget-gib", "3", "--mode", "offload"], capsys)
    rep = json.loads(out)
    assert rep["plan"]["mode"] == "offload"
    assert rep["plan"]["process_cap_bytes"] == 3 * GIB


def test_inspect_with_a_budget_too_small_exits_1_with_the_plan_error(checkpoint, capsys):
    argv = ["inspect", "--model-path", checkpoint, "--json", "--no-metal"]
    rc, out, _ = _run(argv + ["--memory-budget-gib", "0.01"], capsys)
    rep = json.loads(out)
    assert rc == 1 and "plan" not in rep and rep["plan_error"]


def test_inspect_text_prints_the_plan_error_for_a_too_small_budget(checkpoint, capsys):
    argv = ["inspect", "--model-path", checkpoint, "--no-metal", "--memory-budget-gib", "0.01"]
    rc, out, _ = _run(argv, capsys)
    assert rc == 1 and "  plan: " in out


def test_inspect_without_a_model_path_in_the_config_is_not_reachable_from_argparse(capsys):
    # --model-path is required by the parser, so the `cfg["model_path"]` check only guards a
    # namespace built by hand; here it must still refuse cleanly.
    import argparse

    ns = argparse.Namespace(
        profile=None, config=None, model_path=None, json=False, no_metal=True,
        memory_budget_gib=None, memory_budget_gb=None, reserve_system_gib=None, mode=None,
        memory_policy=None, max_slots=None, max_queued=None, max_context=None,
        prefill_chunk=None, prefix_cache_mib=None, expert_pack_path=None,
        build_expert_pack=False, host="127.0.0.1", port=8080,
    )  # fmt: skip
    assert cli.cmd_inspect(ns) == 2
    assert "--model-path is required" in capsys.readouterr().err


def test_inspect_an_unreadable_checkpoint_exits_1(tmp_path, capsys):
    rc, _, err = _run(["inspect", "--model-path", str(tmp_path / "absent"), "--no-metal"], capsys)
    assert rc == 1
    assert "unsupported or unreadable checkpoint" in err


def test_inspect_a_malformed_checkpoint_exits_1(tmp_path, fake_checkpoint, capsys):
    def drop_a_tensor(entries, cfg):
        del entries["language_model.model.norm.weight"]

    path = fake_checkpoint(str(tmp_path / "bad"), mutate=drop_a_tensor)
    rc, _, err = _run(["inspect", "--model-path", path, "--no-metal"], capsys)
    assert rc == 1 and "unsupported or unreadable checkpoint" in err


def test_inspect_with_metal_reads_the_device_limits(checkpoint, monkeypatch, capsys):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    info = {"device_name": "Fake GPU", "architecture": "applegpu_g14g",
            "max_recommended_working_set_size": 10 * GIB}  # fmt: skip
    monkeypatch.setattr(budget, "device_info_or_none", lambda: (info, None))
    rc, out, _ = _run(["inspect", "--model-path", checkpoint, "--json"], capsys)
    rep = json.loads(out)
    assert rc == 0 and rep["metal"]["device_name"] == "Fake GPU"
    assert rep["plan"]["limits"]["metal_bytes"] == 10 * GIB
    rc, out, _ = _run(["inspect", "--model-path", checkpoint], capsys)
    assert rc == 0 and "Metal limits unknown" not in out


# ---------------------------------------------------------------- pack
def test_pack_builds_validates_skips_and_rebuilds(checkpoint, tmp_path, capsys):
    out_path = str(tmp_path / "p.experts")
    argv = ["pack", "--model-path", checkpoint, "--output", out_path]
    rc, out, _ = _run(argv, capsys)
    assert rc == 0 and f"{out_path}: valid" in out and "[pack] building" in out
    rc, out, _ = _run(argv, capsys)
    assert rc == 0 and "nothing to do" in out and "[pack] building" not in out
    rc, out, _ = _run([*argv, "--force"], capsys)
    assert rc == 0 and f"{out_path}: rebuilding" in out and f"{out_path}: valid" in out


def test_pack_rejects_and_replaces_a_file_that_is_not_a_pack(checkpoint, tmp_path, capsys):
    out_path = tmp_path / "p.experts"
    out_path.write_bytes(b"not a pack")
    rc, out, _ = _run(["pack", "--model-path", checkpoint, "--output", str(out_path)], capsys)
    assert rc == 0 and "rejecting" in out and "bad magic" in out and f"{out_path}: valid" in out


def test_pack_default_output_is_beside_the_checkpoint_pack_path_function(
    checkpoint, monkeypatch, tmp_path, capsys
):
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint import pack

    target = str(tmp_path / "default.experts")
    monkeypatch.setattr(pack, "default_pack_path", lambda scan: target)
    rc, out, _ = _run(["pack", "--model-path", checkpoint], capsys)
    assert rc == 0 and f"{target}: valid" in out


def test_pack_of_a_missing_checkpoint_exits_1(tmp_path, capsys):
    rc, _, err = _run(["pack", "--model-path", str(tmp_path / "absent")], capsys)
    assert rc == 1 and "cannot scan" in err


def test_pack_refuses_when_the_disk_is_too_full(checkpoint, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_free_bytes", lambda path: 1024)
    rc, _, err = _run(["pack", "--model-path", checkpoint, "--output", str(tmp_path / "p")], capsys)
    assert rc == 1 and "needs" in err and "slack" in err and "free" in err


def test_pack_reports_a_build_failure(checkpoint, tmp_path, monkeypatch, capsys):
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint import pack

    def boom(scan, out, log=None):
        raise pack.PackError("disk on fire")

    monkeypatch.setattr(pack, "build_pack", boom)
    rc, _, err = _run(["pack", "--model-path", checkpoint, "--output", str(tmp_path / "p")], capsys)
    assert rc == 1 and "disk on fire" in err


def test_pack_exits_1_when_the_built_file_does_not_validate(
    checkpoint, tmp_path, monkeypatch, capsys
):
    from jitengine_qwen3_6_35b_a3b_m4pro.checkpoint import pack

    monkeypatch.setattr(pack, "build_pack", lambda scan, out, log=None: None)
    monkeypatch.setattr(pack, "validate", lambda path, scan: (False, "stale"))
    rc, out, _ = _run(["pack", "--model-path", checkpoint, "--output", str(tmp_path / "p")], capsys)
    assert rc == 1 and "INVALID: stale" in out


def test_free_bytes_of_a_real_directory_and_of_an_impossible_one(tmp_path):
    assert cli._free_bytes(str(tmp_path / "new" / "dir")) > 0
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert cli._free_bytes(str(blocker / "sub")) is None


# ---------------------------------------------------------------- doctor
class _Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode


@pytest.fixture
def no_metal(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    monkeypatch.setattr(budget, "device_info_or_none", lambda: (None, "no Metal here"))


def test_doctor_json_without_metal_reports_the_fields_and_is_unsupported(no_metal, capsys):
    rc, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 1 and rep["supported"] is False
    assert rep["jitengine-qwen3.6-35b-a3b-m4pro"]
    assert rep["python"] == sys.version.split()[0]
    for key in ("platform", "machine", "physical_bytes", "packages", "memory_telemetry"):
        assert key in rep
    assert rep["metal"] == {"error": "no Metal here"}
    assert "tuned_on" not in rep and "quant_ops" not in rep
    assert set(rep["packages"]) == {"mlx", "mlx-metal", "mlx-lm", "numpy", "transformers"}
    assert rep["physical_bytes"] > 0
    tele = rep["memory_telemetry"]
    assert {"process_footprint", "system_memory"} <= set(tele)


def test_doctor_text_without_metal_says_unavailable_and_unsupported(no_metal, capsys):
    rc, out, _ = _run(["doctor"], capsys)
    assert rc == 1
    assert "Metal: unavailable (no Metal here)" in out
    assert "supported for serving: False" in out
    assert "physical memory" in out and "footprint telemetry:" in out


def test_doctor_on_an_apple_silicon_mac_is_supported(monkeypatch, capsys):
    import platform

    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    info = {"device_name": "Apple M2", "architecture": "applegpu_g14g",
            "max_recommended_working_set_size": 16 * GIB, "max_buffer_length": 8 * GIB}  # fmt: skip
    monkeypatch.setattr(budget, "device_info_or_none", lambda: (info, None))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "_probe_quant_ops", lambda bits: {"quantized_matmul": True})
    import subprocess

    def fake_run(cmd, **kw):
        return _Done("Apple M2\n") if "machdep.cpu.brand_string" in cmd else _Done("20000\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 0 and rep["supported"] is True
    assert rep["chip"] == "Apple M2" and rep["iogpu_wired_limit_mb"] == 20000
    assert rep["tuned_on"]["gpu"] == "applegpu_g16s" and rep["tuned_on"]["note"]
    assert rep["quant_ops"] == {"quantized_matmul": True}
    rc, out, _ = _run(["doctor"], capsys)
    assert rc == 0
    assert "Metal: Apple M2 applegpu_g14g, recommended working set 16.00 GiB" in out
    assert "4-bit ops:" in out and "supported for serving: True" in out and "(Apple M2)" in out


def test_doctor_survives_failing_sysctl_and_package_lookups(no_metal, monkeypatch, capsys):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(subprocess, "run", boom)
    rc, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 1
    assert rep["chip"] is None and "sysctl" in rep["chip_error"]
    assert rep["iogpu_wired_limit_mb"] is None


def test_doctor_sysctl_nonzero_exit_is_none(no_metal, monkeypatch, capsys):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Done("", 1))
    _, out, _ = _run(["doctor", "--json"], capsys)
    rep = json.loads(out)
    assert rep["chip"] is None and rep["iogpu_wired_limit_mb"] is None


def test_package_version_helper_returns_none_for_an_absent_package():
    import importlib.metadata as md

    assert cli._version(md, "definitely-not-an-installed-distribution") is None
    assert cli._version(md, "pytest")


def test_quant_probe_runs_on_any_mlx_device(requires_mlx_any):
    rep = cli._probe_quant_ops(cli.SERVED_BITS)
    assert "error" in rep or (
        {k: rep[k] for k in ("quantized_matmul", "gather_qmm", "dequantize")}
        == {"quantized_matmul": True, "gather_qmm": True, "dequantize": True}
        and isinstance(
            rep["gather_equals_matmul"], bool
        )  # rounding of the two kernels varies by GPU
    )


def test_quant_probe_reports_an_op_failure(requires_mlx_any, monkeypatch):
    import mlx.core as mx

    def boom(*a, **k):
        raise RuntimeError("no 4-bit")

    monkeypatch.setattr(mx, "quantize", boom)
    assert cli._probe_quant_ops(cli.SERVED_BITS) == {"error": "RuntimeError('no 4-bit')"}


def test_tuned_on_describes_a_mac_of_the_tuned_family_and_another(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro.kernels import selfcheck

    same = cli._tuned_on({"device_name": "Apple M4 Pro", "architecture": selfcheck.TUNED_GPU})
    assert same["mac"] == selfcheck.TUNED_MAC and same["gpu"] == selfcheck.TUNED_GPU
    assert same["note"]
    other = cli._tuned_on({"device_name": "Apple M2", "architecture": "applegpu_g14g"})
    assert other["note"] and other["note"] != same["note"]


# ---------------------------------------------------------------- status
def _urlopen_returning(routes):
    def urlopen(url, timeout=None):
        for path, result in routes.items():
            if url.endswith(path):
                if isinstance(result, Exception):
                    raise result
                return io.BytesIO(json.dumps(result).encode())
        raise AssertionError(url)

    return urlopen


def test_status_of_an_unreachable_server_reports_the_error_and_exits_1(capsys):
    argv = ["status", "--url", "http://127.0.0.1:1", "--timeout", "0.5", "--json"]
    rc, out, _ = _run(argv, capsys)
    rep = json.loads(out)
    assert rc == 1 and rep["url"] == "http://127.0.0.1:1"
    assert "error" in rep["health"] and "error" in rep["stats"]


def test_status_text_of_an_unreachable_server_still_prints(capsys):
    rc, out, _ = _run(["status", "--url", "http://127.0.0.1:1", "--timeout", "0.5"], capsys)
    assert rc == 1 and out.startswith("http://127.0.0.1:1: ")
    assert "plan: None" in out and "MLX active n/a" in out


def test_status_of_a_healthy_server_exits_0_and_prints_the_memory_state(monkeypatch, capsys):
    stats = {
        "memory_mode": "resident",
        "memory_plan": "resident: fits",
        "memory": {
            "mlx_active_bytes": 3 * GIB,
            "observed": {"process": {"footprint_bytes": 4 * GIB}},
            "last_resize": {"status": "ok"},
        },
    }
    routes = {"/health": {"status": "ok", "admission": "open"}, "/stats": stats}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning(routes))
    rc, out, _ = _run(["status", "--url", "http://host:1/"], capsys)
    assert rc == 0
    assert "http://host:1/: ok; mode resident; admission open" in out
    assert "plan: resident: fits" in out
    assert "MLX active 3.00 GiB, footprint 4.00 GiB, last resize ok" in out
    rc, out, _ = _run(["status", "--url", "http://host:1/", "--json"], capsys)
    assert rc == 0 and json.loads(out)["stats"]["memory_mode"] == "resident"


def test_status_records_an_http_error_body_and_exits_1(monkeypatch, capsys):
    err = urllib.error.HTTPError("http://h/health", 503, "busy", {}, io.BytesIO(b"draining"))
    routes = {"/health": err, "/stats": {"memory_mode": "offload"}}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning(routes))
    rc, out, _ = _run(["status", "--url", "http://h", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 1
    assert rep["health"] == {"http_status": 503, "body": "draining"}
    assert rep["stats"] == {"memory_mode": "offload"}


# ---------------------------------------------------------------- serve (everything before the engine)
def test_serve_without_a_model_path_is_refused(capsys):
    rc, _, err = _run(["serve"], capsys)
    assert rc == 2 and "--model-path is required" in err


def test_serve_of_a_directory_without_config_json_is_refused(tmp_path, capsys):
    rc, _, err = _run(["serve", "--model-path", str(tmp_path)], capsys)
    assert rc == 2 and "not a checkpoint directory" in err


def test_serve_refuses_without_metal(checkpoint, no_metal, capsys):
    rc, _, err = _run(["serve", "--model-path", checkpoint], capsys)
    assert rc == 1 and "Metal is unavailable" in err and "no Metal here" in err


def test_serve_reports_a_budget_that_cannot_hold_the_model(checkpoint, monkeypatch, capsys):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    monkeypatch.setattr(budget, "device_info_or_none", lambda: ({"device_name": "x"}, None))
    rc, _, err = _run(["serve", "--model-path", checkpoint, "--memory-budget-gib", "0.01"], capsys)
    assert rc == 1 and err.startswith(f"{PROG}: error:")


def test_serve_reports_a_checkpoint_without_shards(tmp_path, fake_device, capsys):
    (tmp_path / "config.json").write_text("{}")
    rc, _, err = _run(["serve", "--model-path", str(tmp_path)], capsys)
    assert rc == 1 and "no safetensors shards" in err


@pytest.fixture
def fake_device(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import budget

    info = {"device_name": "Fake", "architecture": "applegpu_g14g"}
    monkeypatch.setattr(budget, "device_info_or_none", lambda: (info, None))


def test_serve_in_offload_mode_needs_an_expert_pack(checkpoint, fake_device, tmp_path, capsys):
    pack = tmp_path / "absent.experts"
    argv = [
        "serve",
        "--model-path",
        checkpoint,
        "--mode",
        "offload",
        "--expert-pack-path",
        str(pack),
    ]
    rc, _, err = _run(argv, capsys)
    assert rc == 1
    assert "offload mode needs an expert pack" in err and "does not exist" in err
    assert f"{PROG} pack --model-path {checkpoint}" in err and "--build-expert-pack" in err


def test_serve_in_offload_mode_rejects_a_stale_pack(checkpoint, fake_device, tmp_path, capsys):
    pack = tmp_path / "stale.experts"
    pack.write_bytes(b"junk")
    argv = [
        "serve",
        "--model-path",
        checkpoint,
        "--mode",
        "offload",
        "--expert-pack-path",
        str(pack),
    ]
    rc, _, err = _run(argv, capsys)
    assert rc == 1 and "offload mode needs an expert pack" in err and "bad magic" in err


def test_serve_refuses_when_the_preflight_fails(checkpoint, fake_device, monkeypatch, capsys):
    seen = {}

    def preflight(planned, wait):
        seen.update(planned=planned, wait=wait)
        return False, "short by 3 GiB"

    monkeypatch.setattr(cli, "_preflight", preflight)
    rc, _, err = _run(["serve", "--model-path", checkpoint, "--preflight-wait", "7"], capsys)
    assert rc == 1 and seen["wait"] == 7.0 and seen["planned"] > 0
    assert "not enough available memory" in err and "short by 3 GiB" in err and "after 7s" in err


class _RecordingEngine:
    def __init__(self, path, **opt):
        self.path, self.opt, self.served, self.closed = path, opt, None, False

    def serve(self, host, port, allowed_hosts=()):
        self.served = (host, port)
        self.allowed_hosts = tuple(allowed_hosts)

    def close(self):
        self.closed = True


@pytest.fixture
def recorded_engine(monkeypatch, fake_device):
    from jitengine_qwen3_6_35b_a3b_m4pro.engine import engine as engine_module

    made = []

    def create(path, **opt):
        made.append(_RecordingEngine(path, **opt))
        return made[-1]

    monkeypatch.setattr(engine_module, "create_engine", create)
    monkeypatch.setattr(cli, "_preflight", lambda planned, wait: (True, "enough"))
    return made


def test_serve_builds_the_engine_from_the_resolved_options_and_closes_it(
    checkpoint, recorded_engine, capsys
):
    argv = ["serve", "--model-path", checkpoint, "--host", "0.0.0.0", "--port", "9123",
            "--max-slots", "2", "--log-level", "warning"]  # fmt: skip
    rc, _, _ = _run(argv, capsys)
    assert rc == 0
    (engine,) = recorded_engine
    assert engine.path == checkpoint and engine.served == ("0.0.0.0", 9123) and engine.closed
    assert engine.opt["max_slots"] == 2
    assert engine.opt["process_budget_bytes"] == 18 * GIB
    assert logging.getLogger("jitengine_qwen3_6_35b_a3b_m4pro").level == logging.WARNING


def test_serve_closes_the_engine_when_serving_raises(checkpoint, recorded_engine, monkeypatch):
    def boom(self, host, port, allowed_hosts=()):
        raise RuntimeError("socket gone")

    monkeypatch.setattr(_RecordingEngine, "serve", boom)
    with pytest.raises(RuntimeError, match="socket gone"):
        cli.main(["serve", "--model-path", checkpoint])
    assert recorded_engine[0].closed


def test_serve_in_offload_mode_hands_a_valid_pack_to_the_engine(
    checkpoint, recorded_engine, tmp_path, capsys
):
    pack = str(tmp_path / "ok.experts")
    assert cli.main(["pack", "--model-path", checkpoint, "--output", pack]) == 0
    capsys.readouterr()
    argv = ["serve", "--model-path", checkpoint, "--mode", "offload", "--expert-pack-path", pack]
    rc, _, _ = _run(argv, capsys)
    assert rc == 0 and recorded_engine[0].opt["expert_pack_path"] == pack


def test_serve_with_build_expert_pack_skips_the_pack_check(
    checkpoint, recorded_engine, tmp_path, capsys
):
    argv = ["serve", "--model-path", checkpoint, "--mode", "offload", "--build-expert-pack",
            "--expert-pack-path", str(tmp_path / "later.experts")]  # fmt: skip
    rc, _, _ = _run(argv, capsys)
    assert rc == 0 and recorded_engine[0].opt["build_expert_pack"] is True


# ---------------------------------------------------------------- preflight
def test_preflight_waits_then_gives_up_and_passes_when_telemetry_is_missing(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import monitor

    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    readings = iter([2 * GIB, 2 * GIB, 9 * GIB])
    monkeypatch.setattr(
        monitor,
        "system_memory",
        lambda: {"available_bytes_estimate": next(readings), "swap_used_bytes": GIB},
    )
    ok, why = cli._preflight(8 * GIB, 60)
    assert ok and sleeps == [2.0, 2.0] and "available (estimate) 9.00 GiB" in why

    monkeypatch.setattr(
        monitor,
        "system_memory",
        lambda: {"available_bytes_estimate": 2 * GIB, "swap_used_bytes": 0},
    )
    ok, why = cli._preflight(8 * GIB, 0)
    assert not ok and "short by 6.00 GiB" in why

    def unavailable():
        raise OSError("no host_statistics64")

    monkeypatch.setattr(monitor, "system_memory", unavailable)
    ok, why = cli._preflight(8 * GIB, 0)
    assert ok and "preflight skipped" in why and "no host_statistics64" in why


def test_preflight_counts_cached_files_as_available(monkeypatch):
    from jitengine_qwen3_6_35b_a3b_m4pro.memory import monitor

    # after reading a large file most memory is cached files, which the inactive-queue estimate misses
    monkeypatch.setattr(
        monitor,
        "system_memory",
        lambda: {
            "available_bytes_estimate": 11 * GIB,
            "available_bytes": 20 * GIB,
            "swap_used_bytes": 0,
        },
    )
    ok, why = cli._preflight(18 * GIB, 0)
    assert ok and "available (estimate) 20.00 GiB" in why


def test_gib_formats_none_and_sizes():
    assert cli._gib(None) == "n/a" and cli._gib(GIB) == "1.00 GiB"


def test_serve_passes_every_allowed_host_to_the_server(checkpoint, recorded_engine, capsys):
    argv = ["serve", "--model-path", checkpoint, "--allowed-host", "mac.local",
            "--allowed-host", "api.internal"]  # fmt: skip
    rc, _, _ = _run(argv, capsys)
    assert rc == 0
    assert recorded_engine[0].allowed_hosts == ("mac.local", "api.internal")
    recorded_engine.clear()
    assert _run(["serve", "--model-path", checkpoint], capsys)[0] == 0
    assert recorded_engine[0].allowed_hosts == ()
