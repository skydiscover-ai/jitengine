"""Each agent's session plan: Pi's provider in an extension, OpenCode's config in its environment,
Claude Code at the engine's Messages API, and Codex at its Responses API with the model's metadata;
none of them writes an agent's own configuration."""

from __future__ import annotations

import json
import tomllib

from jit_launch.agents import CODEX_CATALOG, PI_EXTENSION, PROVIDER, Served, plan

SERVED = Served("http://127.0.0.1:8080", "/Users/demo/models/Qwen3.6-35B-A3B-3bit", 32768)


def test_the_output_budget_is_a_quarter_of_the_context_at_most_32768():
    assert SERVED.max_output == 8192
    assert Served("http://h:1", "flint-int4", 262144).max_output == 32768
    assert SERVED.label == "Qwen3.6-35B-A3B-3bit (JITEngine)"


def test_pi_registers_the_provider_from_a_session_extension():
    p = plan("pi", SERVED, ["-p", "hi"])
    assert p.argv == [
        "pi",
        "-e",
        f"{{dir}}/{PI_EXTENSION}",
        "--provider",
        PROVIDER,
        "--model",
        SERVED.model,
        "-p",
        "hi",
    ]
    assert not p.env and list(p.files) == [PI_EXTENSION]
    text = p.files[PI_EXTENSION]
    call = text[
        text.index("pi.registerProvider(") + len("pi.registerProvider(") : text.rindex(");")
    ]
    name, provider = json.loads(f"[{call}]")
    assert name == PROVIDER and provider["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert provider["api"] == "openai-completions"
    (model,) = provider["models"]
    assert model["id"] == SERVED.model and not model["reasoning"]
    assert (model["contextWindow"], model["maxTokens"]) == (32768, 8192)


def test_opencode_takes_its_config_from_the_environment():
    p = plan("opencode", SERVED, ["run", "hi"])
    assert p.argv == ["opencode", "run", "hi"] and not p.files
    config = json.loads(p.env["OPENCODE_CONFIG_CONTENT"])
    provider = config["provider"][PROVIDER]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert provider["models"][SERVED.model]["limit"] == {"context": 32768, "output": 8192}
    assert config["model"] == f"{PROVIDER}/{SERVED.model}"


def test_claude_code_talks_to_the_engine_s_messages_api():
    p = plan("claude", SERVED, ["-p", "hi"])
    assert p.argv == ["claude", "--model", SERVED.model, "-p", "hi"] and not p.files
    assert p.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"  # the same port as /v1/chat
    assert p.env["ANTHROPIC_API_KEY"] == "" and p.env["ANTHROPIC_AUTH_TOKEN"] == "jitengine"
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert p.env[key] == SERVED.model
    assert p.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "32768"
    # below 64K, a quarter of the context as output would leave Claude Code compacting every turn
    assert p.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    large = plan("claude", Served("http://h:1", "flint-int4", 262144), [])
    assert large.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"


def test_codex_takes_its_provider_and_the_model_s_metadata_as_overrides():
    p = plan("codex", SERVED, ["exec", "hi"])
    assert p.argv[0] == "codex" and p.argv[-4:] == ["-m", SERVED.model, "exec", "hi"]
    overrides = p.argv[1:-4]
    assert overrides[::2] == ["-c"] * 3
    config = tomllib.loads("\n".join(o.replace("=", " = ", 1) for o in overrides[1::2]))
    assert config["model_provider"] == PROVIDER
    assert config["model_providers"][PROVIDER] == {
        "name": "JITEngine",
        "base_url": "http://127.0.0.1:8080/v1",
        "wire_api": "responses",
        "env_key": "JITENGINE_API_KEY",
    }
    assert config["model_catalog_json"] == f"{{dir}}/{CODEX_CATALOG}"
    assert p.env == {"JITENGINE_API_KEY": "jitengine"} and list(p.files) == [CODEX_CATALOG]
    (model,) = json.loads(p.files[CODEX_CATALOG])["models"]
    assert model["slug"] == SERVED.model and model["context_window"] == 32768
    assert model["input_modalities"] == ["text"] and not model["supports_search_tool"]
