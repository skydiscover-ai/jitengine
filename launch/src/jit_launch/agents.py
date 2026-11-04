"""Each agent's configuration for a served model, and the command that starts it, for one session.

`Served` is what the engine reports. `plan(agent, ...)` returns a `Plan`: the command, the
environment it adds, and the files it reads, which cli.py writes to a temporary directory for the
session. Nothing goes into an agent's own configuration, as `ollama launch` and `unsloth start` work:

- Pi takes the provider from an extension (`pi -e`) that registers it for the session.
- OpenCode takes its whole config from `OPENCODE_CONFIG_CONTENT`.
- Claude Code talks to the engine's Anthropic Messages API, on the same port as its OpenAI one,
  through `ANTHROPIC_BASE_URL`.
- Codex talks to the engine's Responses API, with its provider and the model's metadata given as
  `-c` overrides of its config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PROVIDER = "jitengine"
PROVIDER_NAME = "JITEngine"
API_KEY = "jitengine"  # the engines take no key, and the agents refuse an empty one
MAX_OUTPUT_TOKENS = 32768
ZERO_COST = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
PI_EXTENSION = "jitengine.js"
CODEX_CATALOG = "codex-models.json"
CODEX_KEY_VAR = "JITENGINE_API_KEY"
# Claude Code compacts once a request reaches its context less its output tokens and a buffer of
# about 13K, and its first request is about 12.5K (12,490 tokens on our M2, with a short task):
# below 64K, a quarter of the context as output leaves it no room, so it compacts on every turn and
# stops ("Autocompact is thrashing")
CLAUDE_SMALL_CONTEXT = 65536
CLAUDE_SMALL_OUTPUT = 4096
CLAUDE_MODEL_VARS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)
INSTALL = {
    "pi": "npm install -g @earendil-works/pi-coding-agent",
    "claude": "curl -fsSL https://claude.ai/install.sh | bash",
    "opencode": "curl -fsSL https://opencode.ai/install | bash",
    "codex": "npm install -g @openai/codex",
}


@dataclass(frozen=True)
class Served:
    """The engine an agent is pointed at: its origin (scheme, host and port), the model's id, and
    the tokens one request may hold, prompt and output together."""

    origin: str
    model: str
    context: int

    @property
    def base_url(self) -> str:
        return f"{self.origin}/v1"

    @property
    def max_output(self) -> int:
        """Output tokens per request: a quarter of the context, which the agents keep free for
        the reply when they decide to compact, at most 32,768."""
        return max(1, min(self.context // 4, MAX_OUTPUT_TOKENS))

    @property
    def label(self) -> str:
        """The model's name for an agent's model picker: the checkpoint directory's name."""
        return f"{Path(self.model).name or self.model} ({PROVIDER_NAME})"


@dataclass
class Plan:
    """How to start an agent for one session: `argv`, the variables to set, and the files it
    reads, by name, which are written to a temporary directory that `{dir}` in `argv` names."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def _pi(served: Served, args: list[str]) -> Plan:
    provider = {
        "baseUrl": served.base_url,
        "api": "openai-completions",
        "apiKey": API_KEY,
        # the engines take the system role and no store, reasoning-effort or strict fields
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsStore": False,
            "supportsStrictMode": False,
        },
        "models": [
            {
                "id": served.model,
                "name": served.label,
                "reasoning": False,
                "input": ["text"],
                "contextWindow": served.context,
                "maxTokens": served.max_output,
                "cost": ZERO_COST,
            }
        ],
    }
    extension = (
        f"// {PROVIDER_NAME} at {served.base_url}, for one Pi session (jit-launch)\n"
        "export default function (pi) {\n"
        f"  pi.registerProvider({json.dumps(PROVIDER)}, {json.dumps(provider)});\n"
        "}\n"
    )
    argv = ["pi", "-e", f"{{dir}}/{PI_EXTENSION}", "--provider", PROVIDER, "--model", served.model]
    return Plan(argv=[*argv, *args], files={PI_EXTENSION: extension})


def _opencode(served: Served, args: list[str]) -> Plan:
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": PROVIDER_NAME,
                "options": {"baseURL": served.base_url, "apiKey": API_KEY},
                # without a context limit OpenCode reads it as 0 and never compacts
                "models": {
                    served.model: {
                        "name": served.label,
                        "limit": {"context": served.context, "output": served.max_output},
                    }
                },
            }
        },
        "model": f"{PROVIDER}/{served.model}",
    }
    return Plan(argv=["opencode", *args], env={"OPENCODE_CONFIG_CONTENT": json.dumps(config)})


def _claude(served: Served, args: list[str]) -> Plan:
    output = served.max_output
    if served.context < CLAUDE_SMALL_CONTEXT:
        output = min(output, CLAUDE_SMALL_OUTPUT)
    env = dict.fromkeys(CLAUDE_MODEL_VARS, served.model)
    env |= {
        "ANTHROPIC_BASE_URL": served.origin,
        "ANTHROPIC_AUTH_TOKEN": API_KEY,
        "ANTHROPIC_API_KEY": "",  # a key in the shell would win over the token
        # Claude Code takes both for ids that are not Claude's; it would assume 200K otherwise
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(served.context),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output),
        # the engines have no thinking budget, and the attribution header changes the system
        # prompt on every request, which defeats the engines' prefix caches
        "CLAUDE_CODE_DISABLE_THINKING": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    return Plan(argv=["claude", "--model", served.model, *args], env=env)


def _toml(value: str) -> str:
    """`value` as a TOML string: JSON's escapes are TOML's."""
    return json.dumps(value)


def _codex(served: Served, args: list[str]) -> Plan:
    # Codex's metadata for a model it does not know, as FreeToken's `ft launch codex` gives it;
    # without it Codex assumes a context and the tools of its own models
    model = {
        "slug": served.model,
        "display_name": served.label,
        "description": served.label,
        "context_window": served.context,
        "max_context_window": served.context,
        "effective_context_window_percent": 100,
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "input_modalities": ["text"],
        "base_instructions": "",
        "default_reasoning_summary": "none",
        "supported_reasoning_levels": [],
        "supports_reasoning_summaries": False,
        "experimental_supported_tools": [],
        "supports_search_tool": False,
        "web_search_tool_type": "text",
        "supports_image_detail_original": False,
        "support_verbosity": False,
        "default_verbosity": "low",
        "supports_parallel_tool_calls": False,
        "use_responses_lite": False,
        "model_messages": None,
        "upgrade": None,
    }
    provider = {
        "name": PROVIDER_NAME,
        "base_url": served.base_url,
        "wire_api": "responses",
        "env_key": CODEX_KEY_VAR,
    }
    table = ", ".join(f"{key} = {_toml(value)}" for key, value in provider.items())
    argv = [
        "codex",
        "-c",
        f"model_provider={_toml(PROVIDER)}",
        "-c",
        f"model_providers.{PROVIDER}={{{table}}}",
        "-c",
        f"model_catalog_json={_toml(f'{{dir}}/{CODEX_CATALOG}')}",
        "-m",
        served.model,
    ]
    catalog = json.dumps({"models": [model]}, indent=2) + "\n"
    return Plan([*argv, *args], env={CODEX_KEY_VAR: API_KEY}, files={CODEX_CATALOG: catalog})


PLANS = {"pi": _pi, "claude": _claude, "opencode": _opencode, "codex": _codex}


def plan(agent: str, served: Served, args: list[str]) -> Plan:
    """The plan that starts `agent` against `served`, with `args` passed on to the agent."""
    return PLANS[agent](served, list(args))
