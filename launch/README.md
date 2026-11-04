# Use it with any agent

Run Claude Code, Codex, OpenCode, Pi or any other coding agent on a model served from your Mac.

## Quick start

Install and start an engine first; each engine's README has a quick start ([the engines](../engines/README.md)). Then
launch an agent from your project:

```bash
jitengine-flint-m4pro serve --model-path ~/models/flint-int4

# in a second terminal, in the same environment, from the repository root
pip install ./launch          # once
jit-launch claude        # claude / codex / opencode / pi
```

Or `curl -fsSL https://jitengine.ai/install.sh | bash -s -- flint` (or `qwen3.6`, `flash-next`) sets up the engine and
the launcher in one go; then `jitengine claude`.

Any engine works; its README has the command. The Qwen3.6 builds hold 8K tokens a request by
default, too few for an agent, so serve them with `--max-context 32768`; serve the M2 build also with
`--prefix-cache-mib 1024 --build-expert-pack`, so each turn reuses the last turn's prompt (this leaves
memory for only part of the model, so the rest streams from a 14 GB expert pack built on the first start).

## jit-launch

```bash
jit-launch {claude,codex,opencode,pi} [--server URL] [--dry-run] [-- <agent args>]
```

Reads the served model and its context from `/v1/models`, configures the agent for this session,
and starts it. Nothing is written to the agent's own config.

| Flag | Meaning |
|---|---|
| `--server URL` | Engine to point the agent at (default `http://127.0.0.1:8080`) |
| `--dry-run` | Print the settings and the command; run nothing |
| `-- <args>` | Passed to the agent |

```bash
jit-launch claude -- -p "explain this repository"
jit-launch codex -- exec "add a test for the parser"
jit-launch pi --server http://my-mac.local:8080
```

An engine answers only the host names it was told to serve: `localhost`, IP addresses, the address
it listens on, and each `--allowed-host`. For `--server http://my-mac.local:8080`, start the engine
with `serve --host 0.0.0.0 --allowed-host my-mac.local`; without the flag it answers 403 and
`jit-launch` says which host name it refused.

On SIGTERM or SIGHUP, and on SIGINT outside a terminal, the launcher ends the agent and removes the session files. At a terminal, Ctrl-C reaches the agent itself. Run from a script
or a pipe (standard input is not the terminal), the agent runs in its own process group and everything it
started ends with it. At a terminal the agent stays in the terminal's foreground group so that it can
read the keyboard, so only the agent itself is signalled; a process it started and detached keeps
running. The launcher exits with the agent's status, or 128 plus the signal that ended it.

## Manual setup

Every engine serves the OpenAI API (`/v1/chat/completions`, `/v1/responses`) and the Anthropic API
(`/v1/messages`) on one port, so an agent connects to it directly. Below, `<model>` is the id that
`curl http://127.0.0.1:8080/v1/models` lists, and any API key works.

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=jitengine ANTHROPIC_API_KEY="" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096 \
claude --model <model>
```

### Codex

In `~/.codex/config.toml`, then `JITENGINE_API_KEY=jitengine codex`:

```toml
model = "<model>"
model_provider = "jitengine"

[model_providers.jitengine]
name = "JITEngine"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "JITENGINE_API_KEY"
```

### OpenCode

In `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "jitengine": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "<model>": { "limit": { "context": 32768, "output": 8192 } } }
    }
  }
}
```

### Pi

In `~/.pi/agent/models.json`, then `pi --provider jitengine --model <model>`:

```json
{
  "providers": {
    "jitengine": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "jitengine",
      "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
      "models": [{ "id": "<model>", "contextWindow": 32768, "maxTokens": 8192 }]
    }
  }
}
```

### Any other agent

Give it `http://127.0.0.1:8080/v1` as its OpenAI base URL, or `http://127.0.0.1:8080` as its Anthropic one.

### What `jit-launch` adds

`jit-launch` sets a little more than the setups above:

- For Claude Code, it sets the model for its other roles, turns thinking off, and drops the
  attribution header, which would defeat the engine's prefix cache.
- For Codex, it supplies a model catalog, which gives Codex the context window and empty base instructions.

## Context

An agent's own prompt comes first in every request. On our 24 GB M2, through `jit-launch`, the first request of a short
task was about 1.6K tokens with Pi, 3.0K with Codex, 7.2K with OpenCode and 12.5K with Claude Code. The engine's
context limit has to cover that prompt and the conversation after it:

| Engine | Context per request | For an agent |
|---|---|---|
| Qwen3.6 | 8K by default | serve with `--max-context 32768` (the M2 build also `--prefix-cache-mib 1024 --build-expert-pack`) |
| Flash-Next | up to 32K, as its KV budget holds | raise `--kv-budget-mb` if it is less |
| Flint | up to 262K when the whole model is in memory, as the GPU wired limit holds; 32,768 in offload, the 24 GB M2 default | the default |

`jit-launch` warns below 32K. It leaves a quarter of the context for output, except for Claude Code below 64K, where
it leaves 4K. With more output reserved, Claude Code would compact on every turn at 32K and stop, because it also keeps
about 13K free. At 32K, Pi and Codex fit best. Pi's default compaction suits larger windows, so for 32K set it in
`~/.pi/agent/settings.json`:

```json
{ "compaction": { "reserveTokens": 8192, "keepRecentTokens": 8000 } }
```

## Limits

- The engines take text only. An image or a document reaches the model as a note that it was left out.
- A `tool_choice` that requires a tool call is refused on
  `/v1/chat/completions`, and served as `auto` on the other two APIs.
- The Responses API keeps no state: each request carries the whole conversation, as Codex sends it;
  `previous_response_id` is refused, and Codex's web search is unavailable.
- The Qwen builds encode control-token text (`<|im_start|>`, `<tool_call>`, …)
  in a conversation as ordinary text, character for character, so the model reads it as text and a
  file that quotes it does not stop the session.

## Tests

```bash
pip install './launch[dev]' && cd launch && pytest
```

The launcher runs against a stand-in engine, with stand-in agents on `PATH`. Each engine's own
tests cover the three APIs.
