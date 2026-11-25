# jitengine-runtime

The code every engine serves the same way, whatever its model and Mac. Each engine depends on it, and
[`install.sh`](../install.sh) installs it beside the engine.

| Module | What it does |
|:--|:--|
| `jitengine_runtime.anthropic_api` | Anthropic's Messages API (`/v1/messages`, for Claude Code), read into the engine's chat-completions request and written back from its reply, whole or streamed |
| `jitengine_runtime.responses_api` | OpenAI's Responses API (`/v1/responses`, for Codex), the same way |
| `jitengine_runtime.hub` | A Hugging Face model id (`org/name` or `org/name@revision`) for `--model-path`: downloaded into the standard Hugging Face cache and served from its snapshot folder; a directory is used as it is |

Every module but `hub` uses the standard library only, and `hub` imports `huggingface_hub` when it fetches a model id.
The runtime imports no engine; each engine's server maps the API paths to it.

```bash
pip install './runtime[dev]' && cd runtime && pytest
```
