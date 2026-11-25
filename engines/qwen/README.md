# Qwen family

What every Qwen engine shares, whatever the model and the Mac. Each Qwen build depends on it, and
[`install.sh`](../../install.sh) installs it beside the build.

| Module | What it does |
|:--|:--|
| `jitengine_qwen.tokenizer` | the checkpoint's tokenizer and chat template, hardened for client input (control-token text renders as text; an allow-list of template arguments; a bound on the rendered text), and `IncrementalDecoder` for streams |
| `jitengine_qwen.tool_calls` | the tool calls in a reply, in the `<tool_call><function=...><parameter=...>` format the Qwen templates ask for, whole and streamed |

```bash
pip install './qwen[dev]' && cd qwen && pytest
```
