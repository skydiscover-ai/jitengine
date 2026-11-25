# Qwen3.8-Flash-Next, shared

What the two Flash-Next builds share: [`qwen3.8-flash-next-m2`](../qwen3.8-flash-next-m2) and
[`qwen3.8-flash-next-m4pro`](../qwen3.8-flash-next-m4pro). Each build depends on it, and
[`install.sh`](../../install.sh) installs it beside the build.

| Module | What it does |
|:--|:--|
| `checkpoint.pack_format` | the expert pack and the PLE pack on disk: headers, record layout, validation against the checkpoint |
| `model.config` | the model's configuration, read from the checkpoint |
| `kvcache.kv_cache`, `kvcache.linear_state` | the attention history and the linear-attention state of a request |
| `sampler` | drawing a token from the logits |
| `model.moe.waves` | grouping a step's experts into waves |
| `scheduler.faults` | the fault seams the tests inject through |
| `server.openai_api`, `engine.errors`, `kernels.constants` | the OpenAI replies, the errors a request can end with, and kernel constants |

Each build keeps what was tuned per Mac: its kernels, decode loop, expert streaming, pack writers and readers,
memory budget and server. Where a build's packs are stays in the build too (`checkpoint/pack_paths.py`), since its
environment variables carry the build's name. The Qwen engines' shared parts are in [`../qwen`](../qwen), and every
engine's in [`../../runtime`](../../runtime).

```bash
pip install './qwen3.8-flash-next[dev]' && cd qwen3.8-flash-next && pytest
```
