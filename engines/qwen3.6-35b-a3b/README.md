# Qwen3.6-35B-A3B, shared

What the two Qwen3.6 builds share: [`qwen3.6-35b-a3b-m2`](../qwen3.6-35b-a3b-m2) and
[`qwen3.6-35b-a3b-m4pro`](../qwen3.6-35b-a3b-m4pro). Each build depends on it, and [`install.sh`](../../install.sh)
installs it beside the build.

| Module | What it does |
|:--|:--|
| `checkpoint.inventory`, `checkpoint.loader` | the checkpoint's tensors from their safetensors headers, and loading them |
| `engine.io_struct`, `engine.errors` | the request and result types, and the errors a request can end with |
| `scheduler.batch`, `scheduler.retire`, `scheduler.scheduler` | the batch's rows, finishing them, and the scheduling loop |
| `server.api_server` | the HTTP server: the OpenAI, Anthropic and Responses endpoints, `/stats`, `/v1/memory` and Host checks; the build passes its command name |
| `server.protocol`, `server.openai_api`, `server.stats` | the OpenAI request, its replies whole and streamed, and `/stats` |
| `memory.monitor`, `memory.pressure` | sampling the process's memory footprint, and the pressure controller that acts on it |
| `memory.budget`, `memory.resize` | the memory planner (the budget's buckets and the expert pool's size) and resizing a running engine to a new budget; each build keeps its default sizes in `memory/defaults.py` |
| `moe.offload_cache`, `checkpoint.pack` | the expert pool that streams experts from the pack, and the pack format; the build passes its linear class, where its pack lives (`checkpoint/pack_paths.py`) and its command name |
| `model.config`, `moe.expert_store`, `kvcache.linear_state`, `kernels.constants`, `kernels.elementwise` | the model's configuration and the parts both builds compute the same way |

Each build keeps what was tuned per Mac: its kernels, linear layers, decode loop, sampler, memory defaults and
control, and on the M4 Pro multi-token prediction. The two Qwen engines' shared parts are in [`../qwen`](../qwen), and every engine's in
[`../../runtime`](../../runtime).

```bash
pip install './qwen3.6-35b-a3b[dev]' && cd qwen3.6-35b-a3b && pytest
```
