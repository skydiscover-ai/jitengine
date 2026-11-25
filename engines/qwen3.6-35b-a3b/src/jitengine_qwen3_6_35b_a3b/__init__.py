"""Qwen3.6-35B-A3B: what its two builds (jitengine-qwen3.6-35b-a3b-m2, -m4pro) share: the checkpoint inventory and
loader, the request and result types and errors, the scheduler's batch, retire and loop, the OpenAI protocol, its
replies, /stats and the HTTP server, the memory planner, monitor, pressure controller and resizing, the expert
pool and pack format, the model config, the expert store and the linear-attention state. Each build keeps its kernels,
decode loop, sampler, memory control and (the M4 Pro build) multi-token prediction. It imports no build."""

__version__ = "0.1.0a1"
