"""Qwen3.8-Flash-Next: what its two builds (jitengine-qwen3.8-flash-next-m2, -m4pro) share: the expert and PLE
pack formats, the model config, the KV cache and linear-attention state, the sampler, the MoE waves, the fault
seams, the OpenAI replies, the errors and the kernel constants. Each build keeps its kernels, decode, expert
streaming, pack writers and readers, memory budget and server, and where its packs are (checkpoint/pack_paths.py,
whose variables carry the build's name). `pinned` names the checkpoint both builds serve, which `serve` fetches
from Hugging Face when no --model-path names one. It imports no build."""

__version__ = "0.1.0a1"
