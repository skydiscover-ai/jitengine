"""What every Qwen engine shares, whatever the model and the Mac: the chat rendering through the checkpoint's template
(tokenizer) and the tool calls in a reply (tool_calls). The standard library and transformers' tokenizer only; it
imports no engine."""

__version__ = "0.1.0a1"
