"""The checkpoint both Flash-Next builds serve, pinned to the revision they were measured on.

`serve` with no `--model-path` fetches CHECKPOINT at REVISION into the Hugging Face cache, and so does any
`--model-path mlx-community/Qwen3.8-Flash-Next-4bit` that names no revision. install.sh downloads the same revision into ~/models."""

CHECKPOINT = "mlx-community/Qwen3.8-Flash-Next-4bit"
REVISION = "07b5dc6c54600a359b87f1e53e7adf6351c72a2c"

PINNED = {
    CHECKPOINT: REVISION
}  # repository to revision, as jitengine_runtime.hub.resolve_model_path takes it
