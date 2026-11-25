"""The checkpoint this build serves, pinned to the revision it was measured on (the 3-bit MLX conversion).

`serve` with no `--model-path` fetches CHECKPOINT at REVISION into the Hugging Face cache, and so does any
`--model-path andrevp/Qwen3.6-35B-A3B-3bit-MLX` that names no revision. install.sh downloads the same revision into ~/models."""

CHECKPOINT = "andrevp/Qwen3.6-35B-A3B-3bit-MLX"
REVISION = "794408c4b69a90c9a7b96d3fce6949cccf08df85"

PINNED = {
    CHECKPOINT: REVISION
}  # repository to revision, as jitengine_runtime.hub.resolve_model_path takes it
