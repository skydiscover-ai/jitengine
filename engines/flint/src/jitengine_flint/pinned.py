"""The checkpoint both Flint builds serve, pinned to the revision they were measured on.

`serve` with no `--model-path` fetches CHECKPOINT at REVISION into the Hugging Face cache, and so does any
`--model-path RecursiveSI/flint-int4` that names no revision. install.sh downloads the same revision into ~/models."""

CHECKPOINT = "RecursiveSI/flint-int4"
REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"

PINNED = {
    CHECKPOINT: REVISION
}  # repository to revision, as jitengine_runtime.hub.resolve_model_path takes it
