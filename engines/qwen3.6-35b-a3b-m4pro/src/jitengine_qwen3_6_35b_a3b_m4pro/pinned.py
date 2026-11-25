"""The checkpoint this build serves and the MTP head it drafts with, each pinned to the revision it was measured on.

`serve` with no `--model-path` fetches CHECKPOINT at REVISION into the Hugging Face cache, and so does any
`--model-path mlx-community/Qwen3.6-35B-A3B-4bit` that names no revision. The head is a repository of its own: beside a
checkpoint in a directory the engine finds it by name, and for a checkpoint of the Hugging Face cache it fetches
MTP_CHECKPOINT at MTP_REVISION. install.sh downloads both revisions into ~/models."""

CHECKPOINT = "mlx-community/Qwen3.6-35B-A3B-4bit"
REVISION = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
MTP_CHECKPOINT = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
MTP_REVISION = "0295b81421bf4d0fccca9a7c0fcfb1418dda3516"

PINNED = {
    CHECKPOINT: REVISION,
    MTP_CHECKPOINT: MTP_REVISION,
}  # repository to revision, as hub.resolve_model_path takes it
