"""What this engine was measured against. Nothing here imports MLX."""

MODEL_REPO = "mlx-community/Qwen3-1.7B-4bit"
MODEL_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
MLX_LM_COMMIT = "c39139d70f0798834fc721c28a1aa8e1374ea33c"
PINNED = {MODEL_REPO: MODEL_REVISION}  # repository to revision, as hub.resolve_model_path takes it
# The stack of the published measurement.
MEASURED = {
    "python": "3.13.5",
    "mlx": "0.32.4",
    "transformers": "5.20.2",
    "macos": "15.1",
    "hardware": "Apple M4 Pro, 24 GB",
}
