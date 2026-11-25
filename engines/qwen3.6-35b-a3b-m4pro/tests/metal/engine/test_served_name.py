"""The name /v1/models lists: the checkpoint folder's own name, which agents show and send back, not its whole path."""

import os


def test_an_engine_is_named_after_its_checkpoint_folder(tiny_engine, requires_mlx):
    engine = tiny_engine()
    assert (
        engine.model_name == os.path.basename(engine.model_path.rstrip("/"))
        and "/" not in engine.model_name
    )
