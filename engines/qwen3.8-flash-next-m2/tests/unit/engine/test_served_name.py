"""The name /v1/models lists: the checkpoint folder's own name, which agents show and send back, not its whole path."""

import os


def test_an_engine_is_named_after_its_checkpoint_folder(cpu_flashnext_engine, tiny_flashnext):
    model, _pack = tiny_flashnext()
    assert cpu_flashnext_engine().model_name == os.path.basename(model)


def test_a_given_name_wins(cpu_flashnext_engine):
    assert cpu_flashnext_engine(model_name="org/name").model_name == "org/name"
