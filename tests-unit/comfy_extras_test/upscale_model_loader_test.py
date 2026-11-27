from unittest.mock import MagicMock

import pytest
import torch
from spandrel import ImageModelDescriptor

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_upscale_model  # noqa: E402


@pytest.fixture
def loader(monkeypatch):
    descriptor = MagicMock(spec=ImageModelDescriptor)
    descriptor.eval.return_value = descriptor
    monkeypatch.setattr(nodes_upscale_model.folder_paths, "get_full_path_or_raise", lambda folder, name: f"/models/{name}")
    monkeypatch.setattr(nodes_upscale_model.comfy.utils, "load_torch_file", lambda path, safe_load=False: {})
    monkeypatch.setattr(nodes_upscale_model.comfy.storage, "state_dict_fast_disk", lambda sd: False)
    monkeypatch.setattr(nodes_upscale_model.comfy.model_patcher, "CoreModelPatcher", MagicMock())
    monkeypatch.setattr(nodes_upscale_model, "ModelLoader", lambda: MagicMock(load_from_state_dict=lambda sd: descriptor))
    return descriptor


def test_returns_model_and_selected_name(loader):
    model, name = nodes_upscale_model.UpscaleModelLoader.execute("4x_test.pth").result

    assert model is loader
    assert name == "4x_test.pth"


def test_schema_outputs_model_then_name():
    schema = nodes_upscale_model.UpscaleModelLoader.define_schema()

    assert [o.io_type for o in schema.outputs] == ["UPSCALE_MODEL", "STRING"]
    assert schema.outputs[1].display_name == "model_name"
