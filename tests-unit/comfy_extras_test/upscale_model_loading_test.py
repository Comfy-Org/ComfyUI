import pytest
import torch
from spandrel import ModelLoader
from spandrel.architectures.Compact import Compact

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy import model_management, model_patcher  # noqa: E402
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel  # noqa: E402


@pytest.mark.parametrize("prepatched", [False, True])
def test_upscale_spandrel_descriptor(prepatched, monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(model_management, "unet_offload_device", lambda: device)
    monkeypatch.setattr(model_management, "intermediate_device", lambda: device)
    monkeypatch.setattr(model_management, "intermediate_dtype", lambda: torch.float32)
    monkeypatch.setattr(model_management, "current_loaded_models", [])

    model = Compact(num_feat=4, num_conv=1, upscale=2)
    descriptor = ModelLoader().load_from_state_dict(model.state_dict()).eval()
    patcher = None
    if prepatched:
        patcher = model_patcher.CoreModelPatcher(descriptor.model, device, device, fast_disk=True)
        descriptor.patcher = patcher

    image = torch.linspace(0, 1, 2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
    expected = descriptor(image.movedim(-1, -3)).movedim(-3, -1)
    output = ImageUpscaleWithModel.execute(descriptor, image).result[0]

    torch.testing.assert_close(output, expected)
    assert output.shape == (2, 32, 32, 3)
    assert descriptor.patcher.model is descriptor.model
    if prepatched:
        assert descriptor.patcher is patcher
        assert descriptor.patcher.fast_disk

    patcher = descriptor.patcher
    output = ImageUpscaleWithModel.execute(descriptor, image).result[0]
    assert descriptor.patcher is patcher
    torch.testing.assert_close(output, expected)
