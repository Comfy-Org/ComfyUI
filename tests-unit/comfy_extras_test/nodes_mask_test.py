import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_mask import BlurMask, FeatherMask  # noqa: E402


def test_blur_mask_softens_interior_edges():
    mask = torch.zeros(1, 9, 9)
    mask[:, 2:7, 2:7] = 1.0
    result = BlurMask.execute(mask, 1.0)[0]

    assert result.shape == mask.shape
    assert float(result[0, 4, 4]) > 0.9
    assert 0.1 < float(result[0, 2, 4]) < 0.9
    assert float(result[0, 0, 0]) < 0.05


def test_blur_mask_does_not_fade_full_canvas():
    mask = torch.ones(1, 8, 8)
    blurred = BlurMask.execute(mask, 2.0)[0]
    feathered = FeatherMask.execute(mask, 3, 0, 0, 0)[0]

    assert torch.all(blurred > 0.99)
    assert float(feathered[0, 4, 0]) < 0.5


def test_blur_mask_zero_sigma_is_identity():
    mask = torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4)
    result = BlurMask.execute(mask, 0.0)[0]
    assert torch.equal(result, mask)


def test_blur_mask_preserves_batch():
    mask = torch.zeros(2, 5, 5)
    mask[0, :, :2] = 1.0
    mask[1, :, 3:] = 1.0
    result = BlurMask.execute(mask, 0.75)[0]

    assert result.shape == (2, 5, 5)
    assert torch.all((result >= 0.0) & (result <= 1.0 + 1e-6))
    assert float(result[0, 2, 0]) > float(result[1, 2, 0])
    assert float(result[1, 2, 4]) > float(result[0, 2, 4])


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_blur_mask_preserves_device(device):
    mask = torch.zeros(1, 9, 9, device=device)
    mask[:, 2:7, 2:7] = 1.0
    result = BlurMask.execute(mask, 1.0)[0]

    assert result.device == mask.device
    assert result.shape == mask.shape
    assert 0.1 < float(result[0, 2, 4]) < 0.9
