import torch

from node_helpers import image_alpha_fix


def test_image_alpha_fix_pads_to_destination_channel_count():
    destination = torch.zeros(1, 2, 2, 4)
    source = torch.full((1, 2, 2, 1), 0.5)

    _, fixed = image_alpha_fix(destination, source)

    assert fixed.shape == destination.shape
    assert torch.equal(fixed[..., 0], source[..., 0])
    assert torch.equal(fixed[..., 1:-1], torch.zeros_like(fixed[..., 1:-1]))
    assert torch.equal(fixed[..., -1], torch.ones_like(fixed[..., -1]))


def test_image_alpha_fix_pads_rgb_to_rgba():
    destination = torch.zeros(1, 2, 2, 4)
    source = torch.full((1, 2, 2, 3), 0.5)

    _, fixed = image_alpha_fix(destination, source)

    assert fixed.shape == destination.shape
    assert torch.equal(fixed[..., :3], source)
    assert torch.equal(fixed[..., 3], torch.ones_like(fixed[..., 3]))
