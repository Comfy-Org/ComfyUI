import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_wan import _apply_wan_animate_mask  # noqa: E402


@pytest.mark.parametrize(
    ("ref_motion_latent_length", "mask_start"),
    [
        (0, 0),
        (1, 4),
        (2, 8),
    ],
)
def test_wan_animate_mask_uses_causal_temporal_layout(
    ref_motion_latent_length,
    mask_start,
):
    character_mask = torch.arange(77, dtype=torch.float32).view(1, 1, 77, 1, 1)
    expected = torch.cat((
        torch.repeat_interleave(character_mask[:, :, :1], repeats=4, dim=2),
        character_mask[:, :, 1:],
    ), dim=2)
    mask_refmotion = torch.ones_like(expected)
    mask_refmotion[:, :, :mask_start] = 0.0

    actual = _apply_wan_animate_mask(
        mask_refmotion,
        character_mask,
        ref_motion_latent_length,
    )

    assert actual.shape[2] == 80
    assert torch.equal(actual[:, :, :mask_start], torch.zeros_like(actual[:, :, :mask_start]))
    assert torch.equal(actual[:, :, mask_start:], expected[:, :, mask_start:])
