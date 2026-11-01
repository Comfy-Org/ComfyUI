import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_post_processing import ColorTransfer  # noqa: E402


@pytest.mark.parametrize("method", ["reinhard_lab", "mkl_lab", "histogram"])
def test_rgba_transfer_preserves_target_alpha(method):
    target = torch.rand(1, 8, 8, 4)
    reference = torch.rand(1, 8, 8, 4)
    target[..., 3] = 0.25
    reference[..., 3] = 0.75

    result = ColorTransfer.execute(target, reference, method, {"source_stats": "per_frame"}).result[0]

    assert result.shape == target.shape
    assert torch.equal(result[..., 3], target[..., 3])
    assert torch.isfinite(result).all()
    assert not torch.equal(result[..., :3], target[..., :3])


def test_rgb_target_accepts_rgba_reference():
    target = torch.rand(1, 8, 8, 3)
    reference = torch.rand(1, 8, 8, 4)

    result = ColorTransfer.execute(target, reference, "reinhard_lab", {"source_stats": "per_frame"}).result[0]

    assert result.shape == target.shape
    assert torch.isfinite(result).all()
