import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_post_processing import ResizeImageMaskNode, ResizeType  # noqa: E402


def test_scale_total_pixels_snaps_to_multiple():
    resize_type = {"resize_type": ResizeType.SCALE_TOTAL_PIXELS, "megapixels": 1.0, "multiple": 32}

    out = ResizeImageMaskNode.execute(torch.rand(1, 700, 1000, 3), "area", resize_type).result[0]

    assert out.shape == (1, 864, 1216, 3)
