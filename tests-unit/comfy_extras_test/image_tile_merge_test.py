import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from nodes import EmptyImage
from comfy_extras.nodes_images import ImageColorSpace, ImageMergeTileList, SplitImageToTileList


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("overlap", [0, 16])
def test_tile_merge_preserves_image(dtype, overlap):
    image = torch.linspace(0.0001, 0.8, 2 * 96 * 128 * 3, dtype=dtype).reshape(2, 96, 128, 3)
    before = image.clone()
    tiles = SplitImageToTileList.execute(image, 64, 64, overlap)[0]

    result = ImageMergeTileList.execute(tiles, [128], [96], [overlap])[0]

    torch.testing.assert_close(result, image)
    torch.testing.assert_close(image, before)


def test_tile_merge_preserves_fp16_intermediate_image(monkeypatch):
    monkeypatch.setattr(args, "fp16_intermediates", True)
    generated = EmptyImage().generate(128, 96, 1, 0x404040)[0]
    image = ImageColorSpace.execute(generated, "sRGB", "linear")[0]
    assert image.dtype == torch.float16
    tiles = SplitImageToTileList.execute(image, 64, 64, 16)[0]

    result = ImageMergeTileList.execute(tiles, [128], [96], [16])[0]

    torch.testing.assert_close(result, image)
