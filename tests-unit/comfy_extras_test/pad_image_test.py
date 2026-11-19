import torch
import torch.nn.functional as F

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_images import PadImageNode, PadStyle, PadType  # noqa: E402

MIRROR = {"padding_style": PadStyle.MIRROR}


def dims(left=0, top=0, right=0, bottom=0):
    return {"padding_type": PadType.PAD_DIMENSIONS, "left": left, "top": top, "right": right, "bottom": bottom}


def color(value="#000000", feathering=0):
    return {"padding_style": PadStyle.COLOR, "color": value, "feathering": feathering}


def run(image, padding_type, padding_style):
    return PadImageNode.execute(image, padding_type, padding_style).result[0]


def test_mirror_matches_torch_reflect():
    image = torch.rand(2, 7, 9, 3)

    out = run(image, dims(3, 2, 4, 5), MIRROR)

    expected = F.pad(image.movedim(-1, 1), (3, 4, 2, 5), mode="reflect").movedim(1, -1)
    assert torch.equal(out, expected)


def test_mirror_padding_larger_than_image():
    image = torch.arange(4.0).reshape(1, 1, 4, 1)

    out = run(image, dims(left=7, right=16), MIRROR)

    assert out[0, 0, :, 0].tolist() == [1, 0, 1, 2, 3, 2, 1] + [0, 1, 2, 3, 2, 1] * 3 + [0, 1]


def test_mirror_single_pixel_dimension():
    image = torch.rand(1, 1, 5, 3)

    out = run(image, dims(top=2, bottom=3), MIRROR)

    assert out.shape == (1, 6, 5, 3)
    assert torch.equal(out, image.expand(1, 6, 5, 3))


def test_hdr_values_and_dtype_preserved():
    image = torch.rand(1, 8, 8, 3, dtype=torch.float16) * 20.0 - 2.0

    out = run(image, dims(4, 4, 4, 4), MIRROR)

    assert out.dtype == torch.float16
    assert torch.equal(out[:, 4:12, 4:12], image)
    assert out.max() == image.max() and out.min() == image.min()


def test_rgba_mirror_keeps_alpha_and_color_pads_opaque():
    image = torch.rand(1, 6, 6, 4)

    mirrored = run(image, dims(2, 2, 2, 2), MIRROR)
    colored = run(image, dims(2, 2, 2, 2), color("#ff0000"))

    assert torch.equal(mirrored[:, 2:8, 0, 3], image[:, :, 2, 3])
    assert colored[0, 0, 0].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_color_with_alpha_uses_rgb():
    image = torch.rand(1, 4, 4, 3)

    out = run(image, dims(left=1), color("#00ff0080"))

    assert out[0, 0, 0].tolist() == [0.0, 1.0, 0.0]


def test_pad_to_multiple():
    image = torch.rand(1, 33, 64, 3)

    out = run(image, {"padding_type": PadType.PAD_TO_MULTIPLE, "multiple": 32}, MIRROR)

    assert out.shape == (1, 64, 64, 3)
    assert torch.equal(out[:, 15:48], image)


def test_pad_by_multiplier():
    image = torch.rand(1, 10, 20, 3)

    out = run(image, {"padding_type": PadType.PAD_BY, "multiplier": 1.5}, color("#ffffff"))

    assert out.shape == (1, 15, 30, 3)
    assert torch.equal(out[:, 2:12, 5:25], image)
    assert torch.all(out[:, :2] == 1.0) and torch.all(out[:, :, :5] == 1.0)


def test_feathering_only_blends_padded_sides():
    image = torch.ones(1, 10, 10, 3)

    out = run(image, dims(left=4), color(feathering=4))
    region = out[0, :, 4:, 0]

    assert torch.all(out[0, :, :4] == 0.0)
    # weights (4/5)^2, (3/5)^2, (2/5)^2, (1/5)^2: the edge pixel is blended, not replaced
    assert torch.allclose(region[:, :4], torch.tensor([0.36, 0.64, 0.84, 0.96]).expand(10, 4))
    assert torch.all(region[:, 4:] == 1.0)


def test_feathering_is_smooth_for_large_fp16_images():
    """fp16 can't represent pixel indices above 2048 exactly; the bottom/right feather must still match fp32."""
    height = 3000
    reference = run(torch.zeros(1, height, 4, 3), dims(bottom=4), color("#ffffff", feathering=16))
    half = run(torch.zeros(1, height, 4, 3, dtype=torch.float16), dims(bottom=4), color("#ffffff", feathering=16))

    assert torch.allclose(half[0, height - 16:height, 0, 0].float(), reference[0, height - 16:height, 0, 0], atol=1e-3)


def test_no_padding_returns_input():
    image = torch.rand(1, 32, 32, 3)

    assert run(image, {"padding_type": PadType.PAD_TO_MULTIPLE, "multiple": 32}, MIRROR) is image
    assert run(image, dims(), color(feathering=8)) is image
