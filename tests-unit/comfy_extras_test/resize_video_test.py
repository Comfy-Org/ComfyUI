"""Resize Image/Mask must mean the same thing for IMAGE, MASK and VIDEO."""
import io
from fractions import Fraction

import av
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.util.video_types import VideoComponents
from comfy_extras.nodes_post_processing import ResizeImageMaskNode, ResizeType, resolve_resize_ops

SCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


def make_images(frames=3, width=20, height=12):
    """Deliberately non-square and non-power-of-two, so a wrong coordinate choice
    in nearest-exact shows up instead of cancelling out."""
    generator = torch.Generator().manual_seed(11)
    return torch.rand(frames, height, width, 3, generator=generator)


def resize(input, scale_method, **resize_type):
    return ResizeImageMaskNode.execute(input, scale_method, resize_type).result[0]


def video_of(images):
    return VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(30)))


@pytest.mark.parametrize("scale_method", SCALE_METHODS)
@pytest.mark.parametrize(
    "resize_type",
    [
        {"resize_type": ResizeType.SCALE_BY, "multiplier": 0.5},
        {"resize_type": ResizeType.SCALE_BY, "multiplier": 2.0},
        {"resize_type": ResizeType.SCALE_DIMENSIONS, "width": 7, "height": 5, "crop": "disabled"},
        {"resize_type": ResizeType.SCALE_DIMENSIONS, "width": 7, "height": 5, "crop": "center"},
        {"resize_type": ResizeType.SCALE_LONGER_DIMENSION, "longer_size": 9},
        {"resize_type": ResizeType.SCALE_SHORTER_DIMENSION, "shorter_size": 9},
        {"resize_type": ResizeType.SCALE_WIDTH, "width": 11},
        {"resize_type": ResizeType.SCALE_HEIGHT, "height": 7},
        {"resize_type": ResizeType.SCALE_TOTAL_PIXELS, "megapixels": 0.01},
        {"resize_type": ResizeType.SCALE_TO_MULTIPLE, "multiple": 8},
    ],
    ids=lambda value: f"{value['resize_type'].value}-{sorted(value)[0]}",
)
def test_video_matches_image_for_every_mode(scale_method, resize_type):
    """Every resize mode produces the same pixels for IMAGE and VIDEO"""
    images = make_images()
    expected = resize(images, scale_method, **resize_type)
    result = resize(video_of(images), scale_method, **resize_type)
    assert torch.equal(result.get_components().images, expected)


@pytest.mark.parametrize("scale_method", SCALE_METHODS)
def test_video_matches_image_for_match_size(scale_method):
    """match size reads its target from an IMAGE even when the input is a VIDEO"""
    images = make_images()
    match = torch.rand(1, 7, 9, 3)
    resize_type = {"resize_type": ResizeType.MATCH_SIZE, "match": match, "crop": "center"}
    expected = resize(images, scale_method, **resize_type)
    result = resize(video_of(images), scale_method, **resize_type)
    assert torch.equal(result.get_components().images, expected)


def test_mask_still_matches_image_path():
    """The shared geometry must not change MASK behaviour"""
    images = make_images()
    mask = images[..., 0]
    resized = resize(mask, "bilinear", resize_type=ResizeType.SCALE_BY, multiplier=0.5)
    assert resized.shape == (3, 6, 10)


def test_video_alpha_follows_the_images():
    """Alpha gets the same geometry as the images, through the MASK convention"""
    images = make_images()
    alpha = torch.rand(3, 12, 20, 1)
    video = VideoFromComponents(
        VideoComponents(images=images, frame_rate=Fraction(30), alpha=alpha)
    )
    for scale_method in SCALE_METHODS:
        result = resize(video, scale_method, resize_type=ResizeType.SCALE_BY, multiplier=0.5)
        components = result.get_components()
        assert components.images.shape == (3, 6, 10, 3)
        assert components.alpha.shape == (3, 6, 10, 1)


def test_no_op_returns_the_input_unchanged():
    """A mode that resolves to no operation hands the same object back"""
    video = video_of(make_images())
    assert resize(video, "area", resize_type=ResizeType.SCALE_DIMENSIONS, width=0, height=0, crop="disabled") is video
    assert resize(video, "area", resize_type=ResizeType.SCALE_TO_MULTIPLE, multiple=1) is video


def test_video_geometry_uses_display_orientation():
    """A rotated source resizes against the orientation it is displayed at"""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        for index in range(2):
            frame = av.VideoFrame.from_ndarray(
                torch.full((32, 64, 3), index * 40, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    data = bytearray(buffer.getvalue())
    matrix_offset = data.index(b"tkhd", data.rindex(b"moov")) + 4 + 40
    values = [0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30]
    data[matrix_offset:matrix_offset + 36] = b"".join(
        value.to_bytes(4, "big", signed=True) for value in values
    )

    video = VideoFromFile(io.BytesIO(bytes(data)))
    # displayed as 32x64, so the longer dimension is the height
    result = resize(video, "area", resize_type=ResizeType.SCALE_LONGER_DIMENSION, longer_size=32)
    assert result.get_dimensions() == (16, 32)


def test_zero_target_dimensions_match_the_image_path():
    """A multiplier that rounds the target to zero must mean the same thing everywhere.

    VIDEO used to ignore the resize and hand back the source size instead.
    """
    images = make_images(width=32, height=24)
    resize_type = {"resize_type": ResizeType.SCALE_BY, "multiplier": 0.01}

    assert resolve_resize_ops(32, 24, "area", resize_type) == [("scale", 0, 0, "area", "disabled")]
    assert resize(images, "area", **resize_type).shape == (3, 0, 0, 3)
    assert resize(images[..., 0], "area", **resize_type).shape == (3, 0, 0)
    assert resize(video_of(images), "area", **resize_type).get_dimensions() == (0, 0)

    buffer = io.BytesIO()
    source = (images * 255).to(torch.uint8)
    with av.open(buffer, mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv444p"
        for index in range(source.shape[0]):
            frame = av.VideoFrame.from_ndarray(source[index].numpy(), format="rgb24")
            container.mux(stream.encode(frame.reformat(format="yuv444p")))
        container.mux(stream.encode(None))
    buffer.seek(0)
    assert resize(VideoFromFile(buffer), "area", **resize_type).get_dimensions() == (0, 0)


def test_mask_lanczos_to_height_one_keeps_the_mask_shape():
    """lanczos drops the channel axis for single channel input, which used to eat the
    height axis of a mask resized to one row. The result must stay (batch, height, width)."""
    mask = make_images(width=20, height=12)[..., 0]
    resized = resize(mask, "lanczos", resize_type=ResizeType.SCALE_LONGER_DIMENSION, longer_size=1)
    assert resized.shape == (3, 1, 1)


@pytest.mark.parametrize(
    "width,height,resize_type,expected",
    [
        # both edges land on a half pixel; round() breaks the tie to even, not upwards
        (33, 17, {"resize_type": ResizeType.SCALE_BY, "multiplier": 0.5}, (16, 8)),
        # total pixels solves for the area, not for either edge
        (32, 24, {"resize_type": ResizeType.SCALE_TOTAL_PIXELS, "megapixels": 0.01}, (118, 89)),
        # scale to multiple covers the target and then centre-crops
        (20, 12, {"resize_type": ResizeType.SCALE_TO_MULTIPLE, "multiple": 8}, (16, 8)),
        # already a multiple, so nothing happens
        (64, 64, {"resize_type": ResizeType.SCALE_TO_MULTIPLE, "multiple": 64}, (64, 64)),
        # odd source dimensions keep their aspect through the longer edge
        (33, 17, {"resize_type": ResizeType.SCALE_LONGER_DIMENSION, "longer_size": 9}, (9, 5)),
    ],
    ids=["rounding", "total-pixels", "scale-to-multiple", "scale-to-multiple-noop", "odd-longer"],
)
def test_resolved_geometry_is_stable(width, height, resize_type, expected):
    """Pin the dimensions the shared resolver produces, so IMAGE and VIDEO cannot drift
    together. The interpolation method does not take part in this arithmetic."""
    images = make_images(width=width, height=height)
    assert resize(images, "area", **resize_type).shape[1:3] == (expected[1], expected[0])
    assert resize(video_of(images), "area", **resize_type).get_dimensions() == expected


def test_resolved_match_size_geometry_is_stable():
    """match size takes the reference's dimensions verbatim, crop only changes the pixels"""
    images = make_images(width=20, height=12)
    resize_type = {"resize_type": ResizeType.MATCH_SIZE, "match": torch.rand(1, 5, 7, 3), "crop": "center"}
    assert resize(images, "area", **resize_type).shape[1:3] == (5, 7)
    assert resize(video_of(images), "area", **resize_type).get_dimensions() == (7, 5)


# --- match size against a VIDEO reference --------------------------------


def rotated_video(width=64, height=32, frames=2):
    """A source whose display orientation differs from its coded orientation."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                torch.full((height, width, 3), index * 40, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    data = bytearray(buffer.getvalue())
    matrix_offset = data.index(b"tkhd", data.rindex(b"moov")) + 4 + 40
    values = [0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30]
    data[matrix_offset:matrix_offset + 36] = b"".join(
        value.to_bytes(4, "big", signed=True) for value in values
    )
    return VideoFromFile(io.BytesIO(bytes(data)))


def match_size(input, match):
    return resize(input, "area", resize_type=ResizeType.MATCH_SIZE, match=match, crop="center")


def test_match_size_accepts_a_video_reference():
    """A VIDEO can be the reference, for VIDEO, IMAGE and MASK inputs alike"""
    reference = video_of(make_images(width=9, height=7))
    images = make_images(width=20, height=12)

    assert match_size(video_of(images), reference).get_dimensions() == (9, 7)
    assert match_size(images, reference).shape[1:3] == (7, 9)
    assert match_size(images[..., 0], reference).shape[1:3] == (7, 9)


def test_match_size_uses_the_reference_display_dimensions():
    """A rotated reference is matched against the size it is displayed at, not its coded size"""
    reference = rotated_video(width=64, height=32)
    assert reference.get_dimensions() == (64, 32)
    assert reference.get_display_dimensions() == (32, 64)

    assert match_size(make_images(width=20, height=12), reference).shape[1:3] == (64, 32)
    assert match_size(video_of(make_images(width=20, height=12)), reference).get_dimensions() == (32, 64)


def test_match_size_rejects_a_mixed_rotation_reference():
    """A reference with no single display size cannot define a target"""
    reference = VideoFromList([rotated_video(), video_of(make_images(width=64, height=32))])
    with pytest.raises(ValueError, match="incompatible display dimensions"):
        match_size(make_images(width=20, height=12), reference)


@pytest.mark.parametrize("scale_method", SCALE_METHODS)
def test_match_size_with_mask_reference_is_unchanged(scale_method):
    """A MASK reference keeps behaving the same; the IMAGE reference is covered by
    test_video_matches_image_for_match_size."""
    images = make_images(width=20, height=12)
    resize_type = {"resize_type": ResizeType.MATCH_SIZE, "match": torch.rand(1, 7, 9), "crop": "center"}
    expected = resize(images, scale_method, **resize_type)
    assert expected.shape[1:3] == (7, 9)
    assert torch.equal(
        resize(video_of(images), scale_method, **resize_type).get_components().images, expected
    )
