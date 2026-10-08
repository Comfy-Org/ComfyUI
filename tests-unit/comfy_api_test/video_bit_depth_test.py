import pytest
import torch
import av
import numpy as np
from fractions import Fraction
from types import SimpleNamespace
from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents
from comfy_extras.nodes_video import CreateVideo, SaveVideo


@pytest.fixture(scope="module")
def gradient_components():
    """Narrow horizontal ramp (0.25..0.30) that needs more than 8 bits to stay smooth"""
    width, height, frames = 64, 64, 3
    ramp = torch.linspace(0.25, 0.30, width).view(1, 1, width, 1).expand(frames, height, width, 3)
    return VideoComponents(images=ramp.contiguous(), frame_rate=Fraction(30))


@pytest.fixture(scope="module")
def src8(gradient_components, tmp_path_factory):
    """8-bit h264 mp4 (Create Video default)"""
    path = str(tmp_path_factory.mktemp("video") / "src8.mp4")
    VideoFromComponents(gradient_components).save_to(path)
    return path


@pytest.fixture(scope="module")
def src10(gradient_components, tmp_path_factory):
    """10-bit h264 mp4 (Create Video with bit_depth=10)"""
    path = str(tmp_path_factory.mktemp("video") / "src10.mp4")
    VideoFromComponents(gradient_components, bit_depth=10).save_to(path)
    return path


def probe(path):
    """(codec, pix_fmt, bit_depth) of the first video stream"""
    with av.open(path) as container:
        stream = container.streams.video[0]
        return (stream.codec.name, stream.format.name, max(c.bits for c in stream.format.components))


def decoded_levels(path):
    """Unique tonal levels in the first decoded frame (banding measure)"""
    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        return len(np.unique(frame.to_ndarray(format="gbrpf32le")[..., 0]))


def video_packet_bytes(path):
    """Raw video packet payloads; identical to the source's only for a true remux"""
    with av.open(path) as container:
        return [bytes(p) for p in container.demux(container.streams.video[0]) if p.size]


def test_create_video_bit_depth(src8, src10):
    """Create Video's bit_depth picks the encoded depth (default 8-bit); 10-bit reduces banding"""
    assert probe(src8) == ("h264", "yuv420p", 8)
    assert probe(src10) == ("h264", "yuv420p10le", 10)
    assert decoded_levels(src10) > 2 * decoded_levels(src8)


@pytest.mark.parametrize(
    "bit_depth,color_space,expected_bit_depth",
    [
        ("auto", "sRGB", 8),
        ("auto", "HDR", 10),
        ("auto", "HDR PQ", 10),
        (8, "HDR", 8),
        (10, "sRGB", 10),
    ],
)
def test_create_video_node_bit_depth(gradient_components, bit_depth, color_space, expected_bit_depth):
    video = CreateVideo.execute(
        gradient_components.images,
        float(gradient_components.frame_rate),
        bit_depth=bit_depth,
        color_space=color_space,
    ).args[0]
    assert video.get_bit_depth() == expected_bit_depth
    assert video.get_color_space() == color_space


def test_create_video_node_bit_depth_options():
    bit_depth_input = next(input for input in CreateVideo.define_schema().inputs if input.id == "bit_depth")
    assert bit_depth_input.options == ["auto", 8, 10]
    assert bit_depth_input.default == "auto"


@pytest.mark.parametrize(
    "codec,expected_suffix,expected_codec",
    [
        ("auto", "mp4", "h264"),
        ("h264", "mp4", "h264"),
        ("av1", "webm", "av1"),
    ],
)
def test_save_video_auto_format(gradient_components, tmp_path, monkeypatch, codec, expected_suffix, expected_codec):
    monkeypatch.setattr(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "comfy_extras.nodes_video.folder_paths.get_save_image_path",
        lambda *args: (str(tmp_path), "auto", 1, "", "auto"),
    )
    video = VideoFromComponents(gradient_components)

    SaveVideo.execute(
        video,
        "auto",
        {"format": "auto", "codec": {"codec": codec}},
    )

    path = tmp_path / f"auto_00001_.{expected_suffix}"
    with av.open(path) as container:
        assert container.streams.video[0].codec.canonical_name == expected_codec


def test_save_video_has_no_color_space_input():
    schema = SaveVideo.define_schema()
    assert all("color_space" not in str(input.as_dict()) for input in schema.inputs)


def test_save_auto_keeps_source_depth(src8, src10, tmp_path):
    """Save Video (no bit_depth = auto) stream-copies the source, preserving its depth byte-for-byte"""
    for name, src in [("p8", src8), ("p10", src10)]:
        path = str(tmp_path / f"{name}.mp4")
        VideoFromFile(src).save_to(path)
        assert probe(path) == probe(src)
        assert video_packet_bytes(path) == video_packet_bytes(src)


def test_save_explicit_depth_reencodes(src8, src10, tmp_path):
    """An explicit bit_depth different from the source forces a re-encode to that depth"""
    down = str(tmp_path / "down8.mp4")
    VideoFromFile(src10).save_to(down, bit_depth=8)
    assert probe(down) == ("h264", "yuv420p", 8)

    up = str(tmp_path / "up10.mp4")
    VideoFromFile(src8).save_to(up, bit_depth=10)
    assert probe(up) == ("h264", "yuv420p10le", 10)


def test_trim_keeps_source_depth(src10, tmp_path):
    """Video Slice re-encodes (trim) but preserves the source's 10-bit depth"""
    path = str(tmp_path / "trim.mp4")
    VideoFromFile(src10).as_trimmed(start_time=0, duration=1 / 30, strict_duration=False).save_to(path)
    assert probe(path) == ("h264", "yuv420p10le", 10)


def test_get_bit_depth(gradient_components, src8, src10):
    """get_bit_depth reports a video's depth (backs the Get Video Components output)"""
    assert VideoFromFile(src8).get_bit_depth() == 8
    assert VideoFromFile(src10).get_bit_depth() == 10
    assert VideoFromComponents(gradient_components, bit_depth=10).get_bit_depth() == 10
    assert VideoFromComponents(gradient_components).get_bit_depth() == 8
