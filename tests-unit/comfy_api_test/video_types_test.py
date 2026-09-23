import pytest
import torch
import tempfile
import os
import sys
import av
import io
import numpy as np
from fractions import Fraction
from comfy_api.input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._util.video_types import normalize_crop_rect
from comfy_api.util.video_types import VideoComponents, VideoContainer, VideoCodec
from comfy_api.input.basic_types import AudioInput
from av.error import InvalidDataError
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

EPSILON = 0.0001


@pytest.fixture
def sample_images():
    """3-frame 2x2 RGB video tensor"""
    return torch.rand(3, 2, 2, 3)


@pytest.fixture
def sample_audio():
    """Stereo audio with 44.1kHz sample rate"""
    return AudioInput(
        {
            "waveform": torch.rand(1, 2, 1000),
            "sample_rate": 44100,
        }
    )


@pytest.fixture
def video_components(sample_images, sample_audio):
    """VideoComponents with images, audio, and metadata"""
    return VideoComponents(
        images=sample_images,
        audio=sample_audio,
        frame_rate=Fraction(30),
        metadata={"test": "metadata"},
    )


def create_test_video(width=4, height=4, frames=3, fps=30):
    """Helper to create a temporary video file"""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with av.open(tmp.name, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                torch.ones(height, width, 3, dtype=torch.uint8).numpy() * (i * 85),
                format="rgb24",
            )
            frame = frame.reformat(format="yuv420p")
            packet = stream.encode(frame)
            container.mux(packet)

        # Flush
        packet = stream.encode(None)
        container.mux(packet)

    return tmp.name


@pytest.fixture
def simple_video_file():
    """4x4 video with 3 frames at 30fps"""
    file_path = create_test_video()
    yield file_path
    os.unlink(file_path)


def test_video_from_components_get_duration(video_components):
    """Duration calculated correctly from frame count and frame rate"""
    video = VideoFromComponents(video_components)
    duration = video.get_duration()

    expected_duration = 3.0 / 30.0
    assert duration == pytest.approx(expected_duration)


def test_video_from_components_get_duration_different_frame_rates(sample_images):
    """Duration correct for different frame rates including fractional"""
    # Test with 60 fps
    components_60fps = VideoComponents(images=sample_images, frame_rate=Fraction(60))
    video_60fps = VideoFromComponents(components_60fps)
    assert video_60fps.get_duration() == pytest.approx(3.0 / 60.0)

    # Test with fractional frame rate (23.976fps)
    components_frac = VideoComponents(
        images=sample_images, frame_rate=Fraction(24000, 1001)
    )
    video_frac = VideoFromComponents(components_frac)
    expected_frac = 3.0 / (24000.0 / 1001.0)
    assert video_frac.get_duration() == pytest.approx(expected_frac)


def test_video_from_components_get_duration_empty_video():
    """Duration is zero for empty video"""
    empty_components = VideoComponents(
        images=torch.zeros(0, 2, 2, 3), frame_rate=Fraction(30)
    )
    video = VideoFromComponents(empty_components)
    assert video.get_duration() == 0.0


def test_video_from_components_get_dimensions(video_components):
    """Dimensions returned correctly from image tensor shape"""
    video = VideoFromComponents(video_components)
    width, height = video.get_dimensions()
    assert width == 2
    assert height == 2


def test_video_from_file_get_duration(simple_video_file):
    """Duration extracted from file metadata"""
    video = VideoFromFile(simple_video_file)
    duration = video.get_duration()
    assert duration == pytest.approx(0.1, abs=0.01)


def test_video_from_file_get_dimensions(simple_video_file):
    """Dimensions read from stream without decoding frames"""
    video = VideoFromFile(simple_video_file)
    width, height = video.get_dimensions()
    assert width == 4
    assert height == 4


def test_video_color_space_defaults_to_srgb(simple_video_file, video_components):
    assert VideoFromFile(simple_video_file).get_color_space() == "sRGB"
    assert VideoFromComponents(video_components).get_color_space() == "sRGB"


@pytest.mark.parametrize("color_space", ["sRGB", "HDR", "HDR PQ"])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_video_from_components_color_space(video_components, color_space, bit_depth):
    video = VideoFromComponents(video_components, bit_depth=bit_depth, color_space=color_space)
    assert video.get_color_space() == color_space
    assert video.get_bit_depth() == bit_depth


def test_video_from_components_rejects_invalid_color_space(video_components):
    with pytest.raises(ValueError, match="Unsupported video color space"):
        VideoFromComponents(video_components, color_space="Display P3")


def test_video_from_file_bytesio_input():
    """VideoFromFile works with BytesIO input"""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame.from_ndarray(
            torch.zeros(2, 2, 3, dtype=torch.uint8).numpy(), format="rgb24"
        )
        frame = frame.reformat(format="yuv420p")
        packet = stream.encode(frame)
        container.mux(packet)
        packet = stream.encode(None)
        container.mux(packet)

    buffer.seek(0)
    video = VideoFromFile(buffer)

    assert video.get_dimensions() == (2, 2)
    assert video.get_duration() == pytest.approx(1 / 30, abs=0.01)


def test_video_from_file_invalid_file_error():
    """InvalidDataError raised for non-video files"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not a video file")
        tmp.flush()
        tmp_name = tmp.name

    try:
        with pytest.raises(InvalidDataError):
            video = VideoFromFile(tmp_name)
            video.get_dimensions()
    finally:
        os.unlink(tmp_name)


def test_video_from_file_audio_only_error():
    """ValueError raised for audio-only files"""
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        with av.open(tmp_name, mode="w") as container:
            stream = container.add_stream("aac", rate=44100)
            stream.sample_rate = 44100
            stream.format = "fltp"

            audio_data = torch.zeros(1, 1024).numpy()
            audio_frame = av.AudioFrame.from_ndarray(
                audio_data, format="fltp", layout="mono"
            )
            audio_frame.sample_rate = 44100
            audio_frame.pts = 0
            packet = stream.encode(audio_frame)
            container.mux(packet)

            for packet in stream.encode(None):
                container.mux(packet)

        with pytest.raises(ValueError, match="No video stream found"):
            video = VideoFromFile(tmp_name)
            video.get_dimensions()
    finally:
        os.unlink(tmp_name)


def test_single_frame_video():
    """Single frame video has correct duration"""
    components = VideoComponents(
        images=torch.rand(1, 10, 10, 3), frame_rate=Fraction(1)
    )
    video = VideoFromComponents(components)
    assert video.get_duration() == 1.0


@pytest.mark.parametrize(
    "frame_rate,expected_fps",
    [
        (Fraction(24000, 1001), 24000 / 1001),
        (Fraction(30000, 1001), 30000 / 1001),
        (Fraction(25, 1), 25.0),
        (Fraction(50, 2), 25.0),
    ],
)
def test_fractional_frame_rates(frame_rate, expected_fps):
    """Duration calculated correctly for various fractional frame rates"""
    components = VideoComponents(images=torch.rand(100, 4, 4, 3), frame_rate=frame_rate)
    video = VideoFromComponents(components)
    duration = video.get_duration()
    expected_duration = 100.0 / expected_fps
    assert duration == pytest.approx(expected_duration)


def test_duration_consistency(video_components):
    """get_duration() consistent with manual calculation from components"""
    video = VideoFromComponents(video_components)

    duration = video.get_duration()
    components = video.get_components()
    manual_duration = float(components.images.shape[0] / components.frame_rate)

    assert duration == pytest.approx(manual_duration)


def test_save_to_h264_crf_controls_quality(tmp_path):
    generator = torch.Generator().manual_seed(7)
    components = VideoComponents(
        images=torch.rand(12, 64, 64, 3, generator=generator),
        frame_rate=Fraction(30),
    )
    high_quality = str(tmp_path / "high_quality.mp4")
    low_quality = str(tmp_path / "low_quality.mp4")
    transcoded = str(tmp_path / "transcoded.mp4")

    VideoFromComponents(components).save_to(high_quality, codec=VideoCodec.H264, crf=0)
    VideoFromComponents(components).save_to(low_quality, codec=VideoCodec.H264, crf=51)
    assert os.path.getsize(high_quality) > os.path.getsize(low_quality)

    VideoFromFile(high_quality).save_to(transcoded, codec=VideoCodec.H264, crf=51)
    assert os.path.getsize(transcoded) < os.path.getsize(high_quality)


def video_packet_bytes(path):
    with av.open(path) as container:
        return [bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size]


def decoded_video_frames(path):
    with av.open(path) as container:
        frames = []
        for frame in container.decode(video=0):
            bytes_per_sample = max(component.bits for component in frame.format.components)
            bytes_per_sample = (bytes_per_sample + 7) // 8
            plane_sizes = (
                (frame.width * bytes_per_sample, frame.height),
                ((frame.width // 2) * bytes_per_sample, frame.height // 2),
                ((frame.width // 2) * bytes_per_sample, frame.height // 2),
            )
            frames.append(tuple(
                b"".join(
                    bytes(plane)[row * plane.line_size:row * plane.line_size + row_size]
                    for row in range(rows)
                )
                for plane, (row_size, rows) in zip(frame.planes, plane_sizes)
            ))
        return frames


@pytest.mark.parametrize(
    "format,suffix,codec",
    [
        (VideoContainer.MP4, "mp4", VideoCodec.H264),
        (VideoContainer.MKV, "mkv", VideoCodec.H264),
        (VideoContainer.MKV, "mkv", VideoCodec.AV1),
        (VideoContainer.WEBM, "webm", VideoCodec.AV1),
    ],
)
def test_video_from_components_auto_color_space_matches_srgb(tmp_path, format, suffix, codec):
    components = VideoComponents(
        images=torch.rand(3, 64, 64, 3, generator=torch.Generator().manual_seed(23)),
        frame_rate=Fraction(30),
    )
    auto = str(tmp_path / f"auto.{suffix}")
    srgb = str(tmp_path / f"srgb.{suffix}")

    VideoFromComponents(components).save_to(
        auto,
        format=format,
        codec=codec,
        crf=0,
    )
    VideoFromComponents(components).save_to(
        srgb,
        format=format,
        codec=codec,
        crf=0,
        color_space="sRGB",
    )

    assert decoded_video_frames(auto) == decoded_video_frames(srgb)
    for path in (auto, srgb):
        with av.open(path) as container:
            stream = container.streams.video[0]
            assert stream.color_primaries == ColorPrimaries.BT709
            assert stream.color_trc == ColorTrc.IEC61966_2_1
            assert stream.colorspace == 1
            assert stream.color_range == ColorRange.MPEG


def create_hdr_av1_video(path, transfer, color_range):
    images = np.random.default_rng(17).integers(0, 65536, (3, 64, 64, 3), dtype=np.uint16)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libsvtav1", rate=30)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p10le"
        stream.options = {"svtav1-params": "lossless=1"}
        stream.color_primaries = ColorPrimaries.BT2020
        stream.color_trc = transfer
        stream.colorspace = 9
        stream.color_range = color_range
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb48le").reformat(format="yuv420p10le")
            frame.color_primaries = ColorPrimaries.BT2020
            frame.color_trc = transfer
            frame.colorspace = 9
            frame.color_range = color_range
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def test_save_to_av1_crf_controls_quality(tmp_path):
    generator = torch.Generator().manual_seed(11)
    components = VideoComponents(
        images=torch.rand(12, 64, 64, 3, generator=generator),
        frame_rate=Fraction(30),
    )
    high_quality = str(tmp_path / "high_quality.mkv")
    low_quality = str(tmp_path / "low_quality.mkv")

    VideoFromComponents(components).save_to(
        high_quality,
        format=VideoContainer.MKV,
        codec=VideoCodec.AV1,
        crf=0,
    )
    VideoFromComponents(components).save_to(
        low_quality,
        format=VideoContainer.MKV,
        codec=VideoCodec.AV1,
        crf=63,
    )

    assert os.path.getsize(high_quality) > os.path.getsize(low_quality)


@pytest.mark.parametrize(
    "format,suffix,codec,video_codec,audio_codec,audio_rate",
    [
        (VideoContainer.AUTO, "mp4", VideoCodec.AUTO, "h264", "aac", 44100),
        (VideoContainer.MP4, "mp4", VideoCodec.H264, "h264", "aac", 44100),
        (VideoContainer.MP4, "mp4", VideoCodec.AV1, "av1", "aac", 44100),
        (VideoContainer.MKV, "mkv", VideoCodec.AUTO, "h264", "aac", 44100),
        (VideoContainer.MKV, "mkv", VideoCodec.H264, "h264", "aac", 44100),
        (VideoContainer.MKV, "mkv", VideoCodec.AV1, "av1", "aac", 44100),
        (VideoContainer.WEBM, "webm", VideoCodec.AUTO, "av1", "opus", 48000),
        (VideoContainer.WEBM, "webm", VideoCodec.AV1, "av1", "opus", 48000),
    ],
)
def test_save_components_container_codec_and_audio_matrix(
    tmp_path, format, suffix, codec, video_codec, audio_codec, audio_rate
):
    components = VideoComponents(
        images=torch.rand(3, 64, 64, 3),
        audio=AudioInput({
            "waveform": torch.rand(1, 2, 4410),
            "sample_rate": 44100,
        }),
        frame_rate=Fraction(30),
    )
    path = str(tmp_path / f"components.{suffix}")

    VideoFromComponents(components).save_to(
        path,
        format=format,
        codec=codec,
        crf=30,
        metadata={"prompt": {"test": "video"}},
    )

    with av.open(path) as container:
        assert container.streams.video[0].codec.canonical_name == video_codec
        assert container.streams.video[0].format.name == "yuv420p"
        assert container.streams.audio[0].codec.canonical_name == audio_codec
        assert container.streams.audio[0].sample_rate == audio_rate
        prompt = container.metadata.get("PROMPT", container.metadata.get("prompt"))
        assert prompt == '{"test": "video"}'
        assert sum(1 for _ in container.decode(video=0)) == 3
    with av.open(path) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) > 0


@pytest.mark.parametrize(
    "color_space,transfer,primaries,colorspace",
    [
        ("sRGB", ColorTrc.IEC61966_2_1, ColorPrimaries.BT709, 1),
        ("HDR", ColorTrc.ARIB_STD_B67, ColorPrimaries.BT2020, 9),
        ("HDR PQ", ColorTrc.SMPTE2084, ColorPrimaries.BT2020, 9),
    ],
)
@pytest.mark.parametrize("bit_depth,pix_fmt", [(8, "yuv420p"), (10, "yuv420p10le")])
def test_save_to_av1_mkv_color_space(tmp_path, color_space, transfer, primaries, colorspace, bit_depth, pix_fmt):
    components = VideoComponents(
        images=torch.rand(2, 64, 64, 3),
        frame_rate=Fraction(30),
    )
    path = str(tmp_path / "hdr.mkv")
    remuxed = str(tmp_path / "remuxed.mkv")

    VideoFromComponents(components, bit_depth=bit_depth, color_space=color_space).save_to(
        path,
        format=VideoContainer.MKV,
        codec=VideoCodec.AV1,
        crf=30,
        metadata={"prompt": {"test": "hdr"}},
    )

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.codec.canonical_name == "av1"
        assert stream.format.name == pix_fmt
        assert stream.color_primaries == primaries
        assert stream.color_trc == transfer
        assert stream.colorspace == colorspace
        assert stream.color_range == ColorRange.MPEG
        assert container.metadata["PROMPT"] == '{"test": "hdr"}'
    assert VideoFromFile(path).get_color_space() == color_space

    source_packets = video_packet_bytes(path)
    VideoFromFile(path).save_to(
        remuxed,
        format=VideoContainer.MKV,
        codec=VideoCodec.AV1,
    )
    with av.open(remuxed) as container:
        stream = container.streams.video[0]
        assert stream.codec.canonical_name == "av1"
        assert stream.format.name == pix_fmt
        assert stream.color_primaries == primaries
        assert stream.color_trc == transfer
        assert container.metadata["PROMPT"] == '{"test": "hdr"}'
    assert video_packet_bytes(remuxed) == source_packets


@pytest.mark.parametrize(
    "format,suffix",
    [
        (VideoContainer.MP4, "mp4"),
        (VideoContainer.MKV, "mkv"),
    ],
)
@pytest.mark.parametrize(
    "color_space,transfer,primaries,colorspace",
    [
        ("sRGB", ColorTrc.IEC61966_2_1, ColorPrimaries.BT709, 1),
        ("HDR", ColorTrc.ARIB_STD_B67, ColorPrimaries.BT2020, 9),
        ("HDR PQ", ColorTrc.SMPTE2084, ColorPrimaries.BT2020, 9),
    ],
)
@pytest.mark.parametrize("bit_depth,pix_fmt", [(8, "yuv420p"), (10, "yuv420p10le")])
def test_save_to_h264_color_space(tmp_path, format, suffix, color_space, transfer, primaries, colorspace, bit_depth, pix_fmt):
    components = VideoComponents(
        images=torch.rand(2, 64, 64, 3),
        frame_rate=Fraction(30),
    )
    path = str(tmp_path / f"h264.{suffix}")

    VideoFromComponents(components, bit_depth=bit_depth).save_to(
        path,
        format=format,
        codec=VideoCodec.H264,
        crf=23,
        color_space=color_space,
    )

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.codec.canonical_name == "h264"
        assert stream.format.name == pix_fmt
        assert stream.color_primaries == primaries
        assert stream.color_trc == transfer
        assert stream.colorspace == colorspace
        assert stream.color_range == ColorRange.MPEG
        assert sum(1 for _ in container.decode(video=0)) == 2
    assert VideoFromFile(path).get_color_space() == color_space


@pytest.mark.parametrize(
    "transfer,color_range,color_space",
    [
        (ColorTrc.SMPTE2084, ColorRange.MPEG, None),
        (ColorTrc.SMPTE2084, ColorRange.MPEG, "HDR PQ"),
        (ColorTrc.ARIB_STD_B67, ColorRange.MPEG, None),
        (ColorTrc.ARIB_STD_B67, ColorRange.MPEG, "HDR"),
        (ColorTrc.ARIB_STD_B67, ColorRange.JPEG, "HDR"),
    ],
)
def test_save_to_loaded_hdr_preserves_color(tmp_path, transfer, color_range, color_space):
    source = str(tmp_path / "source.mkv")
    remuxed = str(tmp_path / "auto_encoding.mkv")
    reencoded = str(tmp_path / "auto_color_space.webm")
    create_hdr_av1_video(source, transfer, color_range)

    video = VideoFromFile(source)
    video.save_to(remuxed, format=VideoContainer.MKV, codec=VideoCodec.AV1)
    video.save_to(
        reencoded,
        format=VideoContainer.WEBM,
        codec=VideoCodec.AV1,
        crf=0,
        color_space=color_space,
    )

    assert video_packet_bytes(remuxed) == video_packet_bytes(source)
    source_frames = decoded_video_frames(source)
    reencoded_frames = decoded_video_frames(reencoded)
    assert len(source_frames) == len(reencoded_frames)
    assert all(np.array_equal(source_frame, output_frame) for source_frame, output_frame in zip(source_frames, reencoded_frames))

    for path in (remuxed, reencoded):
        with av.open(path) as container:
            stream = container.streams.video[0]
            assert stream.codec.canonical_name == "av1"
            assert stream.format.name == "yuv420p10le"
            assert stream.color_primaries == ColorPrimaries.BT2020
            assert stream.color_trc == transfer
            assert stream.colorspace == 9
            assert stream.color_range == color_range


@pytest.mark.parametrize("color_space", ["sRGB", "HDR PQ"])
def test_save_to_loaded_hdr_rejects_color_conversion(tmp_path, color_space):
    source = str(tmp_path / "source.mkv")
    output = str(tmp_path / "wrong_transfer.webm")
    create_hdr_av1_video(source, ColorTrc.ARIB_STD_B67, ColorRange.MPEG)

    with pytest.raises(ValueError, match=f"Cannot save HDR video as {color_space} without color conversion"):
        VideoFromFile(source).save_to(
            output,
            format=VideoContainer.WEBM,
            codec=VideoCodec.AV1,
            crf=30,
            color_space=color_space,
        )

    assert not os.path.exists(output)


def test_save_to_av1_webm_transcodes_audio(tmp_path):
    components = VideoComponents(
        images=torch.rand(2, 64, 64, 3),
        audio=AudioInput({
            "waveform": torch.rand(1, 2, 4410),
            "sample_rate": 44100,
        }),
        frame_rate=Fraction(30),
    )
    source = str(tmp_path / "source.mp4")
    path = str(tmp_path / "output.webm")
    VideoFromComponents(components).save_to(source, color_space="HDR")

    VideoFromFile(source).save_to(
        path,
        format=VideoContainer.WEBM,
        codec=VideoCodec.AV1,
        crf=30,
        color_space="HDR",
    )

    with av.open(path) as container:
        video_stream = container.streams.video[0]
        assert video_stream.codec.canonical_name == "av1"
        assert video_stream.format.name == "yuv420p"
        assert video_stream.color_primaries == ColorPrimaries.BT2020
        assert video_stream.color_trc == ColorTrc.ARIB_STD_B67
        assert video_stream.colorspace == 9
        assert container.streams.audio[0].codec.name == "opus"
        assert container.streams.audio[0].sample_rate == 48000
        assert sum(1 for _ in container.decode(video=0)) == 2
    with av.open(path) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) > 0


@pytest.mark.parametrize("source_codec", [VideoCodec.H264, VideoCodec.AV1])
def test_save_loaded_mkv_to_webm_auto_transcodes_incompatible_streams(tmp_path, source_codec):
    components = VideoComponents(
        images=torch.rand(3, 64, 64, 3),
        audio=AudioInput({
            "waveform": torch.rand(1, 2, 4410),
            "sample_rate": 44100,
        }),
        frame_rate=Fraction(30),
    )
    source = str(tmp_path / "source.mkv")
    output = str(tmp_path / "output.webm")
    VideoFromComponents(components).save_to(
        source,
        format=VideoContainer.MKV,
        codec=source_codec,
        crf=63 if source_codec == VideoCodec.AV1 else 30,
    )

    VideoFromFile(source).save_to(
        output,
        format=VideoContainer.WEBM,
        codec=VideoCodec.AUTO,
    )

    with av.open(output) as container:
        assert container.streams.video[0].codec.canonical_name == "av1"
        assert container.streams.audio[0].codec.canonical_name == "opus"
        assert container.streams.audio[0].sample_rate == 48000
        assert sum(1 for _ in container.decode(video=0)) == 3
    with av.open(output) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) > 0


def test_save_loaded_h264_mkv_to_webm_h264_rejected_before_creating_output(tmp_path):
    components = VideoComponents(
        images=torch.rand(1, 64, 64, 3),
        frame_rate=Fraction(30),
    )
    source = str(tmp_path / "source.mkv")
    output = tmp_path / "output.webm"
    VideoFromComponents(components).save_to(
        source,
        format=VideoContainer.MKV,
        codec=VideoCodec.H264,
    )

    with pytest.raises(ValueError, match="WebM output requires the AV1 codec"):
        VideoFromFile(source).save_to(
            str(output),
            format=VideoContainer.WEBM,
            codec=VideoCodec.H264,
        )

    assert not output.exists()


def test_save_loaded_webm_auto_remuxes_compatible_streams(tmp_path):
    components = VideoComponents(
        images=torch.rand(3, 64, 64, 3),
        audio=AudioInput({
            "waveform": torch.rand(1, 2, 4410),
            "sample_rate": 44100,
        }),
        frame_rate=Fraction(30),
    )
    source = str(tmp_path / "source.webm")
    output = str(tmp_path / "output.webm")
    VideoFromComponents(components).save_to(
        source,
        format=VideoContainer.WEBM,
        codec=VideoCodec.AV1,
        crf=63,
    )
    source_packets = video_packet_bytes(source)

    VideoFromFile(source).save_to(
        output,
        format=VideoContainer.WEBM,
        codec=VideoCodec.AUTO,
    )

    assert video_packet_bytes(output) == source_packets
    with av.open(output) as container:
        assert container.streams.video[0].codec.canonical_name == "av1"
        assert container.streams.audio[0].codec.canonical_name == "opus"
        assert sum(1 for _ in container.decode(video=0)) == 3


@pytest.mark.parametrize("format", [VideoContainer.MKV, VideoContainer.WEBM])
def test_save_to_av1_file_like_output(format):
    components = VideoComponents(
        images=torch.rand(1, 64, 64, 3),
        frame_rate=Fraction(30),
    )
    output = io.BytesIO()

    VideoFromComponents(components).save_to(
        output,
        format=format,
        codec=VideoCodec.AV1,
        crf=63,
    )

    output.seek(0)
    with av.open(output) as container:
        assert container.streams.video[0].codec.canonical_name == "av1"
        assert sum(1 for _ in container.decode(video=0)) == 1


def test_save_to_rejects_h264_webm_before_creating_output(tmp_path):
    components = VideoComponents(
        images=torch.rand(1, 64, 64, 3),
        frame_rate=Fraction(30),
    )
    path = tmp_path / "invalid.webm"

    with pytest.raises(ValueError, match="WebM output requires the AV1 codec"):
        VideoFromComponents(components).save_to(
            str(path),
            format=VideoContainer.WEBM,
            codec=VideoCodec.H264,
        )

    assert not path.exists()


def test_save_to_rejects_unknown_color_space(tmp_path):
    components = VideoComponents(
        images=torch.rand(1, 64, 64, 3),
        frame_rate=Fraction(30),
    )

    with pytest.raises(ValueError, match="Unsupported video color space: HLG"):
        VideoFromComponents(components).save_to(
            str(tmp_path / "invalid.mkv"),
            format=VideoContainer.MKV,
            codec=VideoCodec.AV1,
            color_space="HLG",
        )


def test_save_to_mp4_writes_metadata_before_media(video_components, tmp_path):
    encoded = tmp_path / "encoded.mp4"
    remuxed = tmp_path / "remuxed.mp4"

    VideoFromComponents(video_components).save_to(str(encoded), metadata={"prompt": {"test": "value"}})
    VideoFromFile(str(encoded)).save_to(str(remuxed), metadata={"prompt": {"test": "value"}})

    for path in (encoded, remuxed):
        data = path.read_bytes()
        assert data.index(b"moov") < data.index(b"mdat")


def create_transcode_source(
    width=64, height=64, frames=30, fps=30, audio_streams=1, undecodable_audio=0, rotation=False,
    container_format="mov", audio_codec="pcm_s16le",
):
    """Create a temp video that save_to must transcode (mpeg4 video, so codec != h264).

    ``undecodable_audio`` trailing PCM streams get their fourcc corrupted so no decoder exists
    (``codec_context is None``), like the APAC track in iPhone spatial-audio recordings.
    ``rotation`` patches a 90-degree display matrix into the video track header.
    """
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        audio = []
        for _ in range(audio_streams + undecodable_audio):
            stream = container.add_stream(audio_codec, rate=44100)
            stream.sample_rate = 44100
            audio.append(stream)

        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(
                torch.full((height, width, 3), (i * 7) % 256, dtype=torch.uint8).numpy(),
                format="rgb24",
            )
            container.mux(video_stream.encode(frame.reformat(format="yuv420p")))
        # write audio in 1024-sample frames, like real decoders produce, so the
        # per-frame skip/cap logic in the transcode path actually runs
        for stream in audio:
            for offset in range(0, 44100 * frames // fps, 1024):
                n = min(1024, 44100 * frames // fps - offset)
                audio_frame = av.AudioFrame.from_ndarray(
                    torch.zeros(1, n, dtype=torch.int16).numpy(), format="s16", layout="mono"
                )
                audio_frame.sample_rate = 44100
                audio_frame.pts = offset
                container.mux(stream.encode(audio_frame))
        for stream in [video_stream, *audio]:
            container.mux(stream.encode(None))

    data = bytearray(buffer.getvalue())
    end = len(data)
    for _ in range(undecodable_audio):
        end = data.rindex(b"sowt", 0, end)
        data[end:end + 4] = b"Xpac"
    if rotation:
        # the 3x3 display matrix sits 40 bytes into the version-0 tkhd payload; first tkhd
        # inside moov = video track (search from moov so mdat bytes can't false-match)
        matrix_offset = data.index(b"tkhd", data.rindex(b"moov")) + 4 + 40
        values = [0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30]
        data[matrix_offset:matrix_offset + 36] = b"".join(v.to_bytes(4, "big", signed=True) for v in values)

    tmp = tempfile.NamedTemporaryFile(suffix=f".{container_format}", delete=False)
    tmp.write(bytes(data))
    tmp.close()
    return tmp.name


def transcode_and_probe(video):
    buffer = io.BytesIO()
    video.save_to(buffer, format=VideoContainer.MP4, codec=VideoCodec.H264)
    buffer.seek(0)
    with av.open(buffer) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        frames = 0
        first_pts = None
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if first_pts is None:
                    first_pts = frame.pts
                frames += 1
        return {
            "codec": video_stream.codec_context.name,
            "width": video_stream.codec_context.width,
            "height": video_stream.codec_context.height,
            "frames": frames,
            "first_pts": first_pts,
            "video_seconds": float(video_stream.duration * video_stream.time_base) if video_stream.duration else None,
            "audio_seconds": float(audio_stream.duration * audio_stream.time_base)
            if audio_stream and audio_stream.duration else None,
            "audio_codecs": [s.codec_context.name for s in container.streams.audio],
        }


def test_save_to_transcode_streams_without_buffering_frames():
    """Transcoding must not decode the whole video into memory first (~2 GiB for this source)"""
    resource = pytest.importorskip("resource")  # no getrusage on Windows
    rss_scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB elsewhere
    # ru_maxrss is a lifetime peak: a heavier test running earlier would shrink the measured
    # delta and quietly defang this canary, so keep this source the biggest thing in the suite
    file_path = create_transcode_source(width=640, height=480, frames=300)
    try:
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
        result = transcode_and_probe(VideoFromFile(file_path))
        rss_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale - rss_before

        assert result["codec"] == "h264"
        assert result["frames"] == 300
        assert rss_delta < 500 * 2**20, f"transcode buffered frames in RAM (peak grew {rss_delta / 2**20:.0f} MiB)"
    finally:
        os.unlink(file_path)


def test_save_to_transcode_honors_trim_window():
    """start_time/duration trim applies to both video and audio on the streaming path"""
    file_path = create_transcode_source(frames=90)  # 3s @ 30fps
    try:
        result = transcode_and_probe(VideoFromFile(file_path, start_time=1, duration=1))
        assert result["frames"] == pytest.approx(30, abs=2)
        assert result["first_pts"] == 0  # trimmed output is rebased to start at zero
        assert result["video_seconds"] == pytest.approx(1.0, abs=0.1)
        assert result["audio_seconds"] == pytest.approx(1.0, abs=0.1)
    finally:
        os.unlink(file_path)


def test_save_to_transcode_keeps_audio_of_sparse_video():
    """Audio that runs ahead of a sparse video track (slideshows, timelapses) must be
    kept in full — it is only clamped to the video's end, never to the video cursor."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("mpeg4", rate=30)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=48000, layout="stereo")
        for t in (0, 30, 60):  # 3 frames spread over 60 seconds
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), t * 4, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = t * 15360
            frame.time_base = Fraction(1, 15360)
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode(None))
        for offset in range(0, 48000 * 60, 1024):
            n = min(1024, 48000 * 60 - offset)
            audio_frame = av.AudioFrame.from_ndarray(
                torch.zeros(2, n, dtype=torch.float32).numpy(), format="fltp", layout="stereo"
            )
            audio_frame.sample_rate = 48000
            audio_frame.pts = offset
            audio_frame.time_base = Fraction(1, 48000)
            container.mux(audio_stream.encode(audio_frame))
        container.mux(audio_stream.encode(None))

    buffer.seek(0)
    result = transcode_and_probe(VideoFromFile(buffer))
    assert result["audio_seconds"] == pytest.approx(60.0, abs=1.0)


def test_save_to_transcode_vfr_audio_covers_video_span():
    """A trim window in the sparse region of a VFR file keeps audio for the true pts span
    of the kept frames. Deriving the span as frames/average_rate undercuts it badly: the
    average is dominated by the dense region (and can be plain wrong on MediaRecorder files)."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("mpeg4", rate=30)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=48000, layout="stereo")
        # 10 frames inside the first second, then one every 1.25 s
        for i, t in enumerate([x / 10 for x in range(10)] + [1.0, 2.25, 3.5, 4.75]):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), (i * 16) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = int(t * 15360)
            frame.time_base = Fraction(1, 15360)
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode(None))
        for offset in range(0, 48000 * 6, 1024):
            n = min(1024, 48000 * 6 - offset)
            audio_frame = av.AudioFrame.from_ndarray(
                torch.zeros(2, n, dtype=torch.float32).numpy(), format="fltp", layout="stereo"
            )
            audio_frame.sample_rate = 48000
            audio_frame.pts = offset
            audio_frame.time_base = Fraction(1, 48000)
            container.mux(audio_stream.encode(audio_frame))
        container.mux(audio_stream.encode(None))

    buffer.seek(0)
    result = transcode_and_probe(VideoFromFile(buffer, start_time=1, duration=5))
    # kept frames: 1.0/2.25/3.5/4.75 s -> rebased span 3.75 s + one nominal interval
    assert result["frames"] == 4
    assert result["audio_seconds"] == pytest.approx(4.0, abs=0.45)


def test_save_to_transcode_trims_audio_in_stream_time_base_units():
    """Matroska audio timestamps tick in 1/1000, not 1/sample_rate; trim and audio timing
    must convert through the frame's time base instead of assuming sample units. AAC audio,
    because it decodes straight to the encoder's format and hits the resampler passthrough
    that keeps the source time base on the frames."""
    file_path = create_transcode_source(frames=90, container_format="matroska", audio_codec="aac")
    try:
        result = transcode_and_probe(VideoFromFile(file_path, start_time=1, duration=1))
        assert result["audio_codecs"] == ["aac"]
        assert result["video_seconds"] == pytest.approx(1.0, abs=0.1)
        assert result["audio_seconds"] == pytest.approx(1.0, abs=0.1)
    finally:
        os.unlink(file_path)


def test_save_to_transcode_learns_unprobed_audio_params():
    """mpegts is only probed a few seconds deep at open, so an audio stream whose first
    packet comes later (live captures where audio kicks in late) still has sample_rate 0
    when the transcode starts; the parameters must be learned from the stream itself."""
    sample_rate, fps, video_seconds, audio_start = 48000, 30, 13, 12
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mpegts") as container:
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        for i in range(video_seconds * fps):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), (i * 7) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            )
            container.mux(video_stream.encode(frame.reformat(format="yuv420p")))
        for offset in range(0, (video_seconds - audio_start) * sample_rate, 1024):
            n = min(1024, (video_seconds - audio_start) * sample_rate - offset)
            audio_frame = av.AudioFrame.from_ndarray(
                torch.zeros(1, n, dtype=torch.float32).numpy(), format="fltp", layout="mono"
            )
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = audio_start * sample_rate + offset
            container.mux(audio_stream.encode(audio_frame))
        for stream in (video_stream, audio_stream):
            container.mux(stream.encode(None))

    buffer.seek(0)
    with av.open(buffer) as container:
        # the scenario requires unprobed parameters; if a future FFmpeg probes deeper,
        # push audio_start/video_seconds further out to restore it
        assert container.streams.audio[0].codec_context.sample_rate == 0
    result = transcode_and_probe(VideoFromFile(buffer))
    assert result["frames"] == video_seconds * fps
    assert result["audio_codecs"] == ["aac"]
    assert result["audio_seconds"] == pytest.approx(1.0, abs=0.1)

    buffer.seek(0)
    trimmed_before_audio = transcode_and_probe(VideoFromFile(buffer, duration=1))
    assert trimmed_before_audio["frames"] == fps
    assert trimmed_before_audio["audio_codecs"] == []
    assert trimmed_before_audio["audio_seconds"] is None

    buffer.seek(0)
    trimmed_crossing_audio = transcode_and_probe(VideoFromFile(buffer, start_time=11.5, duration=1))
    assert trimmed_crossing_audio["frames"] == fps
    assert trimmed_crossing_audio["audio_codecs"] == ["aac"]
    assert trimmed_crossing_audio["video_seconds"] == pytest.approx(1.0, abs=0.05)
    assert trimmed_crossing_audio["audio_seconds"] == pytest.approx(0.5, abs=0.1)


def test_save_to_transcode_trimmed_fragmented_mp4_keeps_audio():
    """Fragmented mp4 (MediaRecorder, DASH/HLS-derived files) delivers audio well behind
    video, so when the trim window's last video frame arrives the audio demuxed so far
    does not cover the window yet; the transcode must keep demuxing audio until it does
    instead of finalizing on the first audio frame it sees afterwards."""
    sample_rate, fps, seconds = 48000, 30, 6
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4", options={"movflags": "frag_keyframe+empty_moov"}) as container:
        video_stream = container.add_stream("h264", rate=fps)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        next_audio_pts = 0
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), (i * 7) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            )
            container.mux(video_stream.encode(frame.reformat(format="yuv420p")))
            while next_audio_pts / sample_rate <= i / fps:  # feed audio alongside, like a live pipeline
                audio_frame = av.AudioFrame.from_ndarray(
                    torch.zeros(1, 1024, dtype=torch.float32).numpy(), format="fltp", layout="mono"
                )
                audio_frame.sample_rate = sample_rate
                audio_frame.pts = next_audio_pts
                container.mux(audio_stream.encode(audio_frame))
                next_audio_pts += 1024
        for stream in (video_stream, audio_stream):
            container.mux(stream.encode(None))

    result = transcode_and_probe(VideoFromFile(buffer, start_time=0.5, duration=1.0))
    assert result["video_seconds"] == pytest.approx(1.0, abs=0.05)
    assert result["audio_seconds"] == pytest.approx(1.0, abs=0.05)


def test_save_to_transcode_sparse_video_keeps_true_duration():
    """average_rate is not a frame duration: a 3-frame video spanning 60 s averages
    0.05 fps, and padding the last frame with 1/average_rate used to extend the
    output — and the audio kept with it — about 20 s past the source span."""
    sample_rate = 48000
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("mpeg4", rate=30)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        for i, second in enumerate((0, 30, 60)):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), i * 80, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = second * 30
            frame.time_base = Fraction(1, 30)
            container.mux(video_stream.encode(frame))
        for offset in range(0, 90 * sample_rate, 1024):
            n = min(1024, 90 * sample_rate - offset)
            audio_frame = av.AudioFrame.from_ndarray(
                torch.zeros(1, n, dtype=torch.float32).numpy(), format="fltp", layout="mono"
            )
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = offset
            container.mux(audio_stream.encode(audio_frame))
        for stream in (video_stream, audio_stream):
            container.mux(stream.encode(None))

    result = transcode_and_probe(VideoFromFile(buffer))
    assert result["frames"] == 3
    # the last frame keeps its true stts duration (1/30 s), not 1/average_rate (~20 s)
    assert result["video_seconds"] == pytest.approx(60.03, abs=0.05)
    assert result["audio_seconds"] == pytest.approx(60.03, abs=0.1)

    trimmed = transcode_and_probe(VideoFromFile(buffer, duration=45))
    assert trimmed["frames"] == 2
    # a kept frame whose source duration crosses the window end is clamped to it
    assert trimmed["video_seconds"] == pytest.approx(45.0, abs=0.05)
    assert trimmed["audio_seconds"] == pytest.approx(45.0, abs=0.1)


def test_save_to_transcode_clamps_final_pts_to_declared_stream_duration():
    """Some iPhone MOVs report a video stream duration that ends before the final
    decoded frame's nominal duration. A transcode must not turn that trailing
    timestamp quirk into an extra frame interval compared to the source/remux path."""
    fps = 30
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        for i, pts in enumerate([*range(31), 32]):
            frame = av.VideoFrame.from_ndarray(
                torch.full((64, 64, 3), (i * 7) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = pts
            frame.time_base = Fraction(1, fps)
            container.mux(video_stream.encode(frame))
        container.mux(video_stream.encode(None))

    class _StreamProxy:
        def __init__(self, stream, duration):
            self._stream = stream
            self.duration = duration

        def __getattr__(self, name):
            return getattr(self._stream, name)

    class _StreamsProxy:
        def __init__(self, video_stream):
            self.video = [video_stream]
            self.audio = []

    class _PacketProxy:
        def __init__(self, packet, stream):
            self._packet = packet
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self._packet, name)

    class _ContainerProxy:
        def __init__(self, container, stream):
            self._container = container
            self._stream = stream
            self.streams = _StreamsProxy(stream)

        def __getattr__(self, name):
            return getattr(self._container, name)

        def demux(self, *streams):
            for packet in self._container.demux(self._stream._stream):
                yield _PacketProxy(packet, self._stream)

    buffer.seek(0)
    output = io.BytesIO()
    with av.open(buffer) as container:
        real_stream = container.streams.video[0]
        declared_duration = 32 * int(round((1 / fps) / real_stream.time_base))
        stream = _StreamProxy(real_stream, declared_duration)
        VideoFromFile(buffer)._save_transcoded(
            _ContainerProxy(container, stream), output, VideoContainer.MP4, VideoCodec.H264, None, 8
        )

    output.seek(0)
    with av.open(output) as container:
        video_stream = container.streams.video[0]
        frames = [f for p in container.demux(video_stream) for f in p.decode()]
        assert len(frames) == 32
        assert float(video_stream.duration * video_stream.time_base) == pytest.approx(32 / fps, abs=0.01)
        assert float(frames[-1].pts * frames[-1].time_base) == pytest.approx(31 / fps, abs=0.01)


def test_save_to_transcode_irregular_vfr_keeps_span():
    """B-frames reorder packets, and mp4 sample durations follow decode order: the dts
    timeline ends before the pts timeline, so an irregular-VFR source's tail holds fell
    out of the container (this 20.23 s span used to come out as 15.27 s, and the 10 s
    trim as 6.03 s). The transcode encodes without B-frames so every sample keeps its
    true display duration."""
    durations = [1, 1, 60, 1, 1, 120, 1, 180, 1, 1, 150, 90]  # 1/30 s ticks, span 20.2333 s
    generator = torch.Generator().manual_seed(7)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video_stream = container.add_stream("mpeg4", rate=30)
        video_stream.width = video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        pts = 0
        for duration in durations:
            # textured frames, so an encoder with default settings has B-frames to gain from
            frame = av.VideoFrame.from_ndarray(
                torch.randint(0, 255, (64, 64, 3), generator=generator, dtype=torch.uint8).numpy(),
                format="rgb24",
            ).reformat(format="yuv420p")
            frame.pts = pts
            frame.time_base = Fraction(1, 30)
            pts += duration
            for packet in video_stream.encode(frame):
                packet.duration = duration  # exact stts in the source
                container.mux(packet)
        container.mux(video_stream.encode(None))

    result = transcode_and_probe(VideoFromFile(buffer))
    assert result["frames"] == len(durations)
    assert result["video_seconds"] == pytest.approx(sum(durations) / 30, abs=0.05)

    trimmed = transcode_and_probe(VideoFromFile(buffer, duration=10))
    assert trimmed["frames"] == 8  # frames at 12.167 s+ fall outside the window
    assert trimmed["video_seconds"] == pytest.approx(10.0, abs=0.05)


def test_save_to_transcode_trim_survives_missing_leading_pts():
    """A trim should survive pts-less kept frames followed by a real-pts frame past the window."""
    nulled_frames = 0

    class _PacketProxy:
        def __init__(self, packet):
            self._packet = packet

        def __getattr__(self, name):
            return getattr(self._packet, name)

        @property
        def stream(self):
            return self._packet.stream

        def decode(self):
            nonlocal nulled_frames
            frames = self._packet.decode()
            for frame in frames:
                if nulled_frames < 2:
                    frame.pts = None
                    nulled_frames += 1
            return frames

    class _ContainerProxy:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def demux(self, *streams):
            for packet in self._real.demux(*streams):
                yield _PacketProxy(packet)

    file_path = create_transcode_source(frames=10, audio_streams=0)
    try:
        buffer = io.BytesIO()
        with av.open(file_path) as container:
            # 0.05 s window: both pts-less frames are kept (synthesized pts 0 and 512),
            # and the first real-pts frame (1024 ticks) already lies past end_pts (768)
            VideoFromFile(file_path, duration=0.05)._save_transcoded(
                _ContainerProxy(container), buffer, VideoContainer.MP4, VideoCodec.H264, None, 8
            )
        assert nulled_frames == 2
        buffer.seek(0)
        with av.open(buffer) as container:
            video_stream = container.streams.video[0]
            frames = [f for p in container.demux(video_stream) for f in p.decode()]
            assert len(frames) == 2
            assert float(video_stream.duration * video_stream.time_base) == pytest.approx(2 / 30, abs=0.01)
    finally:
        os.unlink(file_path)


def test_save_to_transcode_bakes_rotation():
    """A 90-degree display-matrix rotation swaps the output dimensions (portrait video)"""
    file_path = create_transcode_source(width=64, height=32, rotation=True)
    try:
        result = transcode_and_probe(VideoFromFile(file_path))
        assert (result["width"], result["height"]) == (32, 64)
        assert result["frames"] == 30
    finally:
        os.unlink(file_path)


def hevc_encoder_available():
    try:
        av.Codec("libx265", "w")
        return True
    except av.codec.codec.UnknownCodecError:
        return False


hevc_remux_test = pytest.mark.skipif(not hevc_encoder_available(), reason="libx265 encoder not available")


def create_hevc_mp4(codec_tag=None, x265_params=None):
    """In-memory HEVC mp4, which FFmpeg tags 'hev1' unless codec_tag is given."""
    buffer = io.BytesIO()
    options = {"x265-params": ":".join(["log-level=none"] + ([x265_params] if x265_params else []))}
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx265", rate=30, options=options)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        if codec_tag is not None:
            stream.codec_context.codec_tag = codec_tag
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(
                torch.ones(64, 64, 3, dtype=torch.uint8).numpy() * (i * 85),
                format="rgb24",
            ).reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    return buffer


def filter_hvcc_arrays(source: io.BytesIO, keep_nal_types: tuple = ()) -> io.BytesIO:
    """hev1 whose hvcC keeps only the given NAL array types (32=VPS, 33=SPS, 34=PPS, 39=SEI)."""
    output = io.BytesIO()
    with av.open(source) as container, av.open(output, mode="w", format="mp4") as output_container:
        stream = container.streams.video[0]
        extradata = stream.codec_context.extradata
        rebuilt, pos = bytearray(extradata[:22] + b"\x00"), 23
        for _ in range(extradata[22]):
            array_start = pos
            num_nalus = int.from_bytes(extradata[pos + 1:pos + 3], "big")
            pos += 3
            for _ in range(num_nalus):
                pos += 2 + int.from_bytes(extradata[pos:pos + 2], "big")
            if extradata[array_start] & 0x3F in keep_nal_types:
                rebuilt += extradata[array_start:pos]
                rebuilt[22] += 1
        out_stream = output_container.add_stream_from_template(template=stream, opaque=True)
        out_stream.codec_context.extradata = bytes(rebuilt)
        for packet in container.demux(stream):
            if packet.dts is not None:
                packet.stream = out_stream
                output_container.mux(packet)
    output.seek(0)
    return output


def concat_hevc_sample_descriptions(first: io.BytesIO, second: io.BytesIO, codec_tag=None) -> io.BytesIO:
    """mp4 with two sample descriptions, the second created from new_extradata packet side data."""
    output = io.BytesIO()
    with av.open(first) as a, av.open(second) as b, av.open(output, mode="w", format="mp4") as output_container:
        out_stream = output_container.add_stream_from_template(template=a.streams.video[0], opaque=True)
        if codec_tag is not None:
            out_stream.codec_context.codec_tag = codec_tag
        end = 0
        for packet in a.demux(a.streams.video[0]):
            if packet.dts is not None:
                packet.stream = out_stream
                output_container.mux(packet)
                end = max(end, packet.pts + packet.duration)
        extradata = b.streams.video[0].codec_context.extradata
        packets = [packet for packet in b.demux(b.streams.video[0]) if packet.dts is not None]
        sidedata = av.packet.PacketSideData(av.packet.packet_sidedata_type_from_literal("new_extradata"), len(extradata))
        sidedata.update(extradata)
        packets[0].set_sidedata(sidedata)
        for packet in packets:
            packet.pts += end
            packet.dts += end
            packet.stream = out_stream
            output_container.mux(packet)
    output.seek(0)
    return output


def sample_description_count(data: bytes, start: int = 0, end: int | None = None) -> int:
    """Number of 'stsd' entries; a broken orphan entry is invisible to decoders but not to players."""
    pos, end, count = start, len(data) if end is None else end, 0
    while pos + 8 <= end:
        size, box = int.from_bytes(data[pos:pos + 4], "big"), data[pos + 4:pos + 8]
        if box == b"stsd":
            count += int.from_bytes(data[pos + 12:pos + 16], "big")
        elif box in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
            count += sample_description_count(data, pos + 8, pos + size)
        pos += size
    return count


def probe_hevc(source: io.BytesIO) -> dict:
    source.seek(0)
    with av.open(source) as container:
        stream = container.streams.video[0]
        return {
            "tag": stream.codec_context.codec_tag,
            "frames": sum(1 for packet in container.demux(stream) for _ in packet.decode()),
            "sample_descriptions": sample_description_count(source.getvalue()),
        }


def remux_and_probe(source: io.BytesIO, **save_kwargs) -> dict:
    output = io.BytesIO()
    source.seek(0)
    VideoFromFile(source).save_to(output, **save_kwargs)
    return probe_hevc(output)


@hevc_remux_test
@pytest.mark.parametrize("save_kwargs", [pytest.param({}, id="mov"), pytest.param({"format": VideoContainer.MP4}, id="mp4")])
@pytest.mark.parametrize("codec_tag", [None, "dvh1"], ids=["hev1", "dvh1"])
def test_save_to_remux_retags_hevc_as_hvc1(codec_tag, save_kwargs):
    """Remuxed HEVC gets 'hvc1' instead of FFmpeg's default 'hev1', Dolby Vision included: PyAV
    resets the source tag and the muxer drops the DV boxes anyway, so skipping 'dvh1' means 'hev1'."""
    source = create_hevc_mp4(codec_tag=codec_tag)
    assert probe_hevc(source)["tag"] == (codec_tag or "hev1")
    assert remux_and_probe(source, **save_kwargs) == {"tag": "hvc1", "frames": 3, "sample_descriptions": 1}


@hevc_remux_test
@pytest.mark.parametrize(
    "keep_nal_types",
    [pytest.param((), id="empty-hvcc"), pytest.param((33, 34), id="missing-vps"), pytest.param((39,), id="sei-only")],
)
def test_save_to_remux_rebuilds_hvcc_from_inband_parameter_sets(keep_nal_types):
    """Parameter sets missing from hvcC but present in-band yield one valid hvcC, not an empty one."""
    source = filter_hvcc_arrays(create_hevc_mp4(x265_params="repeat-headers=1"), keep_nal_types)
    output = io.BytesIO()
    VideoFromFile(source).save_to(output)
    output.seek(0)
    with av.open(output) as container:
        assert container.streams.video[0].codec_context.extradata[22] > 0  # hvcC numOfArrays
    assert probe_hevc(output) == {"tag": "hvc1", "frames": 3, "sample_descriptions": 1}


@hevc_remux_test
def test_save_to_remux_keeps_hvc1_sources_untouched():
    """'hvc1' sources skip the bitstream filter, so their sample descriptions survive as they are."""
    source = concat_hevc_sample_descriptions(
        create_hevc_mp4(codec_tag="hvc1"), create_hevc_mp4(codec_tag="hvc1", x265_params="no-sao=1"), codec_tag="hvc1"
    )
    assert probe_hevc(source) == {"tag": "hvc1", "frames": 6, "sample_descriptions": 2}
    assert remux_and_probe(source) == {"tag": "hvc1", "frames": 6, "sample_descriptions": 2}


@hevc_remux_test
def test_save_to_remux_rejects_hevc_with_multiple_sample_descriptions():
    """hevc_mp4toannexb ignores mid-stream parameter set changes, so such sources must be re-encoded."""
    source = concat_hevc_sample_descriptions(create_hevc_mp4(), create_hevc_mp4(x265_params="no-sao=1"))
    assert probe_hevc(source) == {"tag": "hev1", "frames": 6, "sample_descriptions": 2}
    with pytest.raises(ValueError, match="multiple sample descriptions"):
        remux_and_probe(source)


def test_save_to_transcode_skips_undecodable_audio():
    """Streaming transcode keeps the decodable audio track and drops undecodable ones;
    with no decodable audio at all the output is video-only instead of crashing."""
    mixed = all_bad = None
    try:
        mixed = create_transcode_source(audio_streams=1, undecodable_audio=1)
        all_bad = create_transcode_source(audio_streams=0, undecodable_audio=2)
        result = transcode_and_probe(VideoFromFile(mixed))
        assert result["audio_codecs"] == ["aac"]
        assert result["audio_seconds"] == pytest.approx(1.0, abs=0.1)
        assert transcode_and_probe(VideoFromFile(all_bad))["audio_codecs"] == []
    finally:
        for path in (mixed, all_bad):
            if path:
                os.unlink(path)


def test_as_trimmed_strict_duration_gates_unavailable_length(simple_video_file):
    video = VideoFromFile(simple_video_file)

    assert video.as_trimmed(0.0, 10.0, strict_duration=True) is None

    relaxed = video.as_trimmed(0.0, 10.0, strict_duration=False)
    assert relaxed is not None
    assert relaxed.get_duration() < 10.0
    assert relaxed.get_duration() == pytest.approx(video.get_duration(), abs=EPSILON)


def test_normalize_crop_rect_aligns_odd_origin_to_chroma_grid():
    assert normalize_crop_rect(1, 1, 100, 100, 1920, 1080) == (0, 0, 100, 100)
    assert normalize_crop_rect(3, 5, 10, 9, 64, 48) == (2, 4, 10, 8)
    assert normalize_crop_rect(0, 0, 64, 48, 64, 48) is None


def _create_marker_video(path, width=64, height=48, marker_x=2, frames=3, fps=8):
    with av.open(path, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        pixels = torch.zeros(height, width, 3, dtype=torch.uint8).numpy()
        pixels[:, marker_x:marker_x + 2, :] = 255
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _brightest_column(images):
    return images[0].float().mean(dim=(0, 2)).argmax().item()


def test_cropped_decode_and_save_paths_select_same_pixels(tmp_path):
    source = str(tmp_path / "marker.mp4")
    _create_marker_video(source, marker_x=4)

    cropped = VideoFromFile(source).as_cropped(3, 1, 16, 16)

    components = cropped.get_components()
    assert tuple(components.images.shape[1:3]) == (16, 16)
    decode_column = _brightest_column(components.images)
    assert decode_column in (2, 3)

    saved = str(tmp_path / "cropped.mp4")
    cropped.save_to(saved)
    saved_components = VideoFromFile(saved).get_components()
    assert tuple(saved_components.images.shape[1:3]) == (16, 16)
    save_column = _brightest_column(saved_components.images)
    assert save_column in (2, 3)

    assert decode_column == save_column


def test_as_cropped_components_releases_uncropped_storage():
    images = torch.rand(2, 8, 8, 3)
    video = VideoFromComponents(
        VideoComponents(images=images, frame_rate=Fraction(8))
    )

    cropped = video.as_cropped(0, 0, 4, 4)
    cropped_images = cropped.get_components().images

    assert tuple(cropped_images.shape[1:3]) == (4, 4)
    assert (
        cropped_images.untyped_storage().data_ptr()
        != images.untyped_storage().data_ptr()
    )


def test_video_encoder_options_applies_h264_preset():
    from comfy_api.latest._input_impl.video_types import video_encoder_options

    assert video_encoder_options(VideoCodec.H264, None, "ultrafast") == {
        "preset": "ultrafast"
    }
    assert video_encoder_options(VideoCodec.H264, 23.0, "ultrafast") == {
        "preset": "ultrafast",
        "crf": "23.0",
    }
    assert video_encoder_options(VideoCodec.H264, 23.0, None) == {"crf": "23.0"}
    assert video_encoder_options(VideoCodec.AV1, None, "ultrafast") == {}
    assert video_encoder_options(VideoCodec.AV1, 0, "ultrafast") == {
        "svtav1-params": "lossless=1"
    }


def test_save_to_preset_transcodes_playable_output(tmp_path):
    source = create_test_video(width=32, height=32)
    try:
        out = str(tmp_path / "preset.mp4")
        VideoFromFile(source).as_cropped(0, 0, 16, 16).save_to(
            out, preset="ultrafast"
        )
        saved = VideoFromFile(out).get_components()
        assert tuple(saved.images.shape[1:3]) == (16, 16)
    finally:
        os.unlink(source)
