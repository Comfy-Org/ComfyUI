"""Tests for VIDEO spatial resizing (VideoInput.as_resized)."""
import io
import os
from fractions import Fraction

import av
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.latest._input import VideoInput
from comfy_api.latest._util.video_types import apply_spatial_ops
import comfy.utils
from comfy_api.latest._input_impl import video_types
from comfy_api.util.video_types import VideoCodec, VideoComponents, VideoContainer
from comfy_api.input.basic_types import AudioInput

from video_types_test import (
    create_hdr_av1_video,
    create_test_video,
    create_transcode_source,
    video_packet_bytes,
)

SCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


class MinimalVideo(VideoInput):
    """A VideoInput specialization that implements nothing beyond the abstract API,
    so it exercises the base-class fallback rather than any optimized override."""

    def __init__(self, components, bit_depth=8, color_space="auto"):
        self._components = components
        self._bit_depth = bit_depth
        self._color_space = color_space

    def get_components(self):
        return self._components

    def get_bit_depth(self):
        return self._bit_depth

    def get_color_space(self):
        return self._color_space

    def save_to(self, path, format=VideoContainer.AUTO, codec=VideoCodec.AUTO, metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None):
        VideoFromComponents(self._components).save_to(
            path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth,
            crf=crf, color_space=color_space, preset=preset,
        )

    def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
        raise NotImplementedError


def make_components(frames=3, width=8, height=6, alpha=False, audio=True):
    generator = torch.Generator().manual_seed(7)
    return VideoComponents(
        images=torch.rand(frames, height, width, 3, generator=generator),
        frame_rate=Fraction(30),
        audio=AudioInput({"waveform": torch.rand(1, 2, 1000, generator=generator), "sample_rate": 44100})
        if audio
        else None,
        metadata={"test": "metadata"},
        alpha=torch.rand(frames, height, width, 1, generator=generator) if alpha else None,
    )


def make_lossless_video(width=32, height=24, frames=3, fps=30):
    """FFV1 in MKV, so decoded pixels round-trip exactly and parity assertions are exact."""
    tmp = io.BytesIO()
    generator = torch.Generator().manual_seed(3)
    source = (torch.rand(frames, height, width, 3, generator=generator) * 255).to(torch.uint8)
    with av.open(tmp, mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv444p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(source[index].numpy(), format="rgb24")
            container.mux(stream.encode(frame.reformat(format="yuv444p")))
        container.mux(stream.encode(None))
    tmp.seek(0)
    return tmp


# --- base class fallback -------------------------------------------------


@pytest.mark.parametrize("scale_method", SCALE_METHODS)
def test_fallback_resizes_and_preserves_components(scale_method):
    """The base fallback works for any VideoInput and keeps everything but geometry"""
    components = make_components(alpha=True)
    video = MinimalVideo(components, bit_depth=10, color_space="HDR")
    resized = video.as_resized(4, 3, scale_method)

    result = resized.get_components()
    assert result.images.shape == (3, 3, 4, 3)
    assert result.alpha.shape == (3, 3, 4, 1)
    assert result.frame_rate == components.frame_rate
    assert result.metadata == components.metadata
    assert torch.equal(result.audio["waveform"], components.audio["waveform"])
    assert result.audio["sample_rate"] == components.audio["sample_rate"]
    assert resized.get_bit_depth() == 10
    assert resized.get_color_space() == "HDR"
    assert resized.get_frame_count() == components.images.shape[0]


def test_fallback_keeps_auto_color_space(monkeypatch):
    """An unspecified color space stays unspecified; resizing must not decide it is sRGB"""
    # large enough for the encoders to accept the frames
    resized = MinimalVideo(make_components(width=64, height=48)).as_resized(32, 24, "area")
    assert resized.get_color_space() == "auto"

    forced = []
    monkeypatch.setattr(
        video_types, "set_video_color_properties", lambda target, color_space: forced.append(color_space)
    )
    resized.save_to(io.BytesIO(), format=VideoContainer.MP4, codec=VideoCodec.H264)
    assert forced == []


@pytest.mark.parametrize("color_space", ["sRGB", "HDR", "HDR PQ"])
def test_fallback_keeps_explicit_color_space(monkeypatch, color_space):
    """A known color space is still carried through the resize and still forced on save"""
    resized = MinimalVideo(
        make_components(width=64, height=48), color_space=color_space
    ).as_resized(32, 24, "area")
    assert resized.get_color_space() == color_space

    forced = []
    monkeypatch.setattr(
        video_types, "set_video_color_properties", lambda target, space: forced.append(space)
    )
    resized.save_to(io.BytesIO(), format=VideoContainer.MKV, codec=VideoCodec.AV1)
    assert set(forced) == {color_space}


def test_fallback_matches_common_upscale():
    """The fallback scales with comfy.utils.common_upscale and nothing else.

    Building the expectation from common_upscale directly, rather than from the same
    helper the implementation uses, keeps this honest about the helper itself.
    """
    components = make_components()
    resized = MinimalVideo(components).as_resized(5, 4, "bicubic")
    expected = comfy.utils.common_upscale(
        components.images.movedim(-1, 1), 5, 4, "bicubic", "disabled"
    ).movedim(1, -1)
    assert torch.equal(resized.get_components().images, expected)


def test_fallback_post_crop_is_exact():
    """post_crop is an exact slice; an odd rectangle is not snapped to an even grid"""
    resized = MinimalVideo(make_components()).as_resized(9, 7, "area", "disabled", (3, 1, 5, 5))
    assert resized.get_dimensions() == (5, 5)


# --- VideoFromFile -------------------------------------------------------


@pytest.mark.parametrize("scale_method", SCALE_METHODS)
def test_file_get_components_reflects_resize(scale_method):
    """Resizing on decode gives the same pixels as decoding then resizing"""
    source = make_lossless_video()
    video = VideoFromFile(source)
    expected = apply_spatial_ops(
        video.get_components().images, [("scale", 7, 5, scale_method, "disabled")]
    )
    resized = video.as_resized(7, 5, scale_method)
    assert torch.equal(resized.get_components().images, expected)
    assert resized.get_dimensions() == (7, 5)


def test_file_save_to_reflects_resize():
    """save_to() must not remux around the resize"""
    video = VideoFromFile(make_lossless_video()).as_resized(8, 6, "area")
    buffer = io.BytesIO()
    video.save_to(buffer, format=VideoContainer.MP4, codec=VideoCodec.H264)
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (8, 6)


def test_file_dimensions_agree_across_paths():
    """get_dimensions(), get_components() and save_to() report one size"""
    video = VideoFromFile(make_lossless_video()).as_resized(6, 4, "bilinear")
    components = video.get_components()
    assert video.get_dimensions() == (6, 4)
    assert (components.images.shape[2], components.images.shape[1]) == (6, 4)
    buffer = io.BytesIO()
    video.save_to(buffer, format=VideoContainer.MP4, codec=VideoCodec.H264)
    buffer.seek(0)
    assert VideoFromFile(buffer).get_dimensions() == (6, 4)


def test_file_keeps_odd_logical_size():
    """An odd resize result stays the logical size; only the encoder trims it"""
    video = VideoFromFile(make_lossless_video()).as_resized(7, 5, "area")
    assert video.get_dimensions() == (7, 5)
    assert video.get_components().images.shape[1:3] == (5, 7)
    buffer = io.BytesIO()
    video.save_to(buffer, format=VideoContainer.MP4, codec=VideoCodec.H264)
    buffer.seek(0)
    with av.open(buffer) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (6, 4)


def test_file_resize_of_unaligned_width():
    """Sources whose width is not a multiple of 32 go through the decode alignment
    padding first; the resize must sit after it, on the real frame."""
    file_path = create_test_video(width=40, height=24, frames=2)
    try:
        video = VideoFromFile(file_path)
        expected = apply_spatial_ops(
            video.get_components().images, [("scale", 10, 6, "area", "disabled")]
        )
        assert torch.equal(video.as_resized(10, 6, "area").get_components().images, expected)
    finally:
        os.unlink(file_path)


def test_file_resize_never_materializes_the_video(monkeypatch):
    """The point of the file path is that no step holds every frame at once.

    get_components() is the only way to do that, so make calling it fail and check the
    resize and the save still go through.
    """
    video = VideoFromFile(make_lossless_video())
    resized = video.as_resized(4, 3, "area")
    assert resized is not video
    assert isinstance(resized, VideoFromFile)

    def forbidden(self):
        raise AssertionError("the whole video was materialized")

    monkeypatch.setattr(VideoFromFile, "get_components", forbidden)
    assert video.as_resized(4, 3, "area").get_dimensions() == (4, 3)
    VideoFromList([video.as_resized(4, 3, "area")]).save_to(
        io.BytesIO(), format=VideoContainer.MP4, codec=VideoCodec.AUTO
    )
    video.as_resized(4, 3, "area").save_to(
        io.BytesIO(), format=VideoContainer.MP4, codec=VideoCodec.AUTO
    )


def test_file_resize_keeps_bit_depth_and_color_space(tmp_path):
    """A 10-bit HDR source stays 10-bit HDR through a resize; the scaling itself runs
    in float RGB, so the extra depth is not quantized away."""
    from av.video.reformatter import ColorRange, ColorTrc

    source = str(tmp_path / "hdr.mkv")
    create_hdr_av1_video(source, ColorTrc.SMPTE2084, ColorRange.MPEG)
    video = VideoFromFile(source)
    assert (video.get_bit_depth(), video.get_color_space()) == (10, "HDR PQ")

    resized = video.as_resized(32, 32, "area")
    assert (resized.get_bit_depth(), resized.get_color_space()) == (10, "HDR PQ")

    output = str(tmp_path / "resized.mkv")
    resized.save_to(output, format=VideoContainer.MKV, codec=VideoCodec.AV1)
    saved = VideoFromFile(output)
    assert (saved.get_bit_depth(), saved.get_color_space()) == (10, "HDR PQ")
    assert saved.get_dimensions() == (32, 32)


@pytest.mark.parametrize(
    "codec,container,bit_depth",
    [(VideoCodec.H264, VideoContainer.MP4, None), (VideoCodec.AV1, VideoContainer.MKV, 10)],
    ids=["h264-8bit", "av1-10bit"],
)
def test_fallback_odd_size_is_trimmed_only_at_the_encoder(codec, container, bit_depth):
    """The fallback returns a VideoFromComponents, which has to honour the same encoder
    boundary VideoFromFile does: the odd size stays the logical size and is rounded down
    only on the way into the encoder."""
    resized = MinimalVideo(make_components(width=16, height=12)).as_resized(7, 5, "area")
    assert resized.get_dimensions() == (7, 5)
    assert resized.get_components().images.shape[1:3] == (5, 7)

    output = io.BytesIO()
    resized.save_to(output, format=container, codec=codec, bit_depth=bit_depth)
    output.seek(0)
    with av.open(output) as opened:
        stream = opened.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (6, 4)


def test_fallback_even_size_is_not_trimmed():
    """An even size reaches the encoder untouched"""
    resized = MinimalVideo(make_components(width=16, height=12)).as_resized(8, 6, "area")
    output = io.BytesIO()
    resized.save_to(output, format=VideoContainer.MP4, codec=VideoCodec.H264)
    output.seek(0)
    with av.open(output) as opened:
        stream = opened.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (8, 6)


def test_fallback_size_with_nothing_left_after_trimming_is_rejected():
    """Nothing can be encoded once both edges round down to zero"""
    resized = MinimalVideo(make_components(width=16, height=12)).as_resized(1, 1, "area")
    with pytest.raises(ValueError, match="even dimensions"):
        resized.save_to(io.BytesIO(), format=VideoContainer.MP4, codec=VideoCodec.H264)


# --- operation ordering --------------------------------------------------


def test_crop_then_resize_differs_from_resize_then_crop():
    """Order is preserved, not normalized"""
    video = VideoFromFile(make_lossless_video())
    crop_first = video.as_cropped(0, 0, 16, 12).as_resized(4, 4, "area")
    resize_first = video.as_resized(16, 12, "area").as_cropped(0, 0, 4, 4)
    assert crop_first.get_dimensions() == resize_first.get_dimensions() == (4, 4)
    assert not torch.equal(
        crop_first.get_components().images, resize_first.get_components().images
    )


def test_resize_chain_is_not_collapsed():
    """Two interpolations must not be folded into one"""
    video = VideoFromFile(make_lossless_video())
    twice = video.as_resized(5, 4, "bilinear").as_resized(12, 9, "bilinear")
    once = video.as_resized(12, 9, "bilinear")
    assert twice.get_dimensions() == once.get_dimensions() == (12, 9)
    assert not torch.equal(twice.get_components().images, once.get_components().images)


def test_resize_crop_resize_applies_in_order():
    """A three step chain matches applying the same operations to the decoded tensor"""
    video = VideoFromFile(make_lossless_video())
    chained = video.as_resized(12, 9, "area").as_resized(6, 5, "area", "disabled", (1, 1, 4, 3))
    expected = apply_spatial_ops(
        video.get_components().images,
        [
            ("scale", 12, 9, "area", "disabled"),
            ("scale", 6, 5, "area", "disabled"),
            ("crop", 1, 1, 4, 3),
        ],
    )
    assert torch.equal(chained.get_components().images, expected)


def test_post_crop_skips_even_alignment_that_as_cropped_applies():
    """as_cropped() snaps to an even grid; the resize node's own crop must not"""
    video = VideoFromFile(make_lossless_video())
    assert video.as_cropped(3, 1, 11, 9).get_dimensions() == (10, 8)
    exact = video.as_resized(16, 12, "area", "disabled", (3, 1, 11, 9))
    assert exact.get_dimensions() == (11, 9)
    assert exact.get_components().images.shape[1:3] == (9, 11)


# --- rotation ------------------------------------------------------------


def test_display_dimensions_are_rotation_aware():
    """Geometry uses the displayed orientation; get_dimensions() keeps its contract"""
    file_path = create_transcode_source(width=64, height=32, rotation=True, frames=2)
    try:
        video = VideoFromFile(file_path)
        assert video.get_dimensions() == (64, 32)
        assert video.get_display_dimensions() == (32, 64)
        resized = video.as_resized(16, 32, "area")
        assert resized.get_dimensions() == (16, 32)
        assert resized.get_components().images.shape[1:3] == (32, 16)
    finally:
        os.unlink(file_path)


# --- VideoFromList -------------------------------------------------------


def test_list_propagates_to_children():
    """Concatenated videos resize per chunk instead of falling back to materializing"""
    videos = [VideoFromFile(make_lossless_video()) for _ in range(2)]
    resized = VideoFromList(videos).as_resized(8, 6, "area")
    assert isinstance(resized, VideoFromList)
    assert all(isinstance(child, VideoFromFile) for child in resized.videos)
    assert all(child.get_dimensions() == (8, 6) for child in resized.videos)
    assert resized.get_dimensions() == (8, 6)
    assert resized.get_display_dimensions() == (8, 6)


def test_list_resize_does_not_reuse_cached_stream():
    """The cached concatenated stream of the original must not leak into the resized one"""
    videos = [VideoFromFile(make_lossless_video()) for _ in range(2)]
    original = VideoFromList(videos)
    original.get_stream_source()  # populates the cache
    resized = original.as_resized(8, 6, "area")
    assert resized is not original
    assert VideoFromFile(resized.get_stream_source()).get_dimensions() == (8, 6)


# --- timing --------------------------------------------------------------


def test_resize_preserves_timing_and_audio():
    """Resizing is spatial only: no frame, rate, duration or audio change"""
    file_path = create_transcode_source(width=64, height=32, frames=30)
    try:
        video = VideoFromFile(file_path)
        before = video.get_components()
        after = video.as_resized(16, 8, "area").get_components()
        assert after.frame_rate == before.frame_rate
        assert after.images.shape[0] == before.images.shape[0]
        assert torch.equal(after.audio["waveform"], before.audio["waveform"])
        assert after.audio["sample_rate"] == before.audio["sample_rate"]
        assert video.as_resized(16, 8, "area").get_duration() == pytest.approx(
            video.get_duration()
        )
    finally:
        os.unlink(file_path)


def test_resize_preserves_vfr_timestamps():
    """A sparse/VFR source keeps its presentation timestamps through a resize"""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        for t in (0, 12, 45):  # unevenly spaced, so this is a real variable frame rate
            frame = av.VideoFrame.from_ndarray(
                torch.full((32, 64, 3), (t * 4) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = t * 15360
            frame.time_base = Fraction(1, 15360)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    buffer.seek(0)

    def pts_of(video):
        out = io.BytesIO()
        video.save_to(out, format=VideoContainer.MP4, codec=VideoCodec.H264)
        out.seek(0)
        with av.open(out) as container:
            video_stream = container.streams.video[0]
            return [
                float(frame.pts * video_stream.time_base)
                for packet in container.demux(video_stream)
                for frame in packet.decode()
            ]

    original = pts_of(VideoFromFile(buffer))
    resized = pts_of(VideoFromFile(buffer).as_resized(16, 8, "area"))
    assert resized == pytest.approx(original)


# --- remux vs transcode -------------------------------------------------


def test_list_transcodes_when_a_chunk_has_pending_edits():
    """A resize chain that lands back on the source size still cannot be remuxed.

    Comparing only the final dimensions would call this chunk unchanged and copy the
    original packets straight through, silently dropping both resizes.
    """
    source = make_lossless_video()
    source_packets = video_packet_bytes(io.BytesIO(source.getvalue()))
    video = VideoFromFile(source)
    original = video.get_components().images
    resized = video.as_resized(16, 12, "area").as_resized(32, 24, "area")
    expected = apply_spatial_ops(
        original,
        [("scale", 16, 12, "area", "disabled"), ("scale", 32, 24, "area", "disabled")],
    )
    assert resized.get_dimensions() == (32, 24)

    output = io.BytesIO()
    VideoFromList([resized]).save_to(output, format=VideoContainer.MKV, codec=VideoCodec.AUTO)
    # remuxing copies packets verbatim, so identical bytes mean the edits were dropped
    assert video_packet_bytes(io.BytesIO(output.getvalue())) != source_packets

    output.seek(0)
    saved = VideoFromFile(output).get_components().images
    assert (saved - expected).abs().mean() < (saved - original).abs().mean()


def test_list_without_pending_edits_still_remuxes():
    """Unedited chunks keep the packet-copy fast path"""
    source = make_lossless_video()
    source_packets = video_packet_bytes(io.BytesIO(source.getvalue()))

    output = io.BytesIO()
    VideoFromList([VideoFromFile(make_lossless_video())]).save_to(
        output, format=VideoContainer.MKV, codec=VideoCodec.AUTO
    )
    assert video_packet_bytes(io.BytesIO(output.getvalue())) == source_packets

    concatenated = io.BytesIO()
    VideoFromList([VideoFromFile(make_lossless_video()), VideoFromFile(make_lossless_video())]).save_to(
        concatenated, format=VideoContainer.MKV, codec=VideoCodec.AUTO
    )
    assert video_packet_bytes(io.BytesIO(concatenated.getvalue())) == source_packets * 2


def test_has_pending_frame_edits_reports_only_frame_edits():
    """The predicate covers this video's own unapplied frame edits, nothing else"""
    video = VideoFromFile(make_lossless_video())
    assert video.has_pending_frame_edits() is False
    assert video.as_resized(8, 6, "area").has_pending_frame_edits() is True
    assert video.as_cropped(0, 0, 16, 12).has_pending_frame_edits() is True
    # trimming is reported through get_active_trim_window(); the two stay separate
    assert video.as_trimmed(0, 0.05, False).has_pending_frame_edits() is False

    assert MinimalVideo(make_components()).has_pending_frame_edits() is False

    plain = VideoFromFile(make_lossless_video())
    assert VideoFromList([plain, VideoFromFile(make_lossless_video())]).has_pending_frame_edits() is False
    assert VideoFromList([plain, VideoFromFile(make_lossless_video()).as_resized(8, 6, "area")]).has_pending_frame_edits() is True


# --- mixed rotation -----------------------------------------------------


def mixed_rotation_list():
    """Two chunks with the same coded size but different display orientation."""
    paths = [
        create_transcode_source(width=64, height=32, frames=3, rotation=False),
        create_transcode_source(width=64, height=32, frames=3, rotation=True),
    ]
    return VideoFromList([VideoFromFile(path) for path in paths]), paths


def test_list_display_dimensions_reject_mixed_rotation():
    """Coded sizes matching is not enough; the displayed sizes are what gets concatenated"""
    videos, paths = mixed_rotation_list()
    try:
        assert [video.get_display_dimensions() for video in videos.videos] == [(64, 32), (32, 64)]
        with pytest.raises(ValueError, match="incompatible display dimensions"):
            videos.get_display_dimensions()
    finally:
        for path in paths:
            os.unlink(path)


def test_list_as_resized_rejects_mixed_rotation():
    """One target size across chunks would stretch the rotated chunk to a different aspect,
    and the resulting equal sizes would also mask the existing get_components() check."""
    videos, paths = mixed_rotation_list()
    try:
        with pytest.raises(ValueError, match="incompatible display dimensions"):
            videos.as_resized(32, 16, "area")
    finally:
        for path in paths:
            os.unlink(path)


def test_list_with_uniform_rotation_still_resizes():
    """Chunks sharing a display orientation keep working"""
    paths = [create_transcode_source(width=64, height=32, frames=3, rotation=True) for _ in range(2)]
    try:
        videos = VideoFromList([VideoFromFile(path) for path in paths])
        assert videos.get_display_dimensions() == (32, 64)
        resized = videos.as_resized(16, 32, "area")
        assert all(isinstance(child, VideoFromFile) for child in resized.videos)
        assert resized.get_display_dimensions() == (16, 32)
    finally:
        for path in paths:
            os.unlink(path)


# --- concatenated VFR timing --------------------------------------------


def make_vfr_video(ticks=(0, 12, 45)):
    """Frames at unevenly spaced second offsets, written with an explicit 1/15360 time
    base. Evenly spaced ticks would only exercise a sparse constant frame rate."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for tick in ticks:
            frame = av.VideoFrame.from_ndarray(
                torch.full((32, 64, 3), (tick * 4) % 256, dtype=torch.uint8).numpy(), format="rgb24"
            ).reformat(format="yuv420p")
            frame.pts = tick * 15360
            frame.time_base = Fraction(1, 15360)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    buffer.seek(0)
    return buffer


def saved_timing(video):
    output = io.BytesIO()
    video.save_to(output, format=VideoContainer.MP4, codec=VideoCodec.AUTO)
    output.seek(0)
    with av.open(output) as container:
        stream = container.streams.video[0]
        frames = [frame for packet in container.demux(stream) for frame in packet.decode()]
        return {
            "time_base": stream.time_base,
            "pts": [float(frame.pts * frame.time_base) for frame in frames],
            "durations": [float((frame.duration or 0) * frame.time_base) for frame in frames],
        }


def test_list_resized_vfr_keeps_pts_and_durations():
    """Resizing sends a chunk down the shared encode path instead of the packet copy.

    That path must still reproduce the source timing: presentation timestamps alone are
    not enough, the per frame durations carry the variable frame rate.
    """
    remuxed = saved_timing(VideoFromList([VideoFromFile(make_vfr_video())]))
    resized = saved_timing(VideoFromList([VideoFromFile(make_vfr_video()).as_resized(16, 8, "area")]))

    assert resized["time_base"] == remuxed["time_base"]
    assert resized["pts"] == pytest.approx(remuxed["pts"])
    assert resized["durations"] == pytest.approx(remuxed["durations"])


def test_list_resized_vfr_chunks_concatenate_in_order():
    """Two resized VFR chunks keep their own timing and follow each other"""
    single = saved_timing(VideoFromList([VideoFromFile(make_vfr_video())]))
    pair = saved_timing(
        VideoFromList([
            VideoFromFile(make_vfr_video()).as_resized(16, 8, "area"),
            VideoFromFile(make_vfr_video()).as_resized(16, 8, "area"),
        ])
    )
    assert len(pair["pts"]) == 2 * len(single["pts"])
    assert pair["pts"][: len(single["pts"])] == pytest.approx(single["pts"])
    assert pair["durations"] == pytest.approx(single["durations"] * 2)
    assert pair["pts"] == sorted(pair["pts"])


def test_list_mixes_edited_and_unedited_chunks():
    """One edited chunk puts the whole list through the shared encode, so every chunk
    has to be encoded with the same settings; differing ones produce incompatible
    extradata and the save is rejected."""
    def edited():
        return VideoFromFile(make_lossless_video()).as_resized(16, 12, "area").as_resized(32, 24, "area")

    for chunks in ([edited(), VideoFromFile(make_lossless_video())],
                   [VideoFromFile(make_lossless_video()), edited()]):
        output = io.BytesIO()
        VideoFromList(chunks).save_to(output, format=VideoContainer.MP4, codec=VideoCodec.AUTO)
        output.seek(0)
        with av.open(output) as container:
            stream = container.streams.video[0]
            frames = sum(1 for packet in container.demux(stream) for _ in packet.decode())
        assert frames == 6


# --- generic fallback alpha shapes --------------------------------------


@pytest.mark.parametrize("alpha_shape", [(2, 6, 8, 1), (2, 6, 8), (2, 6, 1), (2, 1, 8)])
def test_fallback_keeps_the_alpha_shape_it_was_given(alpha_shape):
    """VideoComponents.alpha is a MaskInput, so [B, H, W] is as valid as a trailing
    channel. Either way it comes back in the shape it went in, degenerate sizes included."""
    components = VideoComponents(
        images=torch.rand(2, 6, 8, 3),
        frame_rate=Fraction(30),
        alpha=torch.rand(*alpha_shape),
    )
    alpha = MinimalVideo(components).as_resized(4, 3, "area").get_components().alpha
    assert alpha.ndim == len(alpha_shape)
    assert alpha.shape[1:3] == (3, 4)
