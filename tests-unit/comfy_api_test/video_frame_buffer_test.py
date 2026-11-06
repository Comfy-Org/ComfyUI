import os
import tempfile
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl import video_types
from comfy_api.latest._input_impl.video_types import FrameBuffer, _stream_seconds, expected_frame_count


def test_frame_buffer_grows_when_the_guess_is_short():
    frames = [torch.full((2, 3, 3), float(i)) for i in range(7)]
    buffer = FrameBuffer(2)
    for frame in frames:
        buffer.append(frame)
    assert len(buffer) == 7
    assert torch.equal(buffer.tensor(), torch.stack(frames))


def test_frame_buffer_drops_unused_room_when_the_guess_is_long():
    buffer = FrameBuffer(100)
    for i in range(5):
        buffer.append(torch.full((4, 4, 3), float(i)))
    result = buffer.tensor()
    assert result.shape == (5, 4, 4, 3)
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def test_frame_buffer_keeps_a_close_guess_without_copying():
    buffer = FrameBuffer(10)
    for i in range(10):
        buffer.append(torch.full((2, 2), float(i)))
    assert buffer.tensor().data_ptr() == buffer.data.data_ptr()


def _write_video(path, frames, container_options=None):
    with av.open(path, "w", options=container_options or {}) as out:
        stream = out.add_stream("libx264" if path.endswith(".mp4") else "mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _reference(path):
    with av.open(path) as container:
        return np.stack([f.to_ndarray(format="gbrpf32le") for f in container.decode(video=0)])


@pytest.mark.parametrize("extension", [".mp4", ".mkv"])  # mkv streams carry no frame count
def test_get_components_frames_match_a_plain_decode(extension):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(30)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "video" + extension)
        _write_video(path, frames)
        images = VideoFromFile(path).get_components().images
        expected = _reference(path)
    assert images.shape == (30, 48, 64, 3)
    assert np.array_equal(images.numpy(), expected)


def test_get_components_trim_window_count():
    frames = [np.full((48, 64, 3), i * 8, dtype=np.uint8) for i in range(30)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "video.mp4")
        _write_video(path, frames)
        video = VideoFromFile(path).as_trimmed(start_time=0.5, duration=0.5)
        images = video.get_components().images
    assert images.shape[0] == 12


def test_frame_buffer_short_guess_never_holds_more_than_the_frames_twice():
    """Frames past the guess wait in a list: at the end the buffer, the overflow and the result coexist once."""
    frames = [torch.full((2, 3, 3), float(i)) for i in range(10)]
    buffer = FrameBuffer(4)
    for frame in frames:
        buffer.append(frame)
    assert buffer.data.shape[0] == 4 and len(buffer.overflow) == 6
    assert torch.equal(buffer.tensor(), torch.stack(frames))


def test_expected_frame_count_ignores_a_longer_soundtrack(tmp_path):
    """Matroska and WebM video streams carry no frame count or duration; the container's duration is the longest
    stream's, so a soundtrack longer than the video must not size the frame buffer."""
    for suffix, video_codec, audio_codec in [(".mkv", "mpeg4", "aac"), (".webm", "libvpx-vp9", "libopus")]:
        path = str(tmp_path / f"long-audio{suffix}")
        with av.open(path, "w") as out:
            v = out.add_stream(video_codec, rate=24)
            v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
            a = out.add_stream(audio_codec, rate=48000)
            a.layout = "stereo"
            for i in range(48):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5, np.uint8), format="rgb24")
                frame.pts = i
                for p in v.encode(frame):
                    out.mux(p)
            for p in v.encode():
                out.mux(p)
            for off in range(0, 48000 * 10, 960):
                af = av.AudioFrame.from_ndarray(np.zeros((2, 960), np.float32), format="fltp", layout="stereo")
                af.sample_rate, af.pts = 48000, off
                for p in a.encode(af):
                    out.mux(p)
            for p in a.encode():
                out.mux(p)
        with av.open(path) as container:
            guess = expected_frame_count(container, container.streams.video[0], 0.0, 0.0)
        assert 40 <= guess <= 60, (suffix, guess)
        assert VideoFromFile(path).get_components().images.shape[0] == 48


def test_stream_seconds_uses_the_container_duration_only_for_a_lone_stream():
    video = SimpleNamespace(duration=None, time_base=None, metadata={})
    ten_minutes = 600 * av.time_base
    assert _stream_seconds(SimpleNamespace(streams=[video, object()], duration=ten_minutes), video) == 0.0
    assert _stream_seconds(SimpleNamespace(streams=[video], duration=ten_minutes), video) == 600.0


def test_overstated_metadata_falls_back_to_a_list(monkeypatch):
    """A header claiming more frames than free memory holds: the frames are collected in a list and stacked, as
    before the buffer, instead of one huge allocation."""
    frames = [torch.full((4, 4, 3), float(i)) for i in range(5)]
    frame_bytes = frames[0].numel() * frames[0].element_size()
    monkeypatch.setattr(video_types.psutil, "virtual_memory", lambda: SimpleNamespace(available=frame_bytes * 3))
    buffer = FrameBuffer(10**12)
    for f in frames:
        buffer.append(f)
    assert buffer.data is None and buffer.listed is not None
    assert torch.equal(buffer.tensor(), torch.stack(frames))


def test_a_refused_allocation_falls_back_to_a_list(monkeypatch):
    frames = [torch.full((4, 4, 3), float(i)) for i in range(4)]

    def refuse(*args, **kwargs):
        raise RuntimeError("DefaultCPUAllocator: can't allocate")

    monkeypatch.setattr(video_types.torch, "empty", refuse)
    buffer = FrameBuffer(4)
    for f in frames:
        buffer.append(f)
    monkeypatch.undo()
    assert buffer.data is None and torch.equal(buffer.tensor(), torch.stack(frames))


def test_a_truncated_faststart_mp4_decodes_what_it_has(tmp_path, monkeypatch):
    """A faststart MP4 cut short keeps a header that counts every frame, but only the start decodes. Here free
    memory is set too small for the header's count, so the frames go through the list; they are the frames PyAV
    decodes."""
    full = str(tmp_path / "full.mp4")
    with av.open(full, "w", options={"movflags": "faststart"}) as out:
        v = out.add_stream("mpeg4", rate=24)
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        for i in range(300):
            img = np.full((48, 64, 3), i % 256, dtype=np.uint8)
            for packet in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in v.encode():
            out.mux(packet)
    cut = str(tmp_path / "cut.mp4")
    data = open(full, "rb").read()
    open(cut, "wb").write(data[: len(data) // 5])
    with av.open(cut) as c:
        assert c.streams.video[0].frames == 300  # the header still claims the whole clip
        want = []
        try:
            for f in c.decode(video=0):
                want.append(f.to_ndarray(format="rgb24"))
        except av.error.InvalidDataError:
            pass
    frame_bytes = 48 * 64 * 3 * 4
    monkeypatch.setattr(video_types.psutil, "virtual_memory", lambda: SimpleNamespace(available=frame_bytes * 100))
    images = VideoFromFile(cut).get_components().images
    assert 0 < images.shape[0] == len(want) < 300
    got = (images * 255).round().to(torch.int16)
    # core converts YUV to RGB more carefully than PyAV's rgb24 shortcut, so they agree to within a level or two
    assert (got - torch.from_numpy(np.stack(want)).to(torch.int16)).abs().max() <= 2


def test_expected_frame_count_follows_the_trim(tmp_path):
    from comfy_api.latest._input_impl.video_types import expected_frame_count
    path = str(tmp_path / "t.mp4")
    _write_video(path, [np.full((48, 64, 3), i, np.uint8) for i in range(48)])
    with av.open(path) as container:
        stream = container.streams.video[0]
        assert expected_frame_count(container, stream, 0.0, 0.0) == 48
        assert expected_frame_count(container, stream, 0.5, 0.0) == 36
        assert expected_frame_count(container, stream, 0.0, 0.5) == 13
