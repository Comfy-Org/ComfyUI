import os
import tempfile

import av
import numpy as np
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl.video_types import FrameBuffer


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
