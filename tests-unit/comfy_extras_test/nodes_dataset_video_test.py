"""Tests for the video frame sampling helpers in comfy_extras/nodes_dataset.py."""

import math

import av
import numpy as np
import pytest
import torch

from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy_extras.nodes_dataset import _decode_selected_frames

FPS = 24
FRAMES = 24
WIDTH = 64
HEIGHT = 48


def _mean(images: torch.Tensor, index: int) -> float:
    """Mean pixel value of one returned frame, in 0..255 terms."""
    return float(images[index].mean().item()) * 255.0


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    """A flat-gray test clip where every frame's brightness encodes its index (idx * 10)."""
    path = tmp_path_factory.mktemp("video") / "counting.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        for idx in range(FRAMES):
            frame = av.VideoFrame(WIDTH, HEIGHT, "gray")
            frame.planes[0].update(np.full((HEIGHT, WIDTH), idx * 10, dtype=np.uint8))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def _decode(path, indices, trim=None):
    video = VideoFromFile(path)
    if trim is not None:
        start, duration = trim
        video = video.as_trimmed(start, duration, strict_duration=False)
    out = _decode_selected_frames(video, indices)
    return out.get_components().images


class TestDecodeSelectedFrames:
    def test_untrimmed_samples_requested_indices(self, video_path):
        images = _decode(video_path, [0, 5, 11])
        assert images.shape == (3, HEIGHT, WIDTH, 3)
        # yuv420p gray round-trip loses a little brightness; keep a loose tolerance.
        assert math.isclose(_mean(images, 0), 0, abs_tol=12)
        assert math.isclose(_mean(images, 1), 50, abs_tol=12)
        assert math.isclose(_mean(images, 2), 110, abs_tol=12)

    def test_trim_window_selects_from_window(self, video_path):
        # Window: frames 6..17 (start 6/FPS, duration 12/FPS) — 12 frames,
        # window-relative index 5 is source frame 11.
        images = _decode(video_path, [0, 5], trim=(6 / FPS, 12 / FPS))
        assert math.isclose(_mean(images, 0), 60, abs_tol=12)
        assert math.isclose(_mean(images, 1), 110, abs_tol=12)

    def test_missing_index_clamps_to_last_window_frame(self, video_path):
        # Requesting an index beyond the window must return the window's last frame,
        # not the first requested one.
        images = _decode(video_path, [0, 100], trim=(6 / FPS, 12 / FPS))
        assert math.isclose(_mean(images, 1), 170, abs_tol=12)
        assert not math.isclose(_mean(images, 1), 60, abs_tol=12)

    def test_no_frames_decoded_raises(self, video_path):
        # A window that lies entirely beyond the clip decodes nothing.
        with pytest.raises(ValueError):
            _decode(video_path, [0], trim=(100.0, 1.0))
