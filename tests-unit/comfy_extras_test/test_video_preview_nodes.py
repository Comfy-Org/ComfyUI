"""Tests for video preview generation in comfy_extras/nodes_video.py."""
import os

import av
import torch

import folder_paths
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_extras import nodes_video


def _make_video(path, width=64, height=48, frames=3, fps=8):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(
                torch.zeros(height, width, 3, dtype=torch.uint8).numpy(),
                format="rgb24",
            )
            for packet in stream.encode(frame.reformat(format="yuv420p")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return str(path)


def test_save_video_preview_encodes_cropped_video(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    source = _make_video(tmp_path / "src.mp4")
    cropped = VideoFromFile(source).as_cropped(0, 0, 32, 24)

    preview = nodes_video.save_video_preview(cropped)
    entry = preview.as_dict()["images"][0]

    preview_path = os.path.join(str(tmp_path), entry["subfolder"], entry["filename"])
    assert VideoFromFile(preview_path).get_dimensions() == (32, 24)


def test_save_video_preview_reuses_cached_result(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    source = _make_video(tmp_path / "src.mp4")
    cropped = VideoFromFile(source).as_cropped(0, 0, 32, 24)

    first = nodes_video.save_video_preview(cropped).as_dict()["images"][0]
    second = nodes_video.save_video_preview(cropped).as_dict()["images"][0]

    assert second == first
