import os
import subprocess
import sys

import av
import numpy as np
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl.video_types import FrameChunks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _write_video(path, frames, codec=None, pix_fmt="yuv420p", img_format="rgb24", options=None):
    with av.open(path, "w", options=options or {}) as out:
        stream = out.add_stream(codec or ("libx264" if path.endswith(".mp4") else "mpeg4"), rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, pix_fmt
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format=img_format)
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _write_rgba_video(path, count):
    frames = [np.full((48, 64, 4), i * 16, dtype=np.uint8) for i in range(count)]
    _write_video(path, frames, codec="png", pix_fmt="rgba", img_format="rgba")


def _noise_frames(count):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(count)]


def _truncated_mp4(tmp_path):
    frames = [np.full((48, 64, 3), i % 256, dtype=np.uint8) for i in range(300)]
    full = tmp_path / "full.mp4"
    _write_video(str(full), frames, codec="mpeg4", options={"movflags": "faststart"})
    data = full.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 5])
    return str(cut)


@pytest.mark.parametrize("extension", [".mp4", ".mkv"])
def test_get_components_frames_match_a_plain_decode(tmp_path, extension):
    path = str(tmp_path / ("video" + extension))
    _write_video(path, _noise_frames(30))
    images = VideoFromFile(path).get_components().images
    with av.open(path) as container:
        expected = np.stack([f.to_ndarray(format="gbrpf32le") for f in container.decode(video=0)])
    assert images.shape == (30, 48, 64, 3)
    assert np.array_equal(images.numpy(), expected)


def test_get_components_trim_window_count(tmp_path):
    path = str(tmp_path / "video.mp4")
    _write_video(path, [np.full((48, 64, 3), i * 8, dtype=np.uint8) for i in range(30)])
    images = VideoFromFile(path).as_trimmed(start_time=0.5, duration=0.5).get_components().images
    assert images.shape[0] == 12


def test_rgba_frames_split_into_images_and_alpha(tmp_path):
    path = str(tmp_path / "video.mov")
    _write_rgba_video(path, 4)
    components = VideoFromFile(path).get_components()
    expected = torch.arange(4, dtype=torch.float32) * 16 / 255
    assert components.images.shape == (4, 48, 64, 3)
    assert components.alpha.shape == (4, 48, 64, 1)
    assert components.images.dtype == components.alpha.dtype == torch.float32
    assert torch.equal(components.images[:, 0, 0, 0], expected)
    assert torch.equal(components.alpha[:, 0, 0, 0], expected)


def test_a_truncated_faststart_mp4_decodes_what_it_has(tmp_path):
    cut = _truncated_mp4(tmp_path)
    with av.open(cut) as container:
        assert container.streams.video[0].frames == 300
    assert 0 < VideoFromFile(cut).get_components().images.shape[0] < 300


def test_a_soundtrack_longer_than_the_video_adds_no_frames(tmp_path):
    path = str(tmp_path / "long-audio.mkv")
    with av.open(path, "w") as out:
        video = out.add_stream("mpeg4", rate=24)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("aac", rate=48000)
        audio.layout = "stereo"
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5, np.uint8), format="rgb24")
            frame.pts = i
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        for offset in range(0, 48000 * 10, 960):
            frame = av.AudioFrame.from_ndarray(np.zeros((2, 960), np.float32), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = 48000, offset
            for packet in audio.encode(frame):
                out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)
    assert VideoFromFile(path).get_components().images.shape[0] == 48


def _run_with_aimdo(script, *args):
    result = subprocess.run([sys.executable, "-c", script, *args], cwd=REPO_ROOT,
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": REPO_ROOT})
    if "comfy-aimdo unavailable" in result.stderr:
        pytest.skip("comfy-aimdo unavailable")
    assert result.returncode == 0, result.stderr


HOST_BUFFER_DECODE = """
import sys
import comfy_aimdo.control
if not comfy_aimdo.control.init():
    sys.exit("comfy-aimdo unavailable")
import gc
import numpy as np
import comfy.memory_management
from comfy_api.input_impl.video_types import VideoFromFile
comfy.memory_management.aimdo_enabled = True
components = VideoFromFile(sys.argv[1]).get_components()
arrays = {}
if components.alpha is not None:
    arrays["alpha"] = components.alpha.clone().numpy()
tail = components.images[1:]
del components
gc.collect()
arrays["tail"] = tail.clone().numpy()
del tail
gc.collect()
np.savez(sys.argv[2], **arrays)
"""


@pytest.mark.parametrize("name", ["video.mp4", "video.mov", "cut.mp4"])
def test_host_buffer_and_fromiter_decode_the_same_frames(tmp_path, name):
    path = str(tmp_path / name)
    if name == "video.mov":
        _write_rgba_video(path, 4)
    elif name == "cut.mp4":
        path = _truncated_mp4(tmp_path)
    else:
        _write_video(path, _noise_frames(10))
    output = str(tmp_path / "frames.npz")
    _run_with_aimdo(HOST_BUFFER_DECODE, path, output)
    expected = VideoFromFile(path).get_components()
    with np.load(output) as frames:
        assert np.array_equal(frames["tail"], expected.images[1:].numpy())
        if expected.alpha is not None:
            assert np.array_equal(frames["alpha"], expected.alpha.numpy())


REFUSED_RESERVATION_DECODE = """
import resource
import sys
import comfy_aimdo.control
if not comfy_aimdo.control.init():
    sys.exit("comfy-aimdo unavailable")
import psutil
import comfy.memory_management
from comfy_api.latest._input_impl import video_types
comfy.memory_management.aimdo_enabled = True
address_space = psutil.Process().memory_info().vms + 2**30
resource.setrlimit(resource.RLIMIT_AS, (address_space, resource.getrlimit(resource.RLIMIT_AS)[1]))
assert video_types.host_frames() is None
components = video_types.VideoFromFile(sys.argv[1]).get_components()
assert components.images.shape == (4, 48, 64, 3), components.images.shape
assert components.alpha.shape == (4, 48, 64, 1), components.alpha.shape
"""


@pytest.mark.skipif(sys.platform == "win32", reason="address space limits need the resource module")
def test_a_refused_reservation_falls_back_to_plain_arrays(tmp_path):
    path = str(tmp_path / "video.mov")
    _write_rgba_video(path, 4)
    _run_with_aimdo(REFUSED_RESERVATION_DECODE, path)


# ---- without aimdo: blocks of frames, copied into one tensor at the end ---------------------------------------


def _frames(n, shape=(2, 3, 3)):
    return [np.full(shape, float(i), dtype=np.float32) for i in range(n)]


@pytest.mark.parametrize("n", [1, FrameChunks.CHUNK, 2 * FrameChunks.CHUNK + 5])
def test_frame_chunks_give_the_frames_in_order(n):
    frames = _frames(n)
    chunks = FrameChunks()
    for frame in frames:
        chunks.append(frame)
    assert len(chunks) == n
    out = chunks.tensor()
    assert out.dtype == torch.float32 and torch.equal(out, torch.from_numpy(np.stack(frames)))
    assert chunks.blocks == []  # each block let go once copied


def test_frame_chunks_cap_a_block_by_bytes(monkeypatch):
    frames = _frames(7, shape=(4, 4, 3))  # 192 bytes a frame
    monkeypatch.setattr(FrameChunks, "BLOCK_BYTES", 2 * frames[0].nbytes)
    chunks = FrameChunks()
    for frame in frames:
        chunks.append(frame)
    assert chunks.rows == 2 and len(chunks.blocks) == 4  # one frame never reserves CHUNK frames' worth
    assert torch.equal(chunks.tensor(), torch.from_numpy(np.stack(frames)))
