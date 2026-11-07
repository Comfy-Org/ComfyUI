import os
import subprocess
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl import video_types
from comfy_api.latest._input_impl.video_types import FrameBuffer, _stream_seconds, expected_frame_count

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


# ---- without aimdo: FrameBuffer, sized from the expected frame count ------------------------------------------

def _frames(n, shape=(2, 3, 3)):
    return [np.full(shape, float(i), dtype=np.float32) for i in range(n)]


def _stacked(frames):
    return torch.from_numpy(np.stack(frames))


def _filled(buffer, frames):
    for frame in frames:
        buffer.append(frame)
    return buffer


def test_frame_buffer_grows_when_the_guess_is_short():
    frames = _frames(7)
    buffer = _filled(FrameBuffer(2), frames)
    assert len(buffer) == 7
    assert torch.equal(buffer.tensor(), _stacked(frames))


def test_frame_buffer_short_guess_never_holds_more_than_the_frames_twice():
    """Frames past the guess wait in a list: at the end the buffer, the overflow and the result coexist once."""
    frames = _frames(10)
    buffer = _filled(FrameBuffer(4), frames)
    assert buffer.data.shape[0] == 4 and len(buffer.overflow) == 6
    assert torch.equal(buffer.tensor(), _stacked(frames))


def test_frame_buffer_drops_unused_room_when_the_guess_is_long():
    result = _filled(FrameBuffer(100), _frames(5, (4, 4, 3))).tensor()
    assert result.shape == (5, 4, 4, 3)
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def test_frame_buffer_keeps_a_close_guess_without_copying():
    buffer = _filled(FrameBuffer(10), _frames(10, (2, 2)))
    assert buffer.tensor().data_ptr() == buffer.data.data_ptr()


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


def test_stream_seconds_uses_the_container_duration_only_for_a_lone_stream():
    video = SimpleNamespace(duration=None, time_base=None, metadata={})
    ten_minutes = 600 * av.time_base
    assert _stream_seconds(SimpleNamespace(streams=[video, object()], duration=ten_minutes), video) == 0.0
    assert _stream_seconds(SimpleNamespace(streams=[video], duration=ten_minutes), video) == 600.0


def test_overstated_metadata_falls_back_to_a_list(monkeypatch):
    """A header claiming more frames than free memory holds: the frames are collected in a list and stacked, as
    before the buffer, instead of one huge allocation."""
    frames = _frames(5, (4, 4, 3))
    monkeypatch.setattr(video_types, "_available_memory", lambda: frames[0].nbytes * 3)
    buffer = _filled(FrameBuffer(10**12), frames)
    assert buffer.data is None and buffer.listed is not None
    assert torch.equal(buffer.tensor(), _stacked(frames))


def test_a_refused_allocation_falls_back_to_a_list(monkeypatch):
    frames = _frames(4, (4, 4, 3))

    def refuse(*args, **kwargs):
        raise RuntimeError("DefaultCPUAllocator: can't allocate")

    monkeypatch.setattr(video_types.torch, "empty", refuse)
    buffer = _filled(FrameBuffer(4), frames)
    monkeypatch.undo()
    assert buffer.data is None and torch.equal(buffer.tensor(), _stacked(frames))


def test_a_refused_trim_copy_keeps_the_view(monkeypatch):
    """A guess far over the count normally ends in a trimmed copy; if memory refuses that copy, the frames come back
    as a view of the buffer instead of failing."""
    frames = _frames(3, (4, 4, 3))
    buffer = _filled(FrameBuffer(100), frames)

    def refuse(self):
        raise RuntimeError("DefaultCPUAllocator: can't allocate")

    monkeypatch.setattr(torch.Tensor, "clone", refuse)
    out = buffer.tensor()
    monkeypatch.undo()
    assert torch.equal(out, _stacked(frames)) and out.shape[0] == 3


def test_a_big_claim_is_recounted_from_the_file(monkeypatch):
    """When the header's count would take over a quarter of free memory, the buffer asks for a count from the file
    and sizes itself from that; a modest claim never pays for the extra read."""
    frames = _frames(5, (4, 4, 3))
    monkeypatch.setattr(video_types, "_available_memory", lambda: frames[0].nbytes * 100)
    asked = []
    big = _filled(FrameBuffer(10**6, recount=lambda: asked.append(1) or 5), frames)
    assert asked == [1] and big.data is not None and big.data.shape[0] == 6
    assert torch.equal(big.tensor(), _stacked(frames))
    small = FrameBuffer(5, recount=lambda: asked.append(2) or 5)
    small.append(frames[0])
    assert asked == [1]


def test_without_aimdo_a_truncated_faststart_mp4_with_little_free_memory_decodes_what_it_has(tmp_path, monkeypatch):
    """A faststart MP4 cut short keeps a header that counts every frame, but only the start decodes. With free memory
    too small for the header's count, the frames go through the list; they are the frames PyAV decodes."""
    full = tmp_path / "full.mp4"
    with av.open(str(full), "w", options={"movflags": "faststart"}) as out:
        v = out.add_stream("mpeg4", rate=24)
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        for i in range(300):
            img = np.full((48, 64, 3), i % 256, dtype=np.uint8)
            for packet in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in v.encode():
            out.mux(packet)
    data = full.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 5])
    monkeypatch.setattr(video_types, "_available_memory", lambda: 48 * 64 * 3 * 4 * 100)
    images = VideoFromFile(str(cut)).get_components().images
    counted = VideoFromFile(str(cut))._count_packets(0.0, 0.0)
    assert 0 < images.shape[0] < 300 and abs(counted - images.shape[0]) <= 1
    assert VideoFromFile(str(full))._count_packets(0.0, 0.0) == 300
