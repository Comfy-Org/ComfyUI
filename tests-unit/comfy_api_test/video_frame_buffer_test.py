import os
import subprocess
import sys

import av
import numpy as np
import pytest
import torch
from PIL import Image

import comfy.memory_management
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl.video_types import FrameChunks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(autouse=True)
def _aimdo_off_even_if_an_import_enabled_it(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)


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


AIMDO_PREAMBLE = """
import sys
import comfy_aimdo.control
if not comfy_aimdo.control.init():
    sys.exit("comfy-aimdo unavailable")
import comfy.memory_management
comfy.memory_management.aimdo_enabled = True
"""


def _run_with_aimdo(script, *args):
    result = subprocess.run([sys.executable, "-c", AIMDO_PREAMBLE + script, *args], cwd=REPO_ROOT,
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": REPO_ROOT})
    if "comfy-aimdo unavailable" in result.stderr:
        pytest.skip("comfy-aimdo unavailable")
    assert result.returncode == 0, result.stderr


HOST_BUFFER_DECODE = """
import gc
import numpy as np
from comfy_api.input_impl.video_types import VideoFromFile
components = VideoFromFile(sys.argv[1]).get_components()
assert hasattr(components.images.untyped_storage(), "_comfy_hostbuf"), "images not on a host buffer"
assert components.alpha is None or hasattr(components.alpha.untyped_storage(), "_comfy_hostbuf"), "alpha not on a host buffer"
arrays = {"head": components.images[:1].clone().numpy()}
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
        assert np.array_equal(frames["head"], expected.images[:1].numpy())
        assert np.array_equal(frames["tail"], expected.images[1:].numpy())
        if expected.alpha is not None:
            assert np.array_equal(frames["alpha"], expected.alpha.numpy())


REFUSED_RESERVATION_DECODE = """
import resource
import psutil
from comfy_api.latest._input_impl import video_types
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


ONE_REFUSED_RESERVATION_DECODE = """
import torch
from comfy_api.latest._input_impl import video_types
expected = video_types.VideoFromFile(sys.argv[1]).get_components()
reserve, calls = video_types.reserve_host_buffer, []
def reserve_or_refuse():
    calls.append(None)
    if len(calls) == int(sys.argv[2]):
        raise RuntimeError("out of address space")
    return reserve()
video_types.reserve_host_buffer = reserve_or_refuse
components = video_types.VideoFromFile(sys.argv[1]).get_components()
assert len(calls) == 2
assert torch.equal(components.images, expected.images)
assert (components.alpha is None) == (expected.alpha is None)
assert components.alpha is None or torch.equal(components.alpha, expected.alpha)
"""


@pytest.mark.parametrize("mode, extension", [("RGB", ".png"), ("RGBA", ".png"), ("P", ".png"), ("RGB", ".jpg")])
@pytest.mark.parametrize("refused", [1, 2])  # the images' reservation, then the alpha's
def test_8_bit_images_decode_the_same_when_one_reservation_is_refused(tmp_path, mode, extension, refused):
    path = str(tmp_path / ("image" + extension))
    pixels = np.random.default_rng(0).integers(0, 255, (48, 64, 4), dtype=np.uint8)
    Image.fromarray(pixels).convert(mode).save(path)
    _run_with_aimdo(ONE_REFUSED_RESERVATION_DECODE, path, str(refused))


def _decoded_format(path):
    with av.open(path) as container:
        return next(container.decode(video=0)).format.name


HELD_PICTURES_DECODE = """
import resource
import numpy as np
import psutil
from comfy_api.latest._input_impl import video_types
reservation = psutil.virtual_memory().total + psutil.swap_memory().total
address_space = psutil.Process().memory_info().vms + 3 * reservation
resource.setrlimit(resource.RLIMIT_AS, (address_space, resource.getrlimit(resource.RLIMIT_AS)[1]))
held = [video_types.VideoFromFile(sys.argv[1]).get_components() for _ in range(300)]
video = video_types.VideoFromFile(sys.argv[2]).get_components()
assert hasattr(video.images.untyped_storage(), "_comfy_hostbuf"), "the video's images not on a host buffer"
assert hasattr(video.alpha.untyped_storage(), "_comfy_hostbuf"), "the video's alpha not on a host buffer"
np.savez(sys.argv[3], images=held[-1].images.numpy(), alpha=held[-1].alpha.numpy())
"""


@pytest.mark.skipif(sys.platform == "win32", reason="address space limits need the resource module")
def test_held_pictures_leave_address_space_for_a_video(tmp_path):
    picture = str(tmp_path / "image.png")
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (48, 64, 4), dtype=np.uint8)).save(picture)
    video = str(tmp_path / "video.mov")
    _write_rgba_video(video, 4)
    output = str(tmp_path / "picture.npz")
    _run_with_aimdo(HELD_PICTURES_DECODE, picture, video, output)
    expected = VideoFromFile(picture).get_components()
    with np.load(output) as held:
        assert np.array_equal(held["images"], expected.images.numpy())
        assert np.array_equal(held["alpha"], expected.alpha.numpy())


COUNT_RESERVATIONS = """
from comfy_api.latest._input_impl import video_types
reserve, calls = video_types.reserve_host_buffer, []
def reserve_and_count():  # only reservations that succeed, so a refused one fails the count
    hostbuf = reserve()
    calls.append(None)
    return hostbuf
video_types.reserve_host_buffer = reserve_and_count
"""

STILL_PICTURE_DECODE = COUNT_RESERVATIONS + """
import numpy as np
import psutil
import torch
import weakref
append, make_tensor, last_frame, buffers = video_types.HostFrames.append, video_types.HostFrames.tensor, [], []
def append_and_watch(self, frame):
    last_frame[:] = [weakref.ref(frame)]
    append(self, frame)
def tensor_once_the_rest_has_gone(self):
    assert all(frame() is None for frame in last_frame), "the last decoded frame is still held when a tensor is made"
    assert all(hostbuf() is None for hostbuf in buffers), "the images' host buffer is still held while the alpha is made"
    buffers.append(weakref.ref(self.hostbuf))
    return make_tensor(self)
video_types.HostFrames.append, video_types.HostFrames.tensor = append_and_watch, tensor_once_the_rest_has_gone
address_space = psutil.Process().memory_info().vms
components = video_types.VideoFromFile(sys.argv[1]).get_components()
assert len(calls) == 2, calls
assert last_frame and len(buffers) == 1 + (components.alpha is not None), f"the watched append and tensor did not both run: {len(last_frame)} frames, {len(buffers)} tensors"
grown = psutil.Process().memory_info().vms - address_space
assert grown < 2**30, f"a reservation is still held after the decode: the address space grew by {grown} bytes"
arrays = {}
for name, t in (("images", components.images), ("alpha", components.alpha)):
    if t is not None:
        assert t.dtype == torch.float32 and t.is_contiguous(), (name, t.dtype, t.stride())
        assert not hasattr(t.untyped_storage(), "_comfy_hostbuf"), name
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), name  # storage no bigger than the tensor
        arrays[name] = t.numpy()
np.savez(sys.argv[2], **arrays)
"""


def _write_16_bit_png(path, pixels, pix_fmt):
    with av.open(path, "w") as out:
        stream = out.add_stream("png", rate=1)
        stream.width, stream.height, stream.pix_fmt = pixels.shape[1], pixels.shape[0], pix_fmt
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format=pix_fmt.replace("be", "le"))):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.mark.parametrize("name, pix_fmt", [
    ("rgb.png", "rgb24"), ("rgba.png", "rgba"), ("palette.png", "pal8"), ("lossy.webp", "yuv420p"),
    ("rgb48.png", "rgb48be"), ("rgba64.png", "rgba64be"),
])
def test_still_pictures_decode_the_same_with_aimdo(tmp_path, name, pix_fmt):
    path = str(tmp_path / name)
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (48, 64, 4), dtype=np.uint8)
    if name == "rgb48.png":
        _write_16_bit_png(path, rng.integers(0, 65535, (48, 64, 3), dtype=np.uint16), pix_fmt)
    elif name == "rgba64.png":
        _write_16_bit_png(path, rng.integers(0, 65535, (48, 64, 4), dtype=np.uint16), pix_fmt)
    elif name == "lossy.webp":
        Image.fromarray(pixels).convert("RGB").save(path, lossless=False)
    else:
        Image.fromarray(pixels).convert({"rgb.png": "RGB", "rgba.png": "RGBA", "palette.png": "P"}[name]).save(path)
    assert _decoded_format(path) == pix_fmt
    output = str(tmp_path / "picture.npz")
    _run_with_aimdo(STILL_PICTURE_DECODE, path, output)
    expected = VideoFromFile(path).get_components()
    with np.load(output) as still:
        assert still["images"].dtype == np.float32 and np.array_equal(still["images"], expected.images.numpy())
        assert ("alpha" in still) == (expected.alpha is not None)
        if expected.alpha is not None:
            assert still["alpha"].dtype == np.float32 and np.array_equal(still["alpha"], expected.alpha.numpy())


CROPPED_STILL_DECODE = COUNT_RESERVATIONS + """
import tracemalloc
import numpy as np
tracemalloc.start()
images = video_types.VideoFromFile(sys.argv[1], crop=(0, 64, 1024, 2)).get_components().images
assert len(calls) == 2, calls
numpy_memory = tracemalloc.take_snapshot().filter_traces([tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain)])
held = sum(trace.size for trace in numpy_memory.traces)
assert held <= 2 * images.numel() * images.element_size(), f"the whole decoded frame is still alive behind a full-width crop: {held} bytes held for {tuple(images.shape)}"
assert images.untyped_storage().nbytes() == images.numel() * images.element_size(), "storage bigger than the crop"
np.save(sys.argv[2], images.numpy())
"""


def test_a_cropped_float_still_keeps_only_its_own_pixels(tmp_path):
    path = str(tmp_path / "rgb48.png")
    _write_16_bit_png(path, np.random.default_rng(0).integers(0, 65535, (256, 1024, 3), dtype=np.uint16), "rgb48be")
    output = str(tmp_path / "crop.npy")
    _run_with_aimdo(CROPPED_STILL_DECODE, path, output)
    expected = VideoFromFile(path, crop=(0, 64, 1024, 2)).get_components().images
    assert np.array_equal(np.load(output), expected.numpy())


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


def test_frame_chunks_make_8_bit_frames_float_at_the_end():
    frames = [np.full((2, 3, 3), i * 50, dtype=np.uint8) for i in range(6)]
    chunks = FrameChunks()
    for frame in frames:
        chunks.append(frame)
    assert chunks.blocks[0].dtype == np.uint8  # a quarter of the bytes while decoding
    out = chunks.tensor()
    assert out.dtype == torch.float32 and torch.equal(out, torch.from_numpy(np.stack(frames)).float() / 255.0)
