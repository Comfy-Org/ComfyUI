import gc
import os
import subprocess
import sys
import tracemalloc
import weakref

import av
import numpy as np
import pytest
import torch
from PIL import Image

import comfy.memory_management
from comfy_api.input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl import video_types
from comfy_api.latest._input_impl.video_types import FrameChunks, _float_frame

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
        assert np.array_equal(frames["head"], expected.images[:1].numpy()), "the first frame, held until the second arrives, differs from the plain decode"
        assert np.array_equal(frames["tail"], expected.images[1:].numpy())
        if expected.alpha is not None:
            assert np.array_equal(frames["alpha"], expected.alpha.numpy())


REFUSED_RESERVATION_DECODE = """
import resource
import psutil
from comfy_api.latest._input_impl import video_types
address_space = psutil.Process().memory_info().vms + 2**30
resource.setrlimit(resource.RLIMIT_AS, (address_space, resource.getrlimit(resource.RLIMIT_AS)[1]))
components = video_types.VideoFromFile(sys.argv[1]).get_components()
assert not hasattr(components.images.untyped_storage(), "_comfy_hostbuf")
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
assert len(calls) == int(sys.argv[3]), f"{len(calls)} reservations: two frames reserve once for the images and once more only with alpha"
refused = int(sys.argv[2])  # the images reserve first, then the alpha
assert hasattr(components.images.untyped_storage(), "_comfy_hostbuf") == (refused != 1)
assert components.alpha is None or hasattr(components.alpha.untyped_storage(), "_comfy_hostbuf") == (refused != 2)
assert torch.equal(components.images, expected.images)
assert (components.alpha is None) == (expected.alpha is None)
assert components.alpha is None or torch.equal(components.alpha, expected.alpha)
"""


def _decoded_format(path):
    with av.open(path) as container:
        return next(container.decode(video=0)).format.name


@pytest.mark.parametrize("mode, pix_fmt, refused, reservations", [
    ("RGB", "rgb24", 1, 1), ("RGBA", "rgba", 1, 2), ("RGBA", "rgba", 2, 2), ("P", "pal8", 1, 2), ("P", "pal8", 2, 2),
    ("MJPEG", "yuvj420p", 1, 1),
])
def test_8_bit_images_decode_the_same_when_one_reservation_is_refused(tmp_path, mode, pix_fmt, refused, reservations):
    pixels = np.random.default_rng(0).integers(0, 255, (2, 48, 64, 4), dtype=np.uint8)
    if mode == "MJPEG":
        path = str(tmp_path / "video.mkv")
        _write_video(path, [np.ascontiguousarray(p[..., :3]) for p in pixels], codec="mjpeg", pix_fmt="yuvj420p")
    else:
        path = tmp_path / "image.png"
        with open(path, "wb") as f:
            for p in pixels:
                Image.fromarray(p).convert(mode).save(f, "PNG")
        path = str(path)
    assert _decoded_format(path) == pix_fmt
    _run_with_aimdo(ONE_REFUSED_RESERVATION_DECODE, path, str(refused), str(reservations))


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
assert hasattr(video.images.untyped_storage(), "_comfy_hostbuf")
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


def _decode_still_with_aimdo(monkeypatch, path, crop=None):
    calls, store_kinds, real_store = [], [], video_types.frame_store
    def reserve():
        calls.append(None)
        raise AssertionError("a still picture made a reservation")
    def frame_store():
        store = real_store()
        store_kinds.append(type(store))
        return store
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(video_types, "reserve_host_buffer", reserve)
    monkeypatch.setattr(video_types, "frame_store", frame_store)
    components = VideoFromFile(path, crop=crop).get_components()
    assert not calls, "a still picture tried to reserve; counted here because a fallback may catch what the stub raises"
    assert store_kinds and set(store_kinds) == {video_types.HostFrames}, f"the still was not decoded through HostFrames: {store_kinds}"
    return components


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
def test_still_pictures_decode_the_same_with_aimdo(tmp_path, monkeypatch, name, pix_fmt):
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
    expected = VideoFromFile(path).get_components()
    still = _decode_still_with_aimdo(monkeypatch, path)
    for part, t, e in (("images", still.images, expected.images), ("alpha", still.alpha, expected.alpha)):
        assert (t is None) == (e is None), part
        if t is not None:
            assert t.dtype == torch.float32 and t.is_contiguous(), (part, t.dtype, t.stride())
            assert torch.equal(t, e), part


def test_a_cropped_float_still_keeps_only_its_own_pixels(tmp_path, monkeypatch):
    path = str(tmp_path / "rgb48.png")
    _write_16_bit_png(path, np.random.default_rng(0).integers(0, 65535, (256, 1024, 3), dtype=np.uint16), "rgb48be")
    crop = (0, 64, 1024, 2)
    expected = VideoFromFile(path, crop=crop).get_components().images
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        numpy_domain = [tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain)]
        gc.collect()
        before = tracemalloc.take_snapshot().filter_traces(numpy_domain)
        images = _decode_still_with_aimdo(monkeypatch, path, crop).images
        after = tracemalloc.take_snapshot().filter_traces(numpy_domain)
    finally:
        if started:
            tracemalloc.stop()
    held = sum(max(0, stat.size_diff) for stat in after.compare_to(before, "traceback"))
    assert held <= 2 * images.numel() * images.element_size(), f"the whole decoded frame is still alive behind a full-width crop: {held} bytes held for {tuple(images.shape)}"
    assert torch.equal(images, expected)


@pytest.mark.parametrize("make, view", [
    (lambda frame: frame.copy(), lambda a: a[1:3]),
    (lambda frame: np.concatenate([frame, frame[..., :1]], -1), lambda a: a[..., :3]),
], ids=["rows", "rgb-of-rgba"])
def test_float_frame_copies_a_view(make, view):
    frame = np.random.default_rng(0).random((4, 6, 3), dtype=np.float32)
    source = make(frame)
    expected, kept = view(source).copy(), weakref.ref(source)
    result = _float_frame(view(source))
    source[...] = -1
    del source
    assert kept() is None, "the result keeps its source alive"
    assert result.flags.c_contiguous and np.array_equal(result, expected), "the result is not a contiguous copy, or it follows its source"
    assert isinstance(result.base, torch.Tensor), "the new array is not from _host_empty"


def test_float_frame_keeps_an_array_that_owns_its_pixels_and_divides_8_bit():
    frame = np.random.default_rng(0).random((4, 6, 3), dtype=np.float32)
    assert _float_frame(frame) is frame, "a float frame that owns its pixels was copied"
    eight_bit = _float_frame(np.full((2, 2, 3), 51, dtype=np.uint8))
    assert np.array_equal(eight_bit, np.full((2, 2, 3), 0.2, dtype=np.float32)), "an 8-bit frame was not divided by 255"
    assert isinstance(eight_bit.base, torch.Tensor), "the 8-bit frame's new array is not from _host_empty"


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
