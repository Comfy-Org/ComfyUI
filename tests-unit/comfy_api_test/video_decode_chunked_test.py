import io
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.latest._util.video_types import VideoChunk, VideoComponents


def write_video(path, codec="libx264", pix_fmt="yuv420p", width=64, height=48, frames=20, fps=24,
                audio=True, container_format=None, rotation=False):
    """A video whose frames all differ (a moving gradient), with a sine tone when audio."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format or {"mkv": "matroska"}.get(ext := path.rsplit(".", 1)[-1], ext)) as out:
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        sound = None
        if audio:
            sound = out.add_stream("pcm_s16le" if container_format == "mov" else "aac", rate=48000)
            sound.layout = "stereo"
        has_alpha = pix_fmt in ("rgba", "yuva420p")
        y, x = np.mgrid[0:height, 0:width]
        for i in range(frames):
            rgb = np.stack([(x * 4 + i * 9) % 256, (y * 5 + i * 3) % 256, (x + y + i * 17) % 256], axis=-1)
            if has_alpha:
                rgb = np.concatenate([rgb, ((x * 3 + i * 11) % 256)[..., None]], axis=-1)
            frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgba" if has_alpha else "rgb24")
            frame.pts = i
            for packet in stream.encode(frame.reformat(format=pix_fmt)):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        if sound is not None:
            total = 48000 * frames // fps
            for offset in range(0, total, 1024):
                n = min(1024, total - offset)
                t = (np.arange(offset, offset + n) / 48000.0).astype(np.float32)
                tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
                af = av.AudioFrame.from_ndarray(np.stack([tone, tone // 2], axis=1).reshape(1, -1), format="s16", layout="stereo")
                af.sample_rate = 48000
                af.pts = offset
                for packet in sound.encode(af):
                    out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)
    data = bytearray(buffer.getvalue())
    if rotation:
        matrix_offset = data.index(b"tkhd", data.rindex(b"moov")) + 4 + 40
        values = [0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30]
        data[matrix_offset:matrix_offset + 36] = b"".join(v.to_bytes(4, "big", signed=True) for v in values)
    with open(path, "wb") as f:
        f.write(bytes(data))
    return path


FIXTURES = {
    # name: (filename, write_video kwargs). Between them they take every branch of the decoder.
    "h264": ("a.mp4", {}),
    "h264-odd-width": ("b.mp4", {"width": 70, "height": 46}),
    "full-range": ("c.mkv", {"codec": "mjpeg", "pix_fmt": "yuvj420p"}),
    "png-alpha": ("d.mov", {"codec": "png", "pix_fmt": "rgba", "container_format": "mov"}),
    "ffv1-alpha": ("e.mkv", {"codec": "ffv1", "pix_fmt": "yuva420p"}),
    "ffv1-10bit": ("f.mkv", {"codec": "ffv1", "pix_fmt": "yuv420p10le"}),
    "rotated": ("g.mov", {"codec": "mpeg4", "container_format": "mov", "rotation": True}),
    "no-audio": ("h.mp4", {"audio": False}),
}

VARIANTS = {
    "whole": {},
    "trimmed": {"start_time": 0.3, "duration": 0.4},
    "negative-start": {"start_time": -0.5},
    "cropped": {"crop": (6, 4, 20, 16)},
}


@pytest.fixture(scope="module")
def fixture_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("videos")
    return {name: write_video(str(root / filename), **kwargs) for name, (filename, kwargs) in FIXTURES.items()}


def joined(chunks):
    chunks = list(chunks)
    images = torch.cat([c.images for c in chunks]) if chunks else torch.zeros(0, 0, 0, 3)
    alphas = [c.alpha for c in chunks]
    alpha = torch.cat(alphas) if chunks and all(a is not None for a in alphas) else None
    return chunks, images, alpha


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", FIXTURES)
def test_decode_chunked_matches_get_components(fixture_files, name, variant):
    video = VideoFromFile(fixture_files[name], **VARIANTS[variant])
    components = video.get_components()
    chunks, images, alpha = joined(video.decode_chunked(chunk_frames=3))
    assert images.dtype == torch.float32
    assert torch.equal(images, components.images)
    assert (alpha is None) == (components.alpha is None)
    if alpha is not None:
        assert torch.equal(alpha, components.alpha)
    assert [c.start_frame for c in chunks] == list(range(0, components.images.shape[0], 3))
    assert all(c.images.shape[0] == 3 for c in chunks[:-1])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", FIXTURES)
def test_get_audio_matches_get_components(fixture_files, name, variant):
    video = VideoFromFile(fixture_files[name], **VARIANTS[variant])
    expected = video.get_components().audio
    audio = video.get_audio()
    assert (audio is None) == (expected is None)
    if audio is not None:
        assert audio["sample_rate"] == expected["sample_rate"]
        assert torch.equal(audio["waveform"], expected["waveform"])


@pytest.mark.parametrize("name", ["h264", "full-range", "png-alpha", "ffv1-alpha", "ffv1-10bit"])
def test_decode_chunked_uint8_rounds_the_float_frames(fixture_files, name):
    video = VideoFromFile(fixture_files[name])
    components = video.get_components()
    _, images, alpha = joined(video.decode_chunked(chunk_frames=4, dtype=torch.uint8))
    assert images.dtype == torch.uint8
    assert torch.equal(images, (components.images * 255).round().clamp(0, 255).to(torch.uint8))
    if components.alpha is not None:
        assert torch.equal(alpha, (components.alpha * 255).round().clamp(0, 255).to(torch.uint8))


def test_decode_chunked_from_bytesio_interleaves(fixture_files):
    with open(fixture_files["h264"], "rb") as f:
        video = VideoFromFile(io.BytesIO(f.read()))
    first, second = video.decode_chunked(chunk_frames=5), video.decode_chunked(chunk_frames=5)
    a1, b1, a2 = next(first), next(second), next(first)
    assert torch.equal(a1.images, b1.images)
    assert a2.start_frame == 5
    assert torch.equal(torch.cat([a1.images, a2.images]), video.get_components().images[:10])


@pytest.mark.parametrize("name", ["h264", "png-alpha"])
def test_decode_chunked_holds_one_chunk(fixture_files, name):
    """The generator keeps nothing of a chunk once it is handed over: dropping the chunk frees its storage."""
    import gc
    import weakref
    refs = []
    for chunk in VideoFromFile(fixture_files[name]).decode_chunked(chunk_frames=2):
        owner = chunk.images._base if chunk.images._base is not None else chunk.images
        refs.append(weakref.ref(owner))
        del chunk, owner
        gc.collect()
        assert all(r() is None for r in refs)


def test_decode_chunked_rejects_bad_arguments(fixture_files):
    video = VideoFromFile(fixture_files["h264"])
    with pytest.raises(ValueError):
        next(video.decode_chunked(chunk_frames=0))
    with pytest.raises(ValueError):
        next(video.decode_chunked(dtype=torch.float16))


def test_video_from_components_chunks_are_views():
    images = torch.rand(7, 4, 6, 3)
    alpha = torch.rand(7, 4, 6, 1)
    video = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(24), alpha=alpha))
    chunks = list(video.decode_chunked(chunk_frames=3))
    assert [c.start_frame for c in chunks] == [0, 3, 6]
    assert all(c.images.data_ptr() == images[c.start_frame].data_ptr() for c in chunks)
    assert torch.equal(torch.cat([c.alpha for c in chunks]), alpha)
    _, as_uint8, _ = joined(video.decode_chunked(chunk_frames=3, dtype=torch.uint8))
    assert torch.equal(as_uint8, (images * 255).round().clamp(0, 255).to(torch.uint8))
    assert video.get_audio() is None
    audio = {"waveform": torch.rand(1, 2, 100), "sample_rate": 8000}
    with_audio = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(24), audio=audio))
    assert with_audio.get_audio() is audio


def test_rotated_fixture_decodes_rotated(fixture_files):
    assert VideoFromFile(fixture_files["rotated"]).get_components().images.shape[1:3] == (64, 48)


def test_get_backing_file_only_for_the_whole_file(fixture_files):
    path = fixture_files["h264"]
    assert VideoFromFile(path).get_backing_file() == path
    assert VideoFromFile(path, start_time=0.2).get_backing_file() is None
    assert VideoFromFile(path, duration=0.5).get_backing_file() is None
    assert VideoFromFile(path).as_trimmed(0.1, 0.5).get_backing_file() is None
    assert VideoFromFile(path).as_cropped(0, 0, 16, 16).get_backing_file() is None
    with open(path, "rb") as f:
        assert VideoFromFile(io.BytesIO(f.read())).get_backing_file() is None
    assert VideoFromList([VideoFromFile(path)]).get_backing_file() is None
    assert VideoFromComponents(VideoComponents(images=torch.rand(2, 4, 4, 3), frame_rate=Fraction(24))).get_backing_file() is None


def test_video_from_list_chunks_and_audio_match_get_components(fixture_files):
    parts = [VideoFromFile(fixture_files["h264"]), VideoFromFile(fixture_files["h264"], start_time=0.25)]
    video = VideoFromList(parts)
    components = video.get_components()
    chunks, images, _ = joined(video.decode_chunked(chunk_frames=7))
    assert torch.equal(images, components.images)
    assert [c.start_frame for c in chunks] == [0, 7, 14, 20, 27]
    assert [c.images.shape[0] for c in chunks] == [7, 7, 6, 7, 7]
    assert all(isinstance(c, VideoChunk) for c in chunks)
    audio = video.get_audio()
    assert audio["sample_rate"] == components.audio["sample_rate"]
    assert torch.equal(audio["waveform"], components.audio["waveform"])


def test_video_from_list_audio_needs_every_part(fixture_files):
    video = VideoFromList([VideoFromFile(fixture_files["h264"]), VideoFromFile(fixture_files["no-audio"])])
    with pytest.raises(ValueError, match="missing audio"):
        video.get_audio()


def test_video_from_list_rejects_mixed_frame_rates(fixture_files, tmp_path):
    other = write_video(str(tmp_path / "30.mp4"), fps=30)
    video = VideoFromList([VideoFromFile(fixture_files["h264"]), VideoFromFile(other)])
    with pytest.raises(ValueError, match="frame rates"):
        next(video.decode_chunked())
    with pytest.raises(ValueError, match="frame rates"):
        video.get_audio()


def test_get_audio_skips_decoding_frames(fixture_files, monkeypatch):
    """get_audio never turns a video frame into pixels (each decoded frame's rotation is read on the way)."""
    from comfy_api.latest._input_impl import video_types
    calls = []
    real = video_types._rotation_quadrant
    monkeypatch.setattr(video_types, "_rotation_quadrant", lambda frame: calls.append(1) or real(frame))
    VideoFromFile(fixture_files["h264"]).get_components()
    assert calls
    calls.clear()
    assert VideoFromFile(fixture_files["h264"]).get_audio() is not None
    assert calls == []


def test_video_from_list_cuts_complete_audio_to_the_frames(fixture_files):
    """A Concatenate Video soundtrack longer than its parts is cut where the frames end, as get_components cuts it."""
    long_audio = {"waveform": torch.rand(1, 2, 48000 * 10), "sample_rate": 48000}
    parts = [VideoFromFile(fixture_files["h264"]), VideoFromFile(fixture_files["h264"], start_time=0.25)]
    video = VideoFromList(parts, complete_audio=long_audio)
    expected = video.get_components().audio
    assert expected["waveform"].shape[-1] < long_audio["waveform"].shape[-1]
    assert torch.equal(video.get_audio()["waveform"], expected["waveform"])


def test_video_from_list_audio_cut_matches_a_partial_frame_trim(fixture_files):
    """A trim that ends between frames keeps the frame it cuts into; the soundtrack is cut after it, as in
    get_components."""
    long_audio = {"waveform": torch.rand(1, 2, 48000 * 10), "sample_rate": 48000}
    video = VideoFromList([VideoFromFile(fixture_files["h264"], start_time=0.0, duration=0.1)], complete_audio=long_audio)
    expected = video.get_components()
    assert expected.images.shape[0] == 3
    assert torch.equal(video.get_audio()["waveform"], expected.audio["waveform"])


@pytest.mark.parametrize("name", ["full-range", "ffv1-alpha"])
def test_video_from_list_audio_count_without_frame_metadata(fixture_files, name):
    """Matroska parts carry no frame count; the cut still follows the decoded frames."""
    long_audio = {"waveform": torch.rand(1, 2, 48000 * 10), "sample_rate": 48000}
    video = VideoFromList([VideoFromFile(fixture_files[name]), VideoFromFile(fixture_files[name], start_time=0.3)],
                          complete_audio=long_audio)
    assert torch.equal(video.get_audio()["waveform"], video.get_components().audio["waveform"])


def test_video_from_list_keeps_alpha_only_if_every_part_has_it(fixture_files):
    mixed = VideoFromList([VideoFromFile(fixture_files["png-alpha"]), VideoFromFile(fixture_files["h264"])])
    assert mixed.get_components().alpha is None
    assert all(c.alpha is None for c in mixed.decode_chunked(chunk_frames=8))
    both = VideoFromList([VideoFromFile(fixture_files["png-alpha"]), VideoFromFile(fixture_files["png-alpha"])])
    _, _, alpha = joined(both.decode_chunked(chunk_frames=8))
    assert torch.equal(alpha, both.get_components().alpha)


def test_video_from_list_compares_decoded_sizes(fixture_files, tmp_path):
    """A rotated landscape clip joins a portrait clip of the same decoded size, and not a landscape one."""
    portrait = write_video(str(tmp_path / "portrait.mov"), codec="mpeg4", container_format="mov", width=48, height=64)
    landscape = write_video(str(tmp_path / "landscape.mov"), codec="mpeg4", container_format="mov")
    rotated = VideoFromFile(fixture_files["rotated"])
    ok = VideoFromList([rotated, VideoFromFile(portrait)])
    assert torch.equal(joined(ok.decode_chunked())[1], ok.get_components().images)
    bad = VideoFromList([rotated, VideoFromFile(landscape)])
    with pytest.raises(ValueError, match="incompatible frame dimensions"):
        bad.get_components()
    with pytest.raises(ValueError, match="incompatible frame dimensions"):
        next(bad.decode_chunked())
