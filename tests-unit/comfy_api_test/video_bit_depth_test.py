import pytest
import torch
import av
import numpy as np
import io
import json
from fractions import Fraction
from types import SimpleNamespace
from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents, VideoFromList
from comfy_api.latest._util.video_types import VideoComponents
from comfy_extras.nodes_video import CreateVideo, LoadVideo, SaveVideo


@pytest.fixture(scope="module")
def gradient_components():
    """Narrow horizontal ramp (0.25..0.30) that needs more than 8 bits to stay smooth"""
    width, height, frames = 64, 64, 3
    ramp = torch.linspace(0.25, 0.30, width).view(1, 1, width, 1).expand(frames, height, width, 3)
    return VideoComponents(images=ramp.contiguous(), frame_rate=Fraction(30))


@pytest.fixture(scope="module")
def src8(gradient_components, tmp_path_factory):
    """8-bit h264 mp4 (Create Video default)"""
    path = str(tmp_path_factory.mktemp("video") / "src8.mp4")
    VideoFromComponents(gradient_components).save_to(path)
    return path


@pytest.fixture(scope="module")
def src10(gradient_components, tmp_path_factory):
    """10-bit h264 mp4 (Create Video with bit_depth=10)"""
    path = str(tmp_path_factory.mktemp("video") / "src10.mp4")
    VideoFromComponents(gradient_components, bit_depth=10).save_to(path)
    return path


def probe(path):
    """(codec, pix_fmt, bit_depth) of the first video stream"""
    with av.open(path) as container:
        stream = container.streams.video[0]
        return (stream.codec.canonical_name, stream.format.name, max(c.bits for c in stream.format.components))


def decoded_levels(path):
    """Unique tonal levels in the first decoded frame (banding measure)"""
    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        return len(np.unique(frame.to_ndarray(format="gbrpf32le")[..., 0]))


def video_packet_bytes(path):
    """Raw video packet payloads; identical to the source's only for a true remux"""
    with av.open(path) as container:
        return [bytes(p) for p in container.demux(container.streams.video[0]) if p.size]


def test_create_video_bit_depth(src8, src10):
    """Create Video's bit_depth picks the encoded depth (default 8-bit); 10-bit reduces banding"""
    assert probe(src8) == ("h264", "yuv420p", 8)
    assert probe(src10) == ("h264", "yuv420p10le", 10)
    assert decoded_levels(src10) > 2 * decoded_levels(src8)


@pytest.mark.parametrize(
    "bit_depth,color_space,expected_bit_depth",
    [
        ("auto", "sRGB", 8),
        ("auto", "HDR", 10),
        ("auto", "HDR PQ", 10),
        (8, "HDR", 8),
        (10, "sRGB", 10),
    ],
)
def test_create_video_node_bit_depth(gradient_components, bit_depth, color_space, expected_bit_depth):
    video = CreateVideo.execute(
        gradient_components.images,
        float(gradient_components.frame_rate),
        bit_depth=bit_depth,
        color_space=color_space,
    ).args[0]
    assert video.get_bit_depth() == expected_bit_depth
    assert video.get_color_space() == color_space


def test_create_video_node_bit_depth_options():
    bit_depth_input = next(input for input in CreateVideo.define_schema().inputs if input.id == "bit_depth")
    assert bit_depth_input.options == ["auto", 8, 10]
    assert bit_depth_input.default == "auto"


@pytest.mark.parametrize(
    "codec,expected_suffix,expected_codec",
    [
        ("auto", "mp4", "h264"),
        ("h264", "mp4", "h264"),
        ("av1", "webm", "av1"),
        ("prores", "mov", "prores"),
    ],
)
def test_save_video_auto_format(gradient_components, tmp_path, monkeypatch, codec, expected_suffix, expected_codec):
    monkeypatch.setattr(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "comfy_extras.nodes_video.folder_paths.get_save_image_path",
        lambda *args: (str(tmp_path), "auto", 1, "", "auto"),
    )
    video = VideoFromComponents(gradient_components)

    SaveVideo.execute(
        video,
        "auto",
        {"format": "auto", "codec": {"codec": codec}},
    )

    path = tmp_path / f"auto_00001_.{expected_suffix}"
    with av.open(path) as container:
        assert container.streams.video[0].codec.canonical_name == expected_codec


def test_save_video_has_no_color_space_input():
    schema = SaveVideo.define_schema()
    assert all("color_space" not in str(input.as_dict()) for input in schema.inputs)


def test_save_auto_keeps_source_depth(src8, src10, tmp_path):
    """Save Video (no bit_depth = auto) stream-copies the source, preserving its depth byte-for-byte"""
    for name, src in [("p8", src8), ("p10", src10)]:
        path = str(tmp_path / f"{name}.mp4")
        VideoFromFile(src).save_to(path)
        assert probe(path) == probe(src)
        assert video_packet_bytes(path) == video_packet_bytes(src)


def test_save_explicit_depth_reencodes(src8, src10, tmp_path):
    """An explicit bit_depth different from the source forces a re-encode to that depth"""
    down = str(tmp_path / "down8.mp4")
    VideoFromFile(src10).save_to(down, bit_depth=8)
    assert probe(down) == ("h264", "yuv420p", 8)

    up = str(tmp_path / "up10.mp4")
    VideoFromFile(src8).save_to(up, bit_depth=10)
    assert probe(up) == ("h264", "yuv420p10le", 10)


def test_trim_keeps_source_depth(src10, tmp_path):
    """Video Slice re-encodes (trim) but preserves the source's 10-bit depth"""
    path = str(tmp_path / "trim.mp4")
    VideoFromFile(src10).as_trimmed(start_time=0, duration=1 / 30, strict_duration=False).save_to(path)
    assert probe(path) == ("h264", "yuv420p10le", 10)


def test_get_bit_depth(gradient_components, src8, src10):
    """get_bit_depth reports a video's depth (backs the Get Video Components output)"""
    assert VideoFromFile(src8).get_bit_depth() == 8
    assert VideoFromFile(src10).get_bit_depth() == 10
    assert VideoFromComponents(gradient_components, bit_depth=10).get_bit_depth() == 10
    assert VideoFromComponents(gradient_components).get_bit_depth() == 8


@pytest.mark.parametrize("source_kind", ["components", "file", "list"])
@pytest.mark.parametrize("codec,extension", [("h264", "mp4"), ("av1", "webm")])
def test_12bit_falls_back_with_warning(gradient_components, src10, tmp_path, caplog, source_kind, codec, extension):
    video = VideoFromComponents(gradient_components, bit_depth=12)
    options = {}
    if source_kind == "file":
        video = VideoFromFile(src10)
        options["bit_depth"] = 12
    elif source_kind == "list":
        video = VideoFromList([VideoFromFile(src10), VideoFromFile(src10)])
        options["bit_depth"] = 12
    path = str(tmp_path / f"fallback.{extension}")
    video.save_to(path, codec=codec, **options)

    assert probe(path) == (codec, "yuv420p10le", 10)
    assert "does not support 12-bit color; saving at 10-bit instead" in caplog.text
    assert decoded_levels(path) > 20


@pytest.mark.parametrize("source_kind", ["components", "file", "list"])
@pytest.mark.parametrize("chroma,encoded_profile", [
    ("444", "4444"),
    ("422", "HQ"),
])
def test_prores_chroma_subsampling(gradient_components, src10, tmp_path, caplog, source_kind, chroma, encoded_profile):
    video = VideoFromComponents(gradient_components)
    if source_kind == "file":
        video = VideoFromFile(src10)
    elif source_kind == "list":
        video = VideoFromList([VideoFromFile(src10), VideoFromFile(src10)])
    path = str(tmp_path / "prores.mov")
    video.save_to(path, codec="prores", chroma_subsampling=chroma, bit_depth=12)

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.codec.canonical_name == "prores"
        assert stream.profile == encoded_profile
        assert stream.format.name.startswith(f"yuv{chroma}p")
        assert len(list(container.decode(stream))) == (6 if source_kind == "list" else 3)
    assert "prores_ks does not support 12-bit color; saving at 10-bit instead" in caplog.text
    assert decoded_levels(path) > 20


@pytest.mark.parametrize("alpha_source", ["rgba", "mask"])
def test_prores_alpha_roundtrip(gradient_components, tmp_path, caplog, alpha_source):
    images = gradient_components.images
    alpha = torch.linspace(0, 1, images.shape[2]).expand(images.shape[:3]).contiguous()
    components = VideoComponents(
        images=torch.cat((images, alpha.unsqueeze(-1)), dim=-1) if alpha_source == "rgba" else images,
        alpha=alpha if alpha_source == "mask" else None,
        frame_rate=gradient_components.frame_rate,
    )
    source = str(tmp_path / "alpha.mov")
    VideoFromComponents(components, bit_depth=10).save_to(source, codec="prores", chroma_subsampling="444")
    output = str(tmp_path / "alpha-transcoded.mov")
    VideoFromFile(source).save_to(output, codec="prores", chroma_subsampling="444", bit_depth=10)

    for path in (source, output):
        decoded = VideoFromFile(path).get_components()
        assert decoded.alpha is not None
        torch.testing.assert_close(decoded.alpha.squeeze(-1), alpha, atol=0.003, rtol=0)
    assert "alpha channel will be discarded" not in caplog.text


def test_prores_chroma_change_reencodes(gradient_components, tmp_path):
    source = str(tmp_path / "444.mov")
    VideoFromComponents(gradient_components).save_to(source, codec="prores", chroma_subsampling="444")
    output = str(tmp_path / "422.mov")
    VideoFromFile(source).save_to(output, codec="prores", chroma_subsampling="422", bit_depth=10)
    with av.open(output) as container:
        assert container.streams.video[0].profile == "HQ"
        assert container.streams.video[0].format.name == "yuv422p10le"
    assert video_packet_bytes(output) != video_packet_bytes(source)


@pytest.mark.parametrize("output_kind", ["file", "buffer"])
def test_prores_audio_and_metadata(gradient_components, tmp_path, output_kind):
    components = VideoComponents(
        images=gradient_components.images,
        frame_rate=gradient_components.frame_rate,
        audio={"waveform": torch.zeros(1, 2, 4800), "sample_rate": 48000},
    )
    path = io.BytesIO() if output_kind == "buffer" else str(tmp_path / "metadata.mov")
    VideoFromComponents(components, bit_depth=10).save_to(path, codec="prores", metadata={"prompt": {"test": True}})
    if output_kind == "buffer":
        path.seek(0)
    with av.open(path) as container:
        assert container.format.name.startswith("mov")
        assert json.loads(container.metadata["prompt"]) == {"test": True}
        assert container.streams.video[0].profile == "4444"
        assert container.streams.audio[0].sample_rate == 48000
        assert sum(frame.samples for frame in container.decode(audio=0)) >= 4800


def test_save_video_prores_defaults(gradient_components, tmp_path, monkeypatch, caplog):
    schema = SaveVideo.define_schema()
    assert all("chroma_subsampling" not in str(input.as_dict()) for input in schema.inputs)
    format_input = next(input for input in schema.inputs if input.id == "format")
    mov_option = next(option for option in format_input.options if option.key == "mov")
    prores_option = mov_option.inputs[0].options[0]
    assert prores_option.key == "prores"
    assert prores_option.inputs == []
    chroma_input = next(input for input in CreateVideo.define_schema().inputs if input.id == "chroma_subsampling")
    assert chroma_input.options == ["auto", "444", "422", "420"]
    assert chroma_input.default == "auto"
    monkeypatch.setattr(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "comfy_extras.nodes_video.folder_paths.get_save_image_path",
        lambda *args: (str(tmp_path), "prores", 1, "", "prores"),
    )
    SaveVideo.execute(VideoFromComponents(gradient_components), "prores", "mov")
    with av.open(str(tmp_path / "prores_00001_.mov")) as container:
        assert container.streams.video[0].profile == "4444"
        assert container.streams.video[0].format.name.startswith("yuv444p")
    assert "does not support" not in caplog.text


def test_prores_rejects_mp4(gradient_components, tmp_path):
    with pytest.raises(ValueError, match="ProRes output requires a MOV or MKV container"):
        VideoFromComponents(gradient_components).save_to(str(tmp_path / "bad.mp4"), format="mp4", codec="prores")


@pytest.mark.parametrize("codec,extension", [("prores", "mov"), ("h264", "mp4")])
def test_444_transcode_preserves_odd_dimensions(tmp_path, codec, extension):
    video = VideoFromComponents(
        VideoComponents(images=torch.rand(2, 65, 63, 3), frame_rate=Fraction(30)), bit_depth=10,
    )
    source = str(tmp_path / f"source.{extension}")
    output = str(tmp_path / f"transcoded.{extension}")
    video.save_to(source, codec=codec, chroma_subsampling="444")
    VideoFromFile(source).save_to(output, codec=codec, bit_depth=10, chroma_subsampling="444")
    with av.open(output) as container:
        frames = list(container.decode(video=0))
        assert [(frame.width, frame.height) for frame in frames] == [(63, 65)] * 2


@pytest.mark.parametrize("chroma", ["444", "422"])
def test_prores_remux_preserves_profile(gradient_components, tmp_path, chroma):
    source = str(tmp_path / "source.mov")
    output = str(tmp_path / "remux.mov")
    VideoFromComponents(gradient_components).save_to(source, codec="prores", chroma_subsampling=chroma)
    VideoFromFile(source).save_to(output, format="mov", codec="prores")
    with av.open(source) as original, av.open(output) as remuxed:
        assert remuxed.streams.video[0].profile == original.streams.video[0].profile
        assert remuxed.streams.video[0].codec_context.codec_tag == original.streams.video[0].codec_context.codec_tag
    assert video_packet_bytes(output) == video_packet_bytes(source)


@pytest.mark.parametrize("source_kind", ["components", "file", "list"])
@pytest.mark.parametrize("chroma", ["444", "422", "420"])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_h264_chroma_subsampling(gradient_components, src10, tmp_path, source_kind, chroma, bit_depth):
    video = VideoFromComponents(gradient_components)
    if source_kind == "file":
        video = VideoFromFile(src10)
    elif source_kind == "list":
        video = VideoFromList([VideoFromFile(src10), VideoFromFile(src10)])
    path = str(tmp_path / "chroma.mp4")
    video.save_to(path, codec="h264", bit_depth=bit_depth, chroma_subsampling=chroma)
    assert probe(path) == ("h264", f"yuv{chroma}p" + ("10le" if bit_depth == 10 else ""), bit_depth)
    with av.open(path) as container:
        assert len(list(container.decode(video=0))) == (6 if source_kind == "list" else 3)


@pytest.mark.parametrize("codec,extension,requested,actual", [("av1", "webm", "444", "420"), ("prores", "mov", "420", "422")])
def test_unsupported_chroma_falls_back(gradient_components, tmp_path, caplog, codec, extension, requested, actual):
    path = str(tmp_path / f"fallback.{extension}")
    VideoFromComponents(gradient_components).save_to(path, codec=codec, chroma_subsampling=requested)
    with av.open(path) as container:
        assert container.streams.video[0].format.name.startswith(f"yuv{actual}p")
    assert f"does not support {requested} chroma subsampling; saving as {actual} instead" in caplog.text


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("codec,extension,requested,actual", [
    ("h264", "mp4", "444", "444"),
    ("h264", "mp4", "422", "422"),
    ("h264", "mp4", "420", "420"),
    ("av1", "webm", "444", "420"),
    ("av1", "webm", "422", "420"),
    ("av1", "webm", "420", "420"),
    ("prores", "mov", "444", "444"),
    ("prores", "mov", "422", "422"),
    ("prores", "mov", "420", "422"),
    ("auto", "mp4", "444", "444"),
])
def test_create_video_chroma_control(gradient_components, tmp_path, monkeypatch, caplog, eager, codec, extension, requested, actual):
    monkeypatch.setattr(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(
        "comfy_extras.nodes_video.folder_paths.get_save_image_path",
        lambda *args: (str(tmp_path), "chroma", 1, "", "chroma"),
    )
    video = CreateVideo.execute(
        gradient_components.images, 30, bit_depth=10, chroma_subsampling=requested,
        codec=codec if eager else "none",
    ).args[0]
    assert video.get_chroma_subsampling() == (actual if eager else requested)
    SaveVideo.execute(video, "chroma", {"format": extension, "codec": {"codec": codec}})
    with av.open(str(tmp_path / f"chroma_00001_.{extension}")) as container:
        assert container.streams.video[0].format.name.startswith(f"yuv{actual}p")
        assert len(list(container.decode(video=0))) == 3
    if requested != actual:
        assert f"does not support {requested} chroma subsampling; saving as {actual} instead" in caplog.text
    else:
        assert "chroma subsampling; saving" not in caplog.text


@pytest.mark.parametrize("source_chroma,has_alpha", [("444", False), ("444", True), ("422", False)])
@pytest.mark.parametrize("format_name,codec,extension", [("mov", "prores", "mov"), ("auto", "prores", "mov"), ("auto", "auto", "mp4"), ("mp4", "auto", "mp4")])
def test_load_and_save_prores_nodes(gradient_components, tmp_path, monkeypatch, source_chroma, has_alpha, format_name, codec, extension):
    monkeypatch.setattr(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr("comfy_extras.nodes_video.folder_paths.get_output_directory", lambda: str(tmp_path))
    images = gradient_components.images
    alpha = torch.linspace(0, 1, images.shape[2]).expand(images.shape[:3]).contiguous() if has_alpha else None
    components = VideoComponents(
        images=images, alpha=alpha, frame_rate=gradient_components.frame_rate,
        audio={"waveform": (torch.sin(torch.arange(4800) * (2 * torch.pi * 440 / 48000)) * 0.1).reshape(1, 1, -1), "sample_rate": 48000},
    )
    VideoFromComponents(components, bit_depth=10).save_to(str(tmp_path / "source.mov"), codec="prores", chroma_subsampling=source_chroma)
    assert "source.mov" in LoadVideo.define_schema().inputs[0].options
    loaded = LoadVideo.execute("source.mov").args[0]
    decoded = loaded.get_components()
    assert decoded.images.shape == images.shape
    assert decoded.audio is not None
    assert (decoded.alpha is not None) == has_alpha

    options = {"codec": codec}
    SaveVideo.execute(loaded, "roundtrip", {"format": format_name, "codec": options})
    path = next(tmp_path.glob(f"roundtrip_*.{extension}"))
    saved = VideoFromFile(str(path)).get_components()
    assert saved.images.shape == images.shape
    assert saved.audio is not None
    expected_samples = images.shape[0] * 48000 // int(components.frame_rate)
    # AAC pads the final frame to a block of 1024 samples.
    assert expected_samples <= saved.audio["waveform"].shape[-1] < expected_samples + 1024
    assert saved.audio["waveform"].abs().mean() > 0.01
    with av.open(str(path)) as container:
        assert container.streams.video[0].codec.canonical_name == ("prores" if codec == "prores" else "h264")
    if codec == "prores" and has_alpha:
        torch.testing.assert_close(saved.alpha.squeeze(-1), alpha, atol=0.003, rtol=0)
    else:
        assert saved.alpha is None


@pytest.mark.parametrize("chroma", ["444", "422"])
@pytest.mark.parametrize("operation", ["crop", "trim", "transcode", "concatenate"])
def test_created_chroma_survives_video_operations(gradient_components, tmp_path, chroma, operation):
    video = CreateVideo.execute(gradient_components.images, 30, bit_depth=10, chroma_subsampling=chroma).args[0]
    save_options = {}
    if operation == "crop":
        video = video.as_cropped(0, 0, 32, 32)
    elif operation == "trim":
        video = video.as_trimmed(0, 1 / 30)
    elif operation == "transcode":
        video = VideoFromFile(video.get_stream_source())
        save_options["crf"] = 20
    else:
        second = CreateVideo.execute(gradient_components.images, 30, bit_depth=8, chroma_subsampling=chroma).args[0]
        video = VideoFromList([video, second])
    path = str(tmp_path / "preserved.mp4")
    video.save_to(path, **save_options)
    assert video.get_chroma_subsampling() == chroma
    assert probe(path) == ("h264", f"yuv{chroma}p10le", 10)
