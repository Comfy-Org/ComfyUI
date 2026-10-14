"""ACEScg support on Convert Image Color Space and Save Image (Advanced) EXR."""

import struct
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import folder_paths
from comfy.cli_args import args

previous_nodes = sys.modules.get("nodes")
previous_server = sys.modules.get("server")
sys.modules["nodes"] = MagicMock(MAX_RESOLUTION=16384)
sys.modules["server"] = MagicMock()
from comfy_extras import nodes_images  # noqa: E402
from comfy_extras.nodes_images import (  # noqa: E402
    ImageColorSpace,
    SaveImageAdvanced,
    _ACESCG_TO_REC2020,
    _ACESCCT_CUT,
    _ACESCCT_LOG_OFFSET,
    _ACESCCT_LOG_SCALE,
    _ACESCCT_OFFSET,
    _ACESCCT_SLOPE,
    _CHROMATICITIES,
    _REC2020_TO_REC709,
    _convert_rgb_primaries,
    _encode_image,
    inject_exr_metadata,
)
for name, previous in (("nodes", previous_nodes), ("server", previous_server)):
    if previous is None:
        sys.modules.pop(name)
    else:
        sys.modules[name] = previous


def _convert(image, source, destination):
    return ImageColorSpace.execute(image, source, destination).result[0]


def _compress_acescct(linear):
    low = _ACESCCT_SLOPE * linear + _ACESCCT_OFFSET
    high = (torch.log2(linear.clamp_min(_ACESCCT_CUT)) + _ACESCCT_LOG_OFFSET) / _ACESCCT_LOG_SCALE
    return torch.where(linear > _ACESCCT_CUT, high, low)


def _read_chromaticities(exr_bytes: bytes) -> tuple[float, ...]:
    pos = 8
    while pos < len(exr_bytes) and exr_bytes[pos] != 0:
        name_end = exr_bytes.index(b"\x00", pos)
        name = exr_bytes[pos:name_end].decode("latin-1")
        type_end = exr_bytes.index(b"\x00", name_end + 1)
        size = struct.unpack("<i", exr_bytes[type_end + 1:type_end + 5])[0]
        value = exr_bytes[type_end + 5:type_end + 5 + size]
        if name == "chromaticities":
            return struct.unpack("<8f", value)
        pos = type_end + 5 + size
    raise AssertionError("EXR header has no chromaticities attribute")


def test_acescct_to_acescg_is_the_acescct_decompress():
    # Mid-grey, SDR white, and a superwhite that must stay above 1.0.
    linear = torch.tensor([[[[0.18, 1.0, 4.0]]]], dtype=torch.float32)
    cct = _compress_acescct(linear)

    out = _convert(cct, "HDR ACEScct", "HDR ACEScg")

    torch.testing.assert_close(out, linear, rtol=1e-5, atol=1e-5)


def test_acescg_to_acescct_is_the_acescct_compress():
    linear = torch.tensor([[[[0.18, 1.0, 4.0]]]], dtype=torch.float32)

    out = _convert(linear, "HDR ACEScg", "HDR ACEScct")

    torch.testing.assert_close(out, _compress_acescct(linear), rtol=1e-5, atol=1e-5)


def test_acescg_round_trip_preserves_superwhites():
    linear = torch.tensor([[[[0.02, 0.18, 8.0]]]], dtype=torch.float32)

    cct = _convert(linear, "HDR ACEScg", "HDR ACEScct")
    back = _convert(cct, "HDR ACEScct", "HDR ACEScg")

    torch.testing.assert_close(back, linear, rtol=1e-4, atol=1e-5)
    assert back.amax().item() > 1.0


def test_acescg_to_linear_uses_ap1_to_rec709():
    acescg = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    # Hub: ACEScg → Rec.2020 → Rec.709, which equals AP1 → Rec.709.
    expected = _convert_rgb_primaries(
        _convert_rgb_primaries(acescg, _ACESCG_TO_REC2020),
        _REC2020_TO_REC709,
    )

    out = _convert(acescg, "HDR ACEScg", "linear")

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_same_space_acescg_is_a_no_op_on_values():
    image = torch.tensor([[[[1.5, -0.1, 0.2]]]], dtype=torch.float32)
    out = _convert(image, "HDR ACEScg", "HDR ACEScg")
    torch.testing.assert_close(out.cpu().float(), image)


def test_exr_metadata_stamps_ap1_d60_for_acescg():
    raw = _encode_image(torch.ones(2, 2, 3), "exr", "32-bit float", "linear")
    tagged = inject_exr_metadata(raw, None, None, "HDR ACEScg")
    chroma = _read_chromaticities(tagged)

    assert chroma == pytest.approx(_CHROMATICITIES["ACEScg"])
    assert chroma[6:] == pytest.approx((0.32168, 0.33767))  # ACES white / D60, not D65
    linear_tagged = inject_exr_metadata(raw, None, None, "linear")
    assert _read_chromaticities(linear_tagged)[6:] == pytest.approx((0.3127, 0.3290))  # D65


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    saved = (folder_paths.get_output_directory(), folder_paths.get_previews_directory())
    folder_paths.set_output_directory(str(tmp_path / "output"))
    folder_paths.set_previews_directory(str(tmp_path / "previews"))
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(
        nodes_images.PromptServer,
        "instance",
        SimpleNamespace(asset_manager=SimpleNamespace(enabled=True)),
        raising=False,
    )
    monkeypatch.setattr(args, "disable_metadata", False)
    monkeypatch.setattr(SaveImageAdvanced, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None), raising=False)
    yield tmp_path
    folder_paths.set_output_directory(saved[0])
    folder_paths.set_previews_directory(saved[1])


def test_save_image_advanced_writes_acescg_chromaticities(dirs):
    from comfy_extras import nodes_images

    images = torch.full((1, 8, 8, 3), 1.5)
    fmt = {"format": "exr", "bit_depth": "32-bit float", "input_color_space": "HDR ACEScg"}
    entry = SaveImageAdvanced.execute(images, "acescg", fmt).ui["images"][0]
    path = dirs / "output" / entry["filename"]
    chroma = _read_chromaticities(path.read_bytes())

    assert chroma == pytest.approx(_CHROMATICITIES["ACEScg"])
    assert chroma[6:] == pytest.approx((0.32168, 0.33767))
