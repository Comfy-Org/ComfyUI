"""Tests for /view?res=N downscaled JPEG previews"""

import struct
import sys
import zlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
import server
from utils.mime_types import init_mime_types

init_mime_types()


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(folder_paths, "output_directory", str(out))
    return out


async def view(params, asset_manager=None):
    prompt_server = server.PromptServer(None, asset_manager or MagicMock(enabled=False))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as client:
        resp = await client.get("/view", params=params)
        return resp.status, resp.headers, await resp.read()


def save(path, size=(1000, 500), mode="RGB", color=(200, 30, 30), **kwargs):
    Image.new(mode, size, color).save(path, **kwargs)
    return path.read_bytes()


def break_second_idat_chunk(png):
    out, i, idat = png[:8], 8, 0
    while i < len(png):
        (length,) = struct.unpack(">I", png[i:i + 4])
        ctype, data = png[i + 4:i + 8], png[i + 8:i + 8 + length]
        i += 12 + length
        parts = [(ctype, data)]
        if ctype == b"IDAT" and idat == 0:
            parts = [(b"IDAT", data[:length // 2]), (b"\x00\x01\x02\x03", data[length // 2:])]
            idat += 1
        for t, d in parts:
            out += struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.jpeg", "a.PNG", "a.JPG"])
async def test_downscales_to_jpeg(output_dir, name):
    save(output_dir / name, format="png" if name.lower().endswith("png") else "jpeg")
    status, headers, body = await view({"filename": name, "res": "512"})
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["X-Content-Type-Options"] == "nosniff"
    reference = BytesIO()
    Image.new("RGB", (8, 8)).save(reference, format="jpeg", quality=85)
    with Image.open(BytesIO(body)) as img, Image.open(reference) as ref:
        assert img.format == "JPEG"
        assert img.size == (512, 256)
        assert img.quantization == ref.quantization


@pytest.mark.asyncio
async def test_without_res_serves_original(output_dir):
    original = save(output_dir / "a.png")
    status, headers, body = await view({"filename": "a.png"})
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == original


@pytest.mark.skipif(sys.platform == "win32", reason='" is not allowed in Windows filenames')
@pytest.mark.asyncio
async def test_filename_escaped_in_disposition(output_dir):
    save(output_dir / 'a\\"b.png')
    _, headers, _ = await view({"filename": 'a\\"b.png', "res": "64"})
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Content-Disposition"] == 'filename="a\\\\\\"b.jpg"'


@pytest.mark.asyncio
async def test_portrait_capped_by_height(output_dir):
    save(output_dir / "tall.png", size=(500, 1000))
    _, _, body = await view({"filename": "tall.png", "res": "512"})
    with Image.open(BytesIO(body)) as img:
        assert img.size == (256, 512)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,fmt,content_type", [("small.png", "png", "image/png"), ("small.jpg", "jpeg", "image/jpeg")])
async def test_already_fitting_image_serves_original(output_dir, name, fmt, content_type):
    original = save(output_dir / name, size=(100, 40), format=fmt)
    status, headers, body = await view({"filename": name, "res": "512"})
    assert status == 200 and headers["Content-Type"] == content_type
    assert body == original
    _, _, exact = await view({"filename": name, "res": "100"})
    assert exact == original


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,kwargs", [("RGBA", {}), ("LA", {}), ("I;16", {}), ("RGB", {"transparency": (0, 0, 0)})])
async def test_fitting_image_needing_flattening_is_converted(output_dir, mode, kwargs):
    image = Image.fromarray(np.full((40, 100), 1000, np.uint16)) if mode == "I;16" else Image.new(mode, (100, 40))
    image.save(output_dir / "small.png", **kwargs)
    status, headers, body = await view({"filename": "small.png", "res": "512"})
    assert status == 200 and headers["Content-Type"] == "image/jpeg"
    with Image.open(BytesIO(body)) as img:
        assert img.size == (100, 40)


@pytest.mark.asyncio
async def test_alpha_flattened_onto_black(output_dir):
    # Join Image with Alpha keeps the original RGB under alpha=0, so dropping alpha would show it.
    save(output_dir / "cutout.png", size=(100, 50), mode="RGBA", color=(255, 255, 255, 0))
    _, _, body = await view({"filename": "cutout.png", "res": "512"})
    with Image.open(BytesIO(body)) as img:
        assert max(img.getpixel((10, 10))) < 8


@pytest.mark.asyncio
async def test_colour_key_transparency_not_blended(output_dir):
    stripes = Image.new("RGB", (200, 200), (0, 0, 0))
    stripes.paste((255, 255, 255), (0, 0, 200, 200), mask=Image.fromarray(np.tile([[255, 0]], (200, 100)).astype(np.uint8)))
    stripes.save(output_dir / "keyed.png", transparency=(255, 255, 255))
    _, headers, body = await view({"filename": "keyed.png", "res": "50"})
    assert headers["Content-Type"] == "image/jpeg"
    with Image.open(BytesIO(body)) as img:
        assert img.size == (50, 50)
        assert max(img.getpixel((25, 25))) < 8


@pytest.mark.asyncio
async def test_16_bit_grayscale_scaled(output_dir):
    Image.fromarray(np.full((300, 600), 40000, np.uint16)).save(output_dir / "depth.png")
    status, _, body = await view({"filename": "depth.png", "res": "100"})
    assert status == 200
    with Image.open(BytesIO(body)) as img:
        assert img.size == (100, 50)
        assert abs(img.getpixel((10, 10))[0] - 40000 // 256) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["P", "1"])
async def test_palette_and_bilevel_resampled_with_filter(output_dir, mode):
    checker = Image.fromarray((np.indices((200, 200)).sum(axis=0) % 2 * 255).astype(np.uint8)).convert(mode)
    checker.save(output_dir / "checker.png")
    _, _, body = await view({"filename": "checker.png", "res": "50"})
    with Image.open(BytesIO(body)) as img:
        assert 100 < img.getpixel((25, 25))[0] < 156


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["not an image", "truncated", "broken chunk"])
async def test_undecodable_file_serves_original(output_dir, kind):
    if kind == "broken chunk":  # Pillow raises SyntaxError while decoding
        save(output_dir / "broken.png", size=(64, 64))
        content = break_second_idat_chunk((output_dir / "broken.png").read_bytes())
    elif kind == "truncated":
        noise = np.random.default_rng(0).integers(0, 256, (400, 400, 3), dtype=np.uint8)
        Image.fromarray(noise).save(output_dir / "broken.png")
        full = (output_dir / "broken.png").read_bytes()
        content = full[: len(full) * 6 // 10]
    else:
        content = b"not an image"
    (output_dir / "broken.png").write_bytes(content)
    status, headers, body = await view({"filename": "broken.png", "res": "16"})
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == content


@pytest.mark.asyncio
async def test_oversized_text_chunk_serves_original(output_dir):
    info = PngInfo()
    info.add_text("workflow", "x" * (2 * 1024 * 1024), zip=True)
    original = save(output_dir / "meta.png", pnginfo=info)
    status, headers, body = await view({"filename": "meta.png", "res": "64"})
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == original


@pytest.mark.asyncio
async def test_decompression_bomb_serves_original(output_dir, monkeypatch):
    original = save(output_dir / "huge.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    status, headers, body = await view({"filename": "huge.png", "res": "64"})
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == original


@pytest.mark.asyncio
async def test_exif_orientation_applied(output_dir):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    save(output_dir / "phone.jpg", size=(400, 200), exif=exif)
    _, _, body = await view({"filename": "phone.jpg", "res": "100"})
    with Image.open(BytesIO(body)) as img:
        assert img.size == (50, 100)


@pytest.mark.asyncio
async def test_res_wins_over_preview(output_dir):
    save(output_dir / "a.png")
    _, headers, body = await view({"filename": "a.png", "res": "512", "preview": "webp;75"})
    assert headers["Content-Type"] == "image/jpeg"
    with Image.open(BytesIO(body)) as img:
        assert img.size == (512, 256)


@pytest.mark.asyncio
async def test_channel_disables_res(output_dir):
    save(output_dir / "a.png")
    _, headers, body = await view({"filename": "a.png", "res": "512", "channel": "rgb"})
    assert headers["Content-Type"] == "image/png"
    with Image.open(BytesIO(body)) as img:
        assert img.size == (1000, 500)


@pytest.mark.asyncio
@pytest.mark.parametrize("res", ["abc", "0", "-1", "", " 64", "+64", "6_4", "\u0666\u0664", "\u00b2"])
async def test_invalid_res_serves_original(output_dir, res):
    original = save(output_dir / "a.png")
    status, headers, body = await view({"filename": "a.png", "res": res})
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == original


@pytest.mark.asyncio
@pytest.mark.parametrize("name,fmt,content_type", [
    ("a.webp", "webp", "image/webp"),
    ("a.gif", "gif", "image/gif"),
])
async def test_other_images_pass_through(output_dir, name, fmt, content_type):
    original = save(output_dir / name, format=fmt)
    status, headers, body = await view({"filename": name, "res": "64"})
    assert status == 200
    assert headers["Content-Type"] == content_type
    assert body == original


@pytest.mark.asyncio
async def test_video_passes_through(output_dir):
    (output_dir / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    status, headers, body = await view({"filename": "clip.mp4", "res": "512"})
    assert status == 200
    assert headers["Content-Type"] == "video/mp4"
    assert body == b"\x00\x00\x00\x18ftypmp42"


@pytest.mark.asyncio
async def test_hash_filename_resolves_then_downscales(tmp_path, monkeypatch):
    path = tmp_path / "stored.png"
    save(path)
    monkeypatch.setattr(server, "resolve_hash_to_path", lambda h: SimpleNamespace(
        abs_path=str(path), download_name="pic.png", content_type="image/png"), raising=False)
    status, headers, body = await view({"filename": "blake3:abc", "res": "512"}, MagicMock(enabled=True))
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Content-Disposition"] == 'filename="pic.jpg"'
    with Image.open(BytesIO(body)) as img:
        assert img.size == (512, 256)


@pytest.mark.asyncio
@pytest.mark.parametrize("params,expected", [
    ({"filename": "../secret.png"}, 400),
    ({"filename": "/etc/secret.png"}, 400),
    ({"filename": "secret.png", "subfolder": ".."}, 403),
    ({"filename": "missing.png"}, 404),
])
async def test_path_traversal_still_rejected(output_dir, params, expected):
    save(output_dir.parent / "secret.png")
    status, _, _ = await view({**params, "res": "512"})
    assert status == expected
