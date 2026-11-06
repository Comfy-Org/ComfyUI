from io import BytesIO

import pytest
import pytest_asyncio
import torch
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy.cli_args import args
import folder_paths


_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    from app.assets import manager as assets_manager
    import server
finally:
    args.cpu = _original_cpu


SIZE = (3, 2)
ALPHA = [0, 64, 128, 192, 254, 255]


@pytest_asyncio.fixture
async def image_client(monkeypatch, tmp_path):
    for name in ("input", "output", "temp", "user"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(folder_paths, f"{name}_directory", str(directory))
    monkeypatch.setattr(args, "front_end_root", str(tmp_path))
    monkeypatch.setattr(args, "multi_user", False)
    monkeypatch.setattr(server.PromptServer, "instance", None, raising=False)
    if assets_manager.dependencies_available():
        # NoAssets disables the shared seeder while registering its routes.
        seeder = assets_manager.asset_seeder
        monkeypatch.setattr(seeder, "_disabled", seeder.is_disabled())

    prompt_server = server.PromptServer(None, assets_manager.NoAssets(args))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as client:
        yield client


def encode_image(mode, image_format, pixels, transparency):
    image = Image.new(mode, SIZE)
    if mode == "P":
        image.putpalette([200, 0, 0, 0, 200, 0, 0, 0, 200] + [0] * 759)
    image.putdata(pixels)
    save_options = {}
    if transparency is not None:
        save_options["transparency"] = transparency
    if image_format == "PNG":
        metadata = PngInfo()
        metadata.add_text("workflow", '{"nodes": []}')
        save_options["pnginfo"] = metadata
    buffer = BytesIO()
    image.save(buffer, format=image_format, **save_options)
    data = buffer.getvalue()

    with Image.open(BytesIO(data)) as encoded:
        encoded.load()
        assert encoded.mode == mode
        assert encoded.size == SIZE
        assert encoded.info.get("transparency") == transparency
    return data


async def upload_image(client, data, image_format):
    filename = "alpha.png" if image_format == "PNG" else "alpha.jpg"
    form = FormData()
    form.add_field("image", data, filename=filename, content_type=f"image/{image_format.lower()}")
    form.add_field("subfolder", "alpha-test")
    response = await client.post("/upload/image", data=form)
    assert response.status == 200
    uploaded = await response.json()
    assert uploaded == {"name": filename, "subfolder": "alpha-test", "type": "input"}

    from_disk = folder_paths.get_input_directory()
    with open(f"{from_disk}/alpha-test/{filename}", "rb") as saved:
        assert saved.read() == data
    params = {"filename": uploaded["name"], "subfolder": uploaded["subfolder"], "type": uploaded["type"]}
    response = await client.get("/view", params=params)
    assert response.status == 200
    assert await response.read() == data
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,image_format,pixels,transparency,expected_alpha", [
    pytest.param("LA", "PNG", [(80, a) for a in ALPHA], None, ALPHA, id="la"),
    pytest.param("P", "PNG", [0, 1, 2, 2, 1, 0], bytes([0, 96, 255]), [0, 96, 255, 255, 96, 0], id="palette-table"),
    pytest.param("P", "PNG", [0, 1, 2, 2, 1, 0], 1, [255, 0, 255, 255, 0, 255], id="palette-index"),
    pytest.param("P", "PNG", [0, 1, 2, 2, 1, 0], 0, [0, 255, 255, 255, 255, 0], id="palette-index-zero"),
    pytest.param("RGB", "PNG", [(10, 20, 30), (10, 20, 31)] * 3, (10, 20, 30), [0, 255] * 3, id="rgb-trns"),
    pytest.param("L", "PNG", [64, 65, 0, 255, 64, 63], 64, [0, 255, 255, 255, 0, 255], id="l-trns"),
    pytest.param("RGBA", "PNG", [(80, 80, 80, a) for a in ALPHA], None, ALPHA, id="rgba"),
    pytest.param("RGB", "PNG", [(10, 20, 30)] * 6, None, [255] * 6, id="opaque-rgb"),
    pytest.param("RGB", "JPEG", [(10, 20, 30)] * 6, None, [255] * 6, id="opaque-jpeg"),
])
async def test_view_alpha(image_client, mode, image_format, pixels, transparency, expected_alpha):
    data = encode_image(mode, image_format, pixels, transparency)
    params = await upload_image(image_client, data, image_format)
    response = await image_client.get("/view", params={**params, "channel": "a"})
    assert response.status == 200
    assert response.content_type == "image/png"
    with Image.open(BytesIO(await response.read())) as alpha_image:
        assert alpha_image.mode == "RGBA"
        assert alpha_image.size == SIZE
        assert list(alpha_image.convert("RGB").getdata()) == [(0, 0, 0)] * 6
        assert list(alpha_image.getchannel("A").getdata()) == expected_alpha


@pytest.mark.asyncio
async def test_view_alpha_with_preview(image_client):
    data = encode_image("LA", "PNG", [(80, a) for a in ALPHA], None)
    params = await upload_image(image_client, data, "PNG")
    response = await image_client.get("/view", params={**params, "channel": "a", "preview": "webp;90"})
    assert response.status == 200
    assert response.content_type == "image/webp"
    with Image.open(BytesIO(await response.read())) as preview:
        assert preview.size == SIZE
        assert list(preview.getchannel("A").getdata()) == ALPHA
