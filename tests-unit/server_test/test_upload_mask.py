import io
import json
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import folder_paths
from comfy.cli_args import args


def png(color):
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def mask_form(original_ref):
    form = FormData()
    form.add_field("image", png((0, 0, 0, 0)), filename="mask.png")
    form.add_field("type", "input")
    form.add_field("original_ref", json.dumps(original_ref))
    return form


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    # Importing server initialises the device; these tests never touch a model.
    with patch.object(args, "cpu", True):
        import server
    prompt_server = server.PromptServer(None, MagicMock(enabled=False, register_upload=MagicMock(return_value=None)))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as client:
        yield client


@pytest.mark.asyncio
async def test_upload_mask_applies_alpha_to_original(client, tmp_path):
    (tmp_path / "original.png").write_bytes(png((255, 0, 0, 255)))
    resp = await client.post("/upload/mask", data=mask_form({"filename": "original.png", "type": "input"}))
    assert resp.status == 200
    name = (await resp.json())["name"]
    with Image.open(tmp_path / name) as saved:
        assert saved.getpixel((0, 0)) == (255, 0, 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("original_ref,status", [
    ({"filename": "missing.png", "type": "input"}, 404),
    ({"filename": "../original.png", "type": "input"}, 400),
    ({"filename": "original.png", "type": "input", "subfolder": "../.."}, 403),
])
async def test_upload_mask_rejected_original_returns_error(client, tmp_path, original_ref, status):
    resp = await client.post("/upload/mask", data=mask_form(original_ref))
    assert resp.status == status
    assert list(tmp_path.iterdir()) == []
