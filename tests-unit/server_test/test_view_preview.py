"""/view with ?preview= must not 500 on a file PIL can't identify (e.g. EXR)."""

from io import BytesIO
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import folder_paths
import server


@pytest_asyncio.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda _type: str(tmp_path))
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as client:
        yield client


@pytest.mark.asyncio
async def test_preview_of_undecodable_file_serves_raw_bytes(client, tmp_path):
    # OpenEXR magic number plus padding: enough for the file to be an EXR PIL can't identify.
    raw = b"\x76\x2f\x31\x01" + b"\x00" * 64
    (tmp_path / "render.exr").write_bytes(raw)

    resp = await client.get("/view", params={"filename": "render.exr", "type": "output", "preview": "webp;90"})

    assert resp.status == 200
    assert await resp.read() == raw


@pytest.mark.asyncio
async def test_preview_of_png_is_still_reencoded(client, tmp_path):
    Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp_path / "out.png")

    resp = await client.get("/view", params={"filename": "out.png", "type": "output", "preview": "webp;90"})

    assert resp.status == 200
    assert resp.content_type == "image/webp"
    with Image.open(BytesIO(await resp.read())) as img:
        assert img.format == "WEBP"
