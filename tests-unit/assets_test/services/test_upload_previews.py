import io
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from app.assets.manager import AssetsEnabled
from app.assets.previews import generate_upload_preview

from .test_preview_lifecycle import roots  # noqa: F401 - fixture
from .test_previews import write_exr


class _Args:
    enable_assets = True
    enable_asset_hashing = False


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _client(asset_manager) -> TestClient:
    from comfy.cli_args import args
    from utils.mime_types import init_mime_types

    # Importing server initialises the device; these tests never touch a model.
    with patch.object(args, "cpu", True):
        import server
    init_mime_types()
    prompt_server = server.PromptServer(None, asset_manager)
    prompt_server.app.add_routes(prompt_server.routes)
    return TestClient(TestServer(prompt_server.app))


def _form(name: str, data: bytes, **fields) -> FormData:
    form = FormData()
    form.add_field("image", data, filename=name)
    for key, value in fields.items():
        form.add_field(key, value)
    return form


@pytest.mark.asyncio
async def test_an_exr_upload_returns_its_generated_preview(mock_create_session, roots, tmp_path):  # noqa: F811
    exr = write_exr(tmp_path / "src.exr", 64, 48).read_bytes()
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("frame.exr", exr))
        body = await resp.json()

    asset = body["asset"]
    assert resp.status == 200
    assert asset["preview_id"] != asset["id"]
    assert asset["preview_url"] == f"/api/assets/{asset['preview_id']}/content"


@pytest.mark.asyncio
async def test_a_png_upload_is_its_own_preview(mock_create_session, roots):  # noqa: F811
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("still.png", _png()))
        asset = (await resp.json())["asset"]

    assert asset["preview_id"] == asset["id"]


@pytest.mark.asyncio
async def test_mask_upload_still_answers_with_json(mock_create_session, roots):  # noqa: F811
    (roots / "input" / "base.png").write_bytes(_png())
    original_ref = json.dumps({"filename": "base.png", "type": "input", "subfolder": ""})
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/mask", data=_form("mask.png", _png(), original_ref=original_ref))
        body = await resp.json()

    assert resp.status == 200
    assert body["name"] == "mask.png"


@pytest.mark.asyncio
async def test_with_assets_off_an_exr_upload_has_no_asset(roots):  # noqa: F811
    async with await _client(MagicMock(enabled=False, register_upload=MagicMock(return_value=None))) as client:
        resp = await client.post("/upload/image", data=_form("frame.exr", b"exr"))
        body = await resp.json()

    assert resp.status == 200
    assert "asset" not in body


@pytest.mark.asyncio
async def test_a_preview_the_client_already_set_is_kept(tmp_path):
    exr = write_exr(tmp_path / "frame.exr", 8, 8)

    assert await generate_upload_preview("asset-id", str(exr), "client-preview") == "client-preview"


@pytest.mark.asyncio
async def test_api_asset_uploads_get_a_generated_preview(mock_create_session, roots, tmp_path):  # noqa: F811
    from app.assets.api.routes import _build_asset_response, _resolve_preview_paths, _with_upload_preview
    from app.assets.services.ingest import register_file_in_place
    from utils.mime_types import init_mime_types

    init_mime_types()
    exr = write_exr(roots / "input" / "frame.exr", 64, 48)
    result = register_file_in_place(str(exr), "frame.exr", ["input"], content_written=True)

    result = await _with_upload_preview(result)
    response = _build_asset_response(result, _resolve_preview_paths([result]))

    assert response.preview_id not in (None, response.id)
    assert response.preview_url == f"/api/assets/{response.preview_id}/content"
