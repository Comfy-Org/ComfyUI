import io
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from app.assets.manager import AssetsEnabled

from .preview_helpers import write_exr


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
async def test_an_exr_upload_returns_its_generated_preview(mock_create_session, roots, tmp_path):
    exr = write_exr(tmp_path / "src.exr", 64, 48).read_bytes()
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("frame.exr", exr))
        body = await resp.json()

    asset = body["asset"]
    assert resp.status == 200
    assert asset["preview_id"] != asset["id"]
    assert asset["preview_url"] == f"/api/assets/{asset['preview_id']}/content"


@pytest.mark.asyncio
async def test_a_repeat_upload_of_the_same_exr_reuses_its_preview(mock_create_session, roots, tmp_path):
    from app.assets import previews

    exr = write_exr(tmp_path / "src.exr", 64, 48).read_bytes()
    with patch.object(previews, "_make_preview", wraps=previews._make_preview) as make:
        async with await _client(AssetsEnabled(_Args())) as client:
            first = (await (await client.post("/upload/image", data=_form("frame.exr", exr))).json())["asset"]
            second = (await (await client.post("/upload/image", data=_form("frame.exr", exr))).json())["asset"]

    assert second["id"] != first["id"]
    assert second["preview_id"] == first["preview_id"] is not None
    assert make.call_count == 1, "the same bytes are decoded once"


@pytest.mark.asyncio
async def test_a_png_upload_is_its_own_preview(mock_create_session, roots):
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("still.png", _png()))
        asset = (await resp.json())["asset"]

    assert asset["preview_id"] == asset["id"]


@pytest.mark.asyncio
async def test_mask_upload_still_answers_with_json(mock_create_session, roots):
    (roots / "input" / "base.png").write_bytes(_png())
    original_ref = json.dumps({"filename": "base.png", "type": "input", "subfolder": ""})
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/mask", data=_form("mask.png", _png(), original_ref=original_ref))
        body = await resp.json()

    assert resp.status == 200
    assert body["name"] == "mask.png"


@pytest.mark.asyncio
async def test_with_assets_off_an_exr_upload_has_no_asset(roots):
    from app.assets.manager import NoAssets

    class _Off:
        enable_assets = False
        enable_asset_hashing = False

    async with await _client(NoAssets(_Off())) as client:
        resp = await client.post("/upload/image", data=_form("frame.exr", b"exr"))
        body = await resp.json()

    assert resp.status == 200
    assert "asset" not in body


@pytest.mark.asyncio
async def test_api_asset_uploads_get_a_generated_preview(mock_create_session, roots, tmp_path):
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


async def _assets_client():
    from aiohttp import web

    from app.assets.api import routes
    from utils.mime_types import init_mime_types

    init_mime_types()
    app = web.Application()
    app.add_routes(routes.ROUTES)
    return TestClient(TestServer(app))


@pytest.fixture
def assets_routes_on():
    from app.assets.api import routes

    with (
        patch.object(routes, "_ASSETS_ENABLED", True),
        patch.object(routes, "USER_MANAGER", MagicMock(get_request_user_id=MagicMock(return_value="default"))),
    ):
        yield


@pytest.mark.asyncio
async def test_a_multipart_exr_upload_gets_a_generated_preview(mock_create_session, roots, tmp_path, assets_routes_on):
    exr = write_exr(tmp_path / "src.exr", 64, 48).read_bytes()
    form = FormData()
    form.add_field("file", exr, filename="frame.exr")
    form.add_field("tags", json.dumps(["input"]))
    async with await _assets_client() as client:
        resp = await client.post("/api/assets", data=form)
        body = await resp.json()

    assert resp.status == 201, body
    assert body["preview_id"] not in (None, body["id"])
    assert body["preview_url"] == f"/api/assets/{body['preview_id']}/content"


@pytest.mark.asyncio
async def test_a_from_hash_exr_gets_a_generated_preview(mock_create_session, roots, assets_routes_on, session):
    from app.assets import mode
    from app.assets.database.queries.records import create_content

    class _HashingOn:
        enable_asset_hashing = True

    mode.init(_HashingOn())
    exr = write_exr(roots / "input" / "frame.exr", 64, 48)
    digest = "blake3:" + "ab" * 32
    create_content(session, str(exr), hash=digest, size_bytes=exr.stat().st_size, mtime_ns=exr.stat().st_mtime_ns)
    session.commit()
    async with await _assets_client() as client:
        resp = await client.post("/api/assets/from-hash", json={"hash": digest, "name": "frame.exr", "tags": ["input"]})
        body = await resp.json()

    assert resp.status == 201, body
    assert body["preview_id"] not in (None, body["id"]), "no sibling had a preview, so this one was generated"


def _plain_record(session, path, name, mime_type=None):
    from app.assets.database.queries.records import create_content, create_record

    record = create_record(session, create_content(session, str(path)).id, name, mime_type=mime_type)
    session.commit()
    return record.id


@pytest.mark.asyncio
async def test_the_content_route_serves_byte_ranges(mock_create_session, roots, assets_routes_on, session):
    clip = roots / "output" / "clip.mp4"
    clip.write_bytes(b"0123456789")
    asset_id = _plain_record(session, clip, "clip.mp4", "video/mp4")
    async with await _assets_client() as client:
        resp = await client.get(f"/api/assets/{asset_id}/content", headers={"Range": "bytes=0-3"})
        body = await resp.read()

    assert (resp.status, body) == (206, b"0123"), "video previews must be able to seek"


@pytest.mark.asyncio
async def test_the_content_route_never_serves_a_compressed_sibling(mock_create_session, roots, assets_routes_on, session):
    still = roots / "output" / "still.png"
    still.write_bytes(_png())
    (roots / "output" / "still.png.gz").write_bytes(b"something else entirely")
    asset_id = _plain_record(session, still, "still.png", "image/png")
    async with await _assets_client() as client:
        resp = await client.get(f"/api/assets/{asset_id}/content", headers={"Accept-Encoding": "gzip, br"}, auto_decompress=False)
        body = await resp.read()

    assert body == still.read_bytes()


@pytest.mark.asyncio
async def test_the_content_type_comes_from_the_path_when_the_name_has_none(mock_create_session, roots, assets_routes_on, session):
    still = roots / "output" / "still.png"
    still.write_bytes(_png())
    asset_id = _plain_record(session, still, "untitled")
    async with await _assets_client() as client:
        resp = await client.get(f"/api/assets/{asset_id}/content")

    assert resp.headers["Content-Type"] == "image/png"


@pytest.mark.asyncio
async def test_a_corrupt_exr_upload_still_succeeds(mock_create_session, roots):
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("broken.exr", b"not an exr"))
        asset = (await resp.json())["asset"]

    assert resp.status == 200
    assert "preview_id" not in asset and "preview_url" not in asset


@pytest.mark.parametrize(("dest", "records"), [("image", False), ("video", False), ("audio", False), ("document", True), (None, True)])
@pytest.mark.asyncio
async def test_only_reads_that_are_not_media_renders_record_access(mock_create_session, roots, assets_routes_on, session, dest, records):
    from app.assets.database.models import Asset

    still = roots / "output" / "still.png"
    still.write_bytes(_png())
    asset_id = _plain_record(session, still, "still.png", "image/png")
    async with await _assets_client() as client:
        resp = await client.get(f"/api/assets/{asset_id}/content", headers={"Sec-Fetch-Dest": dest} if dest else {})

    assert resp.status == 200
    session.expire_all()
    assert (session.get(Asset, asset_id).last_access_time is not None) == records
