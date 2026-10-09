import io
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from app.assets.manager import AssetsEnabled
from app.assets.previews import generate_upload_preview
from comfy_api.latest import Previews
from comfy_execution import preview_generators

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
async def test_a_png_upload_is_its_own_preview(mock_create_session, roots):
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("still.png", _png()))
        asset = (await resp.json())["asset"]

    assert asset["preview_id"] == asset["id"]


class _FailingPngGenerator(Previews.PreviewGenerator):
    mime_types = ("image/png",)

    def generate(self, source_path, max_pixels):
        raise ValueError("cannot read this one")


@pytest.mark.asyncio
async def test_a_type_with_a_generator_never_falls_back_to_itself(mock_create_session, roots):
    generator = _FailingPngGenerator()
    preview_generators.register_preview_generator(generator)
    try:
        async with await _client(AssetsEnabled(_Args())) as client:
            resp = await client.post("/upload/image", data=_form("still.png", _png()))
            asset = (await resp.json())["asset"]
    finally:
        preview_generators.unregister_preview_generator(generator)

    assert asset.get("preview_id") is None
    assert asset.get("preview_url") is None


class _TgaPreview(Previews.PreviewGenerator):
    """A custom node's generator, registered the way the public docstring says."""

    mime_types = ("image/x-test-tga",)

    def generate(self, source_path, max_pixels):
        return Image.new("RGB", (6, 4), (0, 128, 255))


@pytest.mark.asyncio
async def test_an_upload_of_a_custom_type_gets_the_registered_generators_preview(mock_create_session, roots):
    import mimetypes

    from comfy_api.latest import ComfyAPI

    generator = _TgaPreview()
    mimetypes.add_type("image/x-test-tga", ".testtga")
    await ComfyAPI().previews.register_generator(generator)
    try:
        async with await _client(AssetsEnabled(_Args())) as client:
            resp = await client.post("/upload/image", data=_form("frame.testtga", b"tga bytes"))
            asset = (await resp.json())["asset"]
            preview = await client.get(asset["preview_url"])
            size = Image.open(io.BytesIO(await preview.read())).size
    finally:
        await ComfyAPI().previews.unregister_generator(generator)
        mimetypes.types_map.pop(".testtga", None)

    assert asset["preview_id"] not in (None, asset["id"])
    assert asset["preview_url"] == f"/api/assets/{asset['preview_id']}/content"
    assert size == (6, 4)


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
    async with await _client(MagicMock(enabled=False, register_upload=MagicMock(return_value=None))) as client:
        resp = await client.post("/upload/image", data=_form("frame.exr", b"exr"))
        body = await resp.json()

    assert resp.status == 200
    assert "asset" not in body


@pytest.mark.asyncio
async def test_a_preview_the_client_already_set_is_kept(tmp_path):
    from utils.mime_types import init_mime_types

    init_mime_types()
    exr = write_exr(tmp_path / "frame.exr", 8, 8)

    with patch("app.assets.previews.submit_preview_job") as submit:
        assert await generate_upload_preview("asset-id", str(exr), "client-preview") == "client-preview"
    submit.assert_not_called()


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


@pytest.mark.asyncio
async def test_put_null_preview_id_clears_the_link(mock_create_session, roots, assets_routes_on, session):
    from app.assets.database.models import Asset
    from app.assets.database.queries.records import create_content, create_record

    thumb = roots / "output" / "thumb.png"
    thumb.write_bytes(_png())
    model = roots / "output" / "m.glb"
    model.write_bytes(b"glb")
    preview = create_record(session, create_content(session, str(thumb)).id, "thumb.png")
    parent = create_record(session, create_content(session, str(model)).id, "m.glb")
    parent.preview_id = preview.id
    session.commit()
    parent_id = parent.id

    async with await _assets_client() as client:
        resp = await client.put(f"/api/assets/{parent_id}", json={"preview_id": None})

    assert resp.status == 200, await resp.text()
    session.expire_all()
    assert session.get(Asset, parent_id).preview_id is None


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


@pytest.mark.asyncio
async def test_an_upload_tagged_preview_lands_in_previews(mock_create_session, roots, assets_routes_on):
    form = FormData()
    form.add_field("file", _png(), filename="thumb.png")
    form.add_field("tags", json.dumps(["preview"]))
    async with await _assets_client() as client:
        resp = await client.post("/api/assets", data=form)

    assert resp.status == 201, await resp.text()
    assert [p.suffix for p in (roots / "previews").iterdir()] == [".png"]


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
