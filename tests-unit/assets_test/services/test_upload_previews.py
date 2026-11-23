import io
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from app.assets.manager import AssetsEnabled

from .preview_helpers import duplicate_data_window, write_exr


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


async def _upload(route: str, exr: bytes, roots, session) -> dict:
    if route == "/upload/image":
        async with await _client(AssetsEnabled(_Args())) as client:
            asset = (await (await client.post(route, data=_form("frame.exr", exr))).json())["asset"]
            return asset | {"preview": await (await client.get(asset["preview_url"])).read()}
    async with await _assets_client() as client:
        if route == "multipart":
            form = FormData()
            form.add_field("file", exr, filename="frame.exr")
            form.add_field("tags", json.dumps(["input"]))
            resp = await client.post("/api/assets", data=form)
        else:  # from-hash, over content a previous run already stored
            from app.assets import mode
            from app.assets.database.queries.records import create_content

            class _HashingOn:
                enable_asset_hashing = True

            mode.init(_HashingOn())
            path = roots / "input" / "frame.exr"
            path.write_bytes(exr)
            digest = "blake3:" + "ab" * 32
            create_content(session, str(path), hash=digest, size_bytes=len(exr), mtime_ns=path.stat().st_mtime_ns)
            session.commit()
            resp = await client.post("/api/assets/from-hash", json={"hash": digest, "name": "frame.exr", "tags": ["input"]})
        asset = await resp.json()
        assert resp.status == 201, asset
        return asset | {"preview": await (await client.get(asset["preview_url"])).read()}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/upload/image", "multipart", "from-hash"])
@pytest.mark.parametrize(("value", "mode"), [((0.5, 0.5, 0.5), "RGB"), ((0.5, 0.5, 0.5, 0.25), "RGBA")])
async def test_an_uploaded_exr_comes_back_with_its_preview(mock_create_session, roots, tmp_path, assets_routes_on, session, route, value, mode):
    from app.assets import previews

    exr = write_exr(tmp_path / "src.exr", 64, 48, value=value).read_bytes()

    with patch.object(previews, "_make_preview", wraps=previews._make_preview) as make:
        asset = await _upload(route, exr, roots, session)
        if route == "/upload/image":
            again = await _upload(route, exr, roots, session)
            assert again["preview_id"] == asset["preview_id"], "the same bytes reuse their preview"
            assert make.call_count == 1, "and aren't decoded again"

    assert asset["preview_id"] not in (None, asset["id"])
    assert asset["preview_url"] == f"/api/assets/{asset['preview_id']}/content"
    preview = Image.open(io.BytesIO(asset["preview"]))
    assert (preview.format, preview.mode, preview.size) == ("WEBP", mode, (64, 48)), "alpha is kept"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [b"not an exr", "duplicated window"])
async def test_an_exr_that_cant_be_previewed_still_uploads(mock_create_session, roots, tmp_path, bad):
    data = bad if isinstance(bad, bytes) else duplicate_data_window(write_exr(tmp_path / "a.exr", 64, 48).read_bytes())
    async with await _client(AssetsEnabled(_Args())) as client:
        resp = await client.post("/upload/image", data=_form("broken.exr", data))
        asset = (await resp.json())["asset"]

    assert resp.status == 200
    assert "preview_id" not in asset and "preview_url" not in asset


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
async def test_clients_cannot_create_a_preview_tagged_asset(mock_create_session, roots, tmp_path, assets_routes_on):
    form = FormData()
    form.add_field("file", b"png", filename="mine.png")
    form.add_field("tags", json.dumps(["input", "preview"]))
    async with await _assets_client() as client:
        upload = await client.post("/api/assets", data=form)
        from_hash = await client.post("/api/assets/from-hash", json={"hash": "blake3:" + "ab" * 32, "tags": ["input", "preview"]})
        bodies = [await upload.json(), await from_hash.json()]

    assert [upload.status, from_hash.status] == [400, 400]
    assert [body["error"]["code"] for body in bodies] == ["SYSTEM_TAG_FORBIDDEN"] * 2
