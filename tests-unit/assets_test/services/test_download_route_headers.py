"""GET /api/assets/{id}/content on the real route: headers, body, and the lookup off the event loop."""

import threading

import pytest
from aiohttp import web

from app.assets.api import routes as asset_routes
from app.assets.services.schemas import DownloadResolutionResult

ASSET_ID = "00000000-0000-4000-8000-000000000002"
CONTENT_URL = f"/api/assets/{ASSET_ID}/content"
PAYLOAD = bytes(range(256)) * 64


class _StubUserManager:
    def get_request_user_id(self, request):
        return "test-user"


@pytest.fixture
def asset_app(monkeypatch, tmp_path):
    def _factory(file_name, stored_mime_type, download_name):
        path = tmp_path / file_name
        path.write_bytes(PAYLOAD)

        monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
        monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
        monkeypatch.setattr(asset_routes, "touch_record_access_time", lambda reference_id: None)
        monkeypatch.setattr(
            asset_routes,
            "resolve_asset_for_download",
            lambda reference_id: DownloadResolutionResult(
                abs_path=str(path),
                content_type=stored_mime_type,
                download_name=download_name,
            ),
        )

        app = web.Application()
        app.add_routes(asset_routes.ROUTES)
        return app

    return _factory


@pytest.mark.asyncio
async def test_dangerous_type_is_forced_to_download(aiohttp_client, asset_app):
    client = await aiohttp_client(asset_app("page.html", "text/html; charset=utf-8", "page.html"))
    resp = await client.get(CONTENT_URL, params={"disposition": "inline"}, headers={"Sec-Fetch-Dest": "document"})

    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/octet-stream"
    assert resp.headers["Content-Disposition"] == "attachment; filename*=UTF-8''page.html"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Vary"] == "Sec-Fetch-Dest"
    assert resp.headers["Cache-Control"] == "no-store"
    assert await resp.read() == PAYLOAD


@pytest.mark.asyncio
async def test_normal_type_is_served_as_stored(aiohttp_client, asset_app):
    # The stored type is served, not one derived from the file's extension.
    client = await aiohttp_client(asset_app("blob.bin", "image/png", "render.png"))
    resp = await client.get(CONTENT_URL, params={"disposition": "inline"})

    assert resp.status == 200
    assert resp.headers["Content-Type"] == "image/png"
    assert resp.headers["Content-Disposition"] == "inline; filename*=UTF-8''render.png"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Content-Length"] == str(len(PAYLOAD))
    assert "Vary" not in resp.headers
    assert "Cache-Control" not in resp.headers
    assert await resp.read() == PAYLOAD


@pytest.mark.asyncio
async def test_resolve_runs_off_the_event_loop_thread(aiohttp_client, monkeypatch, tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PAYLOAD)
    seen = []

    def resolve(reference_id):
        seen.append(threading.current_thread())
        return DownloadResolutionResult(abs_path=str(path), content_type="image/png", download_name="a.png")

    monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
    monkeypatch.setattr(asset_routes, "touch_record_access_time", lambda reference_id: None)
    monkeypatch.setattr(asset_routes, "resolve_asset_for_download", resolve)
    monkeypatch.setattr(asset_routes, "is_memory_db", lambda: False)
    app = web.Application()
    app.add_routes(asset_routes.ROUTES)
    client = await aiohttp_client(app)

    resp = await client.get(CONTENT_URL)

    assert resp.status == 200
    assert seen and seen[0] is not threading.main_thread()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (ValueError("missing"), 404, "ASSET_NOT_FOUND"),
        (FileNotFoundError("gone"), 404, "FILE_NOT_FOUND"),
        (NotImplementedError("remote"), 501, "BACKEND_UNSUPPORTED"),
    ],
)
async def test_resolve_errors_keep_their_status(aiohttp_client, monkeypatch, error, status, code):
    def resolve(reference_id):
        raise error

    monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
    monkeypatch.setattr(asset_routes, "touch_record_access_time", lambda reference_id: None)
    monkeypatch.setattr(asset_routes, "resolve_asset_for_download", resolve)
    app = web.Application()
    app.add_routes(asset_routes.ROUTES)
    client = await aiohttp_client(app)

    resp = await client.get(CONTENT_URL)

    assert resp.status == status
    assert code in await resp.text()


@pytest.mark.asyncio
async def test_memory_db_resolves_inline(aiohttp_client, monkeypatch, tmp_path):
    # An in-memory database is one shared connection, so lookups stay serialized on the loop.
    path = tmp_path / "a.png"
    path.write_bytes(PAYLOAD)
    seen = []

    def resolve(reference_id):
        seen.append(threading.current_thread())
        return DownloadResolutionResult(abs_path=str(path), content_type="image/png", download_name="a.png")

    monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
    monkeypatch.setattr(asset_routes, "touch_record_access_time", lambda reference_id: None)
    monkeypatch.setattr(asset_routes, "resolve_asset_for_download", resolve)
    monkeypatch.setattr(asset_routes, "is_memory_db", lambda: True)
    app = web.Application()
    app.add_routes(asset_routes.ROUTES)
    client = await aiohttp_client(app)

    resp = await client.get(CONTENT_URL)

    assert resp.status == 200
    assert seen == [threading.main_thread()]
