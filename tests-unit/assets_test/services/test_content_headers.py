"""Pins the response headers of GET /api/assets/{id}/content: content type, disposition,
nosniff and the dangerous-type downgrade (see GHSA-779p tests in security_test)."""

import pytest
from aiohttp import web

from app.assets.api import routes as asset_routes
from app.assets.services.schemas import DownloadResolutionResult

ASSET_ID = "00000000-0000-4000-8000-000000000002"
URL = f"/api/assets/{ASSET_ID}/content"
BODY = b"stored bytes"
HEADERS = ("Content-Type", "Content-Disposition", "Content-Length", "X-Content-Type-Options", "Vary", "Cache-Control")


class _StubUserManager:
    def get_request_user_id(self, request):
        return "test-user"


@pytest.fixture
def content_client(aiohttp_client, monkeypatch, tmp_path):
    async def make(content_type, download_name="file.bin"):
        path = tmp_path / "stored"
        path.write_bytes(BODY)
        monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
        monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
        monkeypatch.setattr(
            asset_routes,
            "resolve_asset_for_download",
            lambda reference_id, **_: DownloadResolutionResult(abs_path=str(path), content_type=content_type, download_name=download_name),
        )
        app = web.Application()
        app.add_routes(asset_routes.ROUTES)
        return await aiohttp_client(app)

    return make


async def fetch(client, params=None, dest=None):
    resp = await client.get(URL, params=params or {}, headers={"Sec-Fetch-Dest": dest} if dest else {})
    assert resp.status == 200
    assert await resp.read() == BODY
    return {h: resp.headers.get(h) for h in HEADERS}


def expected(content_type, disposition, dangerous=False):
    return {
        "Content-Type": content_type,
        "Content-Disposition": disposition,
        "Content-Length": str(len(BODY)),
        "X-Content-Type-Options": "nosniff",
        "Vary": "Sec-Fetch-Dest" if dangerous else None,
        "Cache-Control": "no-store" if dangerous else None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params,disposition",
    [(None, "attachment"), ({"disposition": "inline"}, "inline"), ({"disposition": " INLINE "}, "inline"), ({"disposition": "bogus"}, "attachment")],
)
async def test_image_keeps_type_and_requested_disposition(content_client, params, disposition):
    client = await content_client("image/png", "photo.png")
    assert await fetch(client, params) == expected("image/png", f"{disposition}; filename*=UTF-8''photo.png")


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/html", "image/svg+xml", "application/xhtml+xml", "TEXT/HTML; charset=utf-8"])
@pytest.mark.parametrize("dest", [None, "document"])
async def test_dangerous_type_forced_to_download(content_client, content_type, dest):
    client = await content_client(content_type, "page.html")
    got = await fetch(client, {"disposition": "inline"}, dest)
    assert got == expected("application/octet-stream", "attachment; filename*=UTF-8''page.html", dangerous=True)


@pytest.mark.asyncio
async def test_svg_inline_for_image_dest(content_client):
    client = await content_client("image/svg+xml; charset=utf-8", "t.svg")
    got = await fetch(client, {"disposition": "inline"}, "image")
    assert got == expected("image/svg+xml", "inline; filename*=UTF-8''t.svg", dangerous=True)


@pytest.mark.asyncio
async def test_parameters_stripped_and_blank_type_falls_back(content_client):
    client = await content_client("image/png; name=x")
    assert (await fetch(client))["Content-Type"] == "image/png"
    client = await content_client("; charset=utf-8")
    assert (await fetch(client))["Content-Type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_filename_is_percent_encoded_without_line_breaks(content_client):
    client = await content_client("image/png", "a\r\nb ü\"x.png")
    assert (await fetch(client))["Content-Disposition"] == "attachment; filename*=UTF-8''ab%20%C3%BC%22x.png"
