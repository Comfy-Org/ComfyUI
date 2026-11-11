"""
CI unit tests for FIX #4 of GHSA-779p-m5rp-r4h4.

Stored-XSS hardening on GET /userdata/{file} in app/user_manager.py.

User data files are arbitrary user-supplied content and must never render
inline in the app origin. The getuserdata handler:
  - forces Content-Type to application/octet-stream for any type in
    folder_paths.DANGEROUS_CONTENT_TYPES (text/html, image/svg+xml,
    text/javascript, ...),
  - sets X-Content-Type-Options: nosniff,
  - sets Content-Disposition: attachment.

These tests pre-create files in tmp_path and GET them back, asserting the
secure response headers. They mirror the aiohttp_client pattern in
tests-unit/prompt_server_test/user_manager_test.py.
"""

import pytest
import os
from aiohttp import web
from app.user_manager import UserManager

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    um.get_request_user_filepath = lambda req, file, **kwargs: os.path.join(
        tmp_path, file
    ) if file else tmp_path
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_html_served_as_octet_stream(aiohttp_client, app, tmp_path):
    (tmp_path / "evil.html").write_text(
        "<script>console.log('xss-marker-ghsa-779p')</script>"
    )

    client = await aiohttp_client(app)
    resp = await client.get("/userdata/evil.html")

    assert resp.status == 200
    ct = resp.headers.get("Content-Type", "")
    # The load-bearing assertion: a .html file must NOT be served as text/html.
    assert "text/html" not in ct.lower(), (
        f"Content-Type {ct!r} would let a browser render/execute the file (stored XSS)."
    )
    assert ct == "application/octet-stream"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in resp.headers.get("Content-Disposition", "")


async def test_svg_served_as_octet_stream(aiohttp_client, app, tmp_path):
    (tmp_path / "evil.svg").write_text(
        '<?xml version="1.0"?>'
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<script>console.log("xss-marker-ghsa-779p")</script>'
        "</svg>"
    )

    client = await aiohttp_client(app)
    resp = await client.get("/userdata/evil.svg")

    assert resp.status == 200
    ct = resp.headers.get("Content-Type", "")
    # SVG can carry inline <script>; it must not be served as image/svg+xml.
    assert "svg" not in ct.lower(), (
        f"Content-Type {ct!r} would let a browser render the SVG and execute embedded scripts."
    )
    assert ct == "application/octet-stream"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in resp.headers.get("Content-Disposition", "")


async def test_js_served_as_octet_stream(aiohttp_client, app, tmp_path):
    (tmp_path / "evil.js").write_text("alert('xss-marker-ghsa-779p')")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata/evil.js")

    assert resp.status == 200
    ct = resp.headers.get("Content-Type", "").lower()
    # Must not be served as any executable JavaScript content type.
    assert "javascript" not in ct, (
        f"Content-Type {ct!r} is an executable JS type."
    )
    assert "ecmascript" not in ct, (
        f"Content-Type {ct!r} is an executable JS type."
    )
    assert ct == "application/octet-stream"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in resp.headers.get("Content-Disposition", "")


async def test_xml_dialect_served_as_octet_stream(aiohttp_client, app, tmp_path):
    """An XML dialect outside the original blocklist (.xslt -> application/xslt+xml)
    must still be forced to download. This pins the normalised *+xml family rule
    in folder_paths.is_dangerous_content_type(); a plain set-membership test would
    have served this inline."""
    (tmp_path / "evil.xslt").write_text(
        '<?xml version="1.0"?>'
        '<xsl:stylesheet version="1.0" '
        'xmlns:xsl="http://www.w3.org/1999/XSL/Transform">'
        "<!-- xss-marker-ghsa-779p -->"
        "</xsl:stylesheet>"
    )

    client = await aiohttp_client(app)
    resp = await client.get("/userdata/evil.xslt")

    assert resp.status == 200
    ct = resp.headers.get("Content-Type", "")
    assert ct == "application/octet-stream", (
        f"Content-Type {ct!r}: an *+xml dialect must be forced to octet-stream "
        f"(it can carry inline script via stylesheet/entity tricks)."
    )
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in resp.headers.get("Content-Disposition", "")


async def test_benign_txt_still_served(aiohttp_client, app, tmp_path):
    (tmp_path / "note.txt").write_text("just a harmless note")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata/note.txt")

    assert resp.status == 200
    assert await resp.text() == "just a harmless note"
    ct = resp.headers.get("Content-Type", "")
    # text/plain is not in the dangerous set, so it is acceptable here. The
    # defence-in-depth headers must still be present regardless.
    assert "text/plain" in ct.lower()
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "attachment" in resp.headers.get("Content-Disposition", "")
