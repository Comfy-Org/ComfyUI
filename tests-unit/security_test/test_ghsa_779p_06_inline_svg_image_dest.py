"""CI unit guard for the Sec-Fetch-Dest exemption to FIX #5 of GHSA-779p-m5rp-r4h4.

FIX #5 forced every SVG served by /view and the assets download route to
application/octet-stream + Content-Disposition: attachment. That blocked the
stored XSS, but it also stopped SVG node outputs and Media Assets thumbnails
from rendering, because the frontend requests them with a plain <img>.

An SVG referenced by an <img> is loaded in secure static mode: scripting and
external references are disabled, so the payload from vuln #5 cannot execute.
The attack needs the SVG to load as a document, which arrives with a different
Sec-Fetch-Dest. Browsers set that header themselves and page script cannot
override it, so renders_safely_as_image() uses it to re-allow only the <img>
case. Everything else, including a missing header, keeps the forced download.

Because the decision reads a request header, the response must also carry
Vary: Sec-Fetch-Dest and Cache-Control: no-store. Without them a cache keyed on
the URL alone can replay the inline SVG served to an <img> to a later document
navigation of the same URL, which re-enables the very XSS the exemption is
built around. The route tests at the bottom pin those headers.

server.py cannot be imported in a unit test (importing it spins up the full
PromptServer/aiohttp app and its global side effects), so the /view side is
covered by pinning the helper its closure calls.
"""

import pytest
from aiohttp import web

import folder_paths
from app.assets.api import routes as asset_routes
from app.assets.services.schemas import DownloadResolutionResult


# Every Sec-Fetch-Dest that must keep the forced download. 'document' is the
# vuln #5 attack itself; the rest either cannot execute an SVG or have no
# reason to receive one inline. None covers curl and proxies that strip the
# header.
UNSAFE_DESTS = [
    'document',
    'iframe',
    'object',
    'embed',
    'frame',
    'script',
    'style',
    'empty',
    '',
    None,
]


def test_svg_renders_inline_for_image_dest():
    assert folder_paths.renders_safely_as_image('image/svg+xml', 'image')


def test_svg_forced_to_download_for_every_other_dest():
    for dest in UNSAFE_DESTS:
        assert not folder_paths.renders_safely_as_image('image/svg+xml', dest), (
            f"SVG must not be served inline for Sec-Fetch-Dest={dest!r}. Only "
            "an <img> load is safe, everything else can reach document context"
        )


def test_missing_header_fails_closed():
    assert not folder_paths.renders_safely_as_image('image/svg+xml', None)


def test_exemption_normalises_parameters_and_casing():
    for content_type in ('IMAGE/SVG+XML', 'image/svg+xml; charset=utf-8', '  image/svg+xml  '):
        assert folder_paths.renders_safely_as_image(content_type, 'image'), (
            f"{content_type!r} is an SVG and must not be excluded from the "
            "exemption by casing or a charset parameter"
        )


def test_other_dangerous_types_are_never_exempt():
    # Only SVG is safe as an <img>. Nothing else in the blocklist may ride the
    # image dest back into an inline response.
    for content_type in ('text/html', 'text/javascript', 'text/css',
                         'application/xml', 'text/xml', 'application/xhtml+xml',
                         'image/svg+xml.html', 'application/rss+xml'):
        assert not folder_paths.renders_safely_as_image(content_type, 'image'), (
            f"{content_type!r} must stay forced-download regardless of Sec-Fetch-Dest"
        )


def test_exemption_does_not_widen_the_blocklist():
    # The exemption is a call-site gate, not a change to what counts as
    # dangerous. is_dangerous_content_type must still flag SVG on its own.
    assert folder_paths.is_dangerous_content_type('image/svg+xml')


# --- Response-level guards on the assets content route -----------------------
#
# The helper above decides correctly, but the exemption is only safe if the
# response cannot be reused across Sec-Fetch-Dest values. These mount the real
# route and assert on the headers that actually ship.

SVG_PAYLOAD = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
ASSET_ID = "00000000-0000-4000-8000-000000000001"
CONTENT_URL = f"/api/assets/{ASSET_ID}/content"


class _StubUserManager:
    def get_request_user_id(self, request):
        return "test-user"


@pytest.fixture
def asset_app(monkeypatch, tmp_path):
    """Mount the real /api/assets/{id}/content route over a stored SVG."""

    def _factory(stored_mime_type):
        svg = tmp_path / "thumb.svg"
        svg.write_bytes(SVG_PAYLOAD)

        monkeypatch.setattr(asset_routes, "_ASSETS_ENABLED", True)
        monkeypatch.setattr(asset_routes, "USER_MANAGER", _StubUserManager())
        monkeypatch.setattr(
            asset_routes,
            "resolve_asset_for_download",
            lambda reference_id: DownloadResolutionResult(
                abs_path=str(svg),
                content_type=stored_mime_type,
                download_name="thumb.svg",
            ),
        )

        app = web.Application()
        app.add_routes(asset_routes.ROUTES)
        return app

    return _factory


@pytest.mark.asyncio
async def test_inline_svg_response_is_not_cacheable_across_destinations(
    aiohttp_client, asset_app
):
    client = await aiohttp_client(asset_app("image/svg+xml"))
    resp = await client.get(
        CONTENT_URL, params={"disposition": "inline"}, headers={"Sec-Fetch-Dest": "image"}
    )

    assert resp.status == 200
    assert "image/svg+xml" in resp.headers.get("Content-Type", "").lower()
    # The load-bearing assertion: a cache must not be able to hand this inline
    # SVG to a later document navigation of the same URL.
    assert "sec-fetch-dest" in resp.headers.get("Vary", "").lower(), (
        "The response varies on Sec-Fetch-Dest but does not say so, so a cache "
        "keyed on the URL alone can replay the inline SVG into document context."
    )
    assert "no-store" in resp.headers.get("Cache-Control", "").lower()


@pytest.mark.asyncio
async def test_forced_download_response_also_declares_the_variance(
    aiohttp_client, asset_app
):
    # The attachment branch needs the same headers, in both directions: a
    # cached octet-stream replayed to an <img> re-breaks the preview this fix
    # exists to restore.
    client = await aiohttp_client(asset_app("image/svg+xml"))
    resp = await client.get(
        CONTENT_URL,
        params={"disposition": "inline"},
        headers={"Sec-Fetch-Dest": "document"},
    )

    assert resp.status == 200
    assert "application/octet-stream" in resp.headers.get("Content-Type", "").lower()
    assert "attachment" in resp.headers.get("Content-Disposition", "").lower()
    assert "sec-fetch-dest" in resp.headers.get("Vary", "").lower()
    assert "no-store" in resp.headers.get("Cache-Control", "").lower()


@pytest.mark.asyncio
async def test_parameterised_svg_mime_type_does_not_500(aiohttp_client, asset_app):
    # mime_type is uploader-supplied and unvalidated. aiohttp rejects a charset
    # in the content_type argument with ValueError, so the exempt branch must
    # strip parameters before building the response.
    client = await aiohttp_client(asset_app("image/svg+xml; charset=utf-8"))
    resp = await client.get(
        CONTENT_URL, params={"disposition": "inline"}, headers={"Sec-Fetch-Dest": "image"}
    )

    assert resp.status == 200, (
        "A charset parameter on the stored mime type must not turn a valid "
        "inline SVG request into a 500."
    )
    assert "image/svg+xml" in resp.headers.get("Content-Type", "").lower()
