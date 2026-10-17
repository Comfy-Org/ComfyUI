"""Tests for the origin-only middleware in server.py"""

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from server import create_cors_middleware, create_origin_only_middleware

pytestmark = pytest.mark.asyncio

# is_loopback() resolves names, so use literals to keep these offline.
LOOPBACK_HOST = "127.0.0.1:8188"
PUBLIC_HOST = "203.0.113.10:8188"


async def ok_handler(request):
    return web.Response(status=200)


async def status_for(middleware, method, host, **headers):
    request = make_mocked_request(method, "/", headers={"Host": host, **headers})
    response = await middleware(request, ok_handler)
    return response.status


@pytest.fixture
def middleware():
    return create_origin_only_middleware()


@pytest.mark.parametrize("host", [LOOPBACK_HOST, PUBLIC_HOST])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_cross_site_navigation_allowed(middleware, host, method):
    """Following a link here from another site is a navigation, not the attack this blocks."""
    assert await status_for(
        middleware, method, host,
        **{"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"},
    ) == 200


@pytest.mark.parametrize("host", [LOOPBACK_HOST, PUBLIC_HOST])
@pytest.mark.parametrize("method,sec_fetch_mode", [
    ("POST", "navigate"),  # form POST, which is also a navigation
    ("POST", "cors"),      # fetch() / XHR
    ("GET", "no-cors"),    # <img>, <script>
    ("GET", "cors"),       # fetch() / XHR
    ("OPTIONS", "cors"),   # preflight
])
async def test_cross_site_requests_blocked(middleware, host, method, sec_fetch_mode):
    assert await status_for(
        middleware, method, host,
        **{"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": sec_fetch_mode},
    ) == 403


@pytest.mark.parametrize("sec_fetch_site", ["none", "same-origin", "same-site"])
async def test_other_sec_fetch_site_values_untouched(middleware, sec_fetch_site):
    assert await status_for(
        middleware, "GET", PUBLIC_HOST, **{"Sec-Fetch-Site": sec_fetch_site}
    ) == 200


async def test_request_without_the_header_untouched(middleware):
    assert await status_for(middleware, "POST", PUBLIC_HOST) == 200


async def test_loopback_host_origin_mismatch_still_blocked(middleware):
    """The Host/Origin check below the Sec-Fetch-Site one is unrelated and must keep working."""
    assert await status_for(
        middleware, "POST", LOOPBACK_HOST,
        Origin="http://evil.example",
        **{"Sec-Fetch-Site": "same-origin"},
    ) == 403


async def test_cors_middleware_path_has_no_cross_site_check():
    """--enable-cors-header swaps this middleware out for the CORS one."""
    assert await status_for(
        create_cors_middleware("*"), "POST", PUBLIC_HOST,
        **{"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"},
    ) == 200
