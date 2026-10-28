"""Tests for the cross-site check in the origin-only middleware"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import server

pytestmark = pytest.mark.asyncio

NAVIGATION = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
FETCH = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty"}


async def ok(request):
    return web.Response(text="ok")


@pytest.mark.parametrize("method,path,headers,expected", [
    ("GET", "/?code=x&state=y", NAVIGATION, 200),
    ("GET", "/", {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}, 200),
    ("GET", "/", FETCH, 403),
    ("GET", "/", {**NAVIGATION, "Sec-Fetch-Dest": "iframe"}, 403),
    ("GET", "/api/queue", NAVIGATION, 403),
    ("POST", "/", NAVIGATION, 403),
])
async def test_cross_site_requests(method, path, headers, expected):
    app = web.Application(middlewares=[server.create_origin_only_middleware()])
    app.router.add_route("*", "/", ok)
    app.router.add_route("*", "/api/queue", ok)
    async with TestClient(TestServer(app)) as client:
        resp = await client.request(method, path, headers=headers)
        assert resp.status == expected
