"""getHealth, the identity helper's 403 envelope, and that v2 exists only once registered."""

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.api_v2 import routes

from .contract import OPERATIONS, ROUTE_EXTENSIONS, validate


@pytest_asyncio.fixture
async def whoami_client(v2_app):
    async def whoami(request):
        return web.json_response({"user": routes.user_id(request)})

    v2_app.router.add_get("/test/whoami", whoami, allow_head=False)
    async with TestClient(TestServer(v2_app)) as client:
        yield client


@pytest.mark.asyncio
async def test_health(v2_client):
    resp = await v2_client.get("/api/v2/health")
    assert resp.status == 200
    body = await resp.json()
    assert body == {"status": "healthy"}
    validate("Health", body)


@pytest.mark.asyncio
async def test_health_ignores_unknown_user(v2_client, fake_server):
    fake_server.user_manager.multi_user = True
    resp = await v2_client.get("/api/v2/health", headers={"comfy-user": "nobody"})
    assert resp.status == 200
    validate("Health", await resp.json())


@pytest.mark.asyncio
async def test_health_has_no_implicit_head(v2_client):
    assert (await v2_client.head("/api/v2/health")).status == 405


@pytest.mark.asyncio
async def test_unknown_user_is_forbidden(whoami_client, fake_server):
    fake_server.user_manager.multi_user = True
    resp = await whoami_client.get("/test/whoami", headers={"comfy-user": "nobody"})
    assert resp.status == 403
    body = await resp.json()
    validate("ErrorEnvelope", body)
    assert body["error"]["code"] == "forbidden"

    resp = await whoami_client.get("/test/whoami")
    assert resp.status == 200
    assert await resp.json() == {"user": "default"}


@pytest.mark.asyncio
async def test_unregistered_app_has_no_v2_routes():
    async with TestClient(TestServer(web.Application())) as client:
        for method, path in sorted(OPERATIONS | ROUTE_EXTENSIONS):
            url = path.replace("{id}", "00000000-0000-4000-8000-000000000000").replace("{hash}", "blake3:" + "0" * 64)
            resp = await client.request(method, url)
            assert resp.status == 404, (method, path)
