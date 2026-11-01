"""Tests for the OpenAPI spec and API docs routes."""

import pytest
import pytest_asyncio
import yaml
from aiohttp import web

from app.api_docs import add_api_docs_routes

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(aiohttp_client):
    app = web.Application()
    routes = web.RouteTableDef()
    add_api_docs_routes(routes)
    app.add_routes(routes)
    return await aiohttp_client(app)


async def test_get_spec(client):
    resp = await client.get("/openapi.yaml")
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/yaml"
    assert resp.headers["Cache-Control"] == "no-cache"
    spec = yaml.safe_load(await resp.text())
    assert spec["openapi"].startswith("3.")
    assert spec["paths"]


async def test_missing_spec_returns_404(client, monkeypatch):
    monkeypatch.setattr("app.api_docs.SPEC_PATH", "/nonexistent/openapi.yaml")
    resp = await client.get("/openapi.yaml")
    assert resp.status == 404


async def test_docs_page(client):
    resp = await client.get("/api-docs")
    assert resp.status == 200
    assert resp.content_type == "text/html"
    body = await resp.text()
    # Redoc has no request execution; Swagger UI's "Try it out" would give
    # one-click access to destructive endpoints on an unauthenticated server.
    assert "<redoc" in body
    assert "openapi.yaml" in body


async def test_docs_page_has_offline_fallback(client):
    """Offline installs should get a notice, not a blank page."""
    body = await (await client.get("/api-docs")).text()
    assert 'id="fallback"' in body


async def test_spec_url_is_relative(client):
    """What makes the page work under both /api-docs and /api/api-docs.

    server.py re-registers every route under an /api prefix; a relative spec
    URL resolves to the sibling spec from either mount point.
    """
    body = await (await client.get("/api-docs")).text()
    assert 'spec-url="openapi.yaml"' in body


async def test_flag_is_off_by_default():
    """Serving the API surface of an unauthenticated server is opt-in."""
    from comfy.cli_args import parser

    assert parser.parse_args([]).enable_api_docs is False
    assert parser.parse_args(["--enable-api-docs"]).enable_api_docs is True
