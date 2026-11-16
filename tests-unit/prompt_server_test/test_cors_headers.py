"""Tests for the --enable-cors-header headers, including on responses a handler prepares itself (SSE)."""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import server  # noqa: E402

ORIGIN = "http://example.test"
HANDLER_PATHS = ["/json", "/file", "/stream"]
PATHS = HANDLER_PATHS + ["/preset"]


def cors_headers(origin=ORIGIN):
    return {
        "access-control-allow-origin": [origin],
        "access-control-allow-methods": ["POST, GET, DELETE, PUT, OPTIONS, PATCH, HEAD"],
        "access-control-allow-headers": ["Content-Type, Authorization, Idempotency-Key, Range, Comfy-User"],
        "access-control-expose-headers": ["Retry-After, Content-Range, Accept-Ranges, Content-Length, Location, No-Log-Reason"],
        "access-control-allow-credentials": ["true"],
    }


def wire_cors_headers(resp):
    """Access-Control-* headers as sent, one list entry per occurrence."""
    found = {}
    for name, value in resp.raw_headers:
        name = name.decode().lower()
        if name.startswith("access-control-"):
            found.setdefault(name, []).append(value.decode())
    return found


async def json_handler(request):
    return web.json_response({"ok": True})


async def stream_handler(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(b"data: hi\n\n")
    return resp


async def preset_handler(request):
    return web.json_response({"ok": True}, headers={"Access-Control-Allow-Origin": "http://handler.test"})


@pytest.fixture
def cors():
    return ORIGIN


@pytest_asyncio.fixture
async def client(cors, monkeypatch, tmp_path):
    monkeypatch.setattr(args, "enable_cors_header", cors)
    ps = server.PromptServer(None, MagicMock(enabled=False))
    file = tmp_path / "file.txt"
    file.write_text("file body")

    async def file_handler(request):
        return web.FileResponse(file)

    ps.app.router.add_get("/json", json_handler)
    ps.app.router.add_get("/file", file_handler)
    ps.app.router.add_get("/stream", stream_handler)
    ps.app.router.add_get("/preset", preset_handler)
    async with TestClient(TestServer(ps.app)) as c:
        yield c


def preflight(client, path):
    return client.options(path, headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                                         "Access-Control-Request-Headers": "content-type, idempotency-key"})


# Behaviour that held before the headers moved to on_response_prepare.


@pytest.mark.asyncio
async def test_json_and_preflight_get_allow_origin(client):
    resp = await preflight(client, "/json")
    assert resp.status == 200
    assert await resp.read() == b""
    assert resp.headers["Access-Control-Allow-Origin"] == ORIGIN
    assert resp.headers["Access-Control-Allow-Credentials"] == "true"

    resp = await client.get("/json", headers={"Origin": ORIGIN})
    assert resp.status == 200
    assert await resp.json() == {"ok": True}
    assert resp.headers["Access-Control-Allow-Origin"] == ORIGIN


# --enable-cors-header


@pytest.mark.asyncio
@pytest.mark.parametrize("path,status,body", [
    ("/json", 200, b'{"ok": true}'),
    ("/file", 200, b"file body"),
    ("/stream", 200, b"data: hi\n\n"),
    ("/preset", 200, b'{"ok": true}'),
    ("/missing", 404, None),
])
async def test_every_response_carries_cors_headers_once(client, path, status, body):
    resp = await client.get(path, headers={"Origin": ORIGIN})
    assert resp.status == status
    if body is not None:
        assert await resp.read() == body
    assert wire_cors_headers(resp) == cors_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS + ["/missing"])
async def test_preflight_is_empty_200_with_cors_headers_once(client, path):
    resp = await preflight(client, path)
    assert resp.status == 200
    assert await resp.read() == b""
    assert wire_cors_headers(resp) == cors_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {},
    {"Origin": "http://evil.example"},
    {"Origin": "null"},
    {"Origin": "not a url"},
    {"Origin": "http://[::1"},
    [("Origin", ORIGIN), ("Origin", "http://evil.example")],
], ids=["no-origin", "other-origin", "null-origin", "garbage-origin", "malformed-ipv6-origin", "two-origins"])
@pytest.mark.parametrize("path", ["/json", "/stream"])
async def test_allow_origin_is_the_configured_origin_never_the_request_origin(client, path, headers):
    resp = await client.get(path, headers=headers)
    assert resp.status == 200
    assert wire_cors_headers(resp) == cors_headers()


@pytest.mark.asyncio
@pytest.mark.parametrize("cors", ["*"])
@pytest.mark.parametrize("path", PATHS)
async def test_bare_flag_allows_any_origin(client, path):
    resp = await client.get(path, headers={"Origin": ORIGIN})
    assert resp.status == 200
    assert wire_cors_headers(resp) == cors_headers("*")


# CORS disabled: no headers, origin-only middleware still guards loopback.


@pytest.mark.asyncio
@pytest.mark.parametrize("cors", [None, ""])
@pytest.mark.parametrize("path", HANDLER_PATHS)
async def test_disabled_adds_no_cors_headers(client, path):
    resp = await client.get(path)
    assert resp.status == 200
    assert wire_cors_headers(resp) == {}

    resp = await client.options(path, headers={"Access-Control-Request-Method": "POST"})
    assert resp.status == 200
    assert wire_cors_headers(resp) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cors", [None, ""])
@pytest.mark.parametrize("path", HANDLER_PATHS)
async def test_disabled_rejects_cross_site_requests(client, path):
    for headers in ({"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
        resp = await client.get(path, headers=headers)
        assert resp.status == 403
        assert wire_cors_headers(resp) == {}

    resp = await client.options(path, headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert resp.status == 403

    resp = await client.get(path, headers={"Origin": f"http://{client.host}:{client.port}"})
    assert resp.status == 200
    assert wire_cors_headers(resp) == {}
