import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from comfy_api.billing_capabilities import relay_billing_capabilities


pytestmark = pytest.mark.asyncio

AUTH_HEADERS = {"Authorization": "Bearer test-token"}
CAPABILITY_BODY = {
    "resolved_for": {"user_id": "user-1", "workspace_id": "workspace-1"},
    "capabilities": {
        "can_subscribe_self_serve": True,
        "can_top_up": True,
        "can_cancel": False,
        "can_reactivate": False,
        "can_change_seats": False,
        "can_invite_members": True,
        "can_downgrade_to_personal": False,
    },
    "rollout_defaults_applied": {
        "can_downgrade_to_personal": False,
        "can_subscribe_self_serve": False,
        "can_top_up": False,
    },
    "revision": 42,
    "expires_at": "2026-09-08T00:00:00Z",
}


async def _relay_client(aiohttp_client, monkeypatch, upstream_url):
    monkeypatch.setattr(
        "comfy_api.billing_capabilities.args.comfy_api_base", upstream_url
    )
    session = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())
    app = web.Application()

    async def cleanup(_app):
        await session.close()

    async def relay_handler(request):
        return await relay_billing_capabilities(request, session)

    app.on_cleanup.append(cleanup)
    app.router.add_get("/api/billing/capabilities", relay_handler)
    return await aiohttp_client(app)


async def test_relay_uses_fixed_endpoint_and_header_allowlist(
    aiohttp_client, aiohttp_server, monkeypatch
):
    received = {}

    async def upstream_handler(request):
        received["path"] = request.path
        received["headers"] = request.headers
        return web.json_response(CAPABILITY_BODY)

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    monkeypatch.setattr(
        "comfy_api.billing_capabilities.get_deploy_environment", lambda: "local-test"
    )
    monkeypatch.setattr("comfy_api.billing_capabilities.comfyui_version", "1.2.3")
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get(
        "/api/billing/capabilities",
        headers={
            "Authorization": "Bearer secret",
            "X-API-Key": "api-secret",
            "Cookie": "session=secret",
            "Comfy-Env": "browser-value",
            "Comfy-Core-Version": "browser-version",
            "X-Forwarded-Host": "example.com",
        },
    )

    assert response.status == 200
    assert received["path"] == "/api/billing/capabilities"
    assert received["headers"]["Authorization"] == "Bearer secret"
    assert received["headers"]["X-API-Key"] == "api-secret"
    assert received["headers"]["Accept"] == "application/json"
    assert received["headers"]["Comfy-Env"] == "local-test"
    assert received["headers"]["Comfy-Core-Version"] == "1.2.3"
    assert "Cookie" not in received["headers"]
    assert "X-Forwarded-Host" not in received["headers"]


@pytest.mark.parametrize("status", [200, 401, 403, 404, 502])
async def test_relay_preserves_expected_responses(
    aiohttp_client, aiohttp_server, monkeypatch, status
):
    body = (
        CAPABILITY_BODY
        if status == 200
        else {"code": "UPSTREAM_ERROR", "message": f"status {status}"}
    )

    async def upstream_handler(_request):
        return web.json_response(
            body,
            status=status,
            headers={
                "Cache-Control": "private, max-age=30",
                "Vary": "Authorization",
                "X-Capability-Revision": "42",
            },
        )

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == status
    assert await response.json() == body
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert response.headers["Vary"] == "Authorization, X-API-Key"
    assert response.headers["X-Capability-Revision"] == "42"
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"


@pytest.mark.parametrize(
    ("status", "body", "content_type"),
    [
        (500, b'{"internal": "detail"}', "application/json"),
        (200, b"not json", "text/plain"),
        (200, json.dumps(CAPABILITY_BODY).encode(), "text/plain"),
        (200, b"[]", "application/json"),
        (200, b"null", "application/json"),
        (200, b'\xff\xfe{"resolved_for": {}}', "application/json"),
    ],
)
async def test_relay_converts_invalid_upstream_responses_to_502(
    aiohttp_client, aiohttp_server, monkeypatch, status, body, content_type
):
    async def upstream_handler(_request):
        return web.Response(status=status, body=body, content_type=content_type)

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 502
    assert await response.json() == {"error": "Billing capabilities unavailable"}


async def test_relay_refuses_to_follow_upstream_redirects(
    aiohttp_client, aiohttp_server, monkeypatch
):
    redirect_target_requests = []

    async def redirect_target_handler(request):
        redirect_target_requests.append(request.headers)
        return web.json_response(CAPABILITY_BODY)

    target_app = web.Application()
    target_app.router.add_get("/api/billing/capabilities", redirect_target_handler)
    target = await aiohttp_server(target_app)

    async def upstream_handler(_request):
        raise web.HTTPFound(str(target.make_url("/api/billing/capabilities")))

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get(
        "/api/billing/capabilities",
        headers={"Authorization": "Bearer secret", "X-API-Key": "api-secret"},
    )

    assert response.status == 502
    assert await response.json() == {"error": "Billing capabilities unavailable"}
    assert redirect_target_requests == []


async def test_relay_converts_connection_failure_to_502(
    aiohttp_client, monkeypatch, unused_tcp_port
):
    client = await _relay_client(
        aiohttp_client, monkeypatch, f"http://127.0.0.1:{unused_tcp_port}"
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 502
    assert await response.json() == {"error": "Billing capabilities unavailable"}


async def test_relay_converts_upstream_timeout_to_502(
    aiohttp_client, aiohttp_server, monkeypatch
):
    async def upstream_handler(_request):
        await asyncio.sleep(5)
        return web.json_response(CAPABILITY_BODY)

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    monkeypatch.setattr(
        "comfy_api.billing_capabilities._TIMEOUT", aiohttp.ClientTimeout(total=0.05)
    )
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 502
    assert await response.json() == {"error": "Billing capabilities unavailable"}


async def test_relay_drops_unlisted_upstream_response_headers(
    aiohttp_client, aiohttp_server, monkeypatch
):
    async def upstream_handler(_request):
        return web.json_response(
            CAPABILITY_BODY,
            headers={
                "Cache-Control": "private, max-age=30",
                "Set-Cookie": "upstream_session=secret",
                "X-Upstream-Internal": "internal-detail",
                "Access-Control-Allow-Origin": "https://upstream.example",
            },
        )

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 200
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert "Set-Cookie" not in response.headers
    assert "X-Upstream-Internal" not in response.headers
    assert "Access-Control-Allow-Origin" not in response.headers
    assert response.cookies == {}


@pytest.mark.parametrize(
    "request_headers", [{}, {"Authorization": "", "X-API-Key": ""}]
)
async def test_relay_rejects_absent_auth_without_contacting_upstream(
    aiohttp_client, aiohttp_server, monkeypatch, request_headers
):
    requests = []

    async def upstream_handler(request):
        requests.append(request)
        return web.json_response(CAPABILITY_BODY)

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=request_headers)

    assert response.status == 401
    assert await response.json() == {"error": "Authentication required"}
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["Vary"] == "Authorization, X-API-Key"
    assert requests == []


@pytest.mark.parametrize(
    ("status", "body", "content_type"),
    [
        (401, b"", None),
        (403, b"<h1>Access denied</h1>", "text/html"),
        (404, b"not found", "text/plain"),
    ],
)
async def test_relay_preserves_non_json_error_responses(
    aiohttp_client, aiohttp_server, monkeypatch, status, body, content_type
):
    async def upstream_handler(_request):
        return web.Response(status=status, body=body, content_type=content_type)

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == status
    assert await response.read() == body


@pytest.mark.parametrize("status", [429, 503])
async def test_relay_preserves_upstream_backpressure(
    aiohttp_client, aiohttp_server, monkeypatch, status
):
    async def upstream_handler(_request):
        return web.Response(
            status=status,
            text="try later",
            content_type="text/plain",
            headers={"Retry-After": "120"},
        )

    upstream_app = web.Application()
    upstream_app.router.add_get("/api/billing/capabilities", upstream_handler)
    upstream = await aiohttp_server(upstream_app)
    client = await _relay_client(
        aiohttp_client, monkeypatch, str(upstream.make_url("/"))
    )

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == status
    assert await response.text() == "try later"
    assert response.headers["Retry-After"] == "120"


async def test_relay_converts_malformed_base_to_502(aiohttp_client, monkeypatch):
    client = await _relay_client(aiohttp_client, monkeypatch, "http://[::1")

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 502
    assert await response.json() == {"error": "Billing capabilities unavailable"}


async def test_relay_requires_non_persistent_cookie_jar(aiohttp_client, monkeypatch):
    monkeypatch.setattr(
        "comfy_api.billing_capabilities.args.comfy_api_base", "http://127.0.0.1"
    )
    session = aiohttp.ClientSession()
    app = web.Application()

    async def cleanup(_app):
        await session.close()

    async def relay_handler(request):
        return await relay_billing_capabilities(request, session)

    app.on_cleanup.append(cleanup)
    app.router.add_get("/api/billing/capabilities", relay_handler)
    client = await aiohttp_client(app)

    response = await client.get("/api/billing/capabilities", headers=AUTH_HEADERS)

    assert response.status == 500
