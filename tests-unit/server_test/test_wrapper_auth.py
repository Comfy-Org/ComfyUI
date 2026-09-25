"""Tests for the WRAPPER_AUTH_TOKEN bearer auth middleware"""

import time

import aiohttp
import pytest
from aiohttp import web

from middleware.wrapper_auth import (
    AUTH_TOKEN_ENV,
    LOGIN_MAX_TTL_SECONDS,
    LOGIN_PARAM,
    LOGIN_PURPOSE,
    SESSION_COOKIE,
    SESSION_PURPOSE,
    SESSION_TTL_SECONDS,
    UNAUTHORIZED_BODY,
    bearer_auth_middleware_from_env,
    create_bearer_auth_middleware,
    signed_value,
)

TOKEN = "s3cret-token"
GOOD = {"Authorization": f"Bearer {TOKEN}"}


def make_app(auth_middleware, static_dir, calls):
    """A tiny stand-in for PromptServer (whose import is heavy): native, wrapper,
    websocket, sub-app and static routes, with the auth middleware first like
    server.py puts it."""

    @web.middleware
    async def downstream(request, handler):
        # Stands in for server.py's CORS / origin-only middleware, which answer
        # preflights themselves, and records what got past the auth middleware.
        calls.append(request.path)
        if request.method == "OPTIONS":
            return web.Response()
        return await handler(request)

    middlewares = [downstream]
    if auth_middleware is not None:
        middlewares.insert(0, auth_middleware)
    app = web.Application(middlewares=middlewares)

    async def ok(request):
        return web.json_response({"ok": True})

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str("hello")
        await ws.close()
        return ws

    app.router.add_get("/queue", ok)
    app.router.add_post("/prompt", ok)
    app.router.add_get("/api/wrapper/workflows", ok)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/ws", websocket_handler)

    internal = web.Application()
    internal.router.add_get("/logs", ok)
    app.add_subapp("/internal", internal)

    (static_dir / "index.html").write_text("<html></html>")
    app.router.add_static("/", static_dir)
    return app


async def assert_unauthorized(response):
    assert response.status == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.content_type == "application/json"
    body = await response.json()
    assert body == {"error": {"type": "unauthorized", "message": "missing or invalid bearer token"}}
    assert body == UNAUTHORIZED_BODY


class TestFromEnv:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)
        assert bearer_auth_middleware_from_env() is None

    def test_empty_returns_none(self, monkeypatch):
        monkeypatch.setenv(AUTH_TOKEN_ENV, "")
        assert bearer_auth_middleware_from_env() is None

    def test_set_returns_middleware(self, monkeypatch):
        monkeypatch.setenv(AUTH_TOKEN_ENV, TOKEN)
        assert bearer_auth_middleware_from_env() is not None

    def test_factory_rejects_empty_token(self):
        with pytest.raises(ValueError):
            create_bearer_auth_middleware("")


@pytest.mark.asyncio
class TestAuthDisabled:
    async def test_unset_env_leaves_every_route_open(self, aiohttp_client, monkeypatch, tmp_path):
        monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)
        client = await aiohttp_client(make_app(bearer_auth_middleware_from_env(), tmp_path, []))

        for path in ("/queue", "/api/wrapper/workflows", "/internal/logs", "/index.html"):
            response = await client.get(path)
            assert response.status == 200, path
        ws = await client.ws_connect("/ws")
        assert await ws.receive_str() == "hello"
        await ws.close()


@pytest.mark.asyncio
class TestAuthEnabled:
    @pytest.fixture
    def calls(self):
        return []

    @pytest.fixture
    def app(self, monkeypatch, tmp_path, calls):
        monkeypatch.setenv(AUTH_TOKEN_ENV, TOKEN)
        return make_app(bearer_auth_middleware_from_env(), tmp_path, calls)

    @pytest.mark.parametrize("path", ["/queue", "/api/wrapper/workflows", "/internal/logs", "/index.html", "/missing"])
    async def test_missing_header_is_401(self, aiohttp_client, app, path):
        client = await aiohttp_client(app)
        await assert_unauthorized(await client.get(path))

    @pytest.mark.parametrize(
        "authorization",
        [
            "Bearer wrong-token",
            f"Bearer {TOKEN}x",
            f"Bearer {TOKEN[:-1]}",
            "Bearer",
            f"Basic {TOKEN}",
            TOKEN,
        ],
    )
    async def test_wrong_credentials_are_401(self, aiohttp_client, app, authorization):
        client = await aiohttp_client(app)
        await assert_unauthorized(await client.get("/queue", headers={"Authorization": authorization}))

    @pytest.mark.parametrize("path", ["/queue", "/api/wrapper/workflows", "/internal/logs"])
    async def test_correct_token_is_200(self, aiohttp_client, app, path):
        client = await aiohttp_client(app)
        response = await client.get(path, headers=GOOD)
        assert response.status == 200
        assert await response.json() == {"ok": True}

    async def test_correct_token_on_post_and_static(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        response = await client.post("/prompt", headers=GOOD, json={})
        assert response.status == 200
        response = await client.get("/index.html", headers=GOOD)
        assert response.status == 200
        assert await response.text() == "<html></html>"

    async def test_scheme_is_case_insensitive(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        response = await client.get("/queue", headers={"Authorization": f"bearer {TOKEN}"})
        assert response.status == 200

    async def test_query_token_rejected_on_normal_routes(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        for path in ("/queue", "/api/wrapper/workflows", "/index.html"):
            await assert_unauthorized(await client.get(path, params={"token": TOKEN}))

    @pytest.mark.parametrize("path", ["/ws", "/api/ws"])
    async def test_websocket_accepts_query_token(self, aiohttp_client, app, path):
        client = await aiohttp_client(app)
        ws = await client.ws_connect(path, params={"token": TOKEN, "clientId": "abc"})
        assert await ws.receive_str() == "hello"
        await ws.close()

    async def test_websocket_accepts_header(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        ws = await client.ws_connect("/ws", headers=GOOD)
        assert await ws.receive_str() == "hello"
        await ws.close()

    @pytest.mark.parametrize("params", [{}, {"token": "wrong"}, {"token": ""}, {"token": "é"}])
    async def test_websocket_without_valid_token_is_401(self, aiohttp_client, app, params):
        client = await aiohttp_client(app)
        with pytest.raises(aiohttp.WSServerHandshakeError) as exc_info:
            await client.ws_connect("/ws", params=params)
        assert exc_info.value.status == 401

    async def test_options_passes_without_token(self, aiohttp_client, app, calls):
        client = await aiohttp_client(app)
        response = await client.options("/api/wrapper/workflows")
        assert response.status == 200
        assert calls == ["/api/wrapper/workflows"]

    async def test_rejected_before_other_middlewares(self, aiohttp_client, app, calls):
        client = await aiohttp_client(app)
        assert (await client.get("/queue")).status == 401
        assert calls == []
        assert (await client.get("/queue", headers=GOOD)).status == 200
        assert calls == ["/queue"]


def login(exp_in=60, token=TOKEN, purpose=LOGIN_PURPOSE):
    return signed_value(token, purpose, int(time.time()) + exp_in)


@pytest.mark.asyncio
class TestBrowserSession:
    """A signed login link starts an HttpOnly cookie session; the token never reaches the browser."""

    @pytest.fixture
    def calls(self):
        return []

    @pytest.fixture
    def app(self, monkeypatch, tmp_path, calls):
        monkeypatch.setenv(AUTH_TOKEN_ENV, TOKEN)
        return make_app(bearer_auth_middleware_from_env(), tmp_path, calls)

    async def test_login_link_sets_a_session_and_drops_itself_from_the_url(self, aiohttp_client, app, calls):
        client = await aiohttp_client(app)
        response = await client.get("/", params={LOGIN_PARAM: login(), "keep": "1"}, allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/?keep=1"
        assert response.headers["Cache-Control"] == "no-store"
        cookie = response.cookies[SESSION_COOKIE]
        assert cookie["httponly"] is True
        assert cookie["samesite"] == "Lax"
        assert cookie["path"] == "/"
        assert int(cookie["max-age"]) == SESSION_TTL_SECONDS
        assert TOKEN not in cookie.value
        assert calls == [], "the login answer comes from the auth middleware itself"
        # The cookie now authenticates pages, API calls and the WebSocket.
        assert (await client.get("/index.html")).status == 200
        assert (await client.get("/queue")).status == 200
        assert (await client.post("/prompt", json={})).status == 200
        ws = await client.ws_connect("/ws", params={"clientId": "abc"})
        assert await ws.receive_str() == "hello"
        await ws.close()

    async def test_secure_cookie_behind_an_https_proxy(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        response = await client.get("/", params={LOGIN_PARAM: login()}, headers={"X-Forwarded-Proto": "https"}, allow_redirects=False)
        assert response.cookies[SESSION_COOKIE]["secure"] is True

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(lambda: login(exp_in=-1), id="expired"),
            pytest.param(lambda: login(exp_in=LOGIN_MAX_TTL_SECONDS + 60), id="too-far-ahead"),
            pytest.param(lambda: login(token="another-pods-token"), id="other-token"),
            pytest.param(lambda: login(purpose=SESSION_PURPOSE), id="session-value-as-link"),
            pytest.param(lambda: login()[:-1] + ("0" if login()[-1] != "0" else "1"), id="tampered"),
            pytest.param(lambda: "abc." + "0" * 64, id="not-a-time"),
            pytest.param(lambda: "", id="empty"),
            pytest.param(lambda: TOKEN, id="the-raw-token"),
        ],
    )
    async def test_bad_login_links_are_401(self, aiohttp_client, app, calls, value):
        client = await aiohttp_client(app)
        response = await client.get("/queue", params={LOGIN_PARAM: value()}, allow_redirects=False)
        await assert_unauthorized(response)
        assert SESSION_COOKIE not in response.cookies
        assert calls == []

    async def test_login_link_only_on_get(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        await assert_unauthorized(await client.post("/prompt", params={LOGIN_PARAM: login()}, json={}))

    @pytest.mark.parametrize(
        "cookie",
        [
            pytest.param(lambda: signed_value(TOKEN, SESSION_PURPOSE, int(time.time()) - 1), id="expired"),
            pytest.param(lambda: signed_value("another-pods-token", SESSION_PURPOSE, int(time.time()) + 60), id="other-token"),
            pytest.param(lambda: signed_value(TOKEN, LOGIN_PURPOSE, int(time.time()) + 60), id="login-value-as-cookie"),
            pytest.param(lambda: signed_value(TOKEN, SESSION_PURPOSE, int(time.time()) + SESSION_TTL_SECONDS + 3600), id="too-long"),
            pytest.param(lambda: TOKEN, id="the-raw-token"),
        ],
    )
    async def test_bad_session_cookies_are_401(self, aiohttp_client, app, cookie):
        client = await aiohttp_client(app)
        await assert_unauthorized(await client.get("/queue", cookies={SESSION_COOKIE: cookie()}))

    async def test_valid_session_cookie_is_accepted(self, aiohttp_client, app):
        client = await aiohttp_client(app)
        value = signed_value(TOKEN, SESSION_PURPOSE, int(time.time()) + 60)
        assert (await client.get("/queue", cookies={SESSION_COOKIE: value})).status == 200
