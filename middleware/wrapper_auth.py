"""Optional bearer-token auth for every route of the ComfyUI server.

Enabled by setting WRAPPER_AUTH_TOKEN. When it is set, every HTTP route (the
wrapper API, native routes such as /prompt, /queue, /view, /history, /ws, and
the static frontend) requires ``Authorization: Bearer <token>``. When it is
unset or empty, no middleware is installed and nothing changes.

Browser sessions (an admin opening the ComfyUI UI of a pod): a browser cannot
send that header, so whoever holds the token (voxmin-backend) signs a
short-lived login link instead::

    <server>/?comfy_login=<exp>.<HMAC-SHA256(token, "comfy-login:<exp>")>

``<exp>`` is unix seconds, at most LOGIN_MAX_TTL_SECONDS ahead. A valid link
answers with a redirect to the same URL without the parameter and an HttpOnly
session cookie signed the same way ("comfy-session:<exp>", valid for
SESSION_TTL_SECONDS); the UI, its API calls and its WebSocket then
authenticate with the cookie. The token itself never reaches the browser.
"""

import hashlib
import hmac
import os
import time
from typing import Awaitable, Callable

from aiohttp import web

AUTH_TOKEN_ENV = "WRAPPER_AUTH_TOKEN"

# Browsers cannot set headers on a WebSocket handshake, so the socket route also
# accepts ?token=. server.py mirrors every route under /api, hence both paths.
WEBSOCKET_PATHS = ("/ws", "/api/ws")

LOGIN_PARAM = "comfy_login"
SESSION_COOKIE = "comfy_wrapper_session"
LOGIN_PURPOSE = "comfy-login"
SESSION_PURPOSE = "comfy-session"
LOGIN_MAX_TTL_SECONDS = 600
SESSION_TTL_SECONDS = 12 * 3600

UNAUTHORIZED_BODY = {
    "error": {
        "type": "unauthorized",
        "message": "missing or invalid bearer token",
    }
}


def _token_matches(candidate: str, expected: bytes) -> bool:
    # compare_digest on bytes: constant time, and safe for non-ASCII input.
    return hmac.compare_digest(candidate.encode("utf-8"), expected)


def sign(token: str, purpose: str, exp: int) -> str:
    """HMAC-SHA256 hex of ``<purpose>:<exp>`` keyed with the auth token."""
    return hmac.new(token.encode("utf-8"), f"{purpose}:{exp}".encode("ascii"), hashlib.sha256).hexdigest()


def signed_value(token: str, purpose: str, exp: int) -> str:
    """``<exp>.<signature>``: a login link parameter or a session cookie."""
    return f"{exp}.{sign(token, purpose, exp)}"


def _valid_signed(value: str, token: str, purpose: str, now: float, max_ttl: int) -> bool:
    exp_text, _, signature = value.partition(".")
    if not exp_text.isdigit() or len(exp_text) > 12 or not signature:
        return False
    exp = int(exp_text)
    if exp <= now or exp - now > max_ttl:
        return False
    return hmac.compare_digest(signature.encode("utf-8"), sign(token, purpose, exp).encode("ascii"))


def _is_authorized(request: web.Request, token: str, expected: bytes, now: float) -> bool:
    scheme, _, credentials = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer" and _token_matches(credentials.strip(), expected):
        return True
    if request.path in WEBSOCKET_PATHS:
        query_token = request.query.get("token")
        if query_token is not None and _token_matches(query_token, expected):
            return True
    session = request.cookies.get(SESSION_COOKIE)
    return session is not None and _valid_signed(session, token, SESSION_PURPOSE, now, SESSION_TTL_SECONDS)


def _start_session(request: web.Request, token: str, now: float) -> web.StreamResponse:
    """Trade a valid login link for a session cookie, and drop the link from the URL."""
    query = request.rel_url.query.copy()
    query.popall(LOGIN_PARAM)
    response = web.Response(status=302, headers={"Location": str(request.rel_url.with_query(query)), "Cache-Control": "no-store"})
    exp = int(now) + SESSION_TTL_SECONDS
    https = request.secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
    response.set_cookie(
        SESSION_COOKIE,
        signed_value(token, SESSION_PURPOSE, exp),
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="Lax",  # Strict would drop the cookie on the redirect a link from another site starts
        secure=https,
    )
    return response


def create_bearer_auth_middleware(token: str):
    """Middleware that rejects any request without ``Authorization: Bearer <token>``
    (or a browser session started from a signed login link, see the module doc).

    OPTIONS requests are let through unauthenticated: CORS preflights never
    carry credentials, so gating them would break every cross-origin client.
    Whatever answers them next (the CORS or origin-only middleware, which reply
    with an empty 200) exposes no data.
    """
    if not token:
        raise ValueError("bearer auth token must be a non-empty string")
    expected = token.encode("utf-8")

    @web.middleware
    async def bearer_auth(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        now = time.time()
        if request.method == "GET":
            login = request.query.get(LOGIN_PARAM)
            if login is not None and _valid_signed(login, token, LOGIN_PURPOSE, now, LOGIN_MAX_TTL_SECONDS):
                return _start_session(request, token, now)
        if request.method == "OPTIONS" or _is_authorized(request, token, expected, now):
            return await handler(request)
        return web.json_response(
            UNAUTHORIZED_BODY,
            status=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    return bearer_auth


def bearer_auth_middleware_from_env():
    """Bearer auth middleware for WRAPPER_AUTH_TOKEN, or None when it is unset or empty."""
    token = os.environ.get(AUTH_TOKEN_ENV)
    if not token:
        return None
    return create_bearer_auth_middleware(token)
