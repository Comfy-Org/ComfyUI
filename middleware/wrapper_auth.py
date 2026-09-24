"""Optional bearer-token auth for every route of the ComfyUI server.

Enabled by setting WRAPPER_AUTH_TOKEN. When it is set, every HTTP route (the
wrapper API, native routes such as /prompt, /queue, /view, /history, /ws, and
the static frontend) requires ``Authorization: Bearer <token>``. When it is
unset or empty, no middleware is installed and nothing changes.
"""

import hmac
import os
from typing import Awaitable, Callable

from aiohttp import web

AUTH_TOKEN_ENV = "WRAPPER_AUTH_TOKEN"

# Browsers cannot set headers on a WebSocket handshake, so the socket route also
# accepts ?token=. server.py mirrors every route under /api, hence both paths.
WEBSOCKET_PATHS = ("/ws", "/api/ws")

UNAUTHORIZED_BODY = {
    "error": {
        "type": "unauthorized",
        "message": "missing or invalid bearer token",
    }
}


def _token_matches(candidate: str, expected: bytes) -> bool:
    # compare_digest on bytes: constant time, and safe for non-ASCII input.
    return hmac.compare_digest(candidate.encode("utf-8"), expected)


def _is_authorized(request: web.Request, expected: bytes) -> bool:
    scheme, _, credentials = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer" and _token_matches(credentials.strip(), expected):
        return True
    if request.path in WEBSOCKET_PATHS:
        query_token = request.query.get("token")
        if query_token is not None and _token_matches(query_token, expected):
            return True
    return False


def create_bearer_auth_middleware(token: str):
    """Middleware that rejects any request without ``Authorization: Bearer <token>``.

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
        if request.method == "OPTIONS" or _is_authorized(request, expected):
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
