import asyncio
import json

import aiohttp
from aiohttp import web

from comfy.cli_args import args
from comfy.comfy_api_env import comfy_cloud_base_for_api_base
from comfy.deploy_environment import get_deploy_environment
from comfyui_version import __version__ as comfyui_version


_UPSTREAM_PATH = "/api/billing/capabilities"
_EXPECTED_STATUSES = {200, 401, 403, 404, 429, 502, 503}
_RESPONSE_HEADERS = (
    "Cache-Control",
    "X-Capability-Revision",
    "Content-Type",
    "Retry-After",
)
_AUTH_HEADERS = ("Authorization", "X-API-Key")
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _unavailable_response() -> web.Response:
    return web.json_response({"error": "Billing capabilities unavailable"}, status=502)


def _authentication_required_response() -> web.Response:
    return web.json_response(
        {"error": "Authentication required"},
        status=401,
        headers={
            "Cache-Control": "private, no-store",
            "Vary": ", ".join(_AUTH_HEADERS),
        },
    )


def _response_headers(upstream: aiohttp.ClientResponse) -> dict[str, str]:
    headers = {
        name: upstream.headers[name]
        for name in _RESPONSE_HEADERS
        if name in upstream.headers
    }
    vary = [
        name.strip()
        for name in upstream.headers.get("Vary", "").split(",")
        if name.strip()
    ]
    varied_names = {name.lower() for name in vary}
    vary.extend(name for name in _AUTH_HEADERS if name.lower() not in varied_names)
    headers["Vary"] = ", ".join(vary)
    return headers


async def relay_billing_capabilities(
    request: web.Request, client_session: aiohttp.ClientSession
) -> web.Response:
    if not isinstance(client_session.cookie_jar, aiohttp.DummyCookieJar):
        raise RuntimeError(
            "Billing capabilities relay requires a non-persistent cookie jar"
        )

    headers = {
        "Accept": "application/json",
        "Comfy-Env": get_deploy_environment(),
        "Comfy-Core-Version": comfyui_version,
    }
    for name in _AUTH_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value
    if not any(name in headers for name in _AUTH_HEADERS):
        return _authentication_required_response()

    try:
        base_url = comfy_cloud_base_for_api_base(args.comfy_api_base).rstrip("/")
        async with client_session.get(
            f"{base_url}{_UPSTREAM_PATH}",
            headers=headers,
            timeout=_TIMEOUT,
            allow_redirects=False,
        ) as upstream:
            if upstream.status not in _EXPECTED_STATUSES:
                return _unavailable_response()

            body = await upstream.read()
            if upstream.status == 200 and (
                upstream.content_type != "application/json"
                or not isinstance(json.loads(body), dict)
            ):
                return _unavailable_response()
            return web.Response(
                body=body, status=upstream.status, headers=_response_headers(upstream)
            )
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return _unavailable_response()
