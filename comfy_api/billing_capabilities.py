import asyncio
import json

import aiohttp
from aiohttp import web

from comfy.cli_args import args
from comfy.comfy_api_env import normalize_comfy_api_base
from comfy.deploy_environment import get_deploy_environment
from comfyui_version import __version__ as comfyui_version


_UPSTREAM_PATH = "/api/billing/capabilities"
_EXPECTED_STATUSES = {200, 401, 403, 404, 502}
_RESPONSE_HEADERS = ("Cache-Control", "Vary", "X-Capability-Revision", "Content-Type")
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _unavailable_response() -> web.Response:
    return web.json_response({"error": "Billing capabilities unavailable"}, status=502)


async def relay_billing_capabilities(request: web.Request, client_session: aiohttp.ClientSession) -> web.Response:
    headers = {
        "Accept": "application/json",
        "Comfy-Env": get_deploy_environment(),
        "Comfy-Core-Version": comfyui_version,
    }
    for name in ("Authorization", "X-API-Key"):
        if value := request.headers.get(name):
            headers[name] = value

    base_url = normalize_comfy_api_base(args.comfy_api_base).rstrip("/")
    try:
        async with client_session.get(
            f"{base_url}{_UPSTREAM_PATH}", headers=headers, timeout=_TIMEOUT, allow_redirects=False
        ) as upstream:
            if upstream.status not in _EXPECTED_STATUSES:
                return _unavailable_response()

            body = await upstream.read()
            if upstream.content_type != "application/json" or not isinstance(json.loads(body), dict):
                return _unavailable_response()
            response_headers = {name: upstream.headers[name] for name in _RESPONSE_HEADERS if name in upstream.headers}
            return web.Response(body=body, status=upstream.status, headers=response_headers)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
        return _unavailable_response()
