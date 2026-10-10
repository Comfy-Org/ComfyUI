import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp.client_exceptions import ClientError
from yarl import URL

from comfy.cli_args import args
from comfy.comfy_api_env import normalize_comfy_api_base
from comfy.deploy_environment import get_deploy_environment
from comfy.model_management import processing_interrupted
from comfy_api.latest import IO
from comfy_execution.graph_utils import is_link
from comfy_execution.utils import get_executing_context
from comfyui_version import __version__ as comfyui_version

from .common_exceptions import ProcessingInterrupted

_HAS_PCT_ESC = re.compile(r"%[0-9A-Fa-f]{2}")  # any % followed by 2 hex digits
_HAS_BAD_PCT = re.compile(r"%(?![0-9A-Fa-f]{2})")  # any % not followed by 2 hex digits


def is_processing_interrupted() -> bool:
    """Return True if user/runtime requested interruption."""
    return processing_interrupted()


def get_node_id(node_cls: type[IO.ComfyNode]) -> str:
    return node_cls.hidden.unique_id


PARTNER_NODE_TOKEN_AUDIENCE = "comfy-partner-node"
PARTNER_NODE_TOKEN_RENEW_PATH = "/auth/partner-node/renew"
_PARTNER_NODE_TOKEN_RENEW_MARGIN = 300.0  # renew a partner-node token this many seconds before it expires

_partner_token_lineages: dict[str, str] = {}
"""Renewed partner-node token -> the token it descends from. Only tokens comfy-api issued on renewal are entered."""
_renewed_partner_tokens: dict[str, str] = {}
"""Newest renewed partner-node token per lineage. In memory only; renewal does not rotate the token."""
_partner_token_renewals: dict[str, asyncio.Task] = {}


def _partner_token_claims(token: str) -> dict | None:
    """Unverified JWT payload of a partner-node token, or None for any other credential.

    The signature is not checked: the claims only decide when to renew, comfy-api still validates the token.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, binascii.Error):
        return None
    if not isinstance(claims, dict) or claims.get("aud") not in (PARTNER_NODE_TOKEN_AUDIENCE, [PARTNER_NODE_TOKEN_AUDIENCE]):
        return None
    if not isinstance(claims.get("sid"), str) or not isinstance(claims.get("exp"), (int, float)):
        return None
    return claims


def _partner_token_lineage(token: str) -> str:
    """The token a credential was first presented as.

    Unverified claims such as ``sid`` can be forged, so a cached renewal is only shared with the token it was renewed from.
    """
    return _partner_token_lineages.get(token, token)


def _newest_partner_token(token: str, claims: dict) -> tuple[str, float]:
    renewed = _renewed_partner_tokens.get(_partner_token_lineage(token))
    if renewed is not None:
        renewed_exp = _partner_token_claims(renewed)["exp"]
        if renewed_exp >= claims["exp"]:
            return renewed, renewed_exp
    return token, claims["exp"]


async def _request_partner_token_renewal(lineage: str, sid: str, token: str) -> str | None:
    url = urljoin(default_base_url().rstrip("/") + "/", PARTNER_NODE_TOKEN_RENEW_PATH.lstrip("/"))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session:
            async with session.post(url, headers={"Authorization": f"Bearer {token}"}) as resp:
                if resp.status >= 400:
                    logging.warning("Comfy API refused to renew the sign-in token (HTTP %s).", resp.status)
                    if resp.status in (401, 403):
                        _renewed_partner_tokens.pop(lineage, None)
                    return None
                body = await resp.json(content_type=None)
    except (ClientError, OSError, asyncio.TimeoutError, ValueError) as e:
        logging.warning("Could not renew the sign-in token: %s", type(e).__name__)
        return None
    renewed = body.get("token") if isinstance(body, dict) else None
    claims = _partner_token_claims(renewed) if isinstance(renewed, str) else None
    if claims is None or claims["sid"] != sid:
        logging.warning("Comfy API returned an unusable renewed sign-in token.")
        return None
    _partner_token_lineages[renewed] = lineage
    _renewed_partner_tokens[lineage] = renewed
    return renewed


async def _renew_partner_token(token: str, claims: dict) -> str | None:
    """Renew once per lineage at a time; concurrent callers share the in-flight renewal."""
    lineage = _partner_token_lineage(token)
    task = _partner_token_renewals.get(lineage)
    if task is None or task.done() or task.get_loop() is not asyncio.get_running_loop():
        task = asyncio.create_task(_request_partner_token_renewal(lineage, claims["sid"], token))
        _partner_token_renewals[lineage] = task
        task.add_done_callback(
            lambda t: _partner_token_renewals.pop(lineage, None) if _partner_token_renewals.get(lineage) is t else None
        )
    return await asyncio.shield(task)


async def refresh_partner_token(node_cls: type[IO.ComfyNode]) -> None:
    """Renew the node's partner-node token if it expires soon. Any other credential is left untouched."""
    token = node_cls.hidden.auth_token_comfy_org
    claims = _partner_token_claims(token) if token else None
    if claims is None:
        return
    newest, exp = _newest_partner_token(token, claims)
    if exp - time.time() < _PARTNER_NODE_TOKEN_RENEW_MARGIN:
        await _renew_partner_token(newest, claims)


async def renew_rejected_partner_token(authorization: str | None) -> bool:
    """After comfy-api rejected ``authorization`` with a 401, make a fresh partner-node token current.

    Returns True when a resend would carry a different, fresh token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return False
    rejected = authorization[len("Bearer "):]
    claims = _partner_token_claims(rejected)
    if claims is None:
        return False
    newest, exp = _newest_partner_token(rejected, claims)
    if newest != rejected and exp - time.time() >= _PARTNER_NODE_TOKEN_RENEW_MARGIN:
        return True
    return await _renew_partner_token(newest, claims) is not None


def get_auth_header(node_cls: type[IO.ComfyNode]) -> dict[str, str]:
    token = node_cls.hidden.auth_token_comfy_org
    if token:
        claims = _partner_token_claims(token)
        if claims is not None:
            token = _newest_partner_token(token, claims)[0]
        return {"Authorization": f"Bearer {token}"}
    if node_cls.hidden.api_key_comfy_org:
        return {"X-API-KEY": node_cls.hidden.api_key_comfy_org}
    return {}


def get_usage_source(node_cls: type[IO.ComfyNode]) -> str:
    """Source of the prompt that triggered this API node.

    Defaults to "comfyui-api" when the submitting client didn't identify itself,
    i.e. a direct API call to this server.
    """
    return node_cls.hidden.comfy_usage_source or "comfyui-api"


def get_comfy_api_headers(node_cls: type[IO.ComfyNode]) -> dict[str, str]:
    """Common headers (auth, deploy environment, usage source) for Comfy API requests.

    Centralizes the shared header set so every Comfy API request sends a consistent
    set and new shared headers only need to be added in one place. Intended for
    relative/cloud URLs resolved against ``default_base_url()``; because the result
    includes auth, callers must not attach it to arbitrary absolute/presigned URLs.
    """
    headers = {
        **get_auth_header(node_cls),
        "Comfy-Env": get_deploy_environment(),
        "Comfy-Usage-Source": get_usage_source(node_cls),
        "Comfy-Core-Version": comfyui_version,
    }
    ctx = get_executing_context()
    if ctx is not None:
        headers["Comfy-Job-Id"] = ctx.prompt_id
    return headers


def default_base_url() -> str:
    return normalize_comfy_api_base(getattr(args, "comfy_api_base", "https://api.comfy.org"))


async def diagnose_connectivity() -> dict[str, bool]:
    """Best-effort connectivity diagnostics to distinguish local vs. server issues."""
    results = {
        "internet_accessible": False,
        "api_accessible": False,
    }
    timeout = aiohttp.ClientTimeout(total=5.0)

    # Probe Google and Baidu in parallel: Google is blocked by the GFW in mainland China, so a Baidu probe is required
    # to correctly detect that Chinese users with working internet do have working internet.
    internet_probe_urls = ("https://www.google.com", "https://www.baidu.com")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _probe(url: str) -> bool:
            try:
                async with session.get(url) as resp:
                    return resp.status < 500
            except (ClientError, OSError, asyncio.TimeoutError):
                return False

        probe_tasks = [asyncio.create_task(_probe(u)) for u in internet_probe_urls]
        try:
            for fut in asyncio.as_completed(probe_tasks):
                if await fut:
                    results["internet_accessible"] = True
                    break
        finally:
            for t in probe_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*probe_tasks, return_exceptions=True)
        if not results["internet_accessible"]:
            return results

        parsed = urlparse(default_base_url())
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        with contextlib.suppress(ClientError, OSError):
            async with session.get(health_url) as resp:
                results["api_accessible"] = resp.status < 500
    return results


async def sleep_with_interrupt(
    seconds: float,
    node_cls: type[IO.ComfyNode] | None,
    label: str | None = None,
    start_ts: float | None = None,
    *,
    display_callback: Callable[[type[IO.ComfyNode], str, int], None] | None = None,
):
    """
    Sleep in 1s slices while:
      - Checking for interruption (raises ProcessingInterrupted).
      - Optionally emitting time progress via display_callback (if provided).
    """
    end = time.monotonic() + seconds
    while True:
        if is_processing_interrupted():
            raise ProcessingInterrupted("Task cancelled")
        now = time.monotonic()
        if start_ts is not None and label and display_callback:
            with contextlib.suppress(Exception):
                display_callback(node_cls, label, int(now - start_ts))
        if now >= end:
            break
        await asyncio.sleep(min(1.0, end - now))


def _retry_after_wait(value: str | None, fallback: float, max_wait: float) -> float:
    """Delay before the next retry, honoring a server ``Retry-After`` header."""

    seconds: float | None = None
    if value is not None:
        value = value.strip()
        if value.isascii() and value.isdigit():
            # delay-seconds form. The ASCII-digit guard keeps exotic Unicode "digit" characters away from float()
            # an all-digit string always converts (huge values become inf, never raising).
            seconds = float(value)
        elif value:
            # HTTP-date form. parsedate_to_datetime raises OverflowError (not a ValueError) on absurd years/offsets
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:  # naive datetime: HTTP-date is UTC
                    parsed = parsed.replace(tzinfo=timezone.utc)
                delta = (parsed - datetime.now(timezone.utc)).total_seconds()
                seconds = delta if delta > 0 else 0.0
    if seconds is None:
        return fallback
    return min(seconds, max_wait)


def mimetype_to_extension(mime_type: str) -> str:
    """Converts a MIME type to a file extension."""
    return mime_type.split("/")[-1].lower()


def get_fs_object_size(path_or_object: str | BytesIO) -> int:
    if isinstance(path_or_object, str):
        return os.path.getsize(path_or_object)
    return len(path_or_object.getvalue())


def get_output_consumers(node_cls: type[IO.ComfyNode], output_index: int) -> list[str]:
    dynprompt = node_cls.hidden.dynprompt
    if dynprompt is None:
        return []
    node_id = str(node_cls.hidden.unique_id)
    consumers = []
    for consumer_id in dynprompt.all_node_ids():
        consumer = dynprompt.get_node(consumer_id)
        for value in (consumer.get("inputs") or {}).values():
            if is_link(value) and value[0] == node_id and value[1] == output_index:
                title = (consumer.get("_meta") or {}).get("title") or consumer.get("class_type")
                consumers.append(f"{title} #{dynprompt.get_display_node_id(consumer_id)}")
    return sorted(consumers)


def validate_output_unlinked(node_cls: type[IO.ComfyNode], output_index: int, reason: str) -> None:
    consumers = get_output_consumers(node_cls, output_index)
    if consumers:
        raise ValueError(f"{reason} (currently linked: {', '.join(consumers)}).")


def to_aiohttp_url(url: str) -> URL:
    """If `url` appears to be already percent-encoded (contains at least one valid %HH
    escape and no malformed '%' sequences) and contains no raw whitespace/control
    characters preserve the original encoding byte-for-byte (important for signed/presigned URLs).
    Otherwise, return `URL(url)` and allow yarl to normalize/quote as needed."""
    if any(c.isspace() for c in url) or any(ord(c) < 0x20 for c in url):
        # Avoid encoded=True if URL contains raw whitespace/control chars
        return URL(url)
    if _HAS_PCT_ESC.search(url) and not _HAS_BAD_PCT.search(url):
        # Preserve encoding only if it appears pre-encoded AND has no invalid % sequences
        return URL(url, encoded=True)
    return URL(url)
