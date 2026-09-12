import asyncio
import contextlib
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO

from yarl import URL

from comfy.cli_args import args
from comfy.comfy_api_env import normalize_comfy_api_base
from comfy.deploy_environment import get_deploy_environment
from comfy.model_management import processing_interrupted
from comfy_api.latest import IO
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


def get_auth_header(node_cls: type[IO.ComfyNode]) -> dict[str, str]:
    if node_cls.hidden.auth_token_comfy_org:
        return {"Authorization": f"Bearer {node_cls.hidden.auth_token_comfy_org}"}
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
