"""Runtime config the frontend reads from /features to follow --comfy-api-base.

For a non-prod comfy.org backend (staging or an ephemeral preview env), "/features" exposes the api and
platform base so the frontend talks to it without a rebuild, plus the Firebase environment it should use.
Prod bases are left alone and keep their build-time defaults.
"""

from typing import Any
from urllib.parse import urlparse

from comfy.cli_args import args

_STAGING_API_HOST = "stagingapi.comfy.org"
_TESTENV_HOST_SUFFIX = ".testenvs.comfy.org"
_STAGING_PLATFORM_BASE_URL = "https://stagingplatform.comfy.org"
_PROD_PLATFORM_BASE_URL = "https://platform.comfy.org"
_STAGING_BUILDER_BASE_URL = "https://stagingplatformapi.comfy.org/builder"
_PROD_BUILDER_BASE_URL = "https://platformapi.comfy.org/builder"


def _is_staging_tier(host: str) -> bool:
    return host == _STAGING_API_HOST or host.endswith(_TESTENV_HOST_SUFFIX)


def normalize_comfy_api_base(url: str) -> str:
    """Rewrite a testenv's friendly main host to its comfy-api '-registry' sibling."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host.endswith(_TESTENV_HOST_SUFFIX):
        return url
    label = host[: -len(_TESTENV_HOST_SUFFIX)]
    if label.endswith("-registry"):
        return url
    return f"{parsed.scheme or 'https'}://{label}-registry{_TESTENV_HOST_SUFFIX}"


def environment_overrides_for_base(base_url: str) -> dict[str, Any] | None:
    """The /features overrides for a staging-tier base, or None for prod."""
    if not _is_staging_tier(urlparse(base_url).hostname or ""):
        return None
    return {
        "comfy_api_base_url": normalize_comfy_api_base(base_url).rstrip("/"),
        "comfy_platform_base_url": _STAGING_PLATFORM_BASE_URL,
        "firebase_env": "dev",
    }


def get_environment_overrides() -> dict[str, Any] | None:
    return environment_overrides_for_base(getattr(args, "comfy_api_base", "") or "")


def _staging_selected() -> bool:
    return _is_staging_tier(urlparse(getattr(args, "comfy_api_base", "") or "").hostname or "")


def builder_base_url() -> str:
    """comfy-builder API base for the --comfy-api-base tier; fixed mapping, never taken from request data."""
    return _STAGING_BUILDER_BASE_URL if _staging_selected() else _PROD_BUILDER_BASE_URL


def platform_base_url() -> str:
    return _STAGING_PLATFORM_BASE_URL if _staging_selected() else _PROD_PLATFORM_BASE_URL
