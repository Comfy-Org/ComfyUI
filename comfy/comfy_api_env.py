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
_CLOUD_BASE_BY_API_HOST = {
    "api.comfy.org": "https://cloud.comfy.org",
    "stagingapi.comfy.org": "https://stagingcloud.comfy.org",
    "testapi.comfy.org": "https://testcloud.comfy.org",
}


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


def comfy_cloud_base_for_api_base(url: str) -> str:
    """Resolve the Ingest host paired with a configured Comfy API base."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if cloud_base := _CLOUD_BASE_BY_API_HOST.get(host):
        return cloud_base
    if host.endswith(_TESTENV_HOST_SUFFIX):
        label = host[: -len(_TESTENV_HOST_SUFFIX)]
        if label.endswith("-registry"):
            label = label.removesuffix("-registry")
        return f"{parsed.scheme or 'https'}://{label}{_TESTENV_HOST_SUFFIX}"
    return url


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
