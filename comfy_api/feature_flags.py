"""
Feature flags module for ComfyUI WebSocket protocol negotiation.

This module handles capability negotiation between frontend and backend,
allowing graceful protocol evolution while maintaining backward compatibility.
"""

import logging
from typing import Any, TypedDict

from comfy.cli_args import args


class FeatureFlagInfo(TypedDict):
    type: str
    default: Any
    description: str


# Registry of known CLI-settable feature flags.
# Launchers can query this via --list-feature-flags to discover valid flags.
CLI_FEATURE_FLAG_REGISTRY: dict[str, FeatureFlagInfo] = {
    "show_signin_button": {
        "type": "bool",
        "default": False,
        "description": "Show the sign-in button in the frontend even when not signed in",
    },
    "enable_telemetry": {
        "type": "bool",
        "default": False,
        "description": "Signal the frontend that telemetry collection is enabled",
    },
}


def _coerce_bool(v: str) -> bool:
    """Strict bool coercion: only 'true'/'false' (case-insensitive).

    Anything else raises ValueError so the caller can warn and drop the flag,
    rather than silently treating typos like 'ture' or 'yes' as False.
    """
    lower = v.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    raise ValueError(f"expected 'true' or 'false', got {v!r}")


_COERCE_FNS: dict[str, Any] = {
    "bool": _coerce_bool,
    "int": lambda v: int(v),
    "float": lambda v: float(v),
}


def _coerce_flag_value(key: str, raw_value: str) -> Any:
    """Coerce a raw string value using the registry type, or keep as string.

    Returns the raw string if the key is unregistered or the type is unknown.
    Raises ValueError/TypeError if the key is registered with a known type but
    the value cannot be coerced; callers are expected to warn and drop the flag.
    """
    info = CLI_FEATURE_FLAG_REGISTRY.get(key)
    if info is None:
        return raw_value
    coerce = _COERCE_FNS.get(info["type"])
    if coerce is None:
        return raw_value
    return coerce(raw_value)


def _parse_cli_feature_flags() -> dict[str, Any]:
    """Parse --feature-flag key=value pairs from CLI args into a dict.

    Items without '=' default to the value 'true' (bare flag form).
    Flags whose value cannot be coerced to the registered type are dropped
    with a warning, so a typo like '--feature-flag some_bool=ture' does not
    silently take effect as the wrong value.
    """
    result: dict[str, Any] = {}
    for item in getattr(args, "feature_flag", []):
        key, sep, raw_value = item.partition("=")
        key = key.strip()
        if not key:
            continue
        if not sep:
            raw_value = "true"
        try:
            result[key] = _coerce_flag_value(key, raw_value.strip())
        except (ValueError, TypeError) as e:
            info = CLI_FEATURE_FLAG_REGISTRY.get(key, {})
            logging.warning(
                "Could not coerce --feature-flag %s=%r to %s (%s); dropping flag.",
                key, raw_value.strip(), info.get("type", "?"), e,
            )
    return result


# Default server capabilities
_CORE_FEATURE_FLAGS: dict[str, Any] = {
    "supports_preview_metadata": True,
    "max_upload_size": args.max_upload_size * 1024 * 1024, # Convert MB to bytes
    "extension": {"manager": {"supports_v4": True}},
    "node_replacements": True,
    "assets": args.enable_assets,
}

# CLI-provided flags cannot overwrite core flags
_cli_flags = {k: v for k, v in _parse_cli_feature_flags().items() if k not in _CORE_FEATURE_FLAGS}

SERVER_FEATURE_FLAGS: dict[str, Any] = {**_CORE_FEATURE_FLAGS, **_cli_flags}


def get_connection_feature(
    sockets_metadata: dict[str, dict[str, Any]],
    sid: str,
    feature_name: str,
    default: Any = False
) -> Any:
    """
    Get a feature flag value for a specific connection.

    Args:
        sockets_metadata: Dictionary of socket metadata
        sid: Session ID of the connection
        feature_name: Name of the feature to check
        default: Default value if feature not found

    Returns:
        Feature value or default if not found
    """
    if sid not in sockets_metadata:
        return default

    return sockets_metadata[sid].get("feature_flags", {}).get(feature_name, default)


def supports_feature(
    sockets_metadata: dict[str, dict[str, Any]],
    sid: str,
    feature_name: str
) -> bool:
    """
    Check if a connection supports a specific feature.

    Args:
        sockets_metadata: Dictionary of socket metadata
        sid: Session ID of the connection
        feature_name: Name of the feature to check

    Returns:
        Boolean indicating if feature is supported
    """
    return get_connection_feature(sockets_metadata, sid, feature_name, False) is True


def get_server_features() -> dict[str, Any]:
    """
    Get the server's feature flags.

    Returns:
        Dictionary of server feature flags
    """
    return SERVER_FEATURE_FLAGS.copy()
