"""Holds the asset modes for this process, taken from the command-line flags
once at startup: whether hashing is enabled and whether the output directory
is scanned. Callers ask here rather than reading arguments themselves, and
asking before initialization raises instead of defaulting, so a route can never
quietly answer as though a mode were off.
"""

from __future__ import annotations

from typing import Protocol, TypeVar


class _ModeArguments(Protocol):
    enable_asset_hashing: bool
    enable_assets_output_scan: bool


_args: _ModeArguments | None = None

_RootT = TypeVar("_RootT", bound=str)


def init(args: _ModeArguments) -> None:
    global _args
    _args = args


def _initialised_args() -> _ModeArguments:
    if _args is None:
        raise RuntimeError(
            "app.assets.mode.init() was not called before reading asset modes; "
            "asset mode state is uninitialised"
        )
    return _args


def hashing_enabled() -> bool:
    return bool(_initialised_args().enable_asset_hashing)


def output_scan_enabled() -> bool:
    return bool(_initialised_args().enable_assets_output_scan)


def scannable_roots(roots: tuple[_RootT, ...]) -> tuple[_RootT, ...]:
    """Drop the output root from a scan request when output scanning is off."""
    if output_scan_enabled():
        return roots
    return tuple(root for root in roots if root != "output")
