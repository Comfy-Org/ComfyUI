"""Holds whether asset hashing is enabled for this process, taken from the
command-line flag once at startup. Callers ask here rather than reading
arguments themselves, and asking before initialization raises instead of
defaulting, so a route can never quietly answer as though hashing were off.
"""

from __future__ import annotations

from typing import Protocol


class _HashingArguments(Protocol):
    enable_asset_hashing: bool


_args: _HashingArguments | None = None


def init(args: _HashingArguments) -> None:
    global _args
    _args = args


def hashing_enabled() -> bool:
    if _args is None:
        raise RuntimeError(
            "app.assets.mode.init() was not called before hashing_enabled(); "
            "hash-mode state is uninitialised"
        )
    return bool(_args.enable_asset_hashing)
