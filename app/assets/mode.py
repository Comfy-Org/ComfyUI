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
