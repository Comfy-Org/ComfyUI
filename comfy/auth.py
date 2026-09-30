"""Registry for authentication providers (used by ``--require-auth``).

An auth provider is a custom node pack that puts authentication in front of the
server by attaching an aiohttp middleware to ``PromptServer.instance.app``.
It announces itself here so that, when ``--require-auth`` is set, core can
refuse to start if no provider is actually active.

Usage from a custom node pack, at import time::

    from server import PromptServer
    import comfy.auth

    PromptServer.instance.app.middlewares.insert(0, my_auth_middleware)
    comfy.auth.register_auth_provider("my-auth", my_auth_middleware)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple


class AuthProvider(NamedTuple):
    name: str
    middleware: Callable[..., Any]


class AuthRequiredError(RuntimeError):
    """Raised when authentication is required but no provider is active."""


_providers: list[AuthProvider] = []


def register_auth_provider(name: str, middleware: Callable[..., Any]) -> None:
    """Record that ``middleware`` enforces authentication for the server.

    The middleware must also be attached to the server app; registering alone
    does not protect anything.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("auth provider name must be a non-empty string")
    if not callable(middleware):
        raise ValueError("auth provider middleware must be callable")
    _providers.append(AuthProvider(name.strip(), middleware))
    logging.info("Auth provider registered: %s", name.strip())


def get_auth_providers() -> tuple[AuthProvider, ...]:
    return tuple(_providers)


def verify_auth_provider(app_middlewares: Sequence[Any]) -> AuthProvider:
    """Return an active provider, or raise ``AuthRequiredError``.

    A provider is active only if its middleware is attached to the app.
    """
    if not _providers:
        raise AuthRequiredError(
            "--require-auth is set but no auth provider registered. "
            "Install and configure an auth custom node (and make sure custom nodes are not disabled)."
        )
    for provider in _providers:
        if any(m is provider.middleware for m in app_middlewares):
            return provider
    names = ", ".join(p.name for p in _providers)
    raise AuthRequiredError(
        f"--require-auth is set but no registered auth provider ({names}) has its middleware attached to the server."
    )


def _reset_for_tests() -> None:
    _providers.clear()
