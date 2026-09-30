"""Tests for the auth provider registry behind --require-auth."""

import pytest
from aiohttp import web

import comfy.auth
from comfy.cli_args import parser


@pytest.fixture(autouse=True)
def reset_registry():
    comfy.auth._reset_for_tests()
    yield
    comfy.auth._reset_for_tests()


@web.middleware
async def auth_middleware(request, handler):
    return await handler(request)


@web.middleware
async def other_middleware(request, handler):
    return await handler(request)


def test_require_auth_flag_defaults_off():
    assert parser.parse_args([]).require_auth is False
    assert parser.parse_args(["--require-auth"]).require_auth is True


def test_verify_fails_without_provider():
    app = web.Application(middlewares=[other_middleware])
    with pytest.raises(comfy.auth.AuthRequiredError, match="no auth provider registered"):
        comfy.auth.verify_auth_provider(app.middlewares)


def test_verify_fails_when_middleware_not_attached():
    comfy.auth.register_auth_provider("test-auth", auth_middleware)
    app = web.Application(middlewares=[other_middleware])
    with pytest.raises(comfy.auth.AuthRequiredError, match="test-auth"):
        comfy.auth.verify_auth_provider(app.middlewares)


def test_verify_fails_when_not_first():
    app = web.Application(middlewares=[other_middleware, auth_middleware])
    comfy.auth.register_auth_provider("test-auth", auth_middleware)
    with pytest.raises(comfy.auth.AuthRequiredError, match="first middleware"):
        comfy.auth.verify_auth_provider(app.middlewares)


def test_verify_fails_with_no_middlewares():
    comfy.auth.register_auth_provider("test-auth", auth_middleware)
    with pytest.raises(comfy.auth.AuthRequiredError):
        comfy.auth.verify_auth_provider(web.Application().middlewares)


def test_verify_passes_when_first():
    app = web.Application(middlewares=[other_middleware])
    app.middlewares.insert(0, auth_middleware)
    comfy.auth.register_auth_provider("test-auth", auth_middleware)
    provider = comfy.auth.verify_auth_provider(app.middlewares)
    assert provider.name == "test-auth"
    assert provider.middleware is auth_middleware
    assert comfy.auth.get_auth_providers() == (provider,)


@pytest.mark.parametrize("name", ["", "   ", None])
def test_register_rejects_bad_name(name):
    with pytest.raises(ValueError):
        comfy.auth.register_auth_provider(name, auth_middleware)
    assert comfy.auth.get_auth_providers() == ()


def test_register_rejects_non_callable():
    with pytest.raises(ValueError):
        comfy.auth.register_auth_provider("test-auth", "not-callable")
