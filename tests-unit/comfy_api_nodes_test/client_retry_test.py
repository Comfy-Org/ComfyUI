import aiohttp
import pytest

from comfy_api_nodes.util.client import ApiEndpoint, _connection_error_is_retryable


class _ConnectorError(aiohttp.ClientConnectorError):
    """A ClientConnectorError without aiohttp's version-specific ConnectionKey."""

    def __init__(self):  # noqa: D107 - the base __init__ needs internals we don't test
        pass


def _ep(method, idempotent=None):
    return ApiEndpoint("/proxy/x", method, idempotent=idempotent)


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_safe_methods_retry_any_connection_error(method):
    assert _connection_error_is_retryable(_ep(method), aiohttp.ServerDisconnectedError())
    assert _connection_error_is_retryable(_ep(method), _ConnectorError())


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_submissions_are_not_resent_once_the_server_may_have_them(method):
    # The server closed the connection after taking the request: the partner may
    # already be generating, and a resend is a second charge.
    assert not _connection_error_is_retryable(_ep(method), aiohttp.ServerDisconnectedError())
    assert not _connection_error_is_retryable(_ep(method), aiohttp.ClientOSError("broken pipe"))
    assert not _connection_error_is_retryable(_ep(method), aiohttp.SocketTimeoutError())
    assert not _connection_error_is_retryable(_ep(method), aiohttp.ClientPayloadError("truncated body"))


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_submissions_still_retry_when_the_request_was_never_sent(method):
    assert _connection_error_is_retryable(_ep(method), _ConnectorError())
    assert _connection_error_is_retryable(_ep(method), aiohttp.ConnectionTimeoutError())


def test_an_endpoint_declared_idempotent_keeps_retrying():
    # A status poll that happens to use POST: resending it is free.
    assert _connection_error_is_retryable(_ep("POST", idempotent=True), aiohttp.ServerDisconnectedError())


def test_an_endpoint_declared_non_idempotent_is_guarded_whatever_its_method():
    assert not _connection_error_is_retryable(_ep("GET", idempotent=False), aiohttp.ServerDisconnectedError())
