import aiohttp
import pytest

from comfy_api_nodes.util.client import _connection_error_is_retryable


class _ConnectorError(aiohttp.ClientConnectorError):
    """A ClientConnectorError without aiohttp's version-specific ConnectionKey."""

    def __init__(self):  # noqa: D107 - the base __init__ needs internals we don't test
        pass


@pytest.mark.parametrize("method", ["GET", "get", "HEAD", "OPTIONS"])
def test_safe_methods_retry_any_connection_error(method):
    assert _connection_error_is_retryable(method, aiohttp.ServerDisconnectedError())
    assert _connection_error_is_retryable(method, _ConnectorError())


@pytest.mark.parametrize("method", ["POST", "post", "PUT", "PATCH", "DELETE"])
def test_unsafe_methods_do_not_repeat_a_request_the_server_received(method):
    # The server closed the connection after taking the request: the partner may
    # already be generating, and a retry is a second charge.
    assert not _connection_error_is_retryable(method, aiohttp.ServerDisconnectedError())
    assert not _connection_error_is_retryable(method, aiohttp.ClientOSError("broken pipe"))


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_unsafe_methods_still_retry_when_the_request_was_never_sent(method):
    assert _connection_error_is_retryable(method, _ConnectorError())
    assert _connection_error_is_retryable(method, aiohttp.ConnectionTimeoutError())


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_read_timeout_after_send_is_not_repeated(method):
    # sock_read timed out: the request was written, the provider may be running it.
    assert not _connection_error_is_retryable(method, aiohttp.SocketTimeoutError())
    assert not _connection_error_is_retryable(method, aiohttp.ClientPayloadError("truncated body"))


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_status_poll_keeps_retrying(method):
    # The poll loop marks its requests: a status read costs nothing to send again.
    assert _connection_error_is_retryable(method, aiohttp.ServerDisconnectedError(), resend_is_free=True)
    assert _connection_error_is_retryable(method, aiohttp.ClientOSError("broken pipe"), resend_is_free=True)
