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


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_repeatable_call_keeps_retrying(method):
    # Status polls and the upload-URL request cost nothing to repeat, so they keep
    # the retry the paid submits give up.
    assert _connection_error_is_retryable(method, aiohttp.ServerDisconnectedError(), repeatable=True)
    assert _connection_error_is_retryable(method, aiohttp.ClientOSError("broken pipe"), repeatable=True)
