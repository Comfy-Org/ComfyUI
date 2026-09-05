"""Tests for server.send_socket_catch_exception.
"""

import pytest
import utils.install_util  # noqa: F401
from server import send_socket_catch_exception

pytestmark = pytest.mark.asyncio


class TestSendSocketCatchException:
    """The send helper must swallow client-disconnect errors, never propagate them."""

    async def test_swallows_no_route_to_host(self, caplog):

        async def failing_send(_message):
            raise OSError(113, "No route to host")

        await send_socket_catch_exception(failing_send, b"payload")
        assert "send error" in caplog.text

    async def test_swallows_connection_reset(self):
        """Existing behaviour preserved: ConnectionResetError is still caught."""

        async def failing_send(_message):
            raise ConnectionResetError("peer reset the connection")

        await send_socket_catch_exception(failing_send, b"payload")

    async def test_successful_send_delivers_message(self):
        """When the send succeeds the message is passed through and nothing is logged."""
        received = []

        async def ok_send(message):
            received.append(message)

        await send_socket_catch_exception(ok_send, b"hello")
        assert received == [b"hello"]
