import asyncio

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import server as server_module  # noqa: E402
from server import send_socket_catch_exception  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_successful_send_delivers_message():
    sent = []

    async def ok(message):
        sent.append(message)

    await send_socket_catch_exception(ok, "hello")
    assert sent == ["hello"]


async def test_swallows_connection_reset():
    async def raise_connection_reset(message):
        raise ConnectionResetError()

    await send_socket_catch_exception(raise_connection_reset, "test")


async def test_hanging_send_times_out_without_blocking_forever(monkeypatch):
    # A send that never completes (e.g. a stalled/unresponsive client) must not
    # block the caller indefinitely, otherwise it would stall delivery to every
    # other websocket client sharing the same publish loop.
    monkeypatch.setattr(server_module, "SEND_SOCKET_TIMEOUT", 0.05)

    async def hang(message):
        await asyncio.sleep(10)

    await asyncio.wait_for(send_socket_catch_exception(hang, "test"), timeout=1.0)
