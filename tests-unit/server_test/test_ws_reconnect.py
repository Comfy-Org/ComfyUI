"""Reconnecting with the same clientId must not lose the new connection.

When a client reconnects with an existing clientId, the handler of the old connection
can finish after the new one has registered. Its cleanup must only remove its own
socket; otherwise the new registration is deleted and every later message for that
client is dropped without an error."""

import asyncio
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer


class _Args:
    disable_assets = True
    enable_asset_hashing = False


async def _client():
    from comfy.cli_args import args
    from app.assets.manager import NoAssets

    # Importing server initialises the device; these tests never touch a model.
    with patch.object(args, "cpu", True):
        import server
    prompt_server = server.PromptServer(None, NoAssets(_Args()))
    prompt_server.app.add_routes(prompt_server.routes)
    return prompt_server, TestClient(TestServer(prompt_server.app))


@pytest.mark.asyncio
async def test_reconnect_with_same_client_id_keeps_new_socket():
    prompt_server, client = await _client()
    await client.start_server()
    try:
        old = await client.ws_connect("/ws?clientId=c1")
        await old.receive_json()                          # initial status
        new = await client.ws_connect("/ws?clientId=c1")
        await new.receive_json()

        await old.close()                                 # old handler's cleanup runs now
        for _ in range(50):
            await asyncio.sleep(0.01)

        assert "c1" in prompt_server.sockets
        assert "c1" in prompt_server.sockets_metadata
        await prompt_server.send("executing", {"node": "1", "prompt_id": "p1"}, "c1")
        msg = await new.receive_json(timeout=2)
        assert msg == {"type": "executing", "data": {"node": "1", "prompt_id": "p1"}}
        await new.close()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_closing_the_only_socket_still_unregisters_it():
    prompt_server, client = await _client()
    await client.start_server()
    try:
        ws = await client.ws_connect("/ws?clientId=c2")
        await ws.receive_json()
        await ws.close()
        for _ in range(50):
            await asyncio.sleep(0.01)
        assert "c2" not in prompt_server.sockets
        assert "c2" not in prompt_server.sockets_metadata
    finally:
        await client.close()
