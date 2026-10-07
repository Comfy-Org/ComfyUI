"""GET /object_info starts the asset scan only after every node's INPUT_TYPES has run,
and outside the filename cache: a node may register model folders there."""

from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
import nodes
import server


@pytest.mark.asyncio
async def test_the_scan_starts_after_input_types_and_outside_the_cache(monkeypatch):
    calls: list[tuple[str, bool]] = []

    class LazyLoader:
        @classmethod
        def INPUT_TYPES(cls):
            calls.append(("INPUT_TYPES", folder_paths.cache_helper.active))
            return {"required": {}}

        RETURN_TYPES = ()
        FUNCTION = "run"
        CATEGORY = "test"

    asset_manager = MagicMock(enabled=False)
    asset_manager.ensure_scan_started.side_effect = lambda: calls.append(
        ("ensure_scan_started", folder_paths.cache_helper.active)
    )
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {"LazyLoader": LazyLoader})
    prompt_server = server.PromptServer(None, asset_manager)
    prompt_server.add_routes()

    async with TestClient(TestServer(prompt_server.app)) as client:
        resp = await client.get("/object_info")
        assert resp.status == 200
        assert "LazyLoader" in await resp.json()

    *node_calls, last = calls
    assert node_calls and set(node_calls) == {("INPUT_TYPES", True)}
    assert last == ("ensure_scan_started", False)
