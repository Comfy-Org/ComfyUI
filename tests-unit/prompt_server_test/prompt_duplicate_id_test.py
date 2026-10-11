from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

import execution
import server


@pytest.mark.asyncio
async def test_duplicate_prompt_id_returns_conflict(aiohttp_client, monkeypatch):
    monkeypatch.setattr(server.FrontendManager, "init_frontend", lambda _: "")
    asset_manager = MagicMock(enabled=False)
    prompt_server = server.PromptServer(None, asset_manager)
    prompt_server.prompt_queue.put = MagicMock(side_effect=execution.DuplicatePromptIdError)
    monkeypatch.setattr(
        execution,
        "validate_prompt",
        AsyncMock(return_value=(True, None, [], {})),
    )

    prompt_route = next(
        route
        for route in prompt_server.routes
        if route.method == "POST" and route.resource.canonical == "/prompt"
    )
    app = web.Application()
    app.router.add_post("/prompt", prompt_route.handler)
    client = await aiohttp_client(app)

    response = await client.post(
        "/prompt",
        json={
            "prompt_id": "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7",
            "prompt": {},
        },
    )

    assert response.status == 409
    assert (await response.json())["error"]["type"] == "duplicate_prompt_id"
    assert prompt_server.number == 1
