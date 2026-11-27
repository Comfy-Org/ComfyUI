from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

args.cpu = True
import server  # noqa: E402


@pytest_asyncio.fixture
async def client():
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.prompt_queue = MagicMock()
    for route in prompt_server.routes:
        if isinstance(route, web.RouteDef) and route.path in ("/prompt", "/interrupt", "/upload/mask"):
            prompt_server.app.router.add_route(route.method, route.path, route.handler)
    async with TestClient(TestServer(prompt_server.app)) as client:
        client.prompt_queue = prompt_server.prompt_queue
        client.prompt_server = prompt_server
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"number": "abc", "prompt": {}},
    {"prompt": {"1": None}},
    {"prompt": [1]},
    {"prompt": {"1": {"class_type": ["KSampler"], "inputs": {}}}},
])
async def test_malformed_prompt_returns_400_without_queueing(client, body):
    response = await client.post("/prompt", json=body)
    assert response.status == 400
    assert (await response.json())["error"]["message"]
    client.prompt_queue.put.assert_not_called()
    assert client.prompt_server.number == 0


@pytest.mark.asyncio
async def test_interrupt_rejects_array_without_interrupting(client, monkeypatch):
    interrupt = MagicMock()
    monkeypatch.setattr(server.nodes, "interrupt_processing", interrupt)
    response = await client.post("/interrupt", json=[])
    assert response.status == 400
    interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_mask_upload_requires_original_ref(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server.folder_paths, "input_directory", str(tmp_path))
    form = FormData()
    form.add_field("image", b"mask", filename="mask.png")
    response = await client.post("/upload/mask", data=form)
    assert response.status == 400
    assert "original_ref" in await response.text()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_mask_upload_rejects_file_original_ref(client):
    form = FormData()
    form.add_field("original_ref", b"{}", filename="original_ref.json")
    response = await client.post("/upload/mask", data=form)
    assert response.status == 400
