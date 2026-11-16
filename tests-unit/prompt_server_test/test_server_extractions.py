"""Tests for PromptServer.enqueue_prompt, cancel_job_by_id and sid_listeners, and that /prompt and the jobs cancel routes are unchanged."""

import struct
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution  # noqa: E402
import folder_paths  # noqa: E402
import server  # noqa: E402
from comfy_execution.jobs import CANCEL_PENDING, CANCEL_RUNNING, CANCEL_TERMINAL, CANCEL_UNKNOWN  # noqa: E402
from protocol import BinaryEventTypes  # noqa: E402

_UUID_A = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa"
_UUID_B = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
_UUID_C = "cccccccc-cccc-4ccc-cccc-cccccccccccc"
_UUID_MISSING = "ffffffff-ffff-4fff-ffff-ffffffffffff"

ROUTES = {("POST", "/prompt"), ("POST", "/api/jobs/{job_id}/cancel"), ("POST", "/api/jobs/cancel")}


def queue_item(prompt_id):
    return (0, prompt_id, {}, {}, [])


class FakePromptQueue:
    def __init__(self, running=(), pending=(), history=None):
        self.running = list(running)
        self.pending = list(pending)
        self.history = dict(history or {})
        self.put_items = []
        self.interrupted = []

    def put(self, item):
        self.put_items.append(item)

    def get_current_queue(self):
        return list(self.running), list(self.pending)

    def get_history(self, prompt_id=None):
        return dict(self.history)

    def delete_queue_item(self, function):
        for i, item in enumerate(self.pending):
            if function(item):
                self.pending.pop(i)
                return True
        return False

    def interrupt_if_running(self, prompt_id):
        if any(item[1] == prompt_id for item in self.running):
            self.interrupted.append(prompt_id)
            return True
        return False


@pytest.fixture
def validation(monkeypatch):
    result = {"valid": True}

    async def validate_prompt(prompt_id, prompt, partial_execution_targets):
        if result["valid"]:
            return (True, None, ["1"], {})
        return (False, {"type": "prompt_outputs_failed_validation", "message": "bad"}, [], {"1": {"errors": ["x"]}})
    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    return result


@pytest.fixture
def prompt_server(validation, tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "user_directory", str(tmp_path / "user"))
    ps = server.PromptServer(None, MagicMock(enabled=False))
    ps.prompt_queue = FakePromptQueue()
    for route in ps.routes:
        if isinstance(route, web.RouteDef) and (route.method, route.path) in ROUTES:
            ps.app.router.add_route(route.method, route.path, route.handler)
    return ps


@pytest_asyncio.fixture
async def client(prompt_server):
    async with TestClient(TestServer(prompt_server.app)) as c:
        yield c


# /prompt wire behaviour


@pytest.mark.asyncio
async def test_prompt_valid(client, prompt_server):
    resp = await client.post("/prompt", json={"prompt": {"1": {}}, "prompt_id": _UUID_A, "client_id": "c1"},
                             headers={"Comfy-Usage-Source": "sdk"})
    assert resp.status == 200
    assert await resp.json() == {"prompt_id": _UUID_A, "number": 0, "node_errors": {}}
    [item] = prompt_server.prompt_queue.put_items
    number, prompt_id, prompt, extra_data, outputs, sensitive = item
    assert (number, prompt_id, prompt, outputs, sensitive) == (0, _UUID_A, {"1": {}}, ["1"], {})
    assert extra_data["client_id"] == "c1"
    assert extra_data["comfy_usage_source"] == "sdk"
    assert isinstance(extra_data["create_time"], int)


@pytest.mark.asyncio
async def test_prompt_mints_id_and_increments_number(client):
    first = await (await client.post("/prompt", json={"prompt": {}})).json()
    second = await (await client.post("/prompt", json={"prompt": {}, "front": True})).json()
    assert first["number"] == 0 and second["number"] == -1
    assert first["prompt_id"] != second["prompt_id"]


@pytest.mark.asyncio
async def test_prompt_invalid_prompt_id(client, prompt_server):
    resp = await client.post("/prompt", json={"prompt": {}, "prompt_id": "not-a-uuid"})
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["type"] == "invalid_prompt_id"
    assert body["node_errors"] == {}
    assert prompt_server.prompt_queue.put_items == []


@pytest.mark.asyncio
async def test_prompt_no_prompt(client):
    resp = await client.post("/prompt", json={})
    assert resp.status == 400
    assert await resp.json() == {
        "error": {"type": "no_prompt", "message": "No prompt provided", "details": "No prompt provided", "extra_info": {}},
        "node_errors": {},
    }


@pytest.mark.asyncio
async def test_prompt_validation_failure(client, prompt_server, validation):
    validation["valid"] = False
    resp = await client.post("/prompt", json={"prompt": {}})
    assert resp.status == 400
    assert await resp.json() == {
        "error": {"type": "prompt_outputs_failed_validation", "message": "bad"},
        "node_errors": {"1": {"errors": ["x"]}},
    }
    assert prompt_server.prompt_queue.put_items == []


@pytest.mark.asyncio
async def test_prompt_sensitive_keys_moved(client, prompt_server):
    extra = {"api_key_comfy_org": "k", "auth_token_comfy_org": "t", "other": 1}
    resp = await client.post("/prompt", json={"prompt": {}, "extra_data": extra})
    assert resp.status == 200
    [item] = prompt_server.prompt_queue.put_items
    assert item[5] == {"api_key_comfy_org": "k", "auth_token_comfy_org": "t"}
    assert "api_key_comfy_org" not in item[3] and "auth_token_comfy_org" not in item[3]
    assert item[3]["other"] == 1


# jobs cancel routes


@pytest.mark.asyncio
async def test_cancel_single_route(client, prompt_server):
    q = prompt_server.prompt_queue
    q.running = [queue_item(_UUID_A)]
    q.pending = [queue_item(_UUID_B)]
    q.history = {_UUID_C: {"prompt": queue_item(_UUID_C)}}
    for job_id, cancelled in ((_UUID_A, True), (_UUID_B, True), (_UUID_C, False), (_UUID_MISSING, False)):
        resp = await client.post(f"/api/jobs/{job_id}/cancel")
        assert resp.status == 200
        assert await resp.json() == {"cancelled": cancelled}
    assert q.interrupted == [_UUID_A]
    assert q.pending == []


@pytest.mark.asyncio
async def test_cancel_batch_route(client, prompt_server):
    q = prompt_server.prompt_queue
    q.running = [queue_item(_UUID_A)]
    q.pending = [queue_item(_UUID_B)]
    resp = await client.post("/api/jobs/cancel", json={"job_ids": [_UUID_A, _UUID_MISSING, _UUID_B]})
    assert resp.status == 200
    assert await resp.json() == {"cancelled": True}
    assert q.interrupted == [_UUID_A] and q.pending == []

    resp = await client.post("/api/jobs/cancel", json={"job_ids": [_UUID_MISSING]})
    assert await resp.json() == {"cancelled": False}

    resp = await client.post("/api/jobs/cancel", json={"job_ids": ["nope"]})
    assert resp.status == 400
    assert await resp.json() == {"error": "job_ids contains invalid id(s)", "invalid_ids": ["nope"]}


# extracted methods


@pytest.mark.asyncio
async def test_enqueue_prompt_returns_status_and_body(prompt_server):
    status, body = await prompt_server.enqueue_prompt({"prompt": {}, "prompt_id": _UUID_A}, usage_source="sdk")
    assert (status, body) == (200, {"prompt_id": _UUID_A, "number": 0, "node_errors": {}})
    assert prompt_server.prompt_queue.put_items[0][3]["comfy_usage_source"] == "sdk"

    status, body = await prompt_server.enqueue_prompt({})
    assert status == 400 and body["error"]["type"] == "no_prompt"


def test_cancel_job_by_id_classifications(prompt_server):
    q = prompt_server.prompt_queue
    q.running = [queue_item(_UUID_A)]
    q.pending = [queue_item(_UUID_B)]
    q.history = {_UUID_C: {"prompt": queue_item(_UUID_C)}}
    assert prompt_server.cancel_job_by_id(_UUID_B) == CANCEL_PENDING
    assert q.pending == []
    assert prompt_server.cancel_job_by_id(_UUID_A) == CANCEL_RUNNING
    assert q.interrupted == [_UUID_A]
    assert prompt_server.cancel_job_by_id(_UUID_C) == CANCEL_TERMINAL
    assert prompt_server.cancel_job_by_id(_UUID_MISSING) == CANCEL_UNKNOWN


# sid listeners


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)

    async def send_bytes(self, message):
        self.sent.append(bytes(message))


@pytest.fixture
def listened(prompt_server):
    calls = {"a": [], "b": []}
    prompt_server.sockets = {"a": FakeSocket()}
    prompt_server.sid_listeners["a"] = lambda event, data, targeted: calls["a"].append((event, data, targeted))
    prompt_server.sid_listeners["b"] = lambda event, data, targeted: calls["b"].append((event, data, targeted))
    return calls


@pytest.mark.asyncio
async def test_listener_targeted_and_broadcast_json(prompt_server, listened):
    await prompt_server.send_json("executing", {"prompt_id": "p"}, sid="a")
    await prompt_server.send_json("status", {"q": 1})
    await prompt_server.send_json("executing", {"prompt_id": "x"}, sid="other")
    assert listened["a"] == [("executing", {"prompt_id": "p"}, True), ("status", {"q": 1}, False)]
    assert listened["b"] == [("status", {"q": 1}, False)]
    assert prompt_server.sockets["a"].sent == [{"type": "executing", "data": {"prompt_id": "p"}}, {"type": "status", "data": {"q": 1}}]


@pytest.mark.asyncio
async def test_listener_targeted_bytes(prompt_server, listened):
    payload = struct.pack(">I", 2) + b"png-bytes"
    await prompt_server.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, payload, sid="b")
    assert listened["b"] == [(BinaryEventTypes.PREVIEW_IMAGE, payload, True)]
    assert isinstance(listened["b"][0][0], int)
    assert listened["a"] == []


@pytest.mark.asyncio
async def test_raising_listener_does_not_break_delivery(prompt_server, listened, caplog):
    def boom(event, data, targeted):
        raise RuntimeError("listener failed")
    prompt_server.sid_listeners["a"] = boom
    await prompt_server.send_json("status", {"q": 1})
    await prompt_server.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, b"x", sid="a")
    assert prompt_server.sockets["a"].sent == [{"type": "status", "data": {"q": 1}}, prompt_server.encode_bytes(BinaryEventTypes.PREVIEW_IMAGE, b"x")]
    assert listened["b"] == [("status", {"q": 1}, False)]
    assert "listener failed" in caplog.text
