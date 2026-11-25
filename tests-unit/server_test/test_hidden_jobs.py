"""Jobs submitted with "hidden": true in extra_data are left out of the job, queue and history lists and the queue
count, and are still returned by id."""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True  # importing execution reads the device

import execution  # noqa: E402
import server  # noqa: E402
from comfy_execution.jobs import is_hidden_job  # noqa: E402


def item(number, prompt_id, hidden=False):
    extra_data = {"client_id": "c", "create_time": number}
    if hidden:
        extra_data["hidden"] = True
    return (number, prompt_id, {}, extra_data, [], {})


def finish(queue):
    queued, item_id = queue.get(timeout=0)
    status = execution.PromptQueue.ExecutionStatus(status_str="success", completed=True, messages=[])
    queue.task_done(item_id, {}, status, process_item=lambda p: p[:5])
    return queued[1]


def seed(queue):
    """Two finished, two running and two pending jobs; one of each visible and one hidden."""
    queue.put(item(1, "user-done"))
    queue.put(item(2, "agent-done", hidden=True))
    assert [finish(queue), finish(queue)] == ["user-done", "agent-done"]
    queue.put(item(3, "agent-running", hidden=True))
    queue.put(item(4, "user-running"))
    queue.get(timeout=0)
    queue.get(timeout=0)
    queue.put(item(5, "user-pending"))
    queue.put(item(6, "agent-pending", hidden=True))


@pytest.mark.parametrize("extra_data,hidden", [
    ({"hidden": True}, True),
    ({"hidden": "true"}, False),
    ({"hidden": False}, False),
    ({}, False),
])
def test_is_hidden_job(extra_data, hidden):
    assert is_hidden_job((0, "p", {}, extra_data, [])) is hidden


def test_queue_counts_every_job_and_history_filters_before_paging():
    queue = execution.PromptQueue(MagicMock())
    seed(queue)
    assert queue.get_tasks_remaining() == 4  # the worker's idle check counts hidden jobs too
    assert list(queue.get_history()) == ["user-done", "agent-done"]
    visible = server._is_visible_history_entry
    assert list(queue.get_history(filter_function=visible)) == ["user-done"]
    assert list(queue.get_history(max_items=1, filter_function=visible)) == ["user-done"]


@pytest_asyncio.fixture
async def client():
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.prompt_queue = queue = execution.PromptQueue(prompt_server)
    prompt_server.queue_updated = lambda: None
    seed(queue)
    prompt_server.add_routes()
    await prompt_server.setup()
    try:
        async with TestClient(TestServer(prompt_server.app)) as c:
            c.prompt_server = prompt_server
            yield c
    finally:
        await prompt_server.client_session.close()


@pytest.mark.asyncio
async def test_job_list_leaves_hidden_jobs_out(client):
    body = await (await client.get("/api/jobs")).json()
    assert sorted(j["id"] for j in body["jobs"]) == ["user-done", "user-pending", "user-running"]
    assert body["pagination"]["total"] == 3


@pytest.mark.asyncio
async def test_hidden_jobs_are_found_by_id(client):
    for prompt_id in ["agent-done", "agent-running", "agent-pending"]:
        assert (await (await client.get(f"/api/jobs/{prompt_id}")).json())["id"] == prompt_id
    assert list(await (await client.get("/history/agent-done")).json()) == ["agent-done"]


@pytest.mark.asyncio
async def test_queue_and_history_lists_leave_hidden_jobs_out(client):
    queue = await (await client.get("/queue")).json()
    assert [i[1] for i in queue["queue_running"]] == ["user-running"]
    assert [i[1] for i in queue["queue_pending"]] == ["user-pending"]
    assert list(await (await client.get("/history")).json()) == ["user-done"]
    assert list(await (await client.get("/history?max_items=1")).json()) == ["user-done"]


@pytest.mark.asyncio
async def test_status_queue_remaining_leaves_hidden_jobs_out(client):
    assert client.prompt_server.get_queue_info() == {"exec_info": {"queue_remaining": 2}}


@pytest.mark.asyncio
async def test_a_prompt_posted_as_hidden_is_queued_hidden(client, monkeypatch):
    async def valid(prompt_id, prompt, partial_execution_targets):
        return True, None, ["1"], {}

    monkeypatch.setattr(execution, "validate_prompt", valid)
    response = await client.post("/prompt", json={"prompt": {}, "client_id": "c", "extra_data": {"hidden": True}})
    assert response.status == 200
    prompt_id = (await response.json())["prompt_id"]

    queue = await (await client.get("/queue")).json()
    assert [i[1] for i in queue["queue_pending"]] == ["user-pending"]
    assert client.prompt_server.get_queue_info() == {"exec_info": {"queue_remaining": 2}}
    assert (await (await client.get(f"/api/jobs/{prompt_id}")).json())["id"] == prompt_id
