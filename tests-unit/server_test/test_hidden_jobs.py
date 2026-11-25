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
    ({"hidden": 1}, False),
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


def test_history_offset_counts_visible_entries_only():
    queue = execution.PromptQueue(MagicMock())
    queue.put(item(1, "agent-done", hidden=True))
    queue.put(item(2, "user-a"))
    queue.put(item(3, "user-b"))
    assert [finish(queue) for _ in range(3)] == ["agent-done", "user-a", "user-b"]
    visible = server._is_visible_history_entry
    assert list(queue.get_history(offset=1, filter_function=visible)) == ["user-b"]
    assert list(queue.get_history(max_items=2, filter_function=visible)) == ["user-a", "user-b"]


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
async def test_status_is_broadcast_only_when_the_visible_queue_changes(client):
    prompt_server = client.prompt_server
    del prompt_server.queue_updated  # the fixture's no-op
    counts = []
    prompt_server.send_sync = lambda event, data, sid=None: counts.append(data["status"]["exec_info"]["queue_remaining"])
    prompt_server.prompt_queue = queue = execution.PromptQueue(prompt_server)
    status = execution.PromptQueue.ExecutionStatus(status_str="success", completed=True, messages=[])

    queue.put(item(0, "agent-0", hidden=True))  # the first change after startup
    queue.get(timeout=0)
    queue.put(item(1, "user"))
    queue.put(item(2, "agent", hidden=True))
    _, user_id = queue.get(timeout=0)
    _, agent_id = queue.get(timeout=0)
    queue.task_done(user_id, {}, status, process_item=lambda p: p[:5])
    queue.task_done(agent_id, {}, status, process_item=lambda p: p[:5])
    queue.put(item(3, "agent-2", hidden=True))
    # queued, started (the frontend refreshes its queue panel), drained: once each, as for a queue with no hidden jobs
    assert counts == [1, 1, 0]
    queue.put(item(4, "retry"))
    queue.put(item(5, "retry"))  # a client may resubmit with the same prompt_id
    assert counts == [1, 1, 0, 1, 2]


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
