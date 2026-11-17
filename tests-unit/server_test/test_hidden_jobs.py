"""Jobs submitted with "hidden": true in extra_data are left out of the job, queue and history lists and the queue
count, and are still returned by id."""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

args.cpu = True  # importing execution reads the device; CI's torch is CPU-only

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


@pytest.mark.parametrize("extra_data,hidden", [
    ({"hidden": True}, True),
    ({"hidden": "true"}, False),
    ({"hidden": False}, False),
    ({}, False),
    (None, False),
])
def test_is_hidden_job(extra_data, hidden):
    assert is_hidden_job((0, "p", {}, extra_data, [])) is hidden


def test_queue_count_and_history_list_leave_hidden_jobs_out():
    queue = execution.PromptQueue(MagicMock())
    queue.put(item(1, "user-done"))
    queue.put(item(2, "agent-done", hidden=True))
    assert [finish(queue), finish(queue)] == ["user-done", "agent-done"]
    queue.put(item(3, "agent-running", hidden=True))
    queue.get(timeout=0)
    queue.put(item(4, "user-pending"))
    queue.put(item(5, "agent-pending", hidden=True))

    assert queue.get_tasks_remaining() == 3
    assert queue.get_tasks_remaining(include_hidden=False) == 1
    assert list(queue.get_history()) == ["user-done", "agent-done"]
    assert list(queue.get_history(include_hidden=False)) == ["user-done"]
    assert list(queue.get_history(max_items=1, include_hidden=False)) == ["user-done"]
    assert list(queue.get_history(prompt_id="agent-done")) == ["agent-done"]


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.setattr(args, "enable_agent", False, raising=False)
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.prompt_queue = queue = execution.PromptQueue(prompt_server)
    prompt_server.queue_updated = lambda: None
    queue.put(item(1, "user-done"))
    queue.put(item(2, "agent-done", hidden=True))
    finish(queue)
    finish(queue)
    queue.put(item(3, "agent-running", hidden=True))
    queue.get(timeout=0)
    queue.put(item(4, "user-pending"))
    queue.put(item(5, "agent-pending", hidden=True))
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
    assert sorted(j["id"] for j in body["jobs"]) == ["user-done", "user-pending"]
    assert body["pagination"]["total"] == 2


@pytest.mark.asyncio
async def test_hidden_jobs_are_found_by_id(client):
    assert (await (await client.get("/api/jobs/agent-done")).json())["id"] == "agent-done"
    assert (await (await client.get("/api/jobs/agent-pending")).json())["id"] == "agent-pending"
    assert list(await (await client.get("/history/agent-done")).json()) == ["agent-done"]


@pytest.mark.asyncio
async def test_queue_and_history_lists_leave_hidden_jobs_out(client):
    queue = await (await client.get("/queue")).json()
    assert queue["queue_running"] == []
    assert [i[1] for i in queue["queue_pending"]] == ["user-pending"]
    assert list(await (await client.get("/history")).json()) == ["user-done"]
    assert list(await (await client.get("/history?max_items=1")).json()) == ["user-done"]


@pytest.mark.asyncio
async def test_status_queue_remaining_leaves_hidden_jobs_out(client):
    assert client.prompt_server.get_queue_info() == {"exec_info": {"queue_remaining": 1}}
