"""getJob, cancelJob, getJobWorkflow and getJobLogs over the fake queue and history."""

import asyncio
import json
import types
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution  # noqa: E402
import server  # noqa: E402
from app.api_v2 import jobs  # noqa: E402
from app.assets.services.asset_management import delete_asset_reference, get_asset_detail  # noqa: E402
from app.assets.services.ingest import register_executed_output  # noqa: E402
from comfy_execution.jobs import CANCEL_PENDING, CANCEL_RUNNING, CANCEL_TERMINAL, CANCEL_UNKNOWN  # noqa: E402

from .contract import response_schema, validate  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"v2-jobs-read"
WF = {"9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}}
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
MS = int(T0.timestamp() * 1000)


@pytest.fixture
def clock(monkeypatch):
    t = {"now": T0}
    monkeypatch.setattr(jobs, "now", lambda: t["now"])
    return t


@pytest.fixture
def cancels(fake_server):
    """``cancel_job_by_id`` with the real classification over the fake queue; collects (id, outcome)."""
    calls = []
    q = fake_server.prompt_queue

    def cancel(job_id):
        if any(item[1] == job_id for item in q.running):
            outcome = CANCEL_RUNNING
        elif any(item[1] == job_id for item in q.queued):
            q.queued = [item for item in q.queued if item[1] != job_id]
            outcome = CANCEL_PENDING
        else:
            outcome = CANCEL_TERMINAL if job_id in q.history else CANCEL_UNKNOWN
        calls.append((job_id, outcome))
        return outcome

    fake_server.cancel_job_by_id = cancel
    return calls


def new_id() -> str:
    return str(uuid.uuid4())


def queue_item(job_id, number=1, graph=WF, create_time=MS):
    return (number, job_id, graph, {"create_time": create_time, "client_id": job_id}, ["9"], {})


def history_item(job_id, outputs=None, status_str="success", messages=None, graph=WF):
    if messages is None:
        messages = [("execution_start", {"prompt_id": job_id, "timestamp": MS + 1000}), ("execution_cached", {"nodes": [], "prompt_id": job_id, "timestamp": MS + 1000})]
        messages.append(("execution_success" if status_str == "success" else "execution_error", {"prompt_id": job_id, "timestamp": MS + 5000}))
    return {
        "prompt": queue_item(job_id, graph=graph)[:5],
        "outputs": outputs or {},
        "status": {"status_str": status_str, "completed": status_str == "success", "messages": messages},
        "meta": {},
    }


def add_record(v2_app, job_id, user="default", metadata=None, priority=None, created_at=T0) -> jobs.JobRecord:
    record = jobs.JobRecord(job_id, user, created_at, WF, metadata, priority)
    v2_app[jobs.JOBS_KEY].add(record, set)
    return record


def write_output(v2_dirs, name, data=PNG, subfolder=""):
    path = v2_dirs / "output" / subfolder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


async def get(client, job_id, user=None, op="getJob", suffix=""):
    headers = {"comfy-user": user} if user else {}
    resp = await client.get(f"/api/v2/jobs/{job_id}{suffix}", headers=headers)
    data = await resp.json() if resp.status != 204 else None
    schema = response_schema(op, resp.status)
    if schema is not None:
        validate(schema, data)
    return resp, data


async def cancel(client, job_id, user=None):
    headers = {"comfy-user": user} if user else {}
    resp = await client.post(f"/api/v2/jobs/{job_id}/cancel", headers=headers)
    data = await resp.json()
    validate(response_schema("cancelJob", resp.status), data)
    return resp, data


# history: succeeded, failed, canceled


@pytest.mark.asyncio
async def test_succeeded_job_emits_registered_outputs(conforming_client, v2_app, v2_dirs, fake_server, clock):
    job_id = new_id()
    path = write_output(v2_dirs, "ComfyUI_00001_.png")
    ref = register_executed_output(str(path), job_id=job_id).id
    detail = get_asset_detail(ref)
    outputs = {"9": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output", "id": ref}], "animated": [False]}}
    fake_server.prompt_queue.history[job_id] = history_item(job_id, outputs)
    resp, body = await get(conforming_client, job_id)
    assert resp.status == 200, body
    assert body["status"] == "succeeded" and body["error"] is None and body["outputs_reused"] is False
    assert (body["created_at"], body["started_at"], body["completed_at"], body["expires_at"]) == ("2026-11-14T12:00:00Z", "2026-11-14T12:00:01Z", "2026-11-14T12:00:05Z", "2026-11-15T12:00:00Z")
    base = str(conforming_client.make_url("")).rstrip("/")
    [out] = body["outputs"]
    assert out["url_expires_at"].endswith("Z")
    del out["url_expires_at"]
    assert out == {
        "node_id": "9", "name": "ComfyUI_00001_.png", "type": "image", "content_type": "image/png",
        "size_bytes": len(PNG), "id": ref, "hash": detail.asset.hash, "url": f"{base}/api/v2/assets/{ref}/content", "job_id": job_id,
    }
    assert "metadata" not in body and "priority" not in body and "logs" not in body["urls"]


@pytest.mark.asyncio
async def test_output_without_id_is_registered_once(conforming_client, v2_dirs, fake_server, monkeypatch):
    job_id = new_id()
    write_output(v2_dirs, "a.png", subfolder="sub")
    calls = []
    real = fake_server.asset_manager.register_executed_output

    def counting(abs_path, job, preview_ref=None):
        calls.append((abs_path, job))
        return real(abs_path, job)

    monkeypatch.setattr(fake_server.asset_manager, "register_executed_output", counting)
    outputs = {"9": {"images": [{"filename": "a.png", "subfolder": "sub", "type": "output"}]}}
    fake_server.prompt_queue.history[job_id] = history_item(job_id, outputs)
    first = await asyncio.gather(*(get(conforming_client, job_id) for _ in range(3)))
    _, again = await get(conforming_client, job_id)
    ids = {body["outputs"][0]["id"] for _, body in first} | {again["outputs"][0]["id"]}
    assert len(ids) == 1 and calls == [(str(v2_dirs / "output" / "sub" / "a.png"), job_id)]
    [ref] = ids
    assert get_asset_detail(ref).ref.job_id == job_id and again["outputs"][0]["size_bytes"] == len(PNG)


@pytest.mark.asyncio
async def test_output_without_id_reuses_live_content(conforming_client, v2_dirs, fake_server, monkeypatch):
    """A file another record already holds (e.g. from the output scan) is registered like a cached replay: no content is retired."""
    job_id = new_id()
    path = write_output(v2_dirs, "scanned.png")
    other = register_executed_output(str(path), job_id=None).id
    monkeypatch.setattr(fake_server.asset_manager, "register_executed_output", lambda *a, **k: pytest.fail("would retire live content"))
    fake_server.prompt_queue.history[job_id] = history_item(job_id, {"9": {"images": [{"filename": "scanned.png", "subfolder": "", "type": "output"}]}})
    _, body = await get(conforming_client, job_id)
    [out] = body["outputs"]
    mine, theirs = get_asset_detail(out["id"]), get_asset_detail(other)
    assert out["id"] != other and mine.ref.job_id == job_id and mine.ref.file_path == theirs.ref.file_path == str(path)
    assert not theirs.asset.is_missing and [o["id"] for o in (await get(conforming_client, job_id))[1]["outputs"]] == [out["id"]]


@pytest.mark.asyncio
async def test_lazily_registered_output_ids_are_bounded(conforming_client, v2_app, v2_dirs, fake_server, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_OUTPUT_IDS", 2)
    job_ids = [new_id() for _ in range(3)]
    for i, job_id in enumerate(job_ids):
        write_output(v2_dirs, f"b{i}.png")
        fake_server.prompt_queue.history[job_id] = history_item(job_id, {"9": {"images": [{"filename": f"b{i}.png", "subfolder": "", "type": "output"}]}})
        _, body = await get(conforming_client, job_id)
        assert len(body["outputs"]) == 1
    assert [key[0] for key in v2_app[jobs.JOBS_KEY].output_ids] == job_ids[1:]


@pytest.mark.asyncio
async def test_missing_or_escaping_files_are_omitted(conforming_client, v2_dirs, fake_server):
    job_id = new_id()
    (v2_dirs / "outside.png").write_bytes(PNG)
    gone = register_executed_output(str(write_output(v2_dirs, "gone.png")), job_id=job_id).id
    delete_asset_reference(reference_id=gone)
    items = [
        {"filename": "missing.png", "subfolder": "", "type": "output"},
        {"filename": "outside.png", "subfolder": "..", "type": "output"},
        {"filename": "x.png", "subfolder": "", "type": "nowhere"},
        {"filename": 3, "type": "output"},
        {"filename": "y.png", "subfolder": 7, "type": "output"},
        {"filename": "z.png", "subfolder": "", "type": "output", "id": gone},
        {"filename": "w.png", "subfolder": "", "type": "output", "id": "not-a-uuid"},
        "a text value",
    ]
    outputs = {"9": {"images": items, "text": ["hello"], "scalar": 3}, "10": "junk"}
    fake_server.prompt_queue.history[job_id] = history_item(job_id, outputs)
    resp, body = await get(conforming_client, job_id)
    assert resp.status == 200 and body["status"] == "succeeded" and body["outputs"] == []


@pytest.mark.asyncio
async def test_output_type_mapping(conforming_client, v2_dirs, fake_server):
    job_id = new_id()
    keys = {"images": "image", "gifs": "video", "video": "video", "audio": "audio", "text": "text", "files": "file", "latents": "latent", "3d": "file"}
    outputs = {}
    for n, key in enumerate(keys):
        write_output(v2_dirs, f"{n}.bin")
        outputs[str(n)] = {key: [{"filename": f"{n}.bin", "subfolder": "", "type": "output"}]}
    fake_server.prompt_queue.history[job_id] = history_item(job_id, outputs)
    resp, body = await get(conforming_client, job_id)
    assert resp.status == 200, body
    assert [(o["node_id"], o["type"]) for o in body["outputs"]] == [(str(n), t) for n, t in enumerate(keys.values())]


@pytest.mark.asyncio
async def test_failed_job_carries_error_and_reuse(conforming_client, fake_server):
    job_id = new_id()
    messages = [
        ("execution_start", {"prompt_id": job_id, "timestamp": MS + 1000}),
        ("execution_cached", {"nodes": ["1"], "prompt_id": job_id, "timestamp": MS + 1000}),
        ("execution_error", {"prompt_id": job_id, "node_id": "9", "node_type": "SaveImage", "exception_message": "boom", "exception_type": "RuntimeError", "traceback": ["Traceback\n", "  line\n"], "timestamp": MS + 2000}),
    ]
    fake_server.prompt_queue.history[job_id] = history_item(job_id, status_str="error", messages=messages)
    resp, body = await get(conforming_client, job_id)
    assert body["status"] == "failed" and body["outputs_reused"] is True and body["completed_at"] == "2026-11-14T12:00:02Z"
    assert body["error"] == {"code": "node_execution_error", "message": "boom", "node_id": "9", "class_type": "SaveImage", "traceback": "Traceback\n  line\n"}


@pytest.mark.asyncio
async def test_failed_job_without_traceback_or_messages(conforming_client, fake_server):
    job_id = new_id()
    messages = [("execution_error", {"node_id": "9", "node_type": "SaveImage", "exception_message": "boom", "traceback": [], "timestamp": MS})]
    fake_server.prompt_queue.history[job_id] = history_item(job_id, status_str="error", messages=messages)
    _, body = await get(conforming_client, job_id)
    assert body["error"]["traceback"] is None and body["started_at"] is None
    other = new_id()
    fake_server.prompt_queue.history[other] = {**history_item(other), "status": None}
    _, body = await get(conforming_client, other)
    assert (body["status"], body["error"], body["started_at"], body["completed_at"]) == ("failed", None, None, None)


@pytest.mark.asyncio
async def test_interrupted_job_is_canceled(conforming_client, fake_server):
    job_id = new_id()
    messages = [("execution_start", {"timestamp": MS}), ("execution_interrupted", {"prompt_id": job_id, "node_id": "9", "node_type": "SaveImage", "executed": [], "timestamp": MS + 3000})]
    fake_server.prompt_queue.history[job_id] = history_item(job_id, status_str="error", messages=messages)
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "canceled" and body["completed_at"] == "2026-11-14T12:00:03Z" and body["error"] is None


# running and queued


@pytest.mark.asyncio
async def test_running_job_with_live_outputs_and_cancel_requested(conforming_client, v2_app, fake_server, clock):
    job_id = new_id()
    fake_server.prompt_queue.running.append(queue_item(job_id))
    record = add_record(v2_app, job_id)
    live = {"node_id": "9", "name": "a.png", "type": "image", "content_type": "image/png", "size_bytes": 3, "id": new_id(), "hash": None, "url": "http://x/a", "url_expires_at": "2026-11-15T12:00:00Z", "job_id": job_id}
    record.outputs["9:0"] = live
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "running" and body["outputs"] == [live]
    assert body["progress"] == {"value": 0.0, "nodes_done": 0, "nodes_total": 0} and body["queue_position"] is None
    record.cancel_requested = True
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "canceling"


@pytest.mark.asyncio
async def test_terminal_history_outputs_replace_live_outputs(conforming_client, v2_app, fake_server, clock):
    job_id = new_id()
    record = add_record(v2_app, job_id)
    record.outputs["9:0"] = {"id": "stale"}
    fake_server.prompt_queue.history[job_id] = history_item(job_id)
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "succeeded" and body["outputs"] == []


@pytest.mark.asyncio
async def test_queue_position_and_created_at_from_queue(conforming_client, fake_server):
    ids = [new_id() for _ in range(3)]
    fake_server.prompt_queue.queued = [queue_item(ids[0], number=7), queue_item(ids[1], number=-2, create_time=MS + 60_000), queue_item(ids[2], number=3)]
    positions = {}
    for job_id in ids:
        resp, body = await get(conforming_client, job_id)
        assert body["status"] == "queued" and body["progress"] is None and body["outputs"] == []
        positions[job_id] = body["queue_position"]
    assert positions == {ids[1]: 1, ids[2]: 2, ids[0]: 3}
    _, body = await get(conforming_client, ids[1])
    assert body["created_at"] == "2026-11-14T12:01:00Z" and body["expires_at"] == "2026-11-15T12:01:00Z"


@pytest.mark.asyncio
async def test_job_finishing_between_reads_is_not_expired(conforming_client, v2_app, fake_server, clock):
    """Race: history misses, then the worker finishes the job before the queue read."""
    job_id = new_id()
    add_record(v2_app, job_id)
    q = fake_server.prompt_queue
    q.running.append(queue_item(job_id))
    real_volatile = q.get_current_queue_volatile

    def finishing():
        q.running.clear()
        q.history[job_id] = history_item(job_id)
        return real_volatile()

    q.get_current_queue_volatile = finishing
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "succeeded" and body["started_at"] == "2026-11-14T12:00:01Z"


# records without queue or history


@pytest.mark.asyncio
async def test_expired_record_then_not_found(conforming_client, v2_app, clock):
    job_id = new_id()
    add_record(v2_app, job_id, metadata={"client": "acme"})
    _, body = await get(conforming_client, job_id)
    assert body["status"] == "expired" and body["metadata"] == {"client": "acme"} and body["outputs"] == []
    clock["now"] = T0 + jobs.RETENTION
    resp, body = await get(conforming_client, job_id)
    assert resp.status == 404 and body["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_non_owner_sees_no_metadata_or_priority(conforming_client, v2_app, fake_server, clock):
    fake_server.user_manager.multi_user = True
    fake_server.user_manager.users |= {"bob"}
    job_id = new_id()
    add_record(v2_app, job_id, metadata={"client": "acme"}, priority=4)
    fake_server.prompt_queue.queued.append(queue_item(job_id))
    _, own = await get(conforming_client, job_id)
    _, other = await get(conforming_client, job_id, user="bob")
    assert own["metadata"] == {"client": "acme"} and own["priority"] == 4
    assert "metadata" not in other and "priority" not in other and other["status"] == "queued"
    resp, _ = await get(conforming_client, job_id, user="mallory")
    assert resp.status == 403


# cancel


@pytest.mark.asyncio
async def test_cancel_queued_job_then_repeat(conforming_client, v2_app, fake_server, cancels, clock):
    job_id = new_id()
    add_record(v2_app, job_id, metadata={"client": "acme"})
    fake_server.prompt_queue.queued.append(queue_item(job_id))
    clock["now"] = T0 + timedelta(seconds=30)
    resp, body = await cancel(conforming_client, job_id)
    assert resp.status == 200 and body["status"] == "canceled" and body["completed_at"] == "2026-11-14T12:00:30Z"
    assert body["metadata"] == {"client": "acme"}
    _, again = await cancel(conforming_client, job_id)
    _, polled = await get(conforming_client, job_id)
    assert again == body == polled
    assert cancels == [(job_id, CANCEL_PENDING), (job_id, CANCEL_UNKNOWN)]


@pytest.mark.asyncio
async def test_cancel_running_job_is_canceling(conforming_client, v2_app, fake_server, cancels, clock):
    job_id = new_id()
    record = add_record(v2_app, job_id)
    fake_server.prompt_queue.running.append(queue_item(job_id))
    resp, body = await cancel(conforming_client, job_id)
    assert resp.status == 200 and body["status"] == "canceling" and record.cancel_requested
    _, again = await cancel(conforming_client, job_id)
    assert again["status"] == "canceling"


@pytest.mark.asyncio
async def test_cancel_non_v2_jobs_creates_minimal_record(conforming_client, v2_app, fake_server, cancels, clock):
    store = v2_app[jobs.JOBS_KEY]
    pending, running = new_id(), new_id()
    fake_server.prompt_queue.queued.append(queue_item(pending))
    fake_server.prompt_queue.running.append(queue_item(running, number=0))
    _, body = await cancel(conforming_client, pending)
    assert body["status"] == "canceled" and "metadata" not in body
    record = store.records[pending]
    assert (record.user_id, record.metadata, record.priority, record.workflow, record.canceled_at) == ("default", None, None, WF, T0)
    _, body = await get(conforming_client, pending, op="getJobWorkflow", suffix="/workflow")
    assert body == {"workflow": WF, "format": "api"}
    _, body = await cancel(conforming_client, running)
    assert body["status"] == "canceling" and store.records[running].cancel_requested


@pytest.mark.asyncio
async def test_cancel_non_v2_job_keeps_queue_create_time(conforming_client, v2_app, fake_server, cancels, clock):
    clock["now"] = T0 + timedelta(hours=1)
    pending, running = new_id(), new_id()
    fake_server.prompt_queue.queued.append(queue_item(pending))
    fake_server.prompt_queue.running.append(queue_item(running, number=0, create_time=None))
    _, body = await cancel(conforming_client, pending)
    assert body["created_at"] == "2026-11-14T12:00:00Z" and body["expires_at"] == "2026-11-15T12:00:00Z"
    assert v2_app[jobs.JOBS_KEY].records[pending].created_at == T0
    _, body = await cancel(conforming_client, running)
    assert body["created_at"] == "2026-11-14T13:00:00Z"


@pytest.mark.asyncio
async def test_non_v2_cancel_record_expires_behind_newer_record(conforming_client, v2_app, fake_server, cancels, clock):
    clock["now"] = T0 + timedelta(hours=1)
    newer, pending = new_id(), new_id()
    add_record(v2_app, newer, created_at=clock["now"])
    fake_server.prompt_queue.queued.append(queue_item(pending))
    await cancel(conforming_client, pending)
    clock["now"] = T0 + jobs.RETENTION
    resp, body = await get(conforming_client, pending)
    assert resp.status == 404 and body["error"]["code"] == "not_found"
    _, body = await get(conforming_client, newer)
    assert body["status"] == "expired"


@pytest.mark.asyncio
async def test_cancel_terminal_and_unknown(conforming_client, v2_app, fake_server, cancels):
    job_id = new_id()
    fake_server.prompt_queue.history[job_id] = history_item(job_id)
    resp, body = await cancel(conforming_client, job_id)
    assert resp.status == 200 and body["status"] == "succeeded"
    assert v2_app[jobs.JOBS_KEY].records == {}
    unknown = new_id()
    resp, body = await cancel(conforming_client, unknown)
    assert resp.status == 404 and body["error"]["code"] == "not_found"
    assert cancels == [(job_id, CANCEL_TERMINAL)]


# workflow and logs


@pytest.mark.asyncio
async def test_workflow_round_trip_without_secrets(conforming_client, v2_dirs, fake_server, monkeypatch):
    async def validate_prompt(prompt_id, prompt, partial_execution_targets):
        return True, None, ["9"], {}

    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    fake_server.number = 0
    fake_server.node_replace_manager = MagicMock()
    fake_server.trigger_on_prompt = lambda json_data: json_data
    fake_server.enqueue_prompt = types.MethodType(server.PromptServer.enqueue_prompt, fake_server)
    resp = await conforming_client.post("/api/v2/jobs", json={"workflow": WF, "extra_data": {"api_key_comfy_org": "key-secret"}})
    job_id = (await resp.json())["id"]
    q = fake_server.prompt_queue
    q.queued = list(q.put_items)
    for place in ("queued", "history", "record"):
        if place == "history":
            q.queued = []
            q.history[job_id] = {**history_item(job_id), "prompt": q.put_items[0][:5]}
        elif place == "record":
            q.history = {}
        resp, body = await get(conforming_client, job_id, op="getJobWorkflow", suffix="/workflow")
        assert resp.status == 200 and body == {"workflow": WF, "format": "api"}, place
        assert "key-secret" not in json.dumps(body)
    other = new_id()
    q.running.append(queue_item(other, graph={"1": {"class_type": "X", "inputs": {}}}))
    _, body = await get(conforming_client, other, op="getJobWorkflow", suffix="/workflow")
    assert body["workflow"] == {"1": {"class_type": "X", "inputs": {}}}
    resp, body = await get(conforming_client, new_id(), op="getJobWorkflow", suffix="/workflow")
    assert resp.status == 404 and body["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_logs_reasons(conforming_client, v2_app, fake_server, clock):
    finished, queued, running, canceled, expired = (new_id() for _ in range(5))
    fake_server.prompt_queue.history[finished] = history_item(finished)
    fake_server.prompt_queue.queued.append(queue_item(queued))
    fake_server.prompt_queue.running.append(queue_item(running))
    add_record(v2_app, canceled).canceled_at = T0
    add_record(v2_app, expired)
    want = {finished: "finished", queued: "not_finished", running: "not_finished", canceled: "finished", expired: "finished"}
    for job_id, reason in want.items():
        resp, _ = await get(conforming_client, job_id, op="getJobLogs", suffix="/logs")
        assert (resp.status, resp.headers.get("No-Log-Reason"), await resp.read()) == (204, reason, b""), job_id
    resp, body = await get(conforming_client, new_id(), op="getJobLogs", suffix="/logs")
    assert resp.status == 404 and body["error"]["code"] == "not_found"


# malformed ids


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["nope", "123", uuid.uuid4().hex, str(uuid.uuid4()).upper(), "{" + str(uuid.uuid4()) + "}", "%20"])
async def test_non_canonical_ids_are_not_found(conforming_client, fake_server, cancels, raw):
    fake_server.prompt_queue.history[raw] = history_item(raw)
    for method, suffix, op in (("GET", "", "getJob"), ("POST", "/cancel", "cancelJob"), ("GET", "/workflow", "getJobWorkflow"), ("GET", "/logs", "getJobLogs")):
        resp = await conforming_client.request(method, f"/api/v2/jobs/{raw}{suffix}")
        body = await resp.json()
        validate(response_schema(op, resp.status), body)
        assert (resp.status, body["error"]["code"]) == (404, "not_found"), (method, suffix)
    assert cancels == []
