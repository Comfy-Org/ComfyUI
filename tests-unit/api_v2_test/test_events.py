"""Job events driven through the listener and the raw HTTP stream."""
# allow: SIZE_OK — plan 14 keeps its protocol scenarios together in this test module.

import asyncio
import base64
import json
import struct

import pytest

from app.api_v2 import assets, events, jobs
from app.assets.services.ingest import register_executed_output
from comfy_execution.jobs import CANCEL_PENDING, CANCEL_RUNNING

from .contract import validate
from .test_jobs_read import add_record, history_item, new_id, queue_item, write_output

pytestmark = pytest.mark.asyncio
SCHEMAS = {"status": "StatusEvent", "progress": "Progress", "preview": "PreviewEvent", "output": "Output", "error": "ErrorEnvelope"}


def tracked(app, server, graph=None, queued=False):
    jid = new_id()
    record = add_record(app, jid, created_at=jobs.now())
    if graph is not None:
        record.workflow = graph
    q = server.prompt_queue.queued if queued else server.prompt_queue.running
    q.append(queue_item(jid, graph=record.workflow))
    tracker = events.JobTracker(server, app[jobs.JOBS_KEY], record, "http://localhost")
    return jid, record, tracker


def send(tracker, event, **data):
    tracker.on_message(event, {"prompt_id": tracker.record.id, **data}, True)


def broadcast(server):
    for listener in list(server.sid_listeners.values()):
        listener("status", {}, False)


async def frame(response):
    raw = await asyncio.wait_for(response.content.readuntil(b"\n\n"), 1)
    assert raw.endswith(b"\n\n")
    lines = raw[:-2].decode().splitlines()
    assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
    name, data = lines[0][7:], json.loads(lines[1][6:])
    validate(SCHEMAS[name], data)
    return name, data


async def rest(response):
    result = []
    while not response.content.at_eof():
        item = await frame(response)
        result.append(item)
        if item[0] == "error" or item[0] == "status" and item[1]["status"] in events.TERMINAL:
            assert await asyncio.wait_for(response.content.read(), 1) == b""
            break
    return result


async def test_snapshot_order_and_finish(conforming_client, v2_app, fake_server, v2_dirs):
    jid, record, tracker = tracked(v2_app, fake_server)
    send(tracker, "progress", value=1, max=4, node="9")
    tracker.on_message(1, struct.pack(">I", 2) + b"png", True)
    ref = register_executed_output(str(write_output(v2_dirs, "early.png")), job_id=jid).id
    send(tracker, "executed", node="9", output={"images": [{"id": ref, "filename": "early.png"}]})
    await tracker.flush()
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    initial = [await frame(response) for _ in range(4)]
    assert [name for name, _ in initial] == ["status", "progress", "preview", "output"]
    body = await (await conforming_client.get(f"/api/v2/jobs/{jid}")).json()
    assert body["outputs"][0]["id"] == ref and ref in record.outputs
    assert body["progress"] == tracker.latest_progress
    tracker.finish("canceled")
    assert await rest(response) == [("status", {"status": "canceled"})]
    assert response.headers["Cache-Control"] == "no-cache, no-store, no-transform"
    assert response.headers["X-Accel-Buffering"] == "no"
    assert response.headers["Connection"] == "keep-alive"
    assert response.headers["Content-Type"] == "text/event-stream"


async def test_connect_after_complete_is_one_terminal(conforming_client, fake_server):
    jid = new_id()
    fake_server.prompt_queue.history[jid] = history_item(jid)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    assert await rest(response) == [("status", {"status": "succeeded"})]


@pytest.mark.parametrize("total", [0, 5])
async def test_stable_progress_denominator(v2_app, fake_server, total):
    graph = {str(n): {"class_type": "X"} for n in range(total)}
    _, _, tracker = tracked(v2_app, fake_server, graph)
    send(tracker, "progress_state", nodes={"0": {"state": "running", "value": 1, "max": 4}})
    progress = tracker.latest_progress
    validate("Progress", progress)
    assert progress["nodes_total"] == total and progress["value"] == (0.05 if total else 0)
    send(tracker, "execution_cached", nodes=["1", "2"])
    send(tracker, "progress_state", nodes={"1": {"state": "finished"}, "3": {"state": "finished"}})
    assert tracker.latest_progress["nodes_done"] == min(3, total)
    tracker.finish("succeeded")
    assert tracker.latest_progress["value"] == 1 and tracker.latest_progress["nodes_done"] == total


async def test_queued_running_and_position(conforming_client, v2_app, fake_server):
    jid, _, tracker = tracked(v2_app, fake_server, queued=True)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    assert (await frame(response))[1] == {"status": "queued", "queue_position": 1}
    fake_server.prompt_queue.queued.insert(0, queue_item(new_id(), number=-1))
    broadcast(fake_server)
    assert (await frame(response))[1] == {"status": "queued", "queue_position": 2}
    send(tracker, "execution_start")
    assert (await frame(response))[1] == {"status": "running"}
    assert tracker.started_at is not None
    tracker.finish("canceled")
    await rest(response)


@pytest.mark.parametrize("kind", [1, 4])
async def test_preview_formats_and_targeting(v2_app, fake_server, kind):
    _, _, tracker = tracked(v2_app, fake_server)
    send(tracker, "executing", node="9")
    metadata = json.dumps({"node_id": "other", "image_type": "image/png"}).encode()
    payload = struct.pack(">I", 2) + b"image" if kind == 1 else struct.pack(">I", len(metadata)) + metadata + b"image"
    tracker.on_message(kind, payload, False)
    assert tracker.latest_preview is None
    tracker.on_message(kind, payload, True)
    assert tracker.latest_preview == {"node_id": "9" if kind == 1 else "other", "content_type": "image/png", "data_base64": base64.b64encode(b"image").decode()}
    validate("PreviewEvent", tracker.latest_preview)
    tracker.finish("canceled")


async def test_throttle_coalesces_latest(v2_app, fake_server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(events, "monotonic", lambda: clock[0])
    _, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    send(tracker, "progress", value=1, max=10, node="9")
    send(tracker, "progress", value=2, max=10, node="9")
    assert queue.qsize() == 1 and tracker.latest_progress["step"] == 2
    clock[0] += 0.5
    tracker.flush_throttled("progress")
    assert queue.qsize() == 2 and (await queue.get())[1]["step"] == 1
    assert (await queue.get())[1]["step"] == 2
    tracker.finish("canceled")


@pytest.mark.parametrize("event,status", [("execution_success", "succeeded"), ("execution_error", "failed"), ("execution_interrupted", "canceled")])
async def test_waits_for_history_outputs_before_terminal(conforming_client, v2_app, fake_server, v2_dirs, monkeypatch, event, status):
    monkeypatch.setattr(events, "HISTORY_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    send(tracker, event)
    send(tracker, event)
    assert not tracker.closed
    ref = register_executed_output(str(write_output(v2_dirs, "last.png")), job_id=jid).id
    outputs = {"9": {"images": [{"id": ref, "filename": "last.png", "type": "output"}]}}
    messages = [(event, {"timestamp": 1})]
    fake_server.prompt_queue.history[jid] = history_item(jid, outputs, "success" if status == "succeeded" else "error", messages)
    send(tracker, "executing", node=None)
    items = await rest(response)
    assert [n for n, _ in items] == ["output", "status"]
    assert items[-1][1]["status"] == status
    assert jid not in fake_server.sid_listeners and jid not in fake_server.sockets_metadata
    send(tracker, "progress", value=9, max=10)
    assert tracker.closed


async def test_fallback_history_poll(conforming_client, v2_app, fake_server, monkeypatch):
    monkeypatch.setattr(events, "HISTORY_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    send(tracker, "execution_success")
    fake_server.prompt_queue.history[jid] = history_item(jid)
    assert (await rest(response))[-1] == ("status", {"status": "succeeded"})


@pytest.mark.parametrize("disappear", [False, True])
async def test_untracked_polling(conforming_client, fake_server, monkeypatch, disappear):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    jid = new_id()
    fake_server.prompt_queue.running.append(queue_item(jid))
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    assert (await frame(response))[1]["status"] == "running"
    fake_server.prompt_queue.running.clear()
    if not disappear:
        fake_server.prompt_queue.history[jid] = history_item(jid)
    items = await rest(response)
    assert items[-1][0] == ("error" if disappear else "status")
    if disappear:
        assert items[-1][1]["error"]["code"] == "job_not_found"


async def test_stream_cap_disconnect_reconnect(conforming_client, v2_app, fake_server):
    jid, _, tracker = tracked(v2_app, fake_server)
    path = f"/api/v2/jobs/{jid}/events"
    responses = [await conforming_client.get(path) for _ in range(8)]
    for response in responses:
        await frame(response)
    refused = await conforming_client.get(path)
    assert refused.status == 429 and refused.headers["Retry-After"] == "5"
    body = await refused.json()
    validate("ErrorEnvelope", body)
    assert body["error"]["code"] == "too_many_streams"
    responses.pop().close()
    for _ in range(100):
        if len(tracker.subscribers) == 7:
            break
        await asyncio.sleep(0)
    replacement = await conforming_client.get(path)
    assert replacement.status == 200
    await frame(replacement)
    tracker.finish("canceled")
    for response in [*responses, replacement]:
        await rest(response)


async def test_cleanup_replacement_and_backpressure(v2_app, fake_server):
    jid, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    for n in range(100):
        tracker.emit("status", {"status": "queued", "queue_position": n})
    assert queue.qsize() == 64
    replacement = lambda *args: None
    metadata = {"replacement": True}
    fake_server.sid_listeners[jid] = replacement
    fake_server.sockets_metadata[jid] = metadata
    tracker.finish("canceled")
    items = [queue.get_nowait() for _ in range(queue.qsize())]
    assert items[-1] == ("status", {"status": "canceled"})
    assert fake_server.sid_listeners[jid] is replacement and fake_server.sockets_metadata[jid] is metadata


@pytest.mark.parametrize("raw", ["bad", "{" + new_id() + "}", new_id()])
async def test_unknown_and_invalid_id(conforming_client, raw):
    response = await conforming_client.get(f"/api/v2/jobs/{raw}/events")
    assert response.status == 404
    validate("ErrorEnvelope", await response.json())


@pytest.mark.parametrize("status", ["succeeded", "failed", "canceled", "expired"])
async def test_full_queue_keeps_terminal_last_when_later_emits_arrive(v2_app, fake_server, status):
    _, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    for position in range(64):
        tracker.emit("status", {"status": "queued", "queue_position": position})
    assert queue.full()
    tracker.finish(status)
    tracker.emit("progress", {"value": 0.5, "nodes_done": 0, "nodes_total": 1})
    tracker.finish("canceled")
    tracker.on_message("status", {}, False)
    frames = [queue.get_nowait() for _ in range(queue.qsize())]
    assert len(frames) == 64
    assert frames[0] == ("status", {"status": "queued", "queue_position": 1})
    assert frames[-1] == ("status", {"status": status})
    assert sum(name == "status" and payload["status"] in events.TERMINAL for name, payload in frames) == 1


async def test_cleaned_tracker_falls_back_to_polling(conforming_client, v2_app, fake_server, monkeypatch):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    v2_app[jobs.JOBS_KEY].prune(jobs.now() + jobs.RETENTION)
    fake_server.prompt_queue.history[jid] = history_item(jid)
    assert tracker.closed
    assert await rest(response) == [("status", {"status": "succeeded"})]


async def test_preview_throttle_and_cleanup_cancel_timer(v2_app, fake_server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(events, "monotonic", lambda: clock[0])
    _, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    for image in (b"first", b"latest"):
        tracker.on_message(1, struct.pack(">I", 1) + image, True)
    assert queue.qsize() == 1
    assert tracker.latest_preview["data_base64"] == base64.b64encode(b"latest").decode()
    timer = tracker._timers["preview"]
    tracker.finish("canceled")
    assert timer.cancelled() and tracker.latest_preview is None


async def test_pending_cancel_route_closes_stream(conforming_client, v2_app, fake_server, monkeypatch):
    jid, record, tracker = tracked(v2_app, fake_server, queued=True)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)

    def cancel(job_id):
        fake_server.prompt_queue.queued.clear()
        return CANCEL_PENDING

    monkeypatch.setattr(fake_server, "cancel_job_by_id", cancel)
    result = await conforming_client.post(f"/api/v2/jobs/{jid}/cancel")
    assert (await result.json())["status"] == "canceled" and record.canceled_at is not None
    assert await rest(response) == [("status", {"status": "canceled"})]
    assert tracker.closed


def interrupted(server, jid):
    server.prompt_queue.running.clear()
    server.prompt_queue.history[jid] = history_item(jid, status_str="error", messages=[("execution_interrupted", {"timestamp": 1})])


async def test_running_cancel_streams_canceling_once(conforming_client, v2_app, fake_server, monkeypatch):
    jid, _, tracker = tracked(v2_app, fake_server, queued=True)
    q = fake_server.prompt_queue
    monkeypatch.setattr(fake_server, "cancel_job_by_id", lambda job_id: CANCEL_RUNNING)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    q.running.append(q.queued.pop())
    send(tracker, "execution_start")
    assert await frame(response) == ("status", {"status": "running"})
    for _ in range(2):
        assert (await (await conforming_client.post(f"/api/v2/jobs/{jid}/cancel")).json())["status"] == "canceling"
    assert tracker.status == "canceling"
    interrupted(fake_server, jid)
    send(tracker, "execution_interrupted")
    assert await rest(response) == [("status", {"status": "canceling"}), ("status", {"status": "canceled"})]


async def test_cancel_before_execution_start_never_reports_running(v2_app, fake_server, monkeypatch):
    jid, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    monkeypatch.setattr(fake_server, "cancel_job_by_id", lambda job_id: CANCEL_RUNNING)
    await jobs.cancel(fake_server, v2_app[jobs.JOBS_KEY], jid, "default", "http://localhost")
    send(tracker, "execution_start")
    tracker.finish("canceled")
    assert [queue.get_nowait() for _ in range(queue.qsize())] == [("status", {"status": "canceling"}), ("status", {"status": "canceled"})]


async def test_running_cancel_after_terminal_keeps_terminal(v2_app, fake_server, monkeypatch):
    jid, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    tracker.finish("succeeded")
    monkeypatch.setattr(fake_server, "cancel_job_by_id", lambda job_id: CANCEL_RUNNING)
    await jobs.cancel(fake_server, v2_app[jobs.JOBS_KEY], jid, "default", "http://localhost")
    assert tracker.status == "succeeded"
    assert [queue.get_nowait() for _ in range(queue.qsize())] == [("status", {"status": "succeeded"})]


@pytest.mark.parametrize("fallback", [False, True])
async def test_polled_canceling_is_sent_once(conforming_client, v2_app, fake_server, monkeypatch, fallback):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    monkeypatch.setattr(fake_server, "cancel_job_by_id", lambda job_id: CANCEL_RUNNING)
    if fallback:
        jid, _, tracker = tracked(v2_app, fake_server)
    else:
        jid = new_id()
        fake_server.prompt_queue.running.append(queue_item(jid))
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    assert await frame(response) == ("status", {"status": "running"})
    await conforming_client.post(f"/api/v2/jobs/{jid}/cancel")
    assert await frame(response) == ("status", {"status": "canceling"})
    if fallback:
        tracker.cleanup()
    await asyncio.sleep(0.05)
    interrupted(fake_server, jid)
    assert await rest(response) == [("status", {"status": "canceled"})]


async def test_asset_task_failure_falls_back_to_history(conforming_client, v2_app, fake_server, v2_dirs, monkeypatch):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    ref = register_executed_output(str(write_output(v2_dirs, "retry.png")), job_id=jid).id
    original = assets.output_json
    calls = []

    def flaky(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("temporary asset read failure")
        return original(*args)

    monkeypatch.setattr(assets, "output_json", flaky)
    output = {"images": [{"id": ref, "filename": "retry.png", "type": "output"}]}
    send(tracker, "executed", node="9", output=output)
    fake_server.prompt_queue.history[jid] = history_item(jid, {"9": output})
    send(tracker, "executing", node=None)
    items = await rest(response)
    assert [name for name, _ in items] == ["output", "status"]
    assert items[-1] == ("status", {"status": "succeeded"})


async def test_executed_without_ui_output_keeps_live_tracking(conforming_client, v2_app, fake_server, v2_dirs, monkeypatch):
    monkeypatch.setattr(events, "HISTORY_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    assert await frame(response) == ("status", {"status": "running"})
    send(tracker, "executed", node="1", output=None)
    send(tracker, "executed", node="2", output=["not", "a", "dict"])
    await asyncio.sleep(0.01)
    assert not tracker.closed and fake_server.sid_listeners.get(jid) == tracker.on_message
    send(tracker, "progress", value=1, max=2, node="9")
    name, progress = await frame(response)
    assert name == "progress" and progress["step"] == 1
    ref = register_executed_output(str(write_output(v2_dirs, "after.png")), job_id=jid).id
    output = {"images": [{"id": ref, "filename": "after.png", "type": "output"}]}
    send(tracker, "executed", node="9", output=output)
    name, live = await frame(response)
    assert name == "output" and live["id"] == ref and jid not in fake_server.prompt_queue.history
    fake_server.prompt_queue.history[jid] = history_item(jid, {"9": output})
    send(tracker, "executing", node=None)
    assert await rest(response) == [("status", {"status": "succeeded"})]
    assert tracker.closed


@pytest.mark.parametrize("removal", ["delete", "clear", "legacy_cancel"])
async def test_external_queued_removal_closes_stream(conforming_client, v2_app, fake_server, monkeypatch, removal):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server, queued=True)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    if removal == "delete":
        fake_server.prompt_queue.queued[:] = [item for item in fake_server.prompt_queue.queued if item[1] != jid]
    else:
        fake_server.prompt_queue.queued.clear()
    broadcast(fake_server)
    assert await rest(response) == [("status", {"status": "expired"})]
    assert tracker.closed and jid not in fake_server.sid_listeners
    assert not v2_app[jobs.JOBS_KEY].open_streams


@pytest.mark.parametrize("outcome", ["expired", "canceled", "succeeded", "missing"])
async def test_silent_running_removal_uses_poll_fallback(conforming_client, v2_app, fake_server, monkeypatch, outcome):
    monkeypatch.setattr(events, "POLL_INTERVAL", 0.001)
    jid, _, tracker = tracked(v2_app, fake_server)
    response = await conforming_client.get(f"/api/v2/jobs/{jid}/events")
    await frame(response)
    fake_server.prompt_queue.running.clear()
    if outcome in ("canceled", "succeeded"):
        messages = [("execution_interrupted", {})] if outcome == "canceled" else []
        fake_server.prompt_queue.history[jid] = history_item(jid, status_str="error" if outcome == "canceled" else "success", messages=messages)
    elif outcome == "missing":
        v2_app[jobs.JOBS_KEY].records.pop(jid)
    items = await rest(response)
    if outcome == "missing":
        assert items[-1][0] == "error" and items[-1][1]["error"]["code"] == "job_not_found"
    else:
        assert items == [("status", {"status": outcome})]
    assert tracker.closed


async def test_broadcast_snapshots_queue_once_for_all_trackers(v2_app, fake_server, monkeypatch):
    trackers = [tracked(v2_app, fake_server, queued=True)[2] for _ in range(5)]
    fake_server.prompt_queue.queued[:] = [queue_item(tracker.record.id, number=n) for n, tracker in enumerate(trackers)]
    queues = [tracker.subscribe() for tracker in trackers]
    snapshot = fake_server.prompt_queue.get_current_queue_volatile
    calls = []

    def counted():
        calls.append(True)
        return snapshot()

    monkeypatch.setattr(fake_server.prompt_queue, "get_current_queue_volatile", counted)
    broadcast(fake_server)
    assert len(calls) == 1
    assert [queue.get_nowait()[1]["queue_position"] for queue in queues] == [1, 2, 3, 4, 5]
    for tracker in trackers:
        tracker.finish("canceled")


async def test_slow_subscriber_keeps_only_latest_preview(v2_app, fake_server):
    _, _, tracker = tracked(v2_app, fake_server)
    queue = tracker.subscribe()
    for number in range(70):
        tracker.emit("preview", {"data_base64": str(number)})
    assert queue.qsize() == 1
    assert queue.get_nowait() == ("preview", {"data_base64": "69"})
    tracker.emit("output", {"id": "output"})
    tracker.emit("preview", {"data_base64": "pending"})
    tracker.finish("canceled")
    assert [queue.get_nowait() for _ in range(queue.qsize())] == [("output", {"id": "output"}), ("status", {"status": "canceled"})]


async def test_broadcast_during_submit_does_not_expire_new_tracker(v2_app, fake_server):
    jid, _, tracker = tracked(v2_app, fake_server, queued=True)
    fake_server.prompt_queue.queued.clear()
    broadcast(fake_server)
    assert tracker._terminal_task is None and not tracker.closed
    fake_server.prompt_queue.queued.append(queue_item(jid))
    broadcast(fake_server)
    assert tracker.position == 1
    tracker.finish("canceled")


async def test_runtime_expansion_clamps_progress(v2_app, fake_server):
    _, _, tracker = tracked(v2_app, fake_server, {"0": {"class_type": "X"}})
    send(tracker, "execution_cached", nodes=["0", "0:1", "0:2"])
    send(tracker, "progress_state", nodes={"0:3": {"state": "finished"}})
    assert tracker.latest_progress["nodes_done"] == tracker.latest_progress["nodes_total"] == 1
    assert tracker.latest_progress["value"] == 1.0
    tracker.finish("canceled")
