"""Live job events, owned by the submitting job's mutable tracker.

nodes_total counts the resolved workflow, including nodes outside the execution
subgraph. This is an accepted approximation, unlike counting progress_state's
active-only nodes, which would report near-completion as the first node starts.
"""
# allow: SIZE_OK — the plan requires tracker and SSE lifecycle in this module, with no extra package files.

import asyncio
import base64
import json
import logging
import struct
from time import monotonic

from aiohttp import web

from protocol import BinaryEventTypes

from . import assets, jobs

TERMINAL = {"succeeded", "failed", "canceled", "expired"}
THROTTLE_INTERVAL = 0.5
HISTORY_INTERVAL = 0.1
HISTORY_TIMEOUT = 30.0
POLL_INTERVAL = 1.0
KEEPALIVE_INTERVAL = 15.0
QUEUE_LISTENER = "_api_v2_queue_status"


class QueueStatus:
    """One loop-owned queue snapshot per server broadcast, shared by its trackers."""

    def __init__(self, server):
        self.server = server
        self.trackers: set[JobTracker] = set()

    def __call__(self, event, data, targeted: bool) -> None:
        if event != "status" or targeted:
            return
        running, queued = self.server.prompt_queue.get_current_queue_volatile()
        positions = jobs.queue_positions(queued)
        active = set(positions) | {item[1] for item in running}
        for tracker in list(self.trackers):
            if tracker.record.id not in active:
                if tracker.position is not None or tracker.started_at is not None or tracker.subscribers:
                    tracker._start_terminal(wait_history=False)
            elif tracker.status == "queued" and tracker.record.id in positions:
                position = positions[tracker.record.id]
                if position != tracker.position:
                    tracker.position = position
                    tracker.emit("status", {"status": "queued", "queue_position": position})


def drop_previews(queue: asyncio.Queue) -> None:
    frames = [queue.get_nowait() for _ in range(queue.qsize())]
    for name, data in frames:
        if name != "preview":
            queue.put_nowait((name, data))


def status_event(state: jobs.JobState) -> dict:
    result = {"status": state.status}
    if state.queue_position is not None:
        result["queue_position"] = state.queue_position
    return result


def sse_frame(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


class JobTracker:
    """Loop-owned accumulator; listeners never wait for history or asset I/O."""

    def __init__(self, server, store: jobs.JobStore, record: jobs.JobRecord, base: str):
        self.server, self.store, self.record, self.base = server, store, record, base
        self.latest_progress: dict | None = None
        self.latest_preview: dict | None = None
        self.current_node: str | None = None
        self.started_at = None
        self.emitted_output_ids: set[str] = set()
        self.subscribers: list[asyncio.Queue] = []
        self.closed = False
        self.status = "queued"
        self.position = None
        self.done: set[str] = set()
        self.fraction = 0.0
        self._last = {"progress": float("-inf"), "preview": float("-inf")}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._tail: asyncio.Task | None = None
        self._terminal_task: asyncio.Task | None = None
        self._history_ready = asyncio.Event()
        self._metadata = {"feature_flags": {"supports_preview_metadata": True}}
        record.tracker = self
        server.sid_listeners[record.id] = self.on_message
        server.sockets_metadata[record.id] = self._metadata
        self._queue_status = server.sid_listeners.setdefault(QUEUE_LISTENER, QueueStatus(server))
        self._queue_status.trackers.add(self)

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=64)
        self.subscribers.append(queue)
        return queue

    def emit(self, name: str, data: dict) -> None:
        if self.closed:
            return
        for queue in self.subscribers:
            if name in ("preview", "error") or name == "status" and data["status"] in TERMINAL:
                drop_previews(queue)
            if queue.full():
                queue.get_nowait()
            queue.put_nowait((name, data))

    def flush_throttled(self, name: str) -> None:
        timer = self._timers.pop(name, None)
        if timer is not None:
            timer.cancel()
        latest = self.latest_progress if name == "progress" else self.latest_preview
        if latest is not None and not self.closed:
            self._last[name] = monotonic()
            self.emit(name, latest.copy())

    def _throttle(self, name: str) -> None:
        remaining = THROTTLE_INTERVAL - (monotonic() - self._last[name])
        if remaining <= 0:
            self.flush_throttled(name)
        elif name not in self._timers:
            self._timers[name] = asyncio.get_running_loop().call_later(remaining, self.flush_throttled, name)

    def _progress(self, steps: dict | None = None) -> None:
        total = len(self.record.workflow)
        self.latest_progress = {
            **(self.latest_progress or {}),
            "value": min(1.0, max(0.0, (len(self.done) + self.fraction) / total)) if total else 0.0,
            "nodes_done": min(len(self.done), total), "nodes_total": total,
            "current_node": self.current_node,
            "current_node_class": self.record.workflow.get(self.current_node, {}).get("class_type"),
            **(steps or {}),
        }
        self._throttle("progress")

    def _preview(self, event: int, data: bytes) -> None:
        if len(data) < 4:
            return
        size = struct.unpack(">I", data[:4])[0]
        match event:
            case BinaryEventTypes.PREVIEW_IMAGE:
                mime = {1: "image/jpeg", 2: "image/png"}.get(size)
                if mime is None:
                    return
                node, image = self.current_node or "", data[4:]
            case BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
                if size > len(data) - 4:
                    return
                try:
                    metadata = json.loads(data[4:4 + size])
                except (ValueError, UnicodeDecodeError):
                    return
                if not isinstance(metadata, dict):
                    return
                mime, node, image = metadata.get("image_type", "image/jpeg"), str(metadata.get("node_id", "")), data[4 + size:]
            case _:
                return
        self.latest_preview = {"node_id": node, "content_type": mime, "data_base64": base64.b64encode(image).decode("ascii")}
        self._throttle("preview")

    def on_message(self, event, data, targeted: bool) -> None:
        if self.closed:
            return
        if isinstance(event, int):
            if targeted:
                self._preview(event, data)
            return
        if event == "status" and not targeted:
            return
        if not isinstance(data, dict) or data.get("prompt_id") != self.record.id:
            return
        match event:
            case "execution_start":
                self.started_at = jobs.now()
                if self.status == "queued":
                    self.status = "running"
                    self.emit("status", {"status": "running"})
            case "progress":
                self.current_node = str(data["node"]) if data.get("node") is not None else self.current_node
                self.fraction = data["value"] / data["max"] if data.get("max") else 0.0
                self._progress({"step": int(data["value"]), "steps": int(data["max"])})
            case "progress_state":
                nodes = data.get("nodes", {})
                self.done.update(n for n, state in nodes.items() if state.get("state") == "finished")
                running = next(((n, s) for n, s in nodes.items() if s.get("state") == "running"), None)
                self.fraction = 0.0
                steps = {"step": None, "steps": None}
                if running is not None:
                    self.current_node, state = running
                    self.fraction = state.get("value", 0) / state["max"] if state.get("max") else 0.0
                    steps = {"step": int(state.get("value", 0)), "steps": int(state.get("max", 0))}
                self._progress(steps)
            case "execution_cached":
                self.done.update(data.get("nodes", []))
                self._progress()
            case "executing":
                self.current_node = str(data["node"]) if data.get("node") is not None else None
                if self.current_node is None:
                    self._history_ready.set()
                    self._start_terminal()
                elif self.latest_progress is not None:
                    self._progress({"step": None, "steps": None})
            case "executed" if isinstance(data.get("output"), dict):
                # A cached output node without UI output is executed with "output": None.
                previous = self._tail
                self._tail = asyncio.create_task(self._outputs_after(previous, data))
                self._tail.add_done_callback(self._task_done)
            case "execution_success" | "execution_error" | "execution_interrupted":
                self._start_terminal()

    async def _outputs_after(self, previous, data: dict) -> None:
        if previous is not None:
            await previous
        for key, items in data["output"].items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                output = await asyncio.to_thread(assets.output_json, item["id"], self.base, str(data.get("node", "")), item.get("filename", ""), jobs.OUTPUT_TYPES.get(key, "file"), self.record.id)
                if output is not None and not self.closed:
                    self.record.outputs[output["id"]] = output
                    self._output(output)

    def _output(self, output: dict) -> None:
        if output["id"] not in self.emitted_output_ids:
            self.emitted_output_ids.add(output["id"])
            self.emit("output", output)

    async def flush(self) -> None:
        if self._tail is not None:
            await self._tail

    def _start_terminal(self, wait_history: bool = True) -> None:
        if self._terminal_task is None:
            self._terminal_task = asyncio.create_task(self._reconcile(wait_history))
            self._terminal_task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None and not self.closed:
            logging.warning("v2 job %s: event tracking failed; streams will poll history", self.record.id, exc_info=(type(error), error, error.__traceback__))
            self.cleanup()

    async def _reconcile(self, wait_history: bool = True) -> None:
        deadline = monotonic() + HISTORY_TIMEOUT
        while wait_history and not self.closed and not self.server.prompt_queue.get_history(self.record.id):
            if monotonic() >= deadline:
                break
            try:
                await asyncio.wait_for(self._history_ready.wait(), HISTORY_INTERVAL)
            except asyncio.TimeoutError:
                continue
            self._history_ready.clear()
        await self.flush()
        if self.closed:
            return
        state = await jobs.job_state(self.server, self.store, self.record.id, self.base)
        if self.closed:
            return
        if state is None:
            self.emit("error", {"error": {"code": "job_not_found", "message": "job history is unavailable"}})
            self.cleanup()
            return
        if state.status not in TERMINAL:
            if wait_history:
                self.cleanup()
            else:
                self._terminal_task = None
            return
        for output in state.outputs:
            self.record.outputs[output["id"]] = output
            self._output(output)
        self.finish(state.status)

    def finish(self, status: str) -> None:
        if self.closed:
            return
        self.status = status
        if status == "succeeded":
            total = len(self.record.workflow)
            self.latest_progress = {**(self.latest_progress or {}), "value": 1.0, "nodes_done": total, "nodes_total": total}
        self.emit("status", {"status": status})
        self.cleanup()

    def cleanup(self) -> None:
        self.closed = True
        self.latest_preview = None
        for queue in self.subscribers:
            drop_previews(queue)
        self._queue_status.trackers.discard(self)
        if not self._queue_status.trackers and self.server.sid_listeners.get(QUEUE_LISTENER) is self._queue_status:
            del self.server.sid_listeners[QUEUE_LISTENER]
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        if self._terminal_task is not None and self._terminal_task is not asyncio.current_task():
            self._terminal_task.cancel()
        if self._tail is not None and self._tail is not asyncio.current_task():
            self._tail.cancel()
        if self.server.sid_listeners.get(self.record.id) == self.on_message:
            del self.server.sid_listeners[self.record.id]
            if self.server.sockets_metadata.get(self.record.id) is self._metadata:
                del self.server.sockets_metadata[self.record.id]


async def stream(request: web.Request, server, store: jobs.JobStore, user: str, base: str) -> web.StreamResponse:
    job_id = jobs.parse_job_id(request.match_info["id"])
    record = store.records.get(job_id)
    tracker = record.tracker if record else None
    queue = tracker.subscribe() if tracker is not None and not tracker.closed else None
    acquired = False
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-store, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
    try:
        state = await jobs.job_state(server, store, job_id, base)
        if state is None:
            raise assets.ApiError(404, "not_found", "job not found")
        if store.open_streams.get(user, 0) >= 8:
            raise assets.ApiError(429, "too_many_streams", "too many open event streams; poll the job instead", headers={"Retry-After": "5"})
        store.open_streams[user] = store.open_streams.get(user, 0) + 1
        acquired = True
        initial = [("status", status_event(state))]
        if state.status not in TERMINAL:
            if tracker is not None:
                tracker.position = state.queue_position
                for name, latest in (("progress", tracker.latest_progress), ("preview", tracker.latest_preview)):
                    if latest is not None:
                        initial.append((name, latest.copy()))
            initial.extend(("output", output) for output in state.outputs)
        await response.prepare(request)
        seen: set[str] = set()
        last_status = status_event(state)
        heartbeat = monotonic()
        pending = initial
        while True:
            for name, data in pending:
                if name == "output":
                    if data["id"] in seen:
                        continue
                    seen.add(data["id"])
                elif name == "status":
                    last_status = data
                await response.write(sse_frame(name, data))
                if name == "error" or name == "status" and data["status"] in TERMINAL:
                    await response.write_eof()
                    return response
            if request.transport is None or request.transport.is_closing():
                return response
            pending = []
            if monotonic() - heartbeat >= KEEPALIVE_INTERVAL:
                await response.write(b": keepalive\n\n")
                heartbeat = monotonic()
            if queue is not None:
                try:
                    pending = [await asyncio.wait_for(queue.get(), min(POLL_INTERVAL, KEEPALIVE_INTERVAL))]
                    continue
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(min(POLL_INTERVAL, KEEPALIVE_INTERVAL))
            state = await jobs.job_state(server, store, job_id, base)
            if queue is not None and not tracker.closed and (state is None or state.status in TERMINAL):
                tracker._start_terminal(wait_history=False)
                continue
            if state is None:
                pending = [("error", {"error": {"code": "job_not_found", "message": "job not found"}})]
            elif queue is None or tracker.closed:
                pending = [("output", output) for output in state.outputs]
                current = status_event(state)
                if current != last_status:
                    pending.append(("status", current))
                    last_status = current
    except ConnectionResetError:
        return response
    finally:
        if queue is not None:
            tracker.subscribers.remove(queue)
        if acquired:
            store.open_streams[user] -= 1
            if store.open_streams[user] == 0:
                del store.open_streams[user]
