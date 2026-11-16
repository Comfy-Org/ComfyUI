"""In-memory v2 job records and idempotency keys, and the v2 job operations.

The ``JobStore`` lives on the app (``app[JOBS_KEY]``, set by ``register``), so each app
starts empty and nothing survives a restart. Records and idempotency keys are kept for
``RETENTION`` after creation; every handler that reads or writes job state calls
``prune(now())`` first. ``now()`` is the only clock used here, so tests monkeypatch it.

A job's state is core's view of it (``find``, over ``comfy_execution.jobs.get_job``),
and comes from its record only when ComfyUI no longer knows the job: ``job_state``
translates either into a ``JobState`` and ``job_json`` renders it as the contract Job
for a given caller.
"""

import asyncio
import copy
import json
import logging
import math
import re
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import unquote_plus

from aiohttp import web

from app.database.db import dependencies_available
from comfy_execution.asset_enrichment import register_cached_outputs, register_executed_outputs
from comfy_execution.jobs import CANCEL_PENDING, CANCEL_RUNNING, JobStatus, get_job

from . import assets
from .assets import ApiError

if dependencies_available():
    # Importing the services package pulls in sqlalchemy; v2 is only registered when assets are enabled.
    from app.assets.services.cursor import InvalidCursorError, decode_cursor, decode_cursor_time, encode_cursor_from_time

if TYPE_CHECKING:
    from .events import JobTracker

RETENTION = timedelta(hours=24)
MAX_RECORDS = 10_000
MAX_KEYS = 10_000
MAX_OUTPUT_IDS = 10_000
MAX_KEY_LENGTH = 255
MAX_PRIORITY = 1_000_000
MAX_METADATA_PAIRS = 16
MAX_METADATA_VALUE_BYTES = 256
METADATA_KEY_RE = re.compile(r"[A-Za-z0-9_.-]{1,40}")
# C0, DEL, C1, bidi embeddings, overrides and isolates; lone surrogates are not UTF-8 at all.
METADATA_VALUE_BAD_RE = re.compile("[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
EXTRA_DATA_STRINGS = {"api_key_comfy_org", "auth_token_comfy_org", "preview_method"}
STATUSES = {JobStatus.PENDING: "queued", JobStatus.IN_PROGRESS: "running", JobStatus.COMPLETED: "succeeded", JobStatus.FAILED: "failed", JobStatus.CANCELLED: "canceled"}
OUTPUT_TYPES = {"images": "image", "gifs": "video", "video": "video", "audio": "audio", "text": "text", "files": "file", "latents": "latent"}


def now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    id: str
    user_id: str
    created_at: datetime
    workflow: dict
    metadata: dict | None
    priority: int | None
    cancel_requested: bool = False
    canceled_at: datetime | None = None
    # Contract Output dicts of outputs committed while the job runs; GET lists their values until history has the job.
    outputs: dict = field(default_factory=dict)
    tracker: "JobTracker | None" = field(default=None, repr=False)


@dataclass
class JobState:
    """What the contract Job says about one job; ``record`` decides who sees ``metadata`` and ``priority``."""
    id: str
    status: str
    created_at: datetime
    record: JobRecord | None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_position: int | None = None
    progress: dict | None = None
    outputs: list[dict] = field(default_factory=list)
    outputs_reused: bool = False
    error: dict | None = None


class Found(NamedTuple):
    """ComfyUI's knowledge of a job: core's job dict, its API-format graph, the raw history outputs and its queue position."""
    job: dict
    graph: dict
    outputs: dict
    position: int | None


class JobStore:
    """``records`` by job id and claimed ``idempotency`` keys by (user, key), both in insertion order.

    ``output_ids`` maps (job id, node id, subfolder, filename, type) to the reference id
    ``job_state`` registered for a history output that carried none.

    Pruning scans every record, since a record made by cancelling a non-v2 job carries the
    job's older queue create_time. simplification: key pruning pops from the front, assuming
    claim order is time order; a wall clock stepping back only delays pruning of the keys behind it.
    simplification: ``output_ids`` keeps the newest ``MAX_OUTPUT_IDS`` entries; reading an
    evicted output again registers it again, under a new reference id.
    """

    def __init__(self):
        self.records: dict[str, JobRecord] = {}
        self.idempotency: OrderedDict[tuple[str, str], datetime] = OrderedDict()
        self.output_ids: dict[tuple[str, str, str, str, str], str] = {}
        self.output_lock = asyncio.Lock()
        self.open_streams: dict[str, int] = {}

    def prune(self, at: datetime) -> None:
        cutoff = at - RETENTION
        for job_id in [job_id for job_id, record in self.records.items() if record.created_at <= cutoff]:
            record = self.records.pop(job_id)
            if record.tracker is not None:
                record.tracker.cleanup()
        while self.idempotency and next(iter(self.idempotency.values())) <= cutoff:
            self.idempotency.popitem(last=False)

    def add(self, record: JobRecord, active_ids: Callable[[], set[str]]) -> None:
        """At ``MAX_RECORDS``, evict the oldest record whose job is not queued or running, else the oldest."""
        if len(self.records) >= MAX_RECORDS:
            active = active_ids()
            oldest = next(iter(self.records))
            evicted = self.records.pop(next((job_id for job_id in self.records if job_id not in active), oldest))
            if evicted.tracker is not None:
                evicted.tracker.cleanup()
        self.records[record.id] = record

    def claim(self, user: str, key: str, at: datetime) -> None:
        if (user, key) in self.idempotency:
            raise ApiError(422, "idempotency_key_reuse", "this Idempotency-Key was already used; keys are single-use, so poll or list your jobs instead of resubmitting")
        if len(self.idempotency) >= MAX_KEYS:
            retry = math.ceil((next(iter(self.idempotency.values())) + RETENTION - at).total_seconds())
            raise ApiError(429, "rate_limited", "too many live Idempotency-Keys; retry later or submit without a key", headers={"Retry-After": str(max(retry, 1))})
        self.idempotency[(user, key)] = at

    def owned(self, job_id: str, user: str) -> JobRecord | None:
        record = self.records.get(job_id)
        return record if record is not None and record.user_id == user else None


JOBS_KEY = web.AppKey("api_v2_jobs", JobStore)


def _metadata(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ApiError(400, "invalid_request", "'metadata' must be an object")
    if len(value) > MAX_METADATA_PAIRS:
        raise ApiError(422, "metadata_invalid", f"'metadata' has {len(value)} pairs; at most {MAX_METADATA_PAIRS} are allowed")
    for key, v in value.items():
        if not METADATA_KEY_RE.fullmatch(key):
            raise ApiError(422, "metadata_invalid", f"metadata key {key!r} must be 1-40 characters from A-Z a-z 0-9 _ - .", {"key": key})
        if not isinstance(v, str) or METADATA_VALUE_BAD_RE.search(v) or len(v.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
            raise ApiError(422, "metadata_invalid", f"metadata value of {key!r} must be a string of at most {MAX_METADATA_VALUE_BYTES} UTF-8 bytes without control or bidirectional characters", {"key": key})
    return value or None


def _priority(value) -> int | None:
    if value is not None and (type(value) is not int or abs(value) > MAX_PRIORITY):
        raise ApiError(422, "invalid_request", f"'priority' must be an integer from -{MAX_PRIORITY} to {MAX_PRIORITY}")
    return value


def _extra_data(value) -> dict:
    if value is None:
        return {}
    if (
        not isinstance(value, dict)
        or value.keys() - EXTRA_DATA_STRINGS - {"extra_pnginfo"}
        or any(not isinstance(value[k], str) for k in value.keys() & EXTRA_DATA_STRINGS)
        or not isinstance(value.get("extra_pnginfo", {}), dict)
    ):
        raise ApiError(400, "invalid_request", "'extra_data' accepts only string api_key_comfy_org, auth_token_comfy_org and preview_method, and an object extra_pnginfo")
    return dict(value)


def _active_ids(server) -> set[str]:
    running, queued = server.prompt_queue.get_current_queue_volatile()
    return {item[1] for item in running + queued}


def _not_found() -> ApiError:
    return ApiError(404, "not_found", "job not found")


def parse_job_id(raw: str) -> str:
    """``raw`` when it is a canonical (lowercase, hyphenated) UUID, else ApiError 404 ``not_found``."""
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError:
        canonical = None
    if canonical != raw:
        raise _not_found()
    return raw


def _from_ms(value) -> datetime | None:
    return datetime.fromtimestamp(value / 1000, timezone.utc) if type(value) in (int, float) else None


def queue_positions(queued: list) -> dict[str, int]:
    """1-based position of each pending job id, in the order the worker takes them."""
    return {item[1]: position for position, item in enumerate(sorted(queued), 1)}


def find(queue, job_id: str) -> Found | None:
    """``comfy_execution.jobs.get_job``'s view of ``job_id``, or None.

    Reads the queue, then history, like ``/api/jobs/{id}``: the worker moves a job from
    running to history in one locked step, so a job missing from the queue snapshot is
    already in the later history snapshot.
    """
    running, queued = queue.get_current_queue_volatile()
    history = queue.get_history(prompt_id=job_id)
    job = get_job(job_id, [item[:5] for item in running], [item[:5] for item in queued], history)
    if job is None:
        return None
    if job["status"] not in (JobStatus.PENDING, JobStatus.IN_PROGRESS):
        return Found(job, job["workflow"]["prompt"], history[job_id].get("outputs") or {}, None)
    graph = next(item[2] for item in running + queued if item[1] == job_id)
    return Found(job, graph, {}, queue_positions(queued).get(job_id))


def _register_output(asset_manager, job_id: str, item: dict) -> str | None:
    """Reference id for a history output that carries none, through execution-time enrichment.

    Live content at the item's path is reused, as for a cached replay, so no other record
    loses it; only a file with no live content is registered as a fresh output.
    """
    entry = register_cached_outputs({"output": {"items": [item]}}, job_id, asset_manager)["output"]["items"][0]
    if "id" not in entry:
        entry = register_executed_outputs({"items": [item]}, job_id, asset_manager)["items"][0]
    return entry.get("id")


async def _output_id(server, store: JobStore, job_id: str, node_id: str, item: dict) -> str | None:
    """The item's reference id; one it lacks is registered once (``_register_output``)."""
    if item.get("id") is not None:
        return item["id"] if isinstance(item["id"], str) else None
    key = (job_id, node_id, item.get("subfolder") or "", item["filename"], item["type"])
    async with store.output_lock:
        if key not in store.output_ids:
            registered = await asyncio.to_thread(_register_output, server.asset_manager, job_id, item)
            if registered is None:
                return None
            store.output_ids[key] = registered
            if len(store.output_ids) > MAX_OUTPUT_IDS:
                del store.output_ids[next(iter(store.output_ids))]
        return store.output_ids[key]


async def _history_outputs(server, store: JobStore, job_id: str, outputs: dict, base: str) -> list[dict]:
    """Contract Outputs of a history ``outputs`` map; an item that is not a file or has no live asset is left out."""
    result = []
    for node_id, ui in outputs.items():
        if not isinstance(ui, dict):
            continue
        for key, items in ui.items():
            if key == "animated" or not isinstance(items, list):
                continue
            for item in items:
                if not (isinstance(item, dict) and isinstance(item.get("filename"), str) and isinstance(item.get("type"), str) and isinstance(item.get("subfolder") or "", str)):
                    continue
                ref_id = await _output_id(server, store, job_id, node_id, item)
                output = ref_id and await asyncio.to_thread(assets.output_json, ref_id, base, node_id, item["filename"], OUTPUT_TYPES.get(key, "file"), job_id)
                if output:
                    result.append(output)
    return result


async def _history_state(server, store: JobStore, job_id: str, found: Found, created_at: datetime, record: JobRecord | None, base: str) -> JobState:
    job = found.job
    execution = job["execution_status"] or {}
    status = STATUSES[job["status"]]
    if status == "succeeded" and execution.get("status_str") != "success":
        # Core counts a history item without a status_str as completed; only a reported success is succeeded.
        status = "failed"
    failure = job.get("execution_error")
    error = None
    if failure is not None:
        error = {
            "code": "node_execution_error",
            "message": str(failure.get("exception_message", "")),
            "node_id": failure.get("node_id"),
            "class_type": failure.get("node_type"),
            "traceback": "".join(failure.get("traceback") or []) or None,
        }
    messages = [entry for entry in execution.get("messages", []) if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict)]
    return JobState(
        job_id, status, created_at, record,
        started_at=_from_ms(job.get("execution_start_time")),
        completed_at=_from_ms(job.get("execution_end_time")),
        outputs=await _history_outputs(server, store, job_id, found.outputs, base),
        outputs_reused=any(name == "execution_cached" and data.get("nodes") for name, data, *_ in messages),
        error=error,
    )


async def job_state(server, store: JobStore, job_id: str, base: str) -> JobState | None:
    """Current state of ``job_id``, or None when neither ComfyUI nor a live record knows it.

    May register history outputs that carry no asset id (see ``JobStore.output_ids``).
    """
    found = find(server.prompt_queue, job_id)
    record = store.records.get(job_id)
    if found is None:
        if record is None:
            return None
        if record.canceled_at is not None:
            return JobState(job_id, "canceled", record.created_at, record, completed_at=record.canceled_at)
        return JobState(job_id, "expired", record.created_at, record)
    # simplification: a queue item without create_time (never from enqueue_prompt) counts as created now.
    created_at = record.created_at if record else _from_ms(found.job.get("create_time")) or now()
    if found.job["status"] == JobStatus.PENDING:
        return JobState(job_id, "queued", created_at, record, queue_position=found.position)
    if found.job["status"] == JobStatus.IN_PROGRESS:
        tracker = record.tracker if record else None
        return JobState(
            job_id, "canceling" if record and record.cancel_requested else "running", created_at, record,
            started_at=tracker.started_at if tracker else None,
            progress=tracker.latest_progress if tracker and tracker.latest_progress is not None else {"value": 0.0, "nodes_done": 0, "nodes_total": 0},
            outputs=list(record.outputs.values()) if record else [],
        )
    return await _history_state(server, store, job_id, found, created_at, record, base)


def job_json(state: JobState, base: str, user: str) -> dict:
    """The contract Job of ``state`` as ``user`` sees it: ``metadata`` and ``priority`` only for the record's owner."""
    url = f"{base}/api/v2/jobs/{state.id}"
    body = {
        "id": state.id,
        "status": state.status,
        "created_at": assets.iso(state.created_at),
        "started_at": state.started_at and assets.iso(state.started_at),
        "completed_at": state.completed_at and assets.iso(state.completed_at),
        "expires_at": assets.iso(state.created_at + RETENTION),
        "queue_position": state.queue_position,
        "progress": state.progress,
        "outputs": state.outputs,
        "outputs_reused": state.outputs_reused,
        "error": state.error,
        "urls": {"self": url, "events": f"{url}/events", "cancel": f"{url}/cancel"},
    }
    record = state.record
    if record is not None and record.user_id == user:
        if record.metadata:
            body["metadata"] = record.metadata
        if record.priority is not None:
            body["priority"] = record.priority
    return body


async def get(server, store: JobStore, raw_id: str, user: str, base: str) -> dict:
    """``GET /api/v2/jobs/{id}``."""
    job_id = parse_job_id(raw_id)
    state = await job_state(server, store, job_id, base)
    if state is None:
        raise _not_found()
    return job_json(state, base, user)


async def cancel(server, store: JobStore, raw_id: str, user: str, base: str) -> dict:
    """``POST /api/v2/jobs/{id}/cancel``: the Job after the cancel. Idempotent."""
    job_id = parse_job_id(raw_id)
    record = store.records.get(job_id)
    # Only a job without a record needs reading first: for its 404 and the record a cancel creates.
    found = find(server.prompt_queue, job_id) if record is None else None
    if found is None and record is None:
        raise _not_found()
    outcome = server.cancel_job_by_id(job_id)
    if outcome in (CANCEL_RUNNING, CANCEL_PENDING) and record is None:
        # A job submitted outside v2: a record without labels lets GET answer canceling or canceled.
        record = JobRecord(job_id, user, _from_ms(found.job.get("create_time")) or now(), found.graph, None, None)
        store.add(record, lambda: _active_ids(server))
    if outcome == CANCEL_RUNNING:
        if not record.cancel_requested and record.tracker is not None and not record.tracker.closed:
            record.tracker.status = "canceling"
            record.tracker.emit("status", {"status": "canceling"})
        record.cancel_requested = True
    elif outcome == CANCEL_PENDING:
        record.canceled_at = now()
        if record.tracker is not None:
            record.tracker.finish("canceled")
    return await get(server, store, job_id, user, base)


def workflow(server, store: JobStore, raw_id: str) -> dict:
    """``GET /api/v2/jobs/{id}/workflow``: the API-format graph ComfyUI has, else the record's."""
    job_id = parse_job_id(raw_id)
    found = find(server.prompt_queue, job_id)
    if found is not None:
        graph = found.graph
    elif job_id in store.records:
        graph = store.records[job_id].workflow
    else:
        raise _not_found()
    return {"workflow": graph, "format": "api"}


def log_reason(server, store: JobStore, raw_id: str) -> str:
    """``No-Log-Reason`` of ``GET /api/v2/jobs/{id}/logs``; no log is ever captured."""
    job_id = parse_job_id(raw_id)
    found = find(server.prompt_queue, job_id)
    if found is None and job_id not in store.records:
        raise _not_found()
    return "not_finished" if found is not None and found.job["status"] in (JobStatus.PENDING, JobStatus.IN_PROGRESS) else "finished"


async def submit(request: web.Request, server, store: JobStore, user: str, base: str) -> dict:
    """``POST /api/v2/jobs``: the queued Job, or ApiError. Checks run in the contract's order."""
    try:
        body = json.loads(await request.read())
    except (ValueError, RecursionError, web.HTTPRequestEntityTooLarge):
        body = None
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_request", "body must be a JSON object")
    if body.get("webhook_url") is not None or body.get("inputs") is not None:
        raise ApiError(422, "invalid_workflow", "'webhook_url' and 'inputs' are reserved and not accepted")
    workflow = body.get("workflow")
    if not isinstance(workflow, dict):
        raise ApiError(422, "invalid_workflow", "'workflow' must be an API-format graph object")
    if isinstance(workflow.get("nodes"), list) and "links" in workflow:
        raise ApiError(422, "workflow_format_ui", "this is a UI-export graph; export the workflow in API format instead")
    for node_id, node in workflow.items():
        if not (isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict)):
            raise ApiError(422, "invalid_workflow", f"node {node_id!r} must be an object with a string 'class_type' and an object 'inputs'")
    metadata = _metadata(body.get("metadata"))
    priority = _priority(body.get("priority"))
    extra_data = _extra_data(body.get("extra_data"))
    missing: list[str] = []
    try:
        resolved = await asyncio.to_thread(assets.resolve_asset_refs, workflow, missing)
        # enqueue_prompt may rewrite the prompt in place; the record keeps the graph as resolved.
        prompt = copy.deepcopy(resolved)
    except RecursionError:
        raise ApiError(422, "invalid_workflow", "'workflow' is nested too deeply")
    if missing:
        raise ApiError(422, "missing_asset", f"unresolvable core/ASSET reference(s): {', '.join(missing)}", {"missing_ids": missing})

    key = request.headers.get("Idempotency-Key")
    if key is not None:
        if not 1 <= len(key) <= MAX_KEY_LENGTH:
            raise ApiError(400, "invalid_request", f"Idempotency-Key must be 1-{MAX_KEY_LENGTH} characters")
        store.claim(user, key, now())

    job_id = str(uuid.uuid4())
    record = JobRecord(job_id, user, now(), resolved, metadata, priority)
    store.add(record, lambda: _active_ids(server))
    from .events import JobTracker
    tracker = JobTracker(server, store, record, base)
    json_data = {"prompt": prompt, "prompt_id": job_id, "client_id": job_id, "extra_data": extra_data}
    try:
        status, result = await server.enqueue_prompt(json_data, usage_source=request.headers.get("Comfy-Usage-Source"))
    except Exception:
        # Unknown outcome: keep the key claimed and the record, so a retry cannot double-submit.
        logging.exception("v2 submit: enqueueing job %s failed", job_id)
        raise ApiError(500, "upstream_error", "enqueueing the job failed and it may or may not be queued; poll or list your jobs before resubmitting")
    if status != 200:
        if key is not None:
            store.idempotency.pop((user, key), None)
        store.records.pop(job_id, None)
        tracker.cleanup()
        raise ApiError(422, "invalid_workflow", result["error"]["message"], {"node_errors": result.get("node_errors", {})})
    return job_json(JobState(job_id, "queued", record.created_at, record), base, user)


# ---- listJobs: GET /api/v2/jobs (todo 13) ----

MAX_LIST_LIMIT = 500
MAX_METADATA_FILTERS = 3
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "canceling", "canceled", "failed", "expired"})
BAD_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _bad_filter(message: str) -> ApiError:
    return ApiError(400, "invalid_metadata_filter", message)


def _list_query(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    """(other parameters, first value wins; metadata filters) of the undecoded query string ``raw``.

    Pairs split on ``&`` only and are decoded once. A metadata filter whose raw pair holds a
    ``;`` or a ``%`` without two hex digits is refused; other pairs are not checked.
    """
    params: dict[str, str] = {}
    filters: dict[str, str] = {}
    for pair in raw.split("&"):
        raw_name, _, raw_value = pair.partition("=")
        flagged = ";" in pair or BAD_ESCAPE_RE.search(pair) is not None
        name, value = unquote_plus(raw_name), unquote_plus(raw_value)
        if not (name.startswith("metadata[") and name.endswith("]")):
            params.setdefault(name, value)
            continue
        key = name[9:-1]
        if flagged:
            raise _bad_filter("a metadata filter must be URL-encoded: no raw ';', and every '%' followed by two hex digits")
        if not METADATA_KEY_RE.fullmatch(key):
            raise _bad_filter(f"metadata filter key {key!r} must be 1-40 characters from A-Z a-z 0-9 _ - .")
        if key in filters:
            raise _bad_filter(f"metadata filter key {key!r} is repeated")
        if METADATA_VALUE_BAD_RE.search(value):
            raise _bad_filter(f"metadata filter value of {key!r} holds a control or bidirectional character")
        filters[key] = value
    if len(filters) > MAX_METADATA_FILTERS:
        raise _bad_filter(f"at most {MAX_METADATA_FILTERS} metadata filters are allowed")
    return params, filters


def _limit(value: str | None) -> int:
    if value is None:
        return MAX_LIST_LIMIT
    digits = value.lstrip("0")
    if not (value.isascii() and value.isdigit() and digits):
        raise ApiError(422, "invalid_request", "'limit' must be a positive integer")
    return MAX_LIST_LIMIT if len(digits) > 3 else min(int(digits), MAX_LIST_LIMIT)


def _after(cursor: str) -> tuple[datetime, str]:
    """The (created_at, id) position ``cursor`` names; ApiError 400 ``invalid_cursor`` when it is not well-formed."""
    try:
        payload = decode_cursor(cursor, ("created_at",), "desc")
        return decode_cursor_time(payload), payload.id
    except (InvalidCursorError, RecursionError):  # json.loads raises RecursionError on deep nesting
        raise ApiError(400, "invalid_cursor", "'cursor' is not well-formed; pass a page's next_cursor back unchanged")


async def list_jobs(server, store: JobStore, raw_query: str, user: str, base: str) -> dict:
    """``GET /api/v2/jobs``: ``user``'s records, newest first by (created_at, id).

    simplification: a ``status`` filter computes each candidate's state in turn, up to every
    record the user has (``MAX_RECORDS``); an index by status would need event-driven updates.
    """
    params, filters = _list_query(raw_query)
    limit = _limit(params.get("limit"))
    statuses = None
    if "status" in params:
        statuses = set(params["status"].split(","))
        if not statuses <= JOB_STATUSES:
            raise ApiError(400, "invalid_request", f"'status' must be a comma list of {', '.join(sorted(JOB_STATUSES))}")
    after = _after(params["cursor"]) if "cursor" in params else None
    records = sorted(
        (r for r in store.records.values() if r.user_id == user and all((r.metadata or {}).get(k) == v for k, v in filters.items())),
        key=lambda r: (r.created_at, r.id), reverse=True,
    )
    page: list[tuple[JobRecord, JobState]] = []
    more = False
    for record in records:
        if after is not None and (record.created_at, record.id) >= after:
            continue
        state = await job_state(server, store, record.id, base)
        if state is None or (statuses is not None and state.status not in statuses):
            continue
        if len(page) == limit:
            more = True
            break
        page.append((record, state))
    items = []
    for _, state in page:
        job = job_json(state, base, user)
        items.append(job | {"create_time": job["created_at"], "update_time": job["completed_at"] or job["started_at"] or job["created_at"]})
    body = {"jobs": items, "truncated": False}
    if more:
        body["next_cursor"] = encode_cursor_from_time("created_at", page[-1][0].created_at, page[-1][0].id)
    return body
