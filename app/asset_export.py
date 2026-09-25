"""Local implementation of the shared asset export contract.

POST /api/assets/export zips assets selected by job (a job's live output
records, plus its temp records when previews are requested) and by asset id on
a background thread, reporting progress through the ``asset_export`` websocket
event as the cloud runtime does. The finished archive lands in the temp
directory, so like every other temp file it lives until the next start, and
GET /api/assets/exports/{exportName} points the owner at it through /view.
Everything resolves through the assets system, so every route answers 503 when
it is disabled.
"""

import hashlib
import json
import logging
import os
import posixpath
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Sequence, get_args
from urllib.parse import urlencode

from aiohttp import web

import folder_paths
from app.assets.api.routes import _build_error_response, _require_assets_feature_enabled
from app.assets.services.path_utils import get_asset_category_and_relative_path
from app.assets.services.schemas import ExportableAssetFile
from comfy_execution.jobs import validate_job_id

NamingStrategy = Literal["group_by_job_id", "preserve", "asset_id", "group_by_job_time"]

TASK_NAME = "task:export"
EVENT_NAME = "asset_export"
NAMING_STRATEGIES = frozenset(get_args(NamingStrategy))
DEFAULT_NAMING_STRATEGY: NamingStrategy = "group_by_job_time"
EXPORT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+\.zip$")
EXPORT_SUBFOLDER = "exports"
COPY_CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 0.25
MAX_CONCURRENT_EXPORTS = 2
MAX_TRACKED_EXPORTS = 10000
EXPORTABLE_CATEGORIES = ("input", "output", "temp")
JOB_TIME_FOLDER_FORMAT = "%Y-%m-%dT%H-%M-%S"

JobFilesLookup = Callable[[Sequence[str], bool], list[ExportableAssetFile]]
AssetFileLookup = Callable[[str], ExportableAssetFile]
JobCreateTimesLookup = Callable[[Sequence[str]], dict[str, int]]
EventSink = Callable[[str, dict[str, Any]], None]
UserResolver = Callable[[web.Request], str]


class ExportStatus:
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureReason:
    NOT_FOUND = "not_found"
    MODEL_NOT_EXPORTABLE = "model_not_exportable"
    UNSUPPORTED_ASSET_TYPE = "unsupported_asset_type"
    FILENAME_COLLISION = "filename_collision"
    FETCH_FAILED = "fetch_failed"

    REJECTING = frozenset({MODEL_NOT_EXPORTABLE, UNSUPPORTED_ASSET_TYPE})


class ExportRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ExportRequest:
    job_ids: list[str]
    asset_ids: list[str]
    naming_strategy: NamingStrategy
    job_asset_name_filters: dict[str, list[str]]
    include_previews: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "job_ids": self.job_ids,
            "asset_ids": self.asset_ids,
            "naming_strategy": self.naming_strategy,
            "job_asset_name_filters": self.job_asset_name_filters,
            "include_previews": self.include_previews,
        }

    def idempotency_key(self, owner_id: str) -> str:
        """Key under which a repeated request joins the export already running for it."""
        parts = [*self.job_ids, *self.asset_ids]
        for job_id, names in self.job_asset_name_filters.items():
            if job_id in self.job_ids:
                parts.append(f"filter:{job_id}:{'|'.join(sorted(names))}")
        digest = hashlib.sha256(",".join(sorted(parts)).encode("utf-8")).hexdigest()
        previews = "true" if self.include_previews else "false"
        return f"export:{owner_id}:{self.naming_strategy}:{previews}:{digest}"


@dataclass(frozen=True)
class ExportEntry:
    asset_id: str
    abs_path: str
    name: str
    job_id: str | None
    created_at: datetime
    is_preview: bool = False


@dataclass(frozen=True)
class FailedAsset:
    asset_id: str
    error: str
    job_id: str | None = None
    asset_name: str | None = None

    def to_dict(self) -> dict[str, str]:
        failed = {"asset_id": self.asset_id, "error": self.error}
        if self.job_id:
            failed["job_id"] = self.job_id
        if self.asset_name:
            failed["asset_name"] = self.asset_name
        return failed


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _string_list(body: dict, key: str) -> list[str]:
    value = body.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ExportRequestError("INVALID_REQUEST", f"{key} must be an array of strings")
    return value


def parse_export_request(body: Any) -> ExportRequest:
    if not isinstance(body, dict):
        raise ExportRequestError("INVALID_REQUEST", "Request body must be a JSON object")

    try:
        job_ids = _unique([validate_job_id(v) for v in _string_list(body, "job_ids")])
    except ValueError as e:
        raise ExportRequestError("INVALID_REQUEST", f"Invalid job id: {e}") from e

    try:
        asset_ids = _unique([str(uuid.UUID(v)) for v in _string_list(body, "asset_ids")])
    except ValueError as e:
        raise ExportRequestError("INVALID_REQUEST", "asset_ids must contain UUIDs") from e

    if not job_ids and not asset_ids:
        raise ExportRequestError("INVALID_REQUEST", "job_ids and/or asset_ids is required")

    naming_strategy = body.get("naming_strategy", DEFAULT_NAMING_STRATEGY)
    if naming_strategy not in NAMING_STRATEGIES:
        raise ExportRequestError(
            "INVALID_NAMING_STRATEGY", f"Invalid naming strategy: {naming_strategy}"
        )

    raw_filters = body.get("job_asset_name_filters")
    if raw_filters is None:
        raw_filters = {}
    if not isinstance(raw_filters, dict):
        raise ExportRequestError("INVALID_REQUEST", "job_asset_name_filters must be an object")
    filters: dict[str, list[str]] = {}
    for job_id, names in raw_filters.items():
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) for n in names)
        ):
            raise ExportRequestError(
                "INVALID_REQUEST",
                "job_asset_name_filters values must be non-empty arrays of strings",
            )
        filters[job_id] = _unique(names)

    include_previews = body.get("include_previews", False)
    if not isinstance(include_previews, bool):
        raise ExportRequestError("INVALID_REQUEST", "include_previews must be a boolean")

    return ExportRequest(
        job_ids=job_ids,
        asset_ids=asset_ids,
        naming_strategy=naming_strategy,
        job_asset_name_filters=filters,
        include_previews=include_previews,
    )


def export_location(file: ExportableAssetFile) -> tuple[str | None, str | None]:
    """Root category of an asset file and its name inside the archive.

    The archive name is the asset's name in the file's directory under the
    input, output or temp root, so uploads stored under their hash and renamed
    assets keep the name the user sees. It is None for files under any other
    root (models included), and the category is None outside every root.
    """
    try:
        category, relative = get_asset_category_and_relative_path(file.path)
    except ValueError:
        return None, None
    if category not in EXPORTABLE_CATEGORIES:
        return category, None
    directory = posixpath.dirname(relative)
    return category, posixpath.join(directory, file.name) if file.name else relative


def matches_name_filter(
    file: ExportableAssetFile, archive_name: str | None, names: set[str] | None
) -> bool:
    if names is None:
        return True
    return (
        file.name in names
        or archive_name in names
        or (file.hash is not None and file.hash in names)
    )


def safe_archive_path(name: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", name) if p not in ("", ".", "..")]
    return "/".join(parts)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_job_time_folders(
    entries: list[ExportEntry], job_create_times: dict[str, datetime] | None = None
) -> dict[str, str]:
    """Folder name per job, from its creation time in UTC.

    A job whose creation time is unknown is named after its earliest asset.
    Two jobs landing on the same second are bumped a second apart so they never
    share a folder.
    """
    job_times: dict[str, datetime] = {}
    for entry in entries:
        if entry.job_id is None:
            continue
        created = _utc(entry.created_at)
        if entry.job_id not in job_times or created < job_times[entry.job_id]:
            job_times[entry.job_id] = created
    for job_id, create_time in (job_create_times or {}).items():
        if job_id in job_times:
            job_times[job_id] = _utc(create_time)
    folders: dict[str, str] = {}
    used: set[str] = set()
    for job_id, job_time in sorted(job_times.items(), key=lambda item: (item[1], item[0])):
        moment = job_time.replace(microsecond=0)
        folder = moment.strftime(JOB_TIME_FOLDER_FORMAT)
        while folder in used:
            moment += timedelta(seconds=1)
            folder = moment.strftime(JOB_TIME_FOLDER_FORMAT)
        used.add(folder)
        folders[job_id] = folder
    return folders


def _archive_name(entry: ExportEntry, strategy: NamingStrategy, job_folders: dict[str, str]) -> str:
    name = safe_archive_path(entry.name) or os.path.basename(entry.abs_path)
    if strategy == "asset_id":
        return entry.asset_id + os.path.splitext(name)[1]
    if strategy == "group_by_job_id":
        folder = entry.job_id or "unknown"
    elif strategy == "group_by_job_time":
        folder = (
            job_folders[entry.job_id]
            if entry.job_id is not None
            else _utc(entry.created_at).strftime(JOB_TIME_FOLDER_FORMAT)
        )
    else:
        return name
    return f"{folder}/previews/{name}" if entry.is_preview else f"{folder}/{name}"


def assign_archive_names(
    entries: list[ExportEntry],
    strategy: NamingStrategy,
    job_create_times: dict[str, datetime] | None = None,
) -> tuple[list[tuple[ExportEntry, str]], list[ExportEntry]]:
    """Pair entries with archive names, and list the entries dropped on a name collision.

    Entries are taken in asset id order, so the entry kept on a collision (the
    first one) does not depend on the order the entries came in.
    """
    job_folders = (
        build_job_time_folders(entries, job_create_times)
        if strategy == "group_by_job_time"
        else {}
    )
    assigned: list[tuple[ExportEntry, str]] = []
    collided: list[ExportEntry] = []
    used: set[str] = set()
    for entry in sorted(entries, key=lambda e: e.asset_id):
        arcname = _archive_name(entry, strategy, job_folders)
        if arcname in used:
            collided.append(entry)
            continue
        used.add(arcname)
        assigned.append((entry, arcname))
    return assigned, collided


def write_archive(
    assigned: list[tuple[ExportEntry, str]],
    dest_path: str,
    on_bytes: Callable[[int], None],
    on_file_done: Callable[[ExportEntry, bool], None],
) -> int:
    """Write a stored (uncompressed) zip and return the number of files written.

    A file that cannot be opened is skipped and reported as failed. A read
    error after copying started propagates, since the entry could otherwise be
    left truncated inside an otherwise valid archive.
    """
    written = 0
    with zipfile.ZipFile(
        dest_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for entry, arcname in assigned:
            try:
                info = zipfile.ZipInfo.from_file(entry.abs_path, arcname, strict_timestamps=False)
                if info.is_dir():
                    raise IsADirectoryError(entry.abs_path)
                source = open(entry.abs_path, "rb")
            except OSError:
                logging.warning("Asset export skipped unreadable file %s", entry.abs_path)
                on_file_done(entry, False)
                continue
            read_size = max(1, min(info.file_size, COPY_CHUNK_BYTES))
            with source, archive.open(info, "w") as target:
                while chunk := source.read(read_size):
                    target.write(chunk)
                    on_bytes(len(chunk))
            written += 1
            on_file_done(entry, True)
    return written


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _new_export_name() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"comfy-export-{today}-{uuid.uuid4().hex[:8]}.zip"


@dataclass
class ExportTask:
    id: str
    owner_id: str
    request: ExportRequest
    idempotency_key: str
    status: str = ExportStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    export_name: str | None = None
    assets_total: int = 0
    assets_attempted: int = 0
    assets_failed: int = 0
    bytes_total: int = 0
    bytes_processed: int = 0
    used_job_time_fallback: bool = False
    result: dict[str, Any] | None = None
    error_message: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in (ExportStatus.COMPLETED, ExportStatus.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.result["success"]

    def progress(self) -> float:
        if self.succeeded:
            return 1.0
        if self.bytes_total <= 0:
            return 0.0
        return min(self.bytes_processed / self.bytes_total, 1.0)

    def event(self) -> dict[str, Any]:
        """The asset_export websocket payload, shaped like cloud's AssetExportMessage."""
        event: dict[str, Any] = {
            "task_id": self.id,
            "assets_total": self.assets_total,
            "assets_attempted": self.assets_attempted,
            "assets_failed": self.assets_failed,
            "bytes_total": self.bytes_total,
            "bytes_processed": self.bytes_processed,
            "progress": self.progress(),
            "status": self.status,
        }
        if self.succeeded:
            event["export_name"] = self.export_name
        elif self.finished:
            event["status"] = ExportStatus.FAILED
            event["error"] = self.error_message or (self.result or {}).get("error")
        if self.used_job_time_fallback:
            event["used_job_time_fallback"] = True
        return event

    def task_response(self) -> dict[str, Any]:
        response: dict[str, Any] = {
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "task_name": TASK_NAME,
            "payload": self.request.to_payload(),
            "status": self.status,
            "create_time": _iso(self.created_at),
            "update_time": _iso(self.updated_at),
        }
        if self.started_at is not None:
            response["started_at"] = _iso(self.started_at)
        if self.completed_at is not None:
            response["completed_at"] = _iso(self.completed_at)
        if self.result is not None:
            response["result"] = self.result
        if self.error_message:
            response["error_message"] = self.error_message
        return response


def _export_result(
    success: bool,
    *,
    export_name: str | None = None,
    export_size: int = 0,
    uncompressed_size: int = 0,
    assets_total: int = 0,
    assets_succeeded: int = 0,
    used_job_time_fallback: bool = False,
    failed_assets: Sequence[FailedAsset] = (),
    error: str | None = None,
) -> dict[str, Any]:
    """Task result shaped like cloud's ExportResult, omitting empty fields the same way."""
    result: dict[str, Any] = {"success": success}
    if export_name:
        result["export_name"] = export_name
    if export_size > 0:
        result["export_size"] = export_size
    if uncompressed_size > 0:
        result["uncompressed_size"] = uncompressed_size
    if assets_total > 0:
        result["assets_total"] = assets_total
        result["assets_succeeded"] = assets_succeeded
    if used_job_time_fallback:
        result["used_job_time_fallback"] = True
    if failed_assets:
        result["failed_assets"] = [failed.to_dict() for failed in failed_assets]
    if error:
        result["error"] = error
    return result


class AssetExportManager:
    def __init__(
        self,
        list_job_files: JobFilesLookup,
        get_asset_file: AssetFileLookup,
        get_job_create_times: JobCreateTimesLookup,
        event_sink: EventSink,
        get_user_id: UserResolver,
    ):
        self._list_job_files = list_job_files
        self._get_asset_file = get_asset_file
        self._get_job_create_times = get_job_create_times
        self._event_sink = event_sink
        self._get_user_id = get_user_id
        self._tasks: dict[str, ExportTask] = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_EXPORTS)

    def register_routes(self, app: web.Application) -> None:
        gated = _require_assets_feature_enabled
        app.add_routes([
            web.post("/api/assets/export", gated(self._create_export_route)),
            web.get("/api/assets/exports/{exportName}", gated(self._export_url_route)),
            web.get("/api/tasks/{task_id}", gated(self._get_task_route)),
        ])

    @staticmethod
    def export_directory() -> str:
        return os.path.join(folder_paths.get_temp_directory(), EXPORT_SUBFOLDER)

    def export_path(self, export_name: str) -> str:
        return os.path.join(self.export_directory(), export_name)

    def create_export(self, request: ExportRequest, owner_id: str) -> ExportTask:
        """Start an export, or return the one still running for the same request."""
        key = request.idempotency_key(owner_id)
        with self._lock:
            running = next(
                (t for t in self._tasks.values() if t.idempotency_key == key and not t.finished),
                None,
            )
            if running is not None:
                return running
            task = ExportTask(
                id=str(uuid.uuid4()), owner_id=owner_id, request=request, idempotency_key=key
            )
            self._tasks[task.id] = task
            self._forget_oldest_finished()
        threading.Thread(
            target=self._run, args=(task,), name=f"asset-export-{task.id[:8]}", daemon=True
        ).start()
        return task

    def _forget_oldest_finished(self) -> None:
        """Cap the registry the way the prompt history is capped; the zip stays until restart."""
        if len(self._tasks) <= MAX_TRACKED_EXPORTS:
            return
        oldest = next((t for t in self._tasks.values() if t.finished), None)
        if oldest is not None:
            del self._tasks[oldest.id]

    def get_task(self, task_id: str) -> ExportTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _find_by_export_name(self, export_name: str) -> ExportTask | None:
        with self._lock:
            return next((t for t in self._tasks.values() if t.export_name == export_name), None)

    def _update(self, task: ExportTask, **changes: Any) -> dict[str, Any]:
        with self._lock:
            for key, value in changes.items():
                setattr(task, key, value)
            task.updated_at = time.time()
            return task.event()

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self._event_sink(EVENT_NAME, event)
        except Exception:
            logging.warning("Failed to send asset export event", exc_info=True)

    def _resolve_entries(self, request: ExportRequest) -> tuple[list[ExportEntry], list[FailedAsset]]:
        """Entries for the job files and assets of a request, and the assets that failed to resolve."""
        failed: list[FailedAsset] = []
        entries: list[ExportEntry] = []
        seen_ids: set[str] = set()

        def add(file: ExportableAssetFile, category: str | None, archive_name: str | None) -> None:
            if file.id in seen_ids:
                return
            seen_ids.add(file.id)
            if archive_name is None:
                reason = (
                    FailureReason.MODEL_NOT_EXPORTABLE
                    if category == "models"
                    else FailureReason.UNSUPPORTED_ASSET_TYPE
                )
                failed.append(FailedAsset(file.id, reason, file.job_id, file.name))
                return
            entries.append(
                ExportEntry(
                    asset_id=file.id,
                    abs_path=file.path,
                    name=archive_name,
                    job_id=file.job_id,
                    created_at=file.created_at,
                    is_preview=category == "temp",
                )
            )

        for file in self._list_job_files(request.job_ids, request.include_previews):
            names = request.job_asset_name_filters.get(file.job_id or "")
            category, archive_name = export_location(file)
            if matches_name_filter(file, archive_name, set(names) if names is not None else None):
                add(file, category, archive_name)
        for asset_id in request.asset_ids:
            if asset_id in seen_ids:
                continue
            try:
                file = self._get_asset_file(asset_id)
            except (ValueError, OSError):
                failed.append(FailedAsset(asset_id, FailureReason.NOT_FOUND))
                continue
            add(file, *export_location(file))
        return entries, failed

    def _lookup_job_create_times(self, entries: list[ExportEntry]) -> dict[str, datetime] | None:
        """Creation times of the jobs behind the entries, as far as the server still knows them.

        None when the lookup failed, so every job falls back to its asset times.
        """
        job_ids = sorted({entry.job_id for entry in entries if entry.job_id is not None})
        if not job_ids:
            return {}
        try:
            create_times = self._get_job_create_times(job_ids)
        except Exception:
            logging.warning(
                "Asset export could not look up job creation times, using asset times",
                exc_info=True,
            )
            return None
        return {
            job_id: datetime.fromtimestamp(create_time / 1000, tz=timezone.utc)
            for job_id, create_time in create_times.items()
        }

    def _finish(self, task: ExportTask, result: dict[str, Any], **changes: Any) -> None:
        self._emit(
            self._update(
                task,
                status=ExportStatus.COMPLETED,
                result=result,
                completed_at=time.time(),
                **changes,
            )
        )

    def _run(self, task: ExportTask) -> None:
        with self._slots:
            self._emit(self._update(task, status=ExportStatus.RUNNING, started_at=time.time()))
            partial_path = os.path.join(self.export_directory(), f"{task.id}.zip.part")
            try:
                self._export(task, partial_path)
            except Exception as e:
                logging.exception("Asset export %s failed", task.id)
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
                self._emit(
                    self._update(
                        task,
                        status=ExportStatus.FAILED,
                        error_message=str(e) or "Export failed",
                        completed_at=time.time(),
                    )
                )

    def _export(self, task: ExportTask, partial_path: str) -> None:
        entries, failed = self._resolve_entries(task.request)
        if any(f.error in FailureReason.REJECTING for f in failed):
            self._finish(
                task,
                _export_result(
                    False, failed_assets=failed, error="One or more assets are not exportable"
                ),
            )
            return

        strategy = task.request.naming_strategy
        job_create_times = None
        used_fallback = False
        if strategy == "group_by_job_time":
            job_create_times = self._lookup_job_create_times(entries)
            job_ids = {entry.job_id for entry in entries if entry.job_id is not None}
            used_fallback = bool(job_ids) and (
                job_create_times is None or bool(job_ids - job_create_times.keys())
            )
        assigned, collided = assign_archive_names(entries, strategy, job_create_times)
        failed += [
            FailedAsset(entry.asset_id, FailureReason.FILENAME_COLLISION, entry.job_id, entry.name)
            for entry in collided
        ]
        if not assigned:
            self._finish(
                task,
                _export_result(
                    False,
                    used_job_time_fallback=used_fallback,
                    failed_assets=failed,
                    error="No valid assets to export",
                ),
                used_job_time_fallback=used_fallback,
            )
            return

        bytes_total = 0
        for entry, _ in assigned:
            try:
                bytes_total += os.path.getsize(entry.abs_path)
            except OSError:
                pass
        self._update(
            task,
            assets_total=len(assigned),
            bytes_total=bytes_total,
            used_job_time_fallback=used_fallback,
        )
        os.makedirs(self.export_directory(), exist_ok=True)
        last_emit = [0.0]

        def record(bytes_count: int = 0, file_succeeded: bool | None = None) -> None:
            with self._lock:
                task.bytes_processed += bytes_count
                if file_succeeded is not None:
                    task.assets_attempted += 1
                    if not file_succeeded:
                        task.assets_failed += 1
                task.updated_at = time.time()
                now = time.monotonic()
                if now - last_emit[0] < PROGRESS_INTERVAL_SECONDS:
                    return
                last_emit[0] = now
                event = task.event()
            self._emit(event)

        def on_bytes(count: int) -> None:
            record(bytes_count=count)

        def on_file_done(entry: ExportEntry, succeeded: bool) -> None:
            if not succeeded:
                failed.append(
                    FailedAsset(entry.asset_id, FailureReason.FETCH_FAILED, entry.job_id, entry.name)
                )
            record(file_succeeded=succeeded)

        written = write_archive(assigned, partial_path, on_bytes, on_file_done)
        common = {
            "assets_total": len(assigned),
            "assets_succeeded": written,
            "used_job_time_fallback": used_fallback,
            "failed_assets": failed,
        }
        if written == 0:
            os.remove(partial_path)
            self._finish(
                task,
                _export_result(False, error="All assets failed to export", **common),
                bytes_processed=bytes_total,
            )
            return

        export_name = _new_export_name()
        export_path = self.export_path(export_name)
        os.replace(partial_path, export_path)
        self._finish(
            task,
            _export_result(
                True,
                export_name=export_name,
                export_size=os.path.getsize(export_path),
                uncompressed_size=bytes_total,
                **common,
            ),
            export_name=export_name,
            bytes_processed=bytes_total,
        )

    def _owner_or_none(self, request: web.Request) -> str | None:
        try:
            return self._get_user_id(request)
        except KeyError:
            return None

    async def _create_export_route(self, request: web.Request) -> web.Response:
        owner_id = self._owner_or_none(request)
        if owner_id is None:
            return _build_error_response(401, "UNAUTHORIZED", "Unknown user")
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _build_error_response(400, "INVALID_REQUEST", "Request body must be valid JSON")
        try:
            export_request = parse_export_request(body)
        except ExportRequestError as e:
            return _build_error_response(400, e.code, str(e))
        task = self.create_export(export_request, owner_id)
        return web.json_response(
            {
                "task_id": task.id,
                "status": ExportStatus.CREATED,
                "message": "Export task created. Use task_id to track progress.",
            },
            status=202,
        )

    async def _export_url_route(self, request: web.Request) -> web.Response:
        export_name = request.match_info["exportName"]
        if not EXPORT_NAME_RE.match(export_name):
            return _build_error_response(400, "INVALID_EXPORT_NAME", "Invalid export name")
        owner_id = self._owner_or_none(request)
        if owner_id is None:
            return _build_error_response(401, "UNAUTHORIZED", "Unknown user")
        task = self._find_by_export_name(export_name)
        if (
            task is None
            or task.owner_id != owner_id
            or not os.path.isfile(self.export_path(export_name))
        ):
            return _build_error_response(404, "EXPORT_NOT_FOUND", "Export not found")
        query = urlencode({"filename": export_name, "type": "temp", "subfolder": EXPORT_SUBFOLDER})
        return web.json_response({"url": f"/api/view?{query}"})

    async def _get_task_route(self, request: web.Request) -> web.Response:
        try:
            task_id = str(uuid.UUID(request.match_info["task_id"]))
        except ValueError:
            return _build_error_response(404, "TASK_NOT_FOUND", "Task not found")
        owner_id = self._owner_or_none(request)
        if owner_id is None:
            return _build_error_response(401, "UNAUTHORIZED", "Unknown user")
        task = self.get_task(task_id)
        if task is None or task.owner_id != owner_id:
            return _build_error_response(404, "TASK_NOT_FOUND", "Task not found")
        with self._lock:
            response = task.task_response()
        return web.json_response(response)
