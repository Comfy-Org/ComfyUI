"""The `/api/tasks` surface for local model downloads.

Deliberately the same three calls the frontend already makes against Cloud --
``POST /api/assets/download`` to submit, ``GET /api/tasks/{id}`` to read,
``DELETE /api/tasks/{id}`` to cancel -- so the browser needs no local-only code
path, and no knowledge that a download is anything other than a task.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import unquote, urlsplit

from typing import Any, Callable

from aiohttp import web

from app.assets.downloads.backend import (
    CancelOutcome,
    DownloadBackendUnavailable,
    DownloadRejected,
)
from app.assets.downloads.comfy_cli import ComfyCliBackend
from app.assets.downloads.destination import DestinationError
from app.assets.downloads.service import DownloadTaskService

ROUTES = web.RouteTableDef()

MODEL_TYPE_TAG_PREFIX = "model_type:"
MODEL_ROOT_TAG = "models"

# comfy-cli refusals the caller can act on: a different name, or dealing with
# the download already running. Everything else it reports is a server-side
# failure of the transport.
_CONFLICT_CODES = frozenset({"model_file_exists", "model_download_in_flight"})

DOWNLOAD_SERVICE = web.AppKey("download_service", DownloadTaskService)


def create_download_service(
    workspace: str,
    notify: Callable[[str, dict[str, Any]], None],
    refresh_catalog: Callable[[], None] | None = None,
) -> DownloadTaskService:
    return DownloadTaskService(ComfyCliBackend(workspace), notify, refresh_catalog)


def register_download_routes(app: web.Application, service: DownloadTaskService) -> None:
    """Wire the download task API into ``app`` and run its sweeper alongside it.

    The sweeper can only start once the loop is running, which is why it hangs
    off the application's startup signal rather than starting here.
    """
    app[DOWNLOAD_SERVICE] = service
    app.add_routes(ROUTES)

    async def _start(_app: web.Application) -> None:
        service.start_sweeping()

    async def _stop(_app: web.Application) -> None:
        await service.stop_sweeping()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


@ROUTES.post("/api/assets/download")
async def submit_download(request: web.Request) -> web.Response:
    service = request.app[DOWNLOAD_SERVICE]
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error(400, "INVALID_BODY", "Request body must be JSON.")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "Request body must be a JSON object.")

    url = (body.get("source_url") or "").strip()
    if not url:
        return _error(400, "INVALID_BODY", "source_url is required.")
    if urlsplit(url).scheme not in ("http", "https"):
        return _error(400, "INVALID_BODY", "source_url must be an http(s) URL.")

    folder = _folder_from_tags(body.get("tags"))
    if folder is None:
        return _error(
            400,
            "INVALID_BODY",
            f"A '{MODEL_TYPE_TAG_PREFIX}<folder>' tag is required to decide where the model belongs.",
        )

    try:
        task = await service.start(url, folder, body.get("filename") or _filename_from_url(url))
    except DestinationError as e:
        return _error(400, e.code, e.message)
    except DownloadBackendUnavailable as e:
        return _error(503, "DEPENDENCY_MISSING", str(e))
    except DownloadRejected as e:
        status = 409 if e.code in _CONFLICT_CODES else 400
        return _error(status, e.code, e.message)
    except Exception:
        logging.exception("Failed to start model download from %s", urlsplit(url)._replace(query="").geturl())
        return _error(500, "INTERNAL", "Could not start the download.")
    return web.json_response(task, status=202)


@ROUTES.get("/api/tasks/{task_id}")
async def get_task(request: web.Request) -> web.Response:
    task = await request.app[DOWNLOAD_SERVICE].get_task(request.match_info["task_id"])
    if task is None:
        return _error(404, "TASK_NOT_FOUND", "No such task.")
    return web.json_response(task)


@ROUTES.delete("/api/tasks/{task_id}")
async def cancel_task(request: web.Request) -> web.Response:
    try:
        outcome = await request.app[DOWNLOAD_SERVICE].cancel_task(request.match_info["task_id"])
    except DownloadBackendUnavailable as e:
        return _error(503, "DEPENDENCY_MISSING", str(e))
    if outcome is CancelOutcome.MISSING:
        return _error(404, "TASK_NOT_FOUND", "No such task.")
    if outcome is CancelOutcome.NOT_CANCELLABLE:
        return _error(409, "TASK_NOT_CANCELLABLE", "This task can no longer be cancelled.")
    return web.json_response({"status": "cancelling"})


def _folder_from_tags(tags: object) -> str | None:
    """The model folder named by the asset tags the frontend already sends.

    It tags a model upload with ``models`` plus either ``model_type:<folder>``
    or, on older builds, the bare folder name.
    """
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(MODEL_TYPE_TAG_PREFIX):
            return tag[len(MODEL_TYPE_TAG_PREFIX) :]
    for tag in tags:
        if isinstance(tag, str) and tag and tag != MODEL_ROOT_TAG:
            return tag
    return None


def _filename_from_url(url: str) -> str | None:
    """The file name a URL names outright, or None to let the transport resolve it.

    A CivitAI download URL ends in a numeric id and only its API knows the real
    name, so guessing here would save the model under a name no loader lists.
    """
    name = unquote(os.path.basename(urlsplit(url).path))
    return name if os.path.splitext(name)[1] else None
