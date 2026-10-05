"""The `/api/tasks` surface for local model downloads.

Deliberately the same three calls the frontend already makes against Cloud --
``POST /api/assets/download`` to submit, ``GET /api/tasks/{id}`` to read,
``DELETE /api/tasks/{id}`` to cancel -- so the browser needs no local-only code
path, and no knowledge that a download is anything other than a task.
"""

from __future__ import annotations

import asyncio
import functools
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
from app.assets.downloads.destination import DestinationError, known_folder
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


def _error(status: int, code: str, message: str, details: dict[str, Any] | None = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _transport_errors(handler):
    """Answer a transport failure in this API's own error shape.

    Every handler here reaches the backend, including the read ones: resolving
    an unknown task id enumerates, which spawns a subprocess. Without this a
    server whose comfy-cli went missing answers `GET /api/tasks/{id}` with a
    plain-text 500 carrying a stack trace. The frontend treats any non-404 the
    same way either path, so this buys a diagnosable error for whoever is
    reading the response, not different browser behaviour.
    """

    @functools.wraps(handler)
    async def wrapped(request: web.Request) -> web.Response:
        try:
            return await handler(request)
        except DownloadBackendUnavailable as e:
            return _error(503, "DEPENDENCY_MISSING", str(e))
        except DownloadRejected as e:
            status = 409 if e.code in _CONFLICT_CODES else 502
            return _error(status, e.code, e.message, {"hint": e.hint} if e.hint else None)
        except asyncio.TimeoutError:
            return _error(504, "DOWNLOAD_BACKEND_TIMEOUT", "The download backend did not respond in time.")
        except OSError as e:
            return _error(502, "DOWNLOAD_BACKEND_ERROR", f"Could not run the download backend: {e}")

    return wrapped


@ROUTES.post("/api/assets/download")
@_transport_errors
async def submit_download(request: web.Request) -> web.Response:
    service = request.app[DOWNLOAD_SERVICE]
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return _error(400, "INVALID_BODY", "Request body must be JSON.")
    if not isinstance(body, dict):
        return _error(400, "INVALID_BODY", "Request body must be a JSON object.")

    url = body.get("source_url")
    filename = body.get("filename")
    if not isinstance(url, str) or not url.strip():
        return _error(400, "INVALID_BODY", "source_url is required and must be a string.")
    if filename is not None and not isinstance(filename, str):
        return _error(400, "INVALID_BODY", "filename must be a string.")
    url = url.strip()
    try:
        scheme = urlsplit(url).scheme
    except ValueError:
        return _error(400, "INVALID_BODY", "source_url is not a valid URL.")
    if scheme not in ("http", "https"):
        return _error(400, "INVALID_BODY", "source_url must be an http(s) URL.")

    folder = _folder_from_tags(body.get("tags"))
    if folder is None:
        return _error(
            400,
            "INVALID_BODY",
            f"A '{MODEL_TYPE_TAG_PREFIX}<folder>' tag is required to decide where the model belongs.",
        )

    try:
        task = await service.start(url, folder, filename or _filename_from_url(url))
    except DestinationError as e:
        return _error(400, e.code, e.message)
    except DownloadRejected as e:
        # On submit, unlike on a read, a transport refusal is usually something
        # the caller can fix: a different name, or the download already running.
        status = 409 if e.code in _CONFLICT_CODES else 400
        return _error(status, e.code, e.message, {"hint": e.hint} if e.hint else None)
    except (DownloadBackendUnavailable, asyncio.TimeoutError, OSError):
        # Mapped by _transport_errors. Named before the catch-all below because
        # asyncio.TimeoutError is OSError is a subclass of Exception, so a bare
        # handler would swallow them and report an unhelpful 500.
        raise
    except Exception:
        logging.exception("Failed to start model download from %s", _scrub(url))
        return _error(500, "INTERNAL", "Could not start the download.")
    return web.json_response(task, status=202)


@ROUTES.get("/api/tasks/{task_id}")
@_transport_errors
async def get_task(request: web.Request) -> web.Response:
    task = await request.app[DOWNLOAD_SERVICE].get_task(request.match_info["task_id"])
    if task is None:
        return _error(404, "TASK_NOT_FOUND", "No such task.")
    return web.json_response(task)


@ROUTES.delete("/api/tasks/{task_id}")
@_transport_errors
async def cancel_task(request: web.Request) -> web.Response:
    outcome = await request.app[DOWNLOAD_SERVICE].cancel_task(request.match_info["task_id"])
    if outcome is CancelOutcome.MISSING:
        return _error(404, "TASK_NOT_FOUND", "No such task.")
    if outcome is CancelOutcome.NOT_CANCELLABLE:
        return _error(409, "TASK_NOT_CANCELLABLE", "This task can no longer be cancelled.")
    return web.json_response({"status": "cancelling"})


def _scrub(url: str) -> str:
    """Drop the parts of a download url that can carry a credential.

    A resolved url can arrive with a presigned token in its query string or a
    password in its userinfo, and this one is headed for the log.
    """
    parts = urlsplit(url)
    return parts._replace(netloc=parts.hostname or "", query="", fragment="").geturl()


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
    # Without the prefix there is nothing marking which tag is the folder, so
    # skip the ones that do not name one rather than failing on the first
    # descriptive tag that happens to come first.
    for tag in tags:
        if isinstance(tag, str) and tag and tag != MODEL_ROOT_TAG and _names_a_model_folder(tag):
            return tag
    return None


def _names_a_model_folder(tag: str) -> bool:
    try:
        known_folder(tag)
    except DestinationError:
        return False
    return True


def _filename_from_url(url: str) -> str | None:
    """The file name a URL names outright, or None to let the transport resolve it.

    A CivitAI download URL ends in a numeric id and only its API knows the real
    name, so guessing here would save the model under a name no loader lists.
    """
    name = unquote(os.path.basename(urlsplit(url).path))
    return name if os.path.splitext(name)[1] else None
