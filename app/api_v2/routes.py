"""v2 route handlers and the helpers they share.

Every GET route passes ``allow_head=False``. A handler that needs the caller calls
``user_id(request)`` first; it raises a 403 ``forbidden`` envelope for unknown users.
Id and hash segments match ``[^/]+`` so that any segment, braces included, reaches the
handler and gets its 404 envelope.
"""

import asyncio
import json

from aiohttp import web

from . import assets, events, jobs

ROUTES = web.RouteTableDef()
V2_SERVER_KEY = web.AppKey("api_v2_server", object)


def _envelope(code: str, message: str, details: dict | None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


def error(status: int, code: str, message: str, details: dict | None = None, headers: dict | None = None) -> web.Response:
    return web.json_response(_envelope(code, message, details), status=status, headers=headers)


def user_id(request: web.Request) -> str:
    try:
        return request.app[V2_SERVER_KEY].user_manager.get_request_user_id(request)
    except KeyError as e:
        raise web.HTTPForbidden(text=json.dumps(_envelope("forbidden", e.args[0], None)), content_type="application/json")


def base_url(request: web.Request) -> str:
    """Scheme and host the request reached; X-Forwarded-* headers are ignored."""
    return f"{request.scheme}://{request.host}"


@ROUTES.get("/api/v2/health", allow_head=False)
async def get_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy"})


@ROUTES.post("/api/v2/assets")
async def post_assets(request: web.Request) -> web.Response:
    user_id(request)
    try:
        status, body = await assets.upload(request, base_url(request))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)
    return web.json_response(body, status=status)


@ROUTES.post("/api/v2/assets/from-hash")
async def asset_from_hash(request: web.Request) -> web.Response:
    user_id(request)
    try:
        return web.json_response(await assets.from_hash(request, base_url(request)))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.post("/api/v2/assets/from-path")
async def asset_from_path(request: web.Request) -> web.Response:
    return error(501, "not_implemented", "Registering a host file is not implemented by ComfyUI; upload it with POST /api/v2/assets instead.")


@ROUTES.head("/api/v2/assets/by-hash/{hash:[^/]+}")
async def head_asset_by_hash(request: web.Request) -> web.Response:
    user_id(request)
    found = await asyncio.to_thread(assets.mintable_reference, request.match_info["hash"])
    return web.Response(status=200 if found else 404)


@ROUTES.get("/api/v2/assets/{id:[^/]+}", allow_head=False)
async def get_asset(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        return web.json_response(await assets.detail(request.match_info["id"], base_url(request), job_metadata=lambda job_id: (r := store.owned(job_id, uid)) and r.metadata))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.get("/api/v2/assets/{id:[^/]+}/content", allow_head=False)
async def get_asset_content(request: web.Request) -> web.StreamResponse:
    user_id(request)
    try:
        return await assets.content(request)
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.delete("/api/v2/assets/{id:[^/]+}")
async def delete_asset(request: web.Request) -> web.Response:
    user_id(request)
    try:
        await assets.delete(request.match_info["id"])
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)
    return web.Response(status=204)


@ROUTES.get("/api/v2/jobs", allow_head=False)
async def list_jobs(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        return web.json_response(await jobs.list_jobs(request.app[V2_SERVER_KEY], store, request.rel_url.raw_query_string, uid, base_url(request)))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.post("/api/v2/jobs")
async def post_jobs(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        body = await jobs.submit(request, request.app[V2_SERVER_KEY], store, uid, base_url(request))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message, e.details, e.headers)
    return web.json_response(body, status=201)


@ROUTES.get("/api/v2/jobs/{id:[^/]+}", allow_head=False)
async def get_job(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        return web.json_response(await jobs.get(request.app[V2_SERVER_KEY], store, request.match_info["id"], uid, base_url(request)))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.post("/api/v2/jobs/{id:[^/]+}/cancel")
async def cancel_job(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        return web.json_response(await jobs.cancel(request.app[V2_SERVER_KEY], store, request.match_info["id"], uid, base_url(request)))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.get("/api/v2/jobs/{id:[^/]+}/workflow", allow_head=False)
async def get_job_workflow(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    user_id(request)
    try:
        return web.json_response(jobs.workflow(request.app[V2_SERVER_KEY], store, request.match_info["id"]))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.get("/api/v2/jobs/{id:[^/]+}/logs", allow_head=False)
async def get_job_logs(request: web.Request) -> web.Response:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    user_id(request)
    try:
        return web.Response(status=204, headers={"No-Log-Reason": jobs.log_reason(request.app[V2_SERVER_KEY], store, request.match_info["id"])})
    except assets.ApiError as e:
        return error(e.status, e.code, e.message)


@ROUTES.get("/api/v2/jobs/{id:[^/]+}/events", allow_head=False)
async def get_job_events(request: web.Request) -> web.StreamResponse:
    store = request.app[jobs.JOBS_KEY]
    store.prune(jobs.now())
    uid = user_id(request)
    try:
        return await events.stream(request, request.app[V2_SERVER_KEY], store, uid, base_url(request))
    except assets.ApiError as e:
        return error(e.status, e.code, e.message, e.details, e.headers)
