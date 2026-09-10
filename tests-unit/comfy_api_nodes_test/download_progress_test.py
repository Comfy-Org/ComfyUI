import asyncio
from io import BytesIO

from aiohttp import web
from aiohttp.test_utils import TestServer

import comfy.utils
from comfy_api_nodes.util.download_helpers import download_url_to_bytesio

BODY = b"x" * (3 * 1024 * 1024 + 123)


def capture_hooks():
    progress = []
    activity = []
    comfy.utils.set_progress_bar_global_hook(
        lambda value, total, preview, node_id=None: progress.append((value, total))
    )
    comfy.utils.set_progress_activity_global_hook(activity.append)
    return progress, activity


def clear_hooks():
    comfy.utils.set_progress_bar_global_hook(None)
    comfy.utils.set_progress_activity_global_hook(None)


async def serve_and_download(handler):
    app = web.Application()
    app.router.add_get("/file", handler)
    async with TestServer(app) as server:
        dest = BytesIO()
        await download_url_to_bytesio(str(server.make_url("/file")), dest, max_retries=0)
        return dest.getvalue()


def test_download_reports_bytes_against_declared_length():
    async def handler(request):
        return web.Response(body=BODY)

    progress, activity = capture_hooks()
    try:
        data = asyncio.run(serve_and_download(handler))
    finally:
        clear_hooks()

    assert data == BODY
    assert activity == ["downloading", None]
    assert progress[-1] == (len(BODY), len(BODY))
    assert all(total == len(BODY) for _, total in progress)


def test_download_without_declared_length_says_downloading_only():
    async def handler(request):
        resp = web.StreamResponse()
        await resp.prepare(request)
        await resp.write(BODY)
        await resp.write_eof()
        return resp

    progress, activity = capture_hooks()
    try:
        data = asyncio.run(serve_and_download(handler))
    finally:
        clear_hooks()

    assert data == BODY
    assert activity == ["downloading", None]
    assert progress == []
