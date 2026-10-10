"""Tests for /view HTTP caching: Cache-Control: no-cache, validators and 304s"""

import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import folder_paths
import server
from utils.mime_types import init_mime_types

init_mime_types()

VARIANTS = [
    {},
    {"res": "64"},
    {"preview": "webp;75"},
    {"channel": "rgb"},
    {"channel": "a"},
]


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    Image.new("RGBA", (200, 100), (10, 20, 30, 128)).save(tmp_path / "a.png")
    return tmp_path


@pytest_asyncio.fixture
async def client():
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as c:
        yield c


async def get(client, params, headers=None):
    resp = await client.get("/view", params=params, headers=headers or {})
    await resp.read()
    return resp


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", VARIANTS)
async def test_no_cache_with_etag_and_304(output_dir, client, variant):
    params = {"filename": "a.png", **variant}
    first = await get(client, params)
    assert first.status == 200
    assert first.headers["Cache-Control"] == "no-cache"
    etag = first.headers["ETag"]

    again = await get(client, params, {"If-None-Match": etag})
    assert again.status == 304
    assert again.headers["ETag"] == etag
    assert again.headers["Cache-Control"] == "no-cache"

    assert (await get(client, params, {"If-None-Match": "*"})).status == 304
    assert (await get(client, params, {"If-None-Match": f"W/{etag}"})).status == 304
    assert (await get(client, params, {"If-None-Match": f'"other", {etag}'})).status == 304
    assert (await get(client, params, {"If-None-Match": '"other"'})).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("change", ["mtime", "size"])
async def test_etag_changes_when_file_changes(output_dir, client, variant, change):
    params = {"filename": "a.png", **variant}
    path = output_dir / "a.png"
    etag = (await get(client, params)).headers["ETag"]
    st = os.stat(path)
    if change == "mtime":
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    else:
        Image.new("RGBA", (300, 100), (1, 2, 3, 255)).save(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert os.stat(path).st_size != st.st_size
    resp = await get(client, params, {"If-None-Match": etag})
    assert resp.status == 200
    assert resp.headers["ETag"] != etag


@pytest.mark.asyncio
async def test_variants_have_distinct_etags(output_dir, client):
    etags = [(await get(client, {"filename": "a.png", **v})).headers["ETag"] for v in VARIANTS]
    etags.append((await get(client, {"filename": "a.png", "res": "32"})).headers["ETag"])
    etags.append((await get(client, {"filename": "a.png", "preview": "jpeg;75"})).headers["ETag"])
    assert len(set(etags)) == len(etags)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", VARIANTS[1:])
async def test_304_does_not_decode(output_dir, client, monkeypatch, variant):
    params = {"filename": "a.png", **variant}
    etag = (await get(client, params)).headers["ETag"]

    def fail(*args, **kwargs):
        raise AssertionError("decoded on a conditional hit")
    monkeypatch.setattr(server.Image, "open", fail)
    assert (await get(client, params, {"If-None-Match": etag})).status == 304


@pytest.mark.asyncio
async def test_dangerous_type_keeps_no_store(output_dir, client):
    (output_dir / "x.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    resp = await get(client, {"filename": "x.svg"})
    assert resp.headers["Cache-Control"] == "no-store"
    assert resp.headers["Vary"] == "Sec-Fetch-Dest"
