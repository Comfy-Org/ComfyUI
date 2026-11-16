"""headAssetByHash, assetFromHash, getAsset, getAssetContent and deleteAsset."""

import logging
import os
import uuid

import blake3
import pytest
from aiohttp import FormData
from sqlalchemy.exc import OperationalError

from app.api_v2 import assets as v2_assets

from .contract import response_schema, validate

DATA = b"\x89PNG\r\n\x1a\n" + b"v2-read-test-0123456789"
N = len(DATA)
HASH = "blake3:" + blake3.blake3(DATA).hexdigest()
STORED = HASH.removeprefix("blake3:") + ".png"
MISSING_HASH = "blake3:" + "0" * 64
CONDITIONALS = [
    ("If-None-Match", "*"),
    ("If-Modified-Since", "Fri, 01 Jan 2100 00:00:00 GMT"),
    ("If-Match", '"nope"'),
    ("If-Unmodified-Since", "Thu, 01 Jan 1970 00:00:00 GMT"),
]


@pytest.fixture(autouse=True)
def no_aiohttp_errors(caplog):
    """No handler or protocol error, e.g. "Cannot change the response status code", in any test."""
    yield
    bad = [r for r in caplog.get_records("call") if r.levelno >= logging.ERROR and r.name in ("aiohttp.server", "aiohttp.web")]
    assert not bad, [r.getMessage() for r in bad]


async def upload(client, data=DATA, file_path="input/a.png", content_type="image/png") -> dict:
    fd = FormData(default_to_multipart=True)
    fd.add_field("file_path", file_path)
    fd.add_field("content_type", content_type)
    fd.add_field("file", data, filename="f.bin", content_type="application/octet-stream")
    resp = await client.post("/api/v2/assets", data=fd)
    body = await resp.json()
    assert resp.status == 201, body
    return body


async def json_call(client, method, url, op, **kw):
    resp = await client.request(method, url, **kw)
    body = await resp.json()
    validate(response_schema(op, resp.status), body)
    return resp.status, body


async def from_hash(client, payload=None, **kw):
    return await json_call(client, "POST", "/api/v2/assets/from-hash", "assetFromHash", json=payload, **kw)


async def head(client, h):
    resp = await client.head(f"/api/v2/assets/by-hash/{h}")
    assert await resp.read() == b""
    return resp.status


async def content(client, asset_id, headers=None):
    resp = await client.get(f"/api/v2/assets/{asset_id}/content", headers=headers or {})
    return resp, await resp.read()


# by-hash and from-hash


@pytest.mark.asyncio
async def test_hash_lookups_with_hashing_on(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    assert await head(conforming_client, HASH) == 200
    assert await head(conforming_client, MISSING_HASH) == 404
    status, body = await from_hash(conforming_client, {"hash": HASH, "file_path": "input/other/x.png", "tags": ["t"], "expires_in": 60})
    assert status == 200, body
    assert body["id"] == asset["id"] and body["created_new"] is False
    assert body["file_path"] == STORED
    status, body = await from_hash(conforming_client, {"hash": MISSING_HASH})
    assert (status, body["error"]["code"]) == (404, "blob_not_found")


@pytest.mark.asyncio
async def test_hash_lookups_always_miss_with_hashing_off(conforming_client, v2_dirs, hashing):
    """No dedup fast-path: clients fall back to uploading the bytes."""
    hashing.enable_asset_hashing = False
    asset = await upload(conforming_client)
    assert await head(conforming_client, HASH) == 404
    status, body = await from_hash(conforming_client, {"hash": HASH, "file_path": "input/x.png"})
    assert (status, body["error"]["code"]) == (404, "blob_not_found")
    status, body = await from_hash(conforming_client, {"hash": HASH, "tags": [""]})
    assert (status, body["error"]["code"]) == (422, "invalid_request")
    status, body = await json_call(conforming_client, "GET", f"/api/v2/assets/{asset['id']}", "getAsset")
    assert status == 200 and body["hash"] == HASH


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["abc", "blake3:xyz", "blake3:" + "A" * 64, "sha256:" + "0" * 64, "blake3:" + "0" * 63])
async def test_by_hash_malformed_is_404(conforming_client, v2_dirs, bad):
    await upload(conforming_client)
    assert await head(conforming_client, bad) == 404


@pytest.mark.asyncio
async def test_from_hash_malformed_hash_string_is_miss(conforming_client, v2_dirs):
    status, body = await from_hash(conforming_client, {"hash": "not-a-hash"})
    assert (status, body["error"]["code"]) == (404, "blob_not_found")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [HASH],
        "x",
        None,
        {},
        {"hash": 5},
        {"hash": None},
        {"hash": HASH, "file_path": 3},
        {"hash": HASH, "file_path": "x\x00.png"},
        {"hash": HASH, "file_path": "a/x\u202e.png"},
        {"hash": HASH, "file_path": "dir/"},
        {"hash": HASH, "tags": "a"},
        {"hash": HASH, "tags": [1]},
        {"hash": HASH, "tags": [""]},
        {"hash": HASH, "tags": ["a\x00"]},
        {"hash": HASH, "tags": ["x" * 257]},
        {"hash": HASH, "tags": ["t"] * 65},
        {"hash": HASH, "expires_in": 59},
        {"hash": HASH, "expires_in": 604801},
        {"hash": HASH, "expires_in": "60"},
        {"hash": HASH, "expires_in": 60.0},
        {"hash": HASH, "expires_in": True},
    ],
)
async def test_from_hash_malformed_is_422_even_on_hit(conforming_client, v2_dirs, payload):
    await upload(conforming_client)
    status, body = await from_hash(conforming_client, payload)
    assert (status, body["error"]["code"]) == (422, "invalid_request"), body


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [b"", b"{", b"\xff\xfe", b"[]"])
async def test_from_hash_non_json_body_is_422(conforming_client, v2_dirs, raw):
    status, body = await from_hash(conforming_client, data=raw, headers={"Content-Type": "application/json"})
    assert (status, body["error"]["code"]) == (422, "invalid_request")


# getAsset


@pytest.mark.asyncio
async def test_get_asset_conforms(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    status, body = await json_call(conforming_client, "GET", f"/api/v2/assets/{asset['id']}", "getAsset")
    assert status == 200
    assert {k: body[k] for k in ("id", "hash", "size_bytes", "content_type", "file_path")} == {
        "id": asset["id"],
        "hash": HASH,
        "size_bytes": N,
        "content_type": "image/png",
        "file_path": STORED,
    }
    assert "created_new" not in body and "job_metadata" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["nope", "123", "from-hash", str(uuid.uuid4())])
async def test_get_asset_unknown_is_404(conforming_client, v2_dirs, bad):
    status, body = await json_call(conforming_client, "GET", f"/api/v2/assets/{bad}", "getAsset")
    assert (status, body["error"]["code"]) == (404, "not_found")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["{nope}", "{" + str(uuid.uuid4()) + "}"])
async def test_brace_ids_get_envelope_404(conforming_client, v2_dirs, bad):
    for method, url, op in (("GET", f"/api/v2/assets/{bad}", "getAsset"), ("GET", f"/api/v2/assets/{bad}/content", "getAssetContent"), ("DELETE", f"/api/v2/assets/{bad}", "deleteAsset")):
        status, body = await json_call(conforming_client, method, url, op)
        assert (status, body["error"]["code"]) == (404, "not_found")
    assert await head(conforming_client, "{" + HASH + "}") == 404


# getAssetContent


@pytest.mark.asyncio
async def test_content_full(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    resp, data = await content(conforming_client, asset["id"])
    assert resp.status == 200 and data == DATA
    assert resp.headers["Content-Type"] == "image/png"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Content-Disposition"] == "attachment; filename*=UTF-8''a.png"


@pytest.mark.asyncio
async def test_content_dangerous_type_downgraded(conforming_client, v2_dirs):
    asset = await upload(conforming_client, data=b"<svg xmlns='http://www.w3.org/2000/svg'/>", file_path="x.svg", content_type="image/svg+xml")
    resp, _ = await content(conforming_client, asset["id"])
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "application/octet-stream"
    assert resp.headers["Cache-Control"] == "no-store" and resp.headers["Vary"] == "Sec-Fetch-Dest"


@pytest.mark.asyncio
@pytest.mark.parametrize("_round", range(3))
async def test_content_ranges(conforming_client, v2_dirs, _round):
    asset = await upload(conforming_client)
    resp, data = await content(conforming_client, asset["id"], {"Range": "bytes=2-5"})
    assert resp.status == 206 and data == DATA[2:6]
    assert resp.headers["Content-Range"] == f"bytes 2-5/{N}"
    resp, data = await content(conforming_client, asset["id"], {"Range": "bytes=-4"})
    assert resp.status == 206 and data == DATA[-4:]
    resp, data = await content(conforming_client, asset["id"], {"Range": "bytes=999999-"})
    assert resp.status == 416 and data == b""
    assert resp.headers["Content-Range"] == f"bytes */{N}"


@pytest.mark.asyncio
@pytest.mark.parametrize("header", CONDITIONALS, ids=[h for h, _ in CONDITIONALS])
async def test_content_ignores_conditionals(conforming_client, v2_dirs, header):
    asset = await upload(conforming_client)
    resp, data = await content(conforming_client, asset["id"], dict([header]))
    assert resp.status == 200 and data == DATA
    resp, data = await content(conforming_client, asset["id"], dict([header, ("Range", "bytes=2-5")]))
    assert resp.status == 206 and data == DATA[2:6]


@pytest.mark.asyncio
async def test_content_if_range_matching(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    resp, _ = await content(conforming_client, asset["id"])
    resp, data = await content(conforming_client, asset["id"], {"If-Range": resp.headers["Last-Modified"], "Range": "bytes=2-5"})
    assert resp.status == 206 and data == DATA[2:6]


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["gzip", "br", "gzip, br"])
async def test_content_ignores_accept_encoding_siblings(conforming_client, v2_dirs, encoding):
    asset = await upload(conforming_client, file_path="input/demo.bin", content_type="application/octet-stream")
    stored = v2_dirs / "input" / asset["file_path"]
    stored.with_name(stored.name + ".gz").write_bytes(b"\x1f\x8bsibling-gz")
    stored.with_name(stored.name + ".br").write_bytes(b"sibling-br")
    url = f"/api/v2/assets/{asset['id']}/content"
    resp = await conforming_client.get(url, headers={"Accept-Encoding": encoding}, auto_decompress=False)
    data = await resp.read()
    assert resp.status == 200 and data == DATA
    assert "Content-Encoding" not in resp.headers
    resp = await conforming_client.get(url, headers={"Accept-Encoding": encoding, "Range": "bytes=2-5"}, auto_decompress=False)
    data = await resp.read()
    assert resp.status == 206 and data == DATA[2:6]
    assert resp.headers["Content-Range"] == f"bytes 2-5/{N}"


async def assert_content_404(client, asset_id):
    status, body = await json_call(client, "GET", f"/api/v2/assets/{asset_id}/content", "getAssetContent")
    assert (status, body["error"]["code"]) == (404, "not_found")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["nope", str(uuid.uuid4())])
async def test_content_unknown_is_404(conforming_client, v2_dirs, bad):
    await assert_content_404(conforming_client, bad)


@pytest.mark.asyncio
async def test_content_file_deleted_on_disk(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    os.unlink(v2_dirs / "input" / STORED)
    await assert_content_404(conforming_client, asset["id"])


@pytest.mark.asyncio
async def test_content_swapped_for_outside_symlink(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    outside = v2_dirs / "outside.png"
    outside.write_bytes(DATA)
    os.unlink(v2_dirs / "input" / STORED)
    os.symlink(outside, v2_dirs / "input" / STORED)
    await assert_content_404(conforming_client, asset["id"])


# deleteAsset


@pytest.mark.asyncio
async def test_delete_then_gone(conforming_client, v2_dirs):
    asset = await upload(conforming_client)
    resp = await conforming_client.delete(f"/api/v2/assets/{asset['id']}")
    assert resp.status == 204 and await resp.read() == b""
    status, body = await json_call(conforming_client, "GET", f"/api/v2/assets/{asset['id']}", "getAsset")
    assert (status, body["error"]["code"]) == (404, "not_found")
    status, body = await json_call(conforming_client, "DELETE", f"/api/v2/assets/{asset['id']}", "deleteAsset")
    assert (status, body["error"]["code"]) == (404, "not_found")
    assert (v2_dirs / "input" / STORED).read_bytes() == DATA
    assert await head(conforming_client, HASH) == 404
    status, body = await from_hash(conforming_client, {"hash": HASH})
    assert (status, body["error"]["code"]) == (404, "blob_not_found")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["nope", str(uuid.uuid4())])
async def test_delete_unknown_is_404(conforming_client, v2_dirs, bad):
    status, body = await json_call(conforming_client, "DELETE", f"/api/v2/assets/{bad}", "deleteAsset")
    assert (status, body["error"]["code"]) == (404, "not_found")


@pytest.mark.asyncio
async def test_delete_failure_is_enveloped(conforming_client, v2_dirs, monkeypatch):
    asset = await upload(conforming_client)

    def boom(reference_id):
        raise OperationalError("stmt", {}, Exception("locked"))

    monkeypatch.setattr(v2_assets, "delete_asset_reference", boom)
    status, body = await json_call(conforming_client, "DELETE", f"/api/v2/assets/{asset['id']}", "deleteAsset")
    assert (status, body["error"]["code"]) == (500, "upstream_error")
