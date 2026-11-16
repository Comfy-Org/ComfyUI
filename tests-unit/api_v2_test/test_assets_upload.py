"""postAssets: the multipart wire reader, and delegation of storage to the asset service."""

import json
import os

import blake3
import pytest
from aiohttp import FormData
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

import folder_paths
from app.api_v2 import assets as v2_assets
from app.assets.database.models import Asset
from app.assets.services.asset_management import get_asset_detail
from app.assets.services.ingest import register_file_in_place
from app.database.db import create_session
from comfy.cli_args import args

from .contract import response_schema, validate

PNG = b"\x89PNG\r\n\x1a\n" + b"v2-upload-test"
MODEL_TAGS = ("models", "model_type:checkpoints")


def digest(data: bytes = PNG) -> str:
    return blake3.blake3(data).hexdigest()


def form(fields=(), data=PNG, file_first=False) -> FormData:
    """``fields`` is a sequence of (name, value); ``data=None`` omits the file part."""
    fd = FormData(default_to_multipart=True)
    add_file = data is not None and (lambda: fd.add_field("file", data, filename="f.bin", content_type="application/octet-stream"))
    if file_first and add_file:
        add_file()
    for name, value in fields:
        fd.add_field(name, value)
    if not file_first and add_file:
        add_file()
    return fd


def basic(file_path="input/a.png", content_type="image/png", extra=()):
    return [("file_path", file_path), ("content_type", content_type), *extra]


def tagged(file_path, *tags):
    return basic(file_path, extra=[("tags", json.dumps(list(tags)))])


async def post(client, fields=(), data=PNG, file_first=False):
    resp = await client.post("/api/v2/assets", data=form(fields, data, file_first))
    body = await resp.json()
    check_body(resp.status, body)
    return resp.status, body


def check_body(status, body):
    """413 declares no body shape; its description names the self-hosted ``ErrorEnvelope`` form."""
    validate(response_schema("postAssets", status) or "ErrorEnvelope", body)


def assert_error(status, body, expected_status, code="invalid_request"):
    assert status == expected_status, body
    assert body["error"]["code"] == code


def leftovers(root) -> list[str]:
    """Staging files or directories left under temp/uploads."""
    return [str(p) for p in (root / "temp/uploads").rglob("*")]


def stored(root) -> list[str]:
    return sorted(str(p.relative_to(root)) for sub in ("input", "output", "models") for p in (root / sub).rglob("*") if p.is_file())


# delegation


@pytest.mark.asyncio
async def test_upload_delegates_to_asset_service(conforming_client, v2_dirs, monkeypatch):
    calls = []
    real = v2_assets.upload_from_temp_path

    def spy(**kwargs):
        calls.append({**kwargs, "staged": open(kwargs["temp_path"], "rb").read()})
        return real(**kwargs)

    monkeypatch.setattr(v2_assets, "upload_from_temp_path", spy)
    good = "blake3:" + digest()
    status, body = await post(conforming_client, basic("input/dir\\sub/x.png", "image/png; q=1", [("tags", "mine"), ("expected_hash", good)]))
    assert status == 201, body
    (call,) = calls
    assert call.pop("temp_path").startswith(str(v2_dirs / "temp" / "uploads"))
    assert call == {"name": "x.png", "client_filename": "x.png", "tags": ["mine", "input"], "mime_type": "image/png; q=1", "expected_hash": good, "reuse_content": True, "staged": PNG}
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_happy_input_upload(conforming_client, v2_dirs):
    status, body = await post(conforming_client, basic(extra=[("tags", "mine"), ("tags", '["a", "b"]')]))
    assert status == 201, body
    assert body["created_new"] is True
    assert body["hash"] == "blake3:" + digest()
    assert body["size_bytes"] == len(PNG)
    assert body["content_type"] == "image/png"
    assert body["file_path"] == f"{digest()}.png"
    assert body["expires_at"] is None
    assert body["url"].endswith(f"/api/v2/assets/{body['id']}/content")
    assert body["created_at"].endswith("Z") and body["url_expires_at"].endswith("Z")
    assert "job_id" not in body
    assert stored(v2_dirs) == [f"input/{digest()}.png"]
    assert (v2_dirs / "input" / body["file_path"]).read_bytes() == PNG
    detail = get_asset_detail(body["id"])
    assert detail.ref.name == "a.png"
    assert {"input", "mine", "a", "b"} <= set(detail.tags)
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path,stored_ext",
    [("photo.jpeg", ".jpeg"), ("input/dir/x.safetensors", ".safetensors"), ("input\\b\\c.webp", ".webp"), ("noext", ""), ("x." + "e" * 16, "")],
)
async def test_extension_comes_from_file_path(conforming_client, v2_dirs, file_path, stored_ext):
    status, body = await post(conforming_client, basic(file_path))
    assert status == 201, body
    assert body["file_path"] == digest() + stored_ext
    assert stored(v2_dirs) == [f"input/{digest()}{stored_ext}"]


@pytest.mark.asyncio
async def test_model_tags_route_to_model_folder(conforming_client, v2_dirs):
    status, body = await post(conforming_client, tagged("x.safetensors", *MODEL_TAGS))
    assert status == 201, body
    assert body["file_path"] == f"{digest()}.safetensors"
    assert stored(v2_dirs) == [f"models/checkpoints/{digest()}.safetensors"]
    assert {"models", "model_type:checkpoints"} <= set(get_asset_detail(body["id"]).tags)
    assert "input" not in get_asset_detail(body["id"]).tags


@pytest.mark.asyncio
@pytest.mark.parametrize("tags,root", [(["output"], "output"), (["input"], "input"), (["Input"], "input")])
async def test_role_tag_routing(conforming_client, v2_dirs, tags, root):
    status, body = await post(conforming_client, tagged("a.png", *tags))
    assert status == 201, body
    assert stored(v2_dirs) == [f"{root}/{digest()}.png"]


def record_count() -> int:
    with create_session() as session:
        return session.scalar(select(func.count()).select_from(Asset))


@pytest.mark.asyncio
@pytest.mark.parametrize("hashing_enabled", [True, False])
@pytest.mark.parametrize("seed", [None, "input", "output", "checkpoints"])
@pytest.mark.parametrize(
    "file_path,tags",
    [
        ("a.png", ["models"]),
        ("a.png", ["models", "model_type:nope"]),
        ("a.png", ["input", "output"]),
        ("a.png", ["models", "model_type:checkpoints", "model_type:vae"]),
        ("input/a.png", ["input", "input"]),
        ("checkpoints/a.safetensors", ["model_type:checkpoints", "model_type:checkpoints"]),
        ("output/a.png", ["input"]),
        ("checkpoints/a.safetensors", ["models", "model_type:loras"]),
    ],
)
async def test_bad_routing_tags_rejected(conforming_client, v2_dirs, hashing, hashing_enabled, seed, file_path, tags):
    """Rejected whether or not identical bytes are already stored in some folder."""
    hashing.enable_asset_hashing = hashing_enabled
    if seed is not None:
        status, body = await post(conforming_client, basic(f"{seed}/seed.safetensors"))
        assert status == 201, body
    before = stored(v2_dirs), record_count()
    status, body = await post(conforming_client, tagged(file_path, *tags))
    assert_error(status, body, 422)
    assert (stored(v2_dirs), record_count()) == before
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hashing_enabled", [True, False])
@pytest.mark.parametrize(
    "file_path,tags,root",
    [
        ("a.png", ["input"], "input"),
        ("a.png", [], "input"),
        ("input/a.png", ["mine"], "input"),
        ("a.png", ["output"], "output"),
        ("a.safetensors", ["models", "model_type:checkpoints"], "models/checkpoints"),
        ("models/photo.png", ["output"], "output"),
    ],
)
async def test_valid_routing_miss_then_duplicate(conforming_client, v2_dirs, hashing, hashing_enabled, file_path, tags, root):
    hashing.enable_asset_hashing = hashing_enabled
    status, first = await post(conforming_client, tagged(file_path, *tags))
    assert (status, first["created_new"]) == (201, True), first
    status, again = await post(conforming_client, tagged(file_path, *tags))
    assert (status, again["created_new"]) == ((200, False) if hashing_enabled else (201, True)), again
    assert again["file_path"] == first["file_path"]
    assert stored(v2_dirs) == [f"{root}/{first['file_path']}"]
    assert record_count() == 2


# file_path prefix -> destination


def role_tags(asset_id: str) -> set[str]:
    return {t for t in get_asset_detail(asset_id).tags if t in v2_assets.ROLE_TAGS or t.startswith("model_type:")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path,root,roles",
    [
        ("models/checkpoints/x.safetensors", "models/checkpoints", {"models", "model_type:checkpoints"}),
        ("checkpoints/x.safetensors", "models/checkpoints", {"models", "model_type:checkpoints"}),
        ("loras/sub/x.safetensors", "models/loras", {"models", "model_type:loras"}),
        ("models\\loras\\x.safetensors", "models/loras", {"models", "model_type:loras"}),
        ("input/sub/x.png", "input", {"input"}),
        ("output/x.png", "output", {"output"}),
        ("models/x.png", "input", {"input"}),
        ("x.png", "input", {"input"}),
        ("input/x.png", "input", {"input"}),
    ],
)
async def test_file_path_prefix_routes(conforming_client, v2_dirs, file_path, root, roles):
    status, body = await post(conforming_client, basic(file_path))
    assert status == 201, body
    ext = os.path.splitext(file_path)[1]
    assert body["file_path"] == digest() + ext
    assert stored(v2_dirs) == [f"{root}/{digest()}{ext}"]
    assert role_tags(body["id"]) == roles
    assert get_asset_detail(body["id"]).ref.name == "x" + ext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path,tags,root",
    [
        ("checkpoints/x.safetensors", ["mine", "models", "model_type:checkpoints"], "models/checkpoints"),
        ("output/x.png", ["mine", "output"], "output"),
        ("input/x.png", ["mine"], "input"),
    ],
)
async def test_file_path_prefix_merges_tags(conforming_client, v2_dirs, file_path, tags, root):
    status, body = await post(conforming_client, tagged(file_path, *tags))
    assert status == 201, body
    assert stored(v2_dirs) == [f"{root}/{body['file_path']}"]
    assert set(tags) <= set(get_asset_detail(body["id"]).tags)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path,tags",
    [
        ("models/checkpoints/x.safetensors", ["input"]),
        ("checkpoints/x.safetensors", ["models", "model_type:loras"]),
        ("loras/x.safetensors", ["model_type:checkpoints"]),
        ("output/x.png", ["input"]),
        ("input/x.png", [" output"]),
        ("input/x.png", ["output"]),
        ("input/x.png", ["models", "model_type:checkpoints"]),
    ],
)
async def test_file_path_prefix_conflicting_role_tags_rejected(conforming_client, v2_dirs, file_path, tags):
    await post(conforming_client, basic("seed.png"))  # same bytes already stored: must not be reused either
    status, body = await post(conforming_client, tagged(file_path, *tags))
    assert_error(status, body, 422)
    assert stored(v2_dirs) == [f"input/{digest()}.png"]
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_prefix_model_upload_resolves_for_checkpoint_loader(conforming_client, v2_dirs):
    status, body = await post(conforming_client, basic("models/checkpoints/x.safetensors", "application/octet-stream"))
    assert status == 201, body
    missing = []
    value = v2_assets.resolve_asset_refs({"__type": "core/ASSET", "info": {"id": body["id"]}}, missing)
    assert missing == [] and value == body["file_path"] == f"{digest()}.safetensors"
    assert folder_paths.get_full_path("checkpoints", value) == str(v2_dirs / "models" / "checkpoints" / value)
    assert value in folder_paths.get_filename_list("checkpoints")


# dedup


@pytest.mark.asyncio
async def test_reupload_dedups_with_200(conforming_client, v2_dirs):
    status, first = await post(conforming_client, basic())
    assert (status, first["created_new"]) == (201, True)
    status, again = await post(conforming_client, basic("input/other.png"))
    assert status == 200, again
    assert again["created_new"] is False
    assert again["id"] != first["id"] and again["hash"] == first["hash"]
    assert again["file_path"] == first["file_path"]
    assert get_asset_detail(again["id"]).ref.name == "other.png"
    assert stored(v2_dirs) == [f"input/{digest()}.png"]
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_model_reupload_dedups_with_200(conforming_client, v2_dirs):
    status, first = await post(conforming_client, basic("checkpoints/a.safetensors"))
    assert status == 201, first
    status, again = await post(conforming_client, basic("models/checkpoints/b.safetensors"))
    assert (status, again["created_new"], again["file_path"]) == (200, False, first["file_path"])
    assert stored(v2_dirs) == [f"models/checkpoints/{digest()}.safetensors"]


FOLDERS = {"input": "input", "output": "output", "checkpoints": "models/checkpoints"}


@pytest.mark.asyncio
@pytest.mark.parametrize("hashing_enabled", [True, False])
@pytest.mark.parametrize(
    "first,second",
    [("input", "output"), ("output", "input"), ("input", "checkpoints"), ("checkpoints", "input"), ("output", "checkpoints")],
)
async def test_identical_bytes_in_another_folder_are_stored_again(conforming_client, v2_dirs, hashing, hashing_enabled, first, second):
    hashing.enable_asset_hashing = hashing_enabled
    status, earlier = await post(conforming_client, basic(f"{first}/a.safetensors"))
    assert status == 201, earlier
    status, body = await post(conforming_client, basic(f"{second}/b.safetensors"))
    assert (status, body["created_new"]) == (201, True), body
    name = f"{digest()}.safetensors"
    assert body["file_path"] == name
    assert stored(v2_dirs) == sorted(f"{FOLDERS[f]}/{name}" for f in (first, second))
    await assert_downloads(conforming_client, earlier["id"])
    await assert_downloads(conforming_client, body["id"])
    if second == "checkpoints":
        value = v2_assets.resolve_asset_refs({"__type": "core/ASSET", "info": {"id": body["id"]}}, [])
        assert value == name and value in folder_paths.get_filename_list("checkpoints")
        assert folder_paths.get_full_path("checkpoints", value) == str(v2_dirs / "models" / "checkpoints" / name)
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_registered_output_is_not_reused_for_an_input_upload(conforming_client, v2_dirs):
    out = v2_dirs / "output" / "made.png"
    out.write_bytes(PNG)
    register_file_in_place(abs_path=str(out), name="made.png", tags=["output"])
    status, body = await post(conforming_client, basic())
    assert (status, body["created_new"], body["file_path"]) == (201, True, f"{digest()}.png")
    assert stored(v2_dirs) == [f"input/{digest()}.png", "output/made.png"]
    assert leftovers(v2_dirs) == []


async def assert_downloads(client, asset_id, data=PNG):
    resp = await client.get(f"/api/v2/assets/{asset_id}/content")
    assert resp.status == 200 and await resp.read() == data


@pytest.mark.asyncio
async def test_hashing_off_stores_every_upload(conforming_client, v2_dirs, hashing):
    hashing.enable_asset_hashing = False
    bodies = []
    for file_path in ("input/a.png", "input/a.png", "other.png"):
        status, body = await post(conforming_client, basic(file_path))
        assert (status, body["created_new"]) == (201, True), body
        bodies.append(body)
    assert len({b["id"] for b in bodies}) == 3 and {b["hash"] for b in bodies} == {"blake3:" + digest()}
    for body in bodies:
        await assert_downloads(conforming_client, body["id"])
        assert v2_assets.resolve_asset_refs({"__type": "core/ASSET", "info": {"id": body["id"]}}, []) == f"{digest()}.png"
    assert stored(v2_dirs) == [f"input/{digest()}.png"]
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_hashing_off_stores_beside_existing_content(conforming_client, v2_dirs, hashing):
    hashing.enable_asset_hashing = False
    out = v2_dirs / "output" / "made.png"
    out.write_bytes(PNG)
    made = register_file_in_place(abs_path=str(out), name="made.png", tags=["output"])
    status, body = await post(conforming_client, basic())
    assert (status, body["created_new"], body["file_path"]) == (201, True, f"{digest()}.png")
    assert stored(v2_dirs) == [f"input/{digest()}.png", "output/made.png"]
    await assert_downloads(conforming_client, body["id"])
    await assert_downloads(conforming_client, made.ref.id)


@pytest.mark.asyncio
async def test_dedup_skips_deleted_file(conforming_client, v2_dirs):
    status, first = await post(conforming_client, basic("input/d.png"))
    assert status == 201
    os.unlink(v2_dirs / "input" / first["file_path"])
    status, body = await post(conforming_client, basic("input/d.png"))
    assert status == 201, body
    assert body["created_new"] is True and body["id"] != first["id"]
    assert (v2_dirs / "input" / body["file_path"]).read_bytes() == PNG


@pytest.mark.asyncio
async def test_idempotency_key_is_ignored(conforming_client, v2_dirs):
    for data in (PNG, PNG + b"x"):
        resp = await conforming_client.post("/api/v2/assets", data=form(basic(), data), headers={"Idempotency-Key": "k1"})
        body = await resp.json()
        check_body(resp.status, body)
        assert resp.status == 201 and body["file_path"] == f"{digest(data)}.png"


# wire reader


@pytest.mark.asyncio
async def test_expected_hash(conforming_client, v2_dirs):
    good = "blake3:" + digest()
    status, body = await post(conforming_client, basic(extra=[("expected_hash", "blake3:" + "0" * 64)]))
    assert_error(status, body, 409, "hash_mismatch")
    assert stored(v2_dirs) == []
    for bad in ("", "blake3:" + "0" * 63, "BLAKE3:" + "0" * 64, "sha256:" + "0" * 64, good.upper()):
        status, body = await post(conforming_client, basic(extra=[("expected_hash", bad)]))
        assert_error(status, body, 422)
    status, body = await post(conforming_client, basic(extra=[("expected_hash", good)]))
    assert status == 201, body
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["file", "file_path", "content_type"])
async def test_missing_required_field(conforming_client, v2_dirs, missing):
    fields = [f for f in basic() if f[0] != missing]
    status, body = await post(conforming_client, fields, data=None if missing == "file" else PNG)
    assert_error(status, body, 422)
    assert stored(v2_dirs) == []
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["", "   ", "image/png\n", "image/\x7fpng", "image/png\u202e", "image/\x85png"])
async def test_bad_content_type(conforming_client, v2_dirs, content_type):
    status, body = await post(conforming_client, basic(content_type=content_type))
    assert_error(status, body, 422)


@pytest.mark.asyncio
@pytest.mark.parametrize("file_path", ["", "input/", "a\\", "x\x00.png", "x\x1f.png", "x\u202e.png", "dir\u2066/x.png"])
async def test_rejected_file_paths(conforming_client, v2_dirs, file_path):
    status, body = await post(conforming_client, basic(file_path))
    assert_error(status, body, 422)
    assert stored(v2_dirs) == []
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hashing_enabled", [True, False])
@pytest.mark.parametrize(
    "file_path",
    [
        "unknown/x.png",
        "configs/x",
        "custom_nodes/x",
        "models/nope/x.png",
        "models/input/x.png",
        "Output/x.png",
        "../../etc/x.png",
        "/abs/x.png",
        "/input/x.png",
        "\\input\\x.png",
        "./output/x.png",
        "input/../../x.png",
        "input/./x.png",
        "input//x.png",
        "C:\\input\\x.png",
        "C:/x.png",
        "..",
    ],
)
async def test_unknown_or_unsafe_file_path_folders_rejected(conforming_client, v2_dirs, hashing, hashing_enabled, file_path):
    """Rejected before the service, for fresh bytes and when the same bytes are already stored in input."""
    hashing.enable_asset_hashing = hashing_enabled
    status, body = await post(conforming_client, basic(file_path))
    assert_error(status, body, 422)
    assert (stored(v2_dirs), record_count()) == ([], 0)
    status, body = await post(conforming_client, basic("input/seed.png"))
    assert status == 201, body
    status, body = await post(conforming_client, basic(file_path))
    assert_error(status, body, 422)
    assert (stored(v2_dirs), record_count()) == ([f"input/{digest()}.png"], 1)
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [OSError("disk gone"), OperationalError("stmt", {}, Exception("locked")), RuntimeError("move failed")])
async def test_service_failure_is_enveloped_and_cleaned_up(conforming_client, v2_dirs, monkeypatch, exc):
    def boom(**kwargs):
        assert os.path.isfile(kwargs["temp_path"])
        raise exc

    monkeypatch.setattr(v2_assets, "upload_from_temp_path", boom)
    status, body = await post(conforming_client, basic())
    assert_error(status, body, 500, "upstream_error")
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_file_before_file_path(conforming_client, v2_dirs):
    status, body = await post(conforming_client, basic("input/first.png"), file_first=True)
    assert status == 201, body
    assert get_asset_detail(body["id"]).ref.name == "first.png"


@pytest.mark.asyncio
async def test_oversize_upload(conforming_client, v2_dirs, monkeypatch):
    monkeypatch.setattr(args, "max_upload_size", 1)
    status, body = await post(conforming_client, basic(), data=b"\0" * (1024 * 1024 + 1))
    assert_error(status, body, 413, "payload_too_large")
    assert stored(v2_dirs) == []
    assert leftovers(v2_dirs) == []
    status, body = await post(conforming_client, basic(), data=b"\0" * (1024 * 1024))
    assert status == 201, body


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["10", "59", "604801", "abc", "", "6e2", "-60"])
async def test_bad_expires_in(conforming_client, v2_dirs, value):
    status, body = await post(conforming_client, basic(extra=[("expires_in", value)]))
    assert_error(status, body, 422)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["60", "604800"])
async def test_expires_in_accepted_and_ignored(conforming_client, v2_dirs, value):
    status, body = await post(conforming_client, basic(extra=[("expires_in", value)]))
    assert status == 201, body
    assert body["expires_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag_parts",
    [
        [json.dumps([f"t{i}" for i in range(65)])],
        [f"t{i}" for i in range(65)],
        ["x" * 257],
        ["é" * 129],
        ["a\x00b"],
        ["[not json"],
        ['["ok", 1]'],
        ['["a", ["nested"]]'],
        ['[""]'],
        [""],
    ],
)
async def test_bad_tags(conforming_client, v2_dirs, tag_parts):
    status, body = await post(conforming_client, basic(extra=[("tags", t) for t in tag_parts]))
    assert_error(status, body, 422)


@pytest.mark.asyncio
async def test_tag_limits_accepted(conforming_client, v2_dirs):
    status, body = await post(conforming_client, basic(extra=[("tags", json.dumps([f"t{i}" for i in range(63)])), ("tags", "x" * 256)]))
    assert status == 201, body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,raw",
    [
        ({"Content-Type": "application/json"}, b"{}"),
        ({"Content-Type": "multipart/form-data"}, b"--x\r\n\r\n--x--\r\n"),
        ({"Content-Type": "multipart/form-data; boundary=x"}, b"garbage without boundary"),
        ({"Content-Type": "multipart/form-data; boundary=x"}, b"--x\r\nContent-Disposition: form-data; name=\"file_path\"\r\n\r\na.png"),
        ({"Content-Type": "multipart/form-data; boundary=x"}, b""),
        ({"Content-Type": "multipart/form-data; boundary=x"}, b"--x\r\nContent-Dis\x17: form-data\r\n\r\nv\r\n--x--\r\n"),
        ({"Content-Type": "multipart/form-data; boundary=" + "x" * 71}, b""),
    ],
)
async def test_malformed_body(conforming_client, v2_dirs, headers, raw):
    resp = await conforming_client.post("/api/v2/assets", data=raw, headers=headers)
    body = await resp.json()
    check_body(resp.status, body)
    assert_error(resp.status, body, 422)
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
async def test_duplicate_fields_rejected(conforming_client, v2_dirs):
    fd = form(basic())
    fd.add_field("file", b"second", filename="g.bin")
    resp = await conforming_client.post("/api/v2/assets", data=fd)
    body = await resp.json()
    check_body(resp.status, body)
    assert_error(resp.status, body, 422)
    status, body = await post(conforming_client, [*basic(), ("file_path", "b.png")])
    assert_error(status, body, 422)
    assert stored(v2_dirs) == []


@pytest.mark.asyncio
async def test_oversized_text_field_rejected(conforming_client, v2_dirs):
    status, body = await post(conforming_client, basic(extra=[("unknown", "x" * 2_000_000), ("expires_in", "6" * 100_000)]))
    assert_error(status, body, 422)


@pytest.mark.asyncio
async def test_tag_flood_rejected_at_first_excess_tag(conforming_client, v2_dirs, monkeypatch):
    parsed = []
    real_parse = v2_assets._parse_tags
    monkeypatch.setattr(v2_assets, "_parse_tags", lambda raw: parsed.append(raw) or real_parse(raw))
    status, body = await post(conforming_client, basic(extra=[("tags", "t") for _ in range(5000)]))
    assert_error(status, body, 422)
    assert len(parsed) == 65
    assert leftovers(v2_dirs) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        [("unknown", "x" * 60_000) for _ in range(3)],
        [("tags", "[]") for _ in range(200)],
        [("unknown", "") for _ in range(200)],
        [("tags", json.dumps(["x" * 256] * 40)), ("tags", json.dumps(["y" * 256] * 20)), ("unknown", "z" * 60_000), ("unknown", "z" * 60_000)],
    ],
)
async def test_text_parts_budget(conforming_client, v2_dirs, extra):
    status, body = await post(conforming_client, basic(extra=extra))
    assert_error(status, body, 422)
    assert stored(v2_dirs) == []


def test_contained_root(v2_dirs, tmp_path):
    os.symlink(tmp_path / "elsewhere", v2_dirs / "input/out")
    assert v2_assets.contained_root(str(v2_dirs / "input/a.png")) == os.path.realpath(v2_dirs / "input")
    assert v2_assets.contained_root(str(v2_dirs / "models/checkpoints/x")) == os.path.realpath(v2_dirs / "models/checkpoints")
    assert v2_assets.contained_root(str(v2_dirs / "output/o.png")) == os.path.realpath(v2_dirs / "output")
    assert v2_assets.contained_root(str(v2_dirs / "input/out/a.png")) is None
    assert v2_assets.contained_root(str(v2_dirs / "input/../x")) is None
    assert v2_assets.contained_root("/etc/passwd") is None
