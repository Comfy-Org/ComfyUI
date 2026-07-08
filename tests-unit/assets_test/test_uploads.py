import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import pytest

from app.assets.api.schemas_out import Asset, AssetCreated


def test_asset_created_inherits_hash_field():
    """AssetCreated must inherit `hash` from Asset so POST /api/assets responses emit it.

    Schema-level guard: integration tests cover the wire shape, but inheritance
    drift (e.g. AssetCreated ever being redefined to no longer extend Asset)
    would silently drop `hash` from a major endpoint without this check.
    """
    assert "hash" in Asset.model_fields
    assert "hash" in AssetCreated.model_fields
    assert AssetCreated.model_fields["hash"].annotation == Asset.model_fields["hash"].annotation


def test_upload_ok_duplicate_reference(http: requests.Session, api_base: str, make_asset_bytes):
    name = "dup_a.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "alpha"]
    meta = {"purpose": "dup"}
    data = make_asset_bytes(name)
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps(meta)}
    r1 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    a1 = r1.json()
    assert r1.status_code == 201, a1
    assert a1["created_new"] is True
    assert a1["hash"] == a1["asset_hash"]

    # Second upload with the same data and name creates a new AssetReference (duplicates allowed)
    # Returns 200 because Asset already exists, but a new AssetReference is created
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps(meta)}
    r2 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    a2 = r2.json()
    assert r2.status_code in (200, 201), a2
    assert a2["asset_hash"] == a1["asset_hash"]
    assert a2["hash"] == a1["hash"]
    assert a2["id"] != a1["id"]  # new reference with same content

    # Third upload with the same data but different name also creates new AssetReference
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name + "_d", "user_metadata": json.dumps(meta)}
    r3 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    a3 = r3.json()
    assert r3.status_code in (200, 201), a3
    assert a3["asset_hash"] == a1["asset_hash"]
    assert a3["id"] != a1["id"]
    assert a3["id"] != a2["id"]


def test_upload_fastpath_from_existing_hash_no_file(http: requests.Session, api_base: str):
    # Seed a small file first
    name = "fastpath_seed.safetensors"
    tags = ["models", "checkpoints", "unit-tests"]
    meta = {}
    files = {"file": (name, b"B" * 1024, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps(meta)}
    r1 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 201, b1
    h = b1["asset_hash"]
    assert b1["hash"] == h

    # Now POST /api/assets with only hash and no file
    files = [
        ("hash", (None, h)),
        ("tags", (None, json.dumps(tags))),
        ("name", (None, "fastpath_copy.safetensors")),
        ("user_metadata", (None, json.dumps({"purpose": "copy"}))),
    ]
    r2 = http.post(api_base + "/api/assets", files=files, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200, b2  # fast path returns 200 with created_new == False
    assert b2["created_new"] is False
    assert b2["asset_hash"] == h
    assert b2["hash"] == h


def test_upload_fastpath_with_known_hash_and_file(
    http: requests.Session, api_base: str
):
    # Seed
    files = {"file": ("seed.safetensors", b"C" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "checkpoints", "unit-tests", "fp"]), "name": "seed.safetensors", "user_metadata": json.dumps({})}
    r1 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 201, b1
    h = b1["asset_hash"]
    assert b1["hash"] == h

    # Send both file and hash of existing content -> server must drain file and create from hash (200)
    files = {"file": ("ignored.bin", b"ignored" * 10, "application/octet-stream")}
    form = {"hash": h, "tags": json.dumps(["models", "checkpoints", "unit-tests", "fp"]), "name": "copy_from_hash.safetensors", "user_metadata": json.dumps({})}
    r2 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200, b2
    assert b2["created_new"] is False
    assert b2["asset_hash"] == h
    assert b2["hash"] == h


def test_upload_multiple_tags_fields_are_merged(http: requests.Session, api_base: str):
    data = [
        ("tags", "models,checkpoints"),
        ("tags", json.dumps(["unit-tests", "alpha"])),
        ("name", "merge.safetensors"),
        ("user_metadata", json.dumps({"u": 1})),
    ]
    files = {"file": ("merge.safetensors", b"B" * 256, "application/octet-stream")}
    r1 = http.post(api_base + "/api/assets", data=data, files=files, timeout=120)
    created = r1.json()
    assert r1.status_code in (200, 201), created
    aid = created["id"]

    # Verify all tags are present on the resource
    rg = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    detail = rg.json()
    assert rg.status_code == 200, detail
    tags = set(detail["tags"])
    assert {"models", "checkpoints", "unit-tests", "alpha"}.issubset(tags)


@pytest.mark.parametrize("root", ["input", "output"])
def test_concurrent_upload_identical_bytes_different_names(
    root: str,
    http: requests.Session,
    api_base: str,
    make_asset_bytes,
):
    """
    Two concurrent uploads of identical bytes but different names.
    Expect a single Asset (same hash), two AssetReference rows, and exactly one created_new=True.
    """
    scope = f"concupload-{uuid.uuid4().hex[:6]}"
    name1, name2 = "cu_a.bin", "cu_b.bin"
    data = make_asset_bytes("concurrent", 4096)
    tags = [root, "unit-tests", scope]

    def _do_upload(args):
        url, form_data, files_data = args
        with requests.Session() as s:
            return s.post(url, data=form_data, files=files_data, timeout=120)

    url = api_base + "/api/assets"
    form1 = {"tags": json.dumps(tags), "name": name1, "user_metadata": json.dumps({})}
    files1 = {"file": (name1, data, "application/octet-stream")}
    form2 = {"tags": json.dumps(tags), "name": name2, "user_metadata": json.dumps({})}
    files2 = {"file": (name2, data, "application/octet-stream")}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = list(executor.map(_do_upload, [(url, form1, files1), (url, form2, files2)]))
    r1, r2 = futures

    b1, b2 = r1.json(), r2.json()
    assert r1.status_code in (200, 201), b1
    assert r2.status_code in (200, 201), b2
    assert b1["asset_hash"] == b2["asset_hash"]
    assert b1["hash"] == b2["hash"]
    assert b1["hash"] == b1["asset_hash"]
    assert b1["id"] != b2["id"]

    created_flags = sorted([bool(b1.get("created_new")), bool(b2.get("created_new"))])
    assert created_flags == [False, True]

    rl = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "sort": "name"},
        timeout=120,
    )
    bl = rl.json()
    assert rl.status_code == 200, bl
    names = [a["name"] for a in bl.get("assets", [])]
    assert set([name1, name2]).issubset(names)


def test_create_from_hash_endpoint_404(http: requests.Session, api_base: str):
    payload = {
        "hash": "blake3:" + "0" * 64,
        "name": "nonexistent.bin",
        "tags": ["models", "checkpoints", "unit-tests"],
    }
    r = http.post(api_base + "/api/assets/from-hash", json=payload, timeout=120)
    body = r.json()
    assert r.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"


def test_upload_zero_byte_rejected(http: requests.Session, api_base: str):
    files = {"file": ("empty.safetensors", b"", "application/octet-stream")}
    form = {"tags": json.dumps(["models", "checkpoints", "unit-tests", "edge"]), "name": "empty.safetensors", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "EMPTY_UPLOAD"


def test_upload_invalid_root_tag_rejected(http: requests.Session, api_base: str):
    files = {"file": ("badroot.bin", b"A" * 64, "application/octet-stream")}
    form = {"tags": json.dumps(["not-a-root", "whatever"]), "name": "badroot.bin", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_user_metadata_must_be_json(http: requests.Session, api_base: str):
    files = {"file": ("badmeta.bin", b"A" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "checkpoints", "unit-tests", "edge"]), "name": "badmeta.bin", "user_metadata": "{not json}"}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_requires_multipart(http: requests.Session, api_base: str):
    r = http.post(api_base + "/api/assets", json={"foo": "bar"}, timeout=120)
    body = r.json()
    assert r.status_code == 415
    assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_missing_file_and_hash(http: requests.Session, api_base: str):
    files = [
        ("tags", (None, json.dumps(["models", "checkpoints", "unit-tests"]))),
        ("name", (None, "x.safetensors")),
    ]
    r = http.post(api_base + "/api/assets", files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "MISSING_FILE"


def test_upload_models_unknown_category(http: requests.Session, api_base: str):
    files = {"file": ("m.safetensors", b"A" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "no_such_category", "unit-tests"]), "name": "m.safetensors"}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"
    assert body["error"]["message"].startswith("unknown models category")


def test_upload_models_requires_category(http: requests.Session, api_base: str):
    files = {"file": ("nocat.safetensors", b"A" * 64, "application/octet-stream")}
    form = {"tags": json.dumps(["models"]), "name": "nocat.safetensors", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_tags_traversal_guard(http: requests.Session, api_base: str):
    files = {"file": ("evil.safetensors", b"A" * 256, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "checkpoints", "unit-tests", "..", "zzz"]), "name": "evil.safetensors"}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] in ("BAD_REQUEST", "INVALID_BODY")


def test_upload_empty_tags_rejected(http: requests.Session, api_base: str):
    files = {"file": ("notags.bin", b"A" * 64, "application/octet-stream")}
    form = {"tags": json.dumps([]), "name": "notags.bin", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


@pytest.mark.parametrize("root", ["input", "output"])
def test_duplicate_upload_same_display_name_does_not_clobber(
    root: str,
    http: requests.Session,
    api_base: str,
    asset_factory,
    make_asset_bytes,
):
    """
    Two uploads use the same tags and the same display name but different bytes.
    With hash-based filenames, they must NOT overwrite each other. Both assets
    remain accessible and serve their original content.
    """
    scope = f"dup-path-{uuid.uuid4().hex[:6]}"
    display_name = "same_display.bin"

    d1 = make_asset_bytes(scope + "-v1", 1536)
    d2 = make_asset_bytes(scope + "-v2", 2048)
    tags = [root, "unit-tests", scope]

    first = asset_factory(display_name, tags, {}, d1)
    second = asset_factory(display_name, tags, {}, d2)

    assert first["id"] != second["id"]
    assert first["asset_hash"] != second["asset_hash"]  # different content
    assert first["name"] == second["name"] == display_name

    # Both must be independently retrievable
    r1 = http.get(f"{api_base}/api/assets/{first['id']}/content", timeout=120)
    b1 = r1.content
    assert r1.status_code == 200
    assert b1 == d1
    r2 = http.get(f"{api_base}/api/assets/{second['id']}/content", timeout=120)
    b2 = r2.content
    assert r2.status_code == 200
    assert b2 == d2
