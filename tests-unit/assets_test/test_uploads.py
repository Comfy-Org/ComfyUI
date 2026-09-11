import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import pytest

from app.assets.api.schemas_in import UploadAssetSpec
from app.assets.api.schemas_out import Asset, AssetCreated
from helpers import get_asset_filename


def test_asset_created_inherits_hash_field():
    """AssetCreated must inherit `hash` from Asset so POST /api/assets responses emit it.

    Schema-level guard: integration tests cover the wire shape, but inheritance
    drift (e.g. AssetCreated ever being redefined to no longer extend Asset)
    would silently drop `hash` from a major endpoint without this check.
    """
    assert "hash" in Asset.model_fields
    assert "hash" in AssetCreated.model_fields
    assert AssetCreated.model_fields["hash"].annotation == Asset.model_fields["hash"].annotation


def test_upload_asset_spec_ignores_subfolder_field():
    spec = UploadAssetSpec.model_validate(
        {"tags": ["input"], "subfolder": "pasted", "name": "image.png"}
    )

    assert "subfolder" not in UploadAssetSpec.model_fields
    assert not hasattr(spec, "subfolder")


def test_upload_ok_duplicate_reference(http: requests.Session, api_base: str, make_asset_bytes):
    name = "dup_a.safetensors"
    tags = ["models", "model_type:checkpoints", "unit-tests", "alpha"]
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
    assert a2.get("loader_path") is None
    assert a2.get("display_name") is None

    # Third upload with the same data but different name also creates new AssetReference
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name + "_d", "user_metadata": json.dumps(meta)}
    r3 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    a3 = r3.json()
    assert r3.status_code in (200, 201), a3
    assert a3["asset_hash"] == a1["asset_hash"]
    assert a3["id"] != a1["id"]
    assert a3["id"] != a2["id"]
    assert a3.get("loader_path") is None
    assert a3.get("display_name") is None


def test_upload_fastpath_from_existing_hash_no_file(http: requests.Session, api_base: str):
    # Seed a small file first
    name = "fastpath_seed.safetensors"
    tags = ["input", "unit-tests"]
    meta = {}
    files = {"file": (name, b"B" * 1024, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps(meta)}
    r1 = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 201, b1
    h = b1["asset_hash"]
    assert b1["hash"] == h

    # Now POST /api/assets with only hash and no file
    hash_only_tags = ["models", "checkpoints", "unit-tests", "hash-labels"]
    files = [
        ("hash", (None, h)),
        ("tags", (None, json.dumps(hash_only_tags))),
        ("name", (None, "fastpath_copy.safetensors")),
        ("user_metadata", (None, json.dumps({"purpose": "copy"}))),
    ]
    r2 = http.post(api_base + "/api/assets", files=files, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 200, b2  # fast path returns 200 with created_new == False
    assert b2["created_new"] is False
    assert b2["asset_hash"] == h
    assert b2["hash"] == h
    assert "models" in b2["tags"]
    assert "checkpoints" in b2["tags"]
    assert "uploaded" not in b2["tags"]
    assert not any(tag.startswith("model_type:") for tag in b2["tags"])
    assert b2.get("loader_path") is None
    assert b2.get("display_name") is None

    rg = http.get(f"{api_base}/api/assets/{b2['id']}", timeout=120)
    detail = rg.json()
    assert rg.status_code == 200, detail
    assert detail.get("loader_path") is None
    assert detail.get("display_name") is None


def test_create_from_hash_with_model_tags_does_not_synthesize_loader_path(
    http: requests.Session, api_base: str
):
    seed_name = "from_hash_seed.safetensors"
    seed_tags = ["models", "model_type:checkpoints", "unit-tests"]
    files = {"file": (seed_name, b"D" * 1024, "application/octet-stream")}
    form = {
        "tags": json.dumps(seed_tags),
        "name": seed_name,
        "user_metadata": json.dumps({}),
    }
    seed_r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    seed = seed_r.json()
    assert seed_r.status_code == 201, seed

    payload = {
        "hash": seed["asset_hash"],
        "name": "from_hash_copy.safetensors",
        "tags": ["models", "model_type:checkpoints", "unit-tests", "spoofed"],
    }
    created_r = http.post(api_base + "/api/assets/from-hash", json=payload, timeout=120)
    created = created_r.json()
    assert created_r.status_code == 201, created
    assert created["created_new"] is False
    assert created["asset_hash"] == seed["asset_hash"]
    assert created.get("loader_path") is None
    assert created.get("display_name") is None

    detail_r = http.get(f"{api_base}/api/assets/{created['id']}", timeout=120)
    detail = detail_r.json()
    assert detail_r.status_code == 200, detail
    assert detail.get("loader_path") is None
    assert detail.get("display_name") is None


def test_upload_fastpath_with_known_hash_and_file(
    http: requests.Session, api_base: str
):
    # Seed
    files = {"file": ("seed.safetensors", b"C" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "model_type:checkpoints", "unit-tests", "fp"]), "name": "seed.safetensors", "user_metadata": json.dumps({})}
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
    assert "checkpoints" in b2["tags"]
    assert "uploaded" not in b2["tags"]
    assert not any(tag == "model_type:checkpoints" for tag in b2["tags"])


def test_duplicate_byte_upload_is_reference_only_and_does_not_need_destination(
    http: requests.Session, api_base: str
):
    data = b"duplicate-reference-only" * 64
    seed_files = {"file": ("duplicate-seed.bin", data, "application/octet-stream")}
    seed_form = {
        "tags": json.dumps(["input", "unit-tests", "duplicate-seed"]),
        "name": "duplicate-seed.bin",
        "user_metadata": json.dumps({}),
    }
    seed_response = http.post(api_base + "/api/assets", data=seed_form, files=seed_files, timeout=120)
    seed = seed_response.json()
    assert seed_response.status_code == 201, seed

    duplicate_files = {"file": ("duplicate-copy.bin", data, "application/octet-stream")}
    duplicate_form = {
        "tags": json.dumps(["not-a-destination", "unit-tests", "duplicate-copy"]),
        "name": "duplicate-copy.bin",
        "user_metadata": json.dumps({}),
    }
    duplicate_response = http.post(
        api_base + "/api/assets", data=duplicate_form, files=duplicate_files, timeout=120
    )
    duplicate = duplicate_response.json()

    assert duplicate_response.status_code == 200, duplicate
    assert duplicate["created_new"] is False
    assert duplicate["asset_hash"] == seed["asset_hash"]
    assert "not-a-destination" in duplicate["tags"]
    assert "uploaded" not in duplicate["tags"]
    assert "input" not in duplicate["tags"]
    assert duplicate.get("loader_path") is None
    assert duplicate.get("display_name") is None


def test_upload_multiple_tags_fields_are_merged(http: requests.Session, api_base: str):
    data = [
        ("tags", "models,model_type:checkpoints"),
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
    assert {"models", "model_type:checkpoints", "unit-tests", "alpha"}.issubset(tags)


@pytest.mark.parametrize(
    (
        "tags",
        "extension",
        "expected_display_prefix",
    ),
    [
        (["input", "unit-tests"], ".png", ""),
        (
            ["models", "model_type:checkpoints", "unit-tests"],
            ".safetensors",
            "checkpoints/",
        ),
    ],
)
def test_upload_response_includes_loader_path_and_display_name(
    tags: list[str],
    extension: str,
    expected_display_prefix: str,
    http: requests.Session,
    api_base: str,
    make_asset_bytes,
):
    scope = f"response-paths-{uuid.uuid4().hex[:6]}"
    scoped_tags = [*tags, scope]
    name = f"asset_response_path{extension}"

    files = {"file": (name, make_asset_bytes(name, 1024), "application/octet-stream")}
    form = {
        "tags": json.dumps(scoped_tags),
        "name": name,
        "user_metadata": json.dumps({}),
    }
    created_r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    created = created_r.json()
    assert created_r.status_code in (200, 201), created
    stored_filename = get_asset_filename(created["asset_hash"], extension)
    expected_suffix = stored_filename
    expected_display_name = f"{expected_display_prefix}{expected_suffix}"
    # In-root loader path: model category dropped, no subfolders here -> just the filename.
    expected_loader_path = expected_suffix

    assert created["loader_path"] == expected_loader_path
    assert created["display_name"] == expected_display_name
    assert "logical_path" not in created

    detail_r = http.get(f"{api_base}/api/assets/{created['id']}", timeout=120)
    detail = detail_r.json()
    assert detail_r.status_code == 200, detail
    assert detail["loader_path"] == expected_loader_path
    assert detail["display_name"] == expected_display_name

    list_r = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "limit": "50"},
        timeout=120,
    )
    listed = list_r.json()
    assert list_r.status_code == 200, listed
    match = next(a for a in listed["assets"] if a["id"] == created["id"])
    assert match["loader_path"] == expected_loader_path
    assert match["display_name"] == expected_display_name


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


def test_create_from_hash_accepts_arbitrary_system_looking_tags(
    http: requests.Session, api_base: str
):
    files = {"file": ("hash-seed.bin", b"hash-seed" * 64, "application/octet-stream")}
    form = {
        "tags": json.dumps(["input", "unit-tests", "hash-seed"]),
        "name": "hash-seed.bin",
        "user_metadata": json.dumps({}),
    }
    seed_response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    seed = seed_response.json()
    assert seed_response.status_code == 201, seed

    response = http.post(
        api_base + "/api/assets/from-hash",
        json={
            "hash": seed["asset_hash"],
            "name": "hash-copy.bin",
            "tags": [
                "models",
                "model:true",
                "models:foo",
                "temporary:true",
                "unit-tests",
                "hash-copy",
            ],
        },
        timeout=120,
    )
    body = response.json()

    assert response.status_code == 201, body
    assert "models" in body["tags"]
    assert "model:true" in body["tags"]
    assert "models:foo" in body["tags"]
    assert "temporary:true" in body["tags"]
    assert "uploaded" not in body["tags"]


def test_upload_zero_byte_rejected(http: requests.Session, api_base: str):
    files = {"file": ("empty.safetensors", b"", "application/octet-stream")}
    form = {"tags": json.dumps(["models", "model_type:checkpoints", "unit-tests", "edge"]), "name": "empty.safetensors", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "EMPTY_UPLOAD"


def test_upload_rejects_arbitrary_labels_without_required_destination_role(http: requests.Session, api_base: str):
    files = {"file": ("badroot.bin", b"A" * 64, "application/octet-stream")}
    form = {"tags": json.dumps(["not-a-root", "whatever"]), "name": "badroot.bin", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_user_metadata_must_be_json(http: requests.Session, api_base: str):
    files = {"file": ("badmeta.bin", b"A" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "model_type:checkpoints", "unit-tests", "edge"]), "name": "badmeta.bin", "user_metadata": "{not json}"}
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
        ("tags", (None, json.dumps(["models", "model_type:checkpoints", "unit-tests"]))),
        ("name", (None, "x.safetensors")),
    ]
    r = http.post(api_base + "/api/assets", files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "MISSING_FILE"


def test_upload_models_unknown_model_type(http: requests.Session, api_base: str):
    files = {"file": ("m.safetensors", b"A" * 128, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "model_type:no_such_category", "unit-tests"]), "name": "m.safetensors"}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400, body
    assert body["error"]["code"] == "INVALID_BODY"


@pytest.mark.parametrize("model_type", ["configs", "custom_nodes"])
def test_upload_models_rejects_non_model_registered_folder(
    model_type: str, http: requests.Session, api_base: str
):
    files = {"file": ("not-a-model.py", b"A" * 128, "application/octet-stream")}
    form = {
        "tags": json.dumps(["models", f"model_type:{model_type}", "unit-tests"]),
        "name": "not-a-model.py",
    }

    response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 400, body
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_models_requires_model_type(http: requests.Session, api_base: str):
    files = {"file": ("nocat.safetensors", b"A" * 64, "application/octet-stream")}
    form = {"tags": json.dumps(["models"]), "name": "nocat.safetensors", "user_metadata": json.dumps({})}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_upload_extra_tags_are_labels_not_path_components(http: requests.Session, api_base: str):
    files = {"file": ("evil.safetensors", b"A" * 256, "application/octet-stream")}
    form = {"tags": json.dumps(["models", "model_type:checkpoints", "unit-tests", "..", "zzz"]), "name": "evil.safetensors"}
    r = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = r.json()
    assert r.status_code == 201, body
    assert ".." in body["tags"]
    assert "zzz" in body["tags"]
    assert "models" in body["tags"]
    assert "model_type:checkpoints" in body["tags"]


@pytest.mark.parametrize(
    ("subfolder", "expected_tag", "unexpected_tags"),
    [
        ("custom/session", None, {"custom", "session"}),
        ("pasted", "pasted", set()),
    ],
)
def test_upload_image_accepts_arbitrary_subfolder_but_only_known_values_become_tags(
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
    subfolder: str,
    expected_tag: str | None,
    unexpected_tags: set[str],
):
    name = f"upload-image-{uuid.uuid4().hex}.png"
    files = {"image": (name, b"image-upload" * 64, "image/png")}
    form = {"type": "input", "subfolder": subfolder}

    response = http.post(api_base + "/upload/image", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 200, body
    assert body["subfolder"] == subfolder
    assert (comfy_tmp_base_dir / "input" / subfolder / body["name"]).exists()

    asset = body["asset"]
    tags = set(asset["tags"])
    assert "input" in tags
    assert "uploaded" in tags
    if expected_tag:
        assert expected_tag in tags
    assert tags.isdisjoint(unexpected_tags)


@pytest.mark.parametrize("route", ["/upload/image", "/upload/mask"])
@pytest.mark.parametrize("upload_type", ["foo", "", "   "])
def test_upload_rejects_unknown_type(
    http: requests.Session,
    api_base: str,
    route: str,
    upload_type: str,
):
    files = {"image": ("unknown-type.png", b"image-upload" * 64, "image/png")}
    form = {"type": upload_type}

    response = http.post(api_base + route, data=form, files=files, timeout=120)

    assert response.status_code == 400


def test_multipart_upload_accepts_system_looking_extra_labels(
    http: requests.Session, api_base: str
):
    files = {"file": ("relaxed-labels.bin", b"relaxed" * 64, "application/octet-stream")}
    form = {
        "tags": json.dumps(
            [
                "input",
                "unit-tests",
                "model:true",
                "models:foo",
                "temporary",
                "uploaded:true",
            ]
        ),
        "name": "relaxed-labels.bin",
        "user_metadata": json.dumps({}),
    }
    response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 201, body
    assert "input" in body["tags"]
    assert "model:true" in body["tags"]
    assert "models:foo" in body["tags"]
    assert "temporary" in body["tags"]
    assert "uploaded:true" in body["tags"]


def test_multipart_upload_rejects_ambiguous_destination_roles(
    http: requests.Session, api_base: str
):
    files = {"file": ("ambiguous.bin", b"ambiguous" * 64, "application/octet-stream")}
    form = {
        "tags": json.dumps(["input", "output", "unit-tests"]),
        "name": "ambiguous.bin",
        "user_metadata": json.dumps({}),
    }
    response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 400, body
    assert body["error"]["code"] == "INVALID_BODY"


def test_multipart_upload_rejects_multiple_model_types_for_models_destination(
    http: requests.Session, api_base: str
):
    files = {"file": ("ambiguous-model.safetensors", b"ambiguous-model" * 64, "application/octet-stream")}
    form = {
        "tags": json.dumps(
            ["models", "model_type:checkpoints", "model_type:loras", "unit-tests"]
        ),
        "name": "ambiguous-model.safetensors",
        "user_metadata": json.dumps({}),
    }
    response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 400, body
    assert body["error"]["code"] == "INVALID_BODY"


@pytest.mark.parametrize(
    ("tags", "expected_root", "extension"),
    [
        (["input", "unit-tests", "upload-location-input"], "input", ".bin"),
        (["output", "unit-tests", "upload-location-output"], "output", ".bin"),
        (
            ["models", "model_type:checkpoints", "unit-tests", "upload-location-model"],
            "models/checkpoints",
            ".safetensors",
        ),
    ],
)
def test_multipart_upload_role_selects_write_location(
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
    tags: list[str],
    expected_root: str,
    extension: str,
):
    role = next(tag for tag in tags if tag in {"input", "models", "output"})
    name = f"{role}-role-upload{extension}"
    files = {"file": (name, f"{role}-role-bytes".encode() * 64, "application/octet-stream")}
    form = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps({}),
    }

    response = http.post(api_base + "/api/assets", data=form, files=files, timeout=120)
    body = response.json()

    assert response.status_code == 201, body
    stored_name = get_asset_filename(body["asset_hash"], extension)
    expected_disk_path = comfy_tmp_base_dir / expected_root / stored_name
    assert expected_disk_path.exists()


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
