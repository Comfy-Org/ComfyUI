from __future__ import annotations

import os
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import requests
from aiohttp import web
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as SASession

import app.assets.mode as mode_module
import folder_paths
from app.assets.api import routes, schemas_in
from app.assets.database.models import AssetContent, Base
from app.assets.services.ingest import register_executed_output

from .helpers import trigger_sync_seed_assets


def _db_path(comfy_tmp_base_dir: Path, request: pytest.FixtureRequest) -> str:
    url = request.config.getoption("--db-url")
    if url and url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return str(comfy_tmp_base_dir / "assets-test.sqlite3")


def _query(db_path: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    last_err: Exception | None = None
    for _ in range(20):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                return list(con.execute(sql, params))
            finally:
                con.close()
        except sqlite3.OperationalError as e:
            last_err = e
            time.sleep(0.1)
    raise AssertionError(f"sqlite read failed: {last_err}")


def _content_hash_for_asset(db_path: str, asset_id: str) -> str | None:
    rows = _query(
        db_path,
        "SELECT c.hash FROM asset_contents c "
        "JOIN assets a ON a.content_id = c.id "
        "WHERE a.id = ?",
        (asset_id,),
    )
    assert rows, f"no content row found for asset {asset_id}"
    return rows[0][0]


def _content_id_for_asset(db_path: str, asset_id: str) -> str:
    rows = _query(db_path, "SELECT content_id FROM assets WHERE id = ?", (asset_id,))
    assert rows, f"no asset row found for {asset_id}"
    return rows[0][0]


def _is_stored_blake3_hash(value: str | None) -> bool:
    if not isinstance(value, str) or not value.startswith("blake3:"):
        return False
    digest = value[len("blake3:") :]
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())


def _upload_via_api(
    http: requests.Session, base: str, *, name: str, tags: list[str], data: bytes
) -> tuple[int, dict[str, Any]]:
    files = {"file": (name, data, "application/octet-stream")}
    form = {"tags": json.dumps(tags), "name": name, "user_metadata": json.dumps({})}
    r = http.post(base + "/api/assets", files=files, data=form, timeout=120)
    return r.status_code, r.json()


def _upload_via_image(
    http: requests.Session, base: str, *, name: str, data: bytes, upload_type: str = "input"
) -> tuple[int, dict[str, Any]]:
    files = {"image": (name, data, "image/png")}
    form = {"type": upload_type, "subfolder": ""}
    r = http.post(base + "/upload/image", files=files, data=form, timeout=120)
    return r.status_code, r.json()


def _unique_bytes(seed: str, size: int = 4096) -> bytes:
    return uuid.uuid4().bytes + seed.encode("utf-8").ljust(size, b"\0")[: size - 16]


@pytest.mark.asyncio
async def test_hash_only_multipart_upload_off_mode_returns_400(monkeypatch):
    hash_value = f"blake3:{'a' * 64}"
    parsed = schemas_in.ParsedUpload(
        file_present=False,
        file_written=0,
        file_client_name=None,
        tmp_path=None,
        tags_raw=[],
        provided_name=None,
        user_metadata_raw=None,
        provided_hash=hash_value,
        provided_hash_exists=True,
    )
    monkeypatch.setattr(mode_module, "hashing_enabled", lambda: False)
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(
        routes, "parse_multipart_upload", AsyncMock(return_value=parsed)
    )
    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "test-user"),
    )

    response = await routes.upload_asset(AsyncMock(spec=web.Request))

    assert isinstance(response, web.Response)
    assert response.status == 400
    response_body = response.body
    assert isinstance(response_body, bytes | bytearray)
    assert json.loads(response_body)["error"]["code"] == "FEATURE_DISABLED"


@pytest.fixture(scope="session")
def server_hashing_enabled(request: pytest.FixtureRequest) -> bool:
    markexpr = request.config.getoption("markexpr") or ""
    return bool(
        request.config.getoption("--enable-asset-hashing")
        or "hashing_on" in markexpr
        or any(item.get_closest_marker("hashing_on") for item in request.session.items)
    )


def test_upload_via_api_hashes_in_on_mode(http, api_base, comfy_tmp_base_dir, request):
    db_path = _db_path(comfy_tmp_base_dir, request)
    data = _unique_bytes("api-hashes")
    status, body = _upload_via_api(
        http, api_base, name="api_hashes.bin", tags=["output", "unit-tests"], data=data
    )
    assert status in (200, 201), body
    digest = _content_hash_for_asset(db_path, body["id"])
    assert _is_stored_blake3_hash(digest), f"expected a stored blake3 hash, got {digest!r}"


def test_upload_via_image_hashes_in_on_mode(http, api_base, comfy_tmp_base_dir, request):
    db_path = _db_path(comfy_tmp_base_dir, request)
    data = _unique_bytes("image-hashes")
    status, body = _upload_via_image(http, api_base, name="image_hashes.png", data=data)
    assert status == 200, body
    asset = body.get("asset")
    assert asset and asset.get("id"), f"/upload/image did not register an asset: {body}"
    digest = _content_hash_for_asset(db_path, asset["id"])
    assert _is_stored_blake3_hash(digest), f"expected a stored blake3 hash, got {digest!r}"


def test_repeat_upload_mints_a_new_record_in_on_mode(
    http, api_base, comfy_tmp_base_dir, request
):
    db_path = _db_path(comfy_tmp_base_dir, request)
    data = _unique_bytes("dedup-on-mode")
    status1, first = _upload_via_api(
        http, api_base, name="dedup_on.bin", tags=["output", "unit-tests"], data=data
    )
    status2, second = _upload_via_api(
        http, api_base, name="dedup_on.bin", tags=["output", "unit-tests"], data=data
    )
    assert status1 == 201, first
    assert status2 == 201, second
    assert second["id"] != first["id"], "every upload is its own delivery record"
    assert second.get("created_new") is True
    assert _content_id_for_asset(db_path, second["id"]) == _content_id_for_asset(
        db_path, first["id"]
    ), "same bytes stay one content row (hashes alone can collide across rows)"


def test_seeded_file_not_hashed_in_on_mode(
    http, api_base, comfy_tmp_base_dir, request, server_hashing_enabled
):
    if server_hashing_enabled:
        pytest.skip(
            "scanner hashing gate only observable with the hashing flag OFF; "
            "a full-suite run forces the shared server into hash mode"
        )
    db_path = _db_path(comfy_tmp_base_dir, request)
    ckpt_dir = comfy_tmp_base_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    disk_path = ckpt_dir / f"seeded_{uuid.uuid4().hex}.safetensors"
    disk_path.write_bytes(uuid.uuid4().bytes + b"S" * (4096 - 16))

    row_hash: str | None = None
    for _ in range(5):
        trigger_sync_seed_assets(http, api_base)
        rows = _query(
            db_path,
            "SELECT hash FROM asset_contents WHERE path = ? AND is_missing = 0",
            (str(disk_path),),
        )
        if rows:
            row_hash = rows[0][0]
            break
    else:
        raise AssertionError(f"seeded file never produced a content row: {disk_path}")

    assert row_hash is None, "scanner must not hash seeded files while the flag is off"


def test_output_not_hashed_in_on_mode(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    @contextmanager
    def _fake_create_session():
        with SASession(engine) as session:
            yield session

    monkeypatch.setattr(mode_module, "hashing_enabled", lambda: False)
    monkeypatch.setattr(
        "app.assets.services.ingest.create_session", _fake_create_session
    )

    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"render_{uuid.uuid4().hex}.png")
    with open(out_path, "wb") as fh:
        fh.write(b"fake-output-bytes")

    try:
        register_executed_output(out_path, job_id="job-out")
        with SASession(engine) as session:
            content = session.execute(
                select(AssetContent).where(
                    AssetContent.path == os.path.abspath(out_path)
                )
            ).scalar_one()
            assert content.hash is None, (
                "workflow output must not be hashed while the hashing flag is off"
            )
            assert content.is_missing is False
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
