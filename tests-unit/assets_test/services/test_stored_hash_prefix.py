from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from sqlalchemy import func, select
from sqlalchemy.orm import Session as SASession

from app.assets import mode
from app.assets.api import routes
import app.assets.mode as mode_module
import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_record
from app.assets.scanner import enrich_asset
from app.assets.scanner_changes import recover_missing_content
from app.assets.services import asset_management, ingest
from app.assets.services.asset_management import get_asset_detail
from app.assets.services.ingest import (
    HashMismatchError,
    upload_from_temp_path,
)
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture
def hashing_on():
    class FakeArgs:
        enable_asset_hashing = True

    class DisabledArgs:
        enable_asset_hashing = False

    mode_module.init(FakeArgs())
    yield
    mode_module.init(DisabledArgs())


def _write_temp(content: bytes) -> str:
    uploads_root = os.path.join(
        folder_paths.get_temp_directory(), "uploads", uuid.uuid4().hex
    )
    os.makedirs(uploads_root, exist_ok=True)
    path = os.path.join(uploads_root, ".upload.part")
    with open(path, "wb") as file:
        file.write(content)
    return path


def test_upload_stores_prefixed_hash_expected_hash_succeeds_and_reuses_content(
    mock_create_session, hashing_on
):
    content_bytes = b"prefixed-stored-upload-bytes"
    temp1 = _write_temp(content_bytes)
    snapshot = snapshot_hash(temp1)
    assert snapshot is not None
    digest, _ = snapshot
    expected = f"blake3:{digest}"
    assert expected.startswith("blake3:")

    temp2 = _write_temp(content_bytes)
    try:
        r1 = upload_from_temp_path(
            temp_path=temp1,
            name="pref.bin",
            tags=["output"],
            client_filename="pref.bin",
            expected_hash=expected,
        )
        assert r1.created_new is True

        with mock_create_session() as session:
            live = list(
                session.scalars(
                    select(AssetContent).where(AssetContent.is_missing.is_(False))
                )
            )
            assert len(live) == 1
            assert live[0].hash == expected
            assert live[0].hash.startswith("blake3:")

        assert r1.asset.hash == expected

        r2 = upload_from_temp_path(
            temp_path=temp2,
            name="pref.bin",
            tags=["output"],
            client_filename="pref.bin",
            expected_hash=expected,
        )
        assert r2.created_new is True
        assert r2.ref.id != r1.ref.id, "every upload is its own delivery record"
        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(AssetContent)) == 1
            assert session.scalar(select(func.count()).select_from(Asset)) == 2
    finally:
        for path in (temp1, temp2):
            if os.path.exists(path):
                os.unlink(path)


def test_upload_expected_hash_mismatch_still_rejected(mock_create_session, hashing_on):
    temp = _write_temp(b"mismatch-bytes")
    try:
        with pytest.raises(HashMismatchError):
            upload_from_temp_path(
                temp_path=temp,
                name="mismatch.bin",
                tags=["output"],
                client_filename="mismatch.bin",
                expected_hash=f"blake3:{'0' * 64}",
            )
        with mock_create_session() as session:
            assert session.scalar(select(func.count()).select_from(Asset)) == 0
            assert session.scalar(select(func.count()).select_from(AssetContent)) == 0
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def test_enrich_fills_deferred_hash_prefixed(session, temp_dir):
    path = temp_dir / "deferred.bin"
    path.write_bytes(b"deferred output bytes")
    stat = path.stat()

    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
    session.add(content)
    session.flush()
    record = create_record(session, content.id, path.name)
    session.commit()

    enriched = enrich_asset(
        session,
        file_path=str(path),
        content_id=content.id,
        record_id=record.id,
        extract_metadata=False,
        compute_hash=True,
    )

    assert enriched is True
    stored = session.get(AssetContent, content.id).hash
    assert stored is not None
    assert stored.startswith("blake3:")
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    assert stored == f"blake3:{digest}"


def test_recovery_matches_prefixed_stored_hash(session, temp_dir):
    path = temp_dir / "recover.bin"
    original_bytes = b"recover-me-bytes"
    path.write_bytes(original_bytes)
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    stored = f"blake3:{digest}"
    assert stored.startswith("blake3:")

    content = AssetContent(
        path=str(path),
        hash=stored,
        is_missing=True,
        size_bytes=0,
        mtime_ns=None,
    )
    session.add(content)
    session.flush()
    create_record(session, content.id, path.name)
    session.commit()

    path.unlink()
    path.write_bytes(original_bytes)
    stat = os.stat(str(path))
    result = recover_missing_content(
        session, str(path), stat, hashing_is_enabled=True
    )

    assert result == "recovered"
    recovered = session.get(AssetContent, content.id)
    assert recovered is not None
    assert recovered.is_missing is False
    assert recovered.hash == stored


@pytest.mark.asyncio
async def test_all_read_surfaces_agree_on_prefixed_hash(
    db_engine, monkeypatch, hashing_on
):
    @contextmanager
    def _factory():
        with SASession(db_engine) as sess:
            yield sess

    monkeypatch.setattr(routes, "create_session", lambda: SASession(db_engine))
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(mode, "hashing_enabled", lambda: True)
    monkeypatch.setattr(ingest, "create_session", _factory)
    monkeypatch.setattr(asset_management, "create_session", _factory)

    content_bytes = b"one-asset-all-surfaces-agree"
    temp = _write_temp(content_bytes)
    snapshot = snapshot_hash(temp)
    assert snapshot is not None
    digest, _ = snapshot
    expected = f"blake3:{digest}"

    try:
        upload_result = ingest.upload_from_temp_path(
            temp_path=temp,
            name="surf.bin",
            tags=["output"],
            client_filename="surf.bin",
        )
        asset_id = upload_result.ref.id

        assert upload_result.asset is not None
        assert upload_result.asset.hash == expected
        upload_resp = routes._build_asset_response(upload_result, {})
        assert upload_resp.hash == expected

        detail = get_asset_detail(asset_id)
        assert detail is not None
        assert detail.asset is not None
        assert detail.asset.hash == expected

        get_resp = await routes.get_asset_route(
            make_mocked_request(
                "GET", f"/api/assets/{asset_id}", match_info={"id": asset_id}
            )
        )
        assert isinstance(get_resp, web.Response)
        assert isinstance(get_resp.body, bytes | bytearray)
        get_body = json.loads(get_resp.body)
        assert get_body["hash"] == expected

        list_resp = await routes.list_assets_route(
            make_mocked_request("GET", "/api/assets")
        )
        assert isinstance(list_resp, web.Response)
        assert isinstance(list_resp.body, bytes | bytearray)
        list_body = json.loads(list_resp.body)
        item = next(a for a in list_body["assets"] if a["id"] == asset_id)
        assert item["hash"] == expected

        head_resp = await routes.head_asset_by_hash(
            make_mocked_request(
                "HEAD",
                f"/api/assets/hash/{expected}",
                match_info={"hash": expected},
            )
        )
        assert isinstance(head_resp, web.Response)
        assert head_resp.status == 200

        fh_req = AsyncMock(spec=web.Request)
        fh_req.json.return_value = {
            "hash": expected,
            "name": "fromhash.bin",
            "tags": ["output"],
        }
        fh_resp = await routes.create_asset_from_hash_route(fh_req)
        assert isinstance(fh_resp, web.Response)
        assert fh_resp.status == 201
        assert isinstance(fh_resp.body, bytes | bytearray)
        fh_body = json.loads(fh_resp.body)
        assert fh_body["hash"] == expected
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
