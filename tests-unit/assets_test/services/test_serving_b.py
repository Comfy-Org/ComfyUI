import os
from datetime import datetime
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from sqlalchemy import select, update

from app.assets.api import routes
from app.assets.api.routes import _build_asset_response
from app.assets.database.models import Asset
from app.assets.database.queries.records import (
    RecordPageSpec,
    create_content,
    create_record,
    get_record_by_id,
    list_records_page,
    mark_content_missing,
)
from app.assets.helpers import to_stored_hash
from app.assets.services.asset_management import (
    asset_exists,
    delete_asset_reference,
    get_asset_detail,
    resolve_asset_for_download,
    resolve_hash_to_path,
)
from app.assets.services.lookup import lookup_for_view
from app.assets.services.schemas import AssetData, AssetDetailResult, ReferenceData

_TS = datetime(2024, 1, 1, 0, 0, 0)


def test_missing_content_record_metadata_200_with_tag_content_404(
    mock_create_session, session, temp_dir
):
    content = create_content(session, path=str(temp_dir / "gone.bin"))
    record = create_record(session, content_id=content.id, name="gone.bin")
    mark_content_missing(session, content.id)
    session.commit()
    record_id = record.id

    detail = get_asset_detail(record_id)
    assert detail is not None
    assert "missing" in detail.tags

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_missing_record_is_listed_but_never_served(
    mock_create_session, session, temp_dir
):
    path = temp_dir / "listed_gone.bin"
    path.write_bytes(b"x")
    digest = to_stored_hash("a" * 64)
    content = create_content(session, path=str(path), hash=digest)
    record = create_record(session, content_id=content.id, name="listed_gone.bin")
    mark_content_missing(session, content.id)
    session.commit()
    record_id = record.id

    results, tag_map, total = list_records_page(session, RecordPageSpec())
    assert any(r.id == record_id for r in results)
    assert total == 1
    assert "missing" in tag_map.get(record_id, [])

    assert lookup_for_view(session, digest) is None
    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_deleted_record_metadata_and_content_404(mock_create_session, session):
    record_id = "00000000-0000-0000-0000-000000000000"
    assert get_asset_detail(record_id) is None
    with pytest.raises(ValueError, match="not found"):
        resolve_asset_for_download(record_id)


def test_content_404_when_file_absent_not_marked_missing(
    mock_create_session, session, temp_dir
):
    path = temp_dir / "absent.bin"
    path.write_bytes(b"x")
    content = create_content(session, path=str(path))
    record = create_record(session, content_id=content.id, name="absent.bin")
    session.commit()
    record_id = record.id
    path.unlink()

    detail = get_asset_detail(record_id)
    assert detail is not None
    assert "missing" not in detail.tags

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_content_fail_closed_no_sibling_fallback(mock_create_session, session, temp_dir):
    digest = "d" * 64
    missing_path = temp_dir / "missing.bin"
    live_path = temp_dir / "live.bin"
    live_path.write_bytes(b"live")

    missing_content = create_content(session, path=str(missing_path), hash=digest)
    live_content = create_content(session, path=str(live_path), hash=digest)
    record = create_record(session, content_id=missing_content.id, name="missing.bin")
    mark_content_missing(session, missing_content.id)
    session.commit()
    record_id = record.id

    assert live_content.is_missing is False
    assert os.path.isfile(live_path)

    with pytest.raises(FileNotFoundError):
        resolve_asset_for_download(record_id)


def test_resolve_hash_to_path_refuses_temp_content(mock_create_session, session, temp_dir):
    digest = "e" * 64
    f = temp_dir / "temp_only.bin"
    f.write_bytes(b"temp")
    create_content(session, path=str(f), hash=to_stored_hash(digest))
    session.commit()

    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = resolve_hash_to_path(f"blake3:{digest}")

    assert result is None


def test_resolve_hash_to_path_uses_newest_record_for_name_and_content_type(
    mock_create_session, session, temp_dir
):
    digest = "c" * 64
    f = temp_dir / "shared.bin"
    f.write_bytes(b"data")
    content = create_content(session, path=str(f), hash=to_stored_hash(digest), size_bytes=f.stat().st_size)
    older = create_record(
        session,
        content_id=content.id,
        name="older.txt",
        mime_type="text/plain",
    )
    newer = create_record(
        session,
        content_id=content.id,
        name="newer.png",
        mime_type="image/png",
    )
    older.created_at = _TS
    older.updated_at = _TS
    older.last_access_time = _TS
    newer.created_at = _TS.replace(microsecond=1)
    newer.updated_at = _TS.replace(microsecond=1)
    newer.last_access_time = _TS.replace(microsecond=1)
    session.commit()

    result = resolve_hash_to_path(f"blake3:{digest}")

    assert result is not None
    assert result.download_name == "newer.png"
    assert result.content_type == "image/png"


def test_resolve_hash_to_path_serves_content_with_zero_records(
    mock_create_session, session, temp_dir
):
    digest = "d" * 64
    f = temp_dir / "orphan.png"
    f.write_bytes(b"data")
    content = create_content(session, path=str(f), hash=to_stored_hash(digest), size_bytes=f.stat().st_size)
    record = create_record(
        session,
        content_id=content.id,
        name="renamed.txt",
        mime_type="text/plain",
    )
    session.commit()

    assert delete_asset_reference(record.id) is True
    session.expire_all()
    assert lookup_for_view(session, f"blake3:{digest}") is not None
    assert session.scalars(
        select(Asset).where(Asset.content_id == content.id)
    ).all() == []

    result = resolve_hash_to_path(f"blake3:{digest}")

    assert result is not None
    assert result.download_name == "orphan.png"
    assert result.content_type == "image/png"


def test_resolve_hash_to_path_unknown_hash(mock_create_session):
    assert resolve_hash_to_path("blake3:" + "f" * 64) is None


def test_content_read_updates_last_access_time(
    mock_create_session, session, temp_dir
):
    f = temp_dir / "read.bin"
    f.write_bytes(b"data")
    content = create_content(session, path=str(f))
    record = create_record(session, content_id=content.id, name="read.bin")
    session.commit()
    record_id = record.id

    session.execute(
        update(Asset).where(Asset.id == record_id).values(last_access_time=None)
    )
    session.commit()

    before_record = get_record_by_id(session, record_id)
    assert before_record is not None
    before = before_record.last_access_time
    assert before is None

    resolve_asset_for_download(record_id)

    session.expire_all()
    after_record = get_record_by_id(session, record_id)
    assert after_record is not None
    after = after_record.last_access_time
    assert after is not None


def test_view_hash_read_updates_last_access_time(
    mock_create_session, session, temp_dir
):
    digest = "a" * 64
    f = temp_dir / "view.bin"
    f.write_bytes(b"view")
    content = create_content(session, path=str(f), hash=to_stored_hash(digest), size_bytes=f.stat().st_size)
    record = create_record(session, content_id=content.id, name="view.bin")
    session.commit()
    record_id = record.id

    session.execute(
        update(Asset).where(Asset.id == record_id).values(last_access_time=None)
    )
    session.commit()

    result = resolve_hash_to_path(f"blake3:{digest}")
    assert result is not None

    session.expire_all()
    record_after = get_record_by_id(session, record_id)
    assert record_after is not None
    assert record_after.last_access_time is not None


def test_lookup_for_view_returns_none_for_temp_content(session, temp_dir):
    digest = "b" * 64
    f = temp_dir / "view_temp.bin"
    f.write_bytes(b"temp")
    create_content(session, path=str(f), hash=to_stored_hash(digest))
    session.commit()

    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        assert lookup_for_view(session, to_stored_hash(digest)) is None


def test_asset_exists_false_for_temp_content(mock_create_session, session, temp_dir):
    digest = "c" * 64
    f = temp_dir / "exists_temp.bin"
    f.write_bytes(b"temp")
    create_content(session, path=str(f), hash=to_stored_hash(digest))
    session.commit()

    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        assert asset_exists(f"blake3:{digest}") is False


@pytest.mark.asyncio
async def test_head_hash_route_404_for_temp_content(
    mock_create_session, session, temp_dir, monkeypatch
):
    digest = "d" * 64
    f = temp_dir / "head_temp.bin"
    f.write_bytes(b"temp")
    create_content(session, path=str(f), hash=to_stored_hash(digest))
    session.commit()

    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        response = await routes.head_asset_by_hash(
            make_mocked_request(
                "HEAD",
                f"/api/assets/hash/blake3:{digest}",
                match_info={"hash": f"blake3:{digest}"},
            )
        )

    assert isinstance(response, web.Response)
    assert response.status == 404


def test_resolve_hash_to_path_temp_does_not_bump_last_access_time(
    mock_create_session, session, temp_dir
):
    digest = "f" * 64
    f = temp_dir / "noaccess_temp.bin"
    f.write_bytes(b"temp")
    content = create_content(session, path=str(f), hash=to_stored_hash(digest))
    record = create_record(session, content_id=content.id, name="noaccess_temp.bin")
    session.commit()
    record_id = record.id

    session.execute(
        update(Asset).where(Asset.id == record_id).values(last_access_time=None)
    )
    session.commit()

    with patch("app.assets.services.lookup.is_temp_path", return_value=True):
        result = resolve_hash_to_path(f"blake3:{digest}")

    assert result is None
    session.expire_all()
    record_after = get_record_by_id(session, record_id)
    assert record_after is not None
    assert record_after.last_access_time is None


@pytest.fixture
def sandboxed_comfy_roots(tmp_path):
    with patch("app.assets.services.path_utils.folder_paths") as fp:
        fp.get_input_directory.return_value = str(tmp_path / "input")
        fp.get_output_directory.return_value = str(tmp_path / "output")
        fp.get_temp_directory.return_value = str(tmp_path / "temp")
        fp.models_dir = str(tmp_path / "models")
        yield tmp_path


def test_temp_asset_preview_url_still_resolves_type_temp(sandboxed_comfy_roots):
    name = "ComfyUI_temp_abcde_00001_.png"
    result = AssetDetailResult(
        ref=ReferenceData(
            id="ref-temp",
            name=name,
            file_path=str(sandboxed_comfy_roots / "temp" / name),
            loader_path=None,
            user_metadata=None,
            preview_id=None,
            created_at=_TS,
            updated_at=_TS,
            last_access_time=_TS,
        ),
        asset=AssetData(hash="blake3:abc", size_bytes=1024, mime_type="image/png"),
        tags=[],
    )

    resp = _build_asset_response(result, {})

    assert resp.preview_url == f"/api/view?type=temp&filename={name}"
