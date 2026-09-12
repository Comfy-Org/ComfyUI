from unittest.mock import patch

from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, mark_content_missing
from app.assets.helpers import to_stored_hash
from app.assets.services.ingest import create_from_hash
from app.assets.services.lookup import (
    claim_qualified_content as _real_claim_qualified_content,
    refresh_qualified_content as _real_refresh_qualified_content,
)


def test_create_from_hash_with_prefixed_hash_finds_existing_content(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "a" * 64
    path = temp_dir / "existing.bin"
    path.write_bytes(b"existing bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)

    with mock_create_session() as session:
        content = create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
        content_id = content.id
        session.commit()

    result = create_from_hash(f"blake3:{digest}", "derived.bin")

    assert result is not None
    with mock_create_session() as session:
        records = list(session.scalars(select(Asset)))
        created_record = session.get(Asset, result.ref.id)
    assert created_record is not None
    assert created_record.content_id == content_id
    assert [record.content_id for record in records] == [content_id]


def test_create_from_hash_with_bare_hash_also_works(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "b" * 64
    path = temp_dir / "existing.bin"
    path.write_bytes(b"existing bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)

    with mock_create_session() as session:
        content = create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
        content_id = content.id
        session.commit()

    result = create_from_hash(digest, "derived.bin")

    assert result is not None
    with mock_create_session() as session:
        records = list(session.scalars(select(Asset)))
        created_record = session.get(Asset, result.ref.id)
    assert created_record is not None
    assert created_record.content_id == content_id
    assert [record.content_id for record in records] == [content_id]


def _seed_live_content(mock_create_session, path, digest):
    with mock_create_session() as session:
        content = create_content(session, str(path), to_stored_hash(digest), path.stat().st_size)
        content_id = content.id
        session.commit()
    return content_id


def test_content_retired_between_lookup_and_claim_mints_nothing(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "c" * 64
    path = temp_dir / "retired.bin"
    path.write_bytes(b"retired bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)
    content_id = _seed_live_content(mock_create_session, path, digest)

    def retire_then_claim(session, claimed_id, hash):
        mark_content_missing(session, claimed_id)
        session.commit()
        return _real_claim_qualified_content(session, claimed_id, hash)

    with patch("app.assets.services.ingest.claim_qualified_content", retire_then_claim):
        result = create_from_hash(f"blake3:{digest}", "derived.bin")

    assert result is None
    with mock_create_session() as session:
        assert list(session.scalars(select(Asset))) == []
        retired = session.get(AssetContent, content_id)
        assert retired is not None
        assert retired.hash == to_stored_hash(digest)
        assert retired.path == str(path)


def test_file_vanishing_between_claim_and_refresh_mints_nothing(
    mock_create_session, monkeypatch, temp_dir
):
    digest = "d" * 64
    path = temp_dir / "vanishing.bin"
    path.write_bytes(b"vanishing bytes")
    monkeypatch.setattr("app.assets.mode.hashing_enabled", lambda: True)
    content_id = _seed_live_content(mock_create_session, path, digest)

    def delete_file_then_refresh(session, refreshed_id):
        path.unlink()
        return _real_refresh_qualified_content(session, refreshed_id)

    with patch(
        "app.assets.services.ingest.refresh_qualified_content", delete_file_then_refresh
    ):
        result = create_from_hash(f"blake3:{digest}", "derived.bin")

    assert result is None
    with mock_create_session() as session:
        assert list(session.scalars(select(Asset))) == []
        assert session.get(AssetContent, content_id) is not None
