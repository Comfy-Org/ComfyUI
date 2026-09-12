from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent, AssetTag, Tag
from app.assets.helpers import to_stored_hash
from app.assets.scanner import (
    SeedAssetSpec,
    clear_pending_verifications,
    pending_recovery_count,
    seed_asset_specs,
)
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def _clear_queues():
    clear_pending_verifications()
    yield
    clear_pending_verifications()


def _missing_content(session, path: Path, hash_value: str) -> tuple[AssetContent, Asset]:
    content = AssetContent(path=str(path), hash=hash_value, is_missing=True)
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name=path.name)
    session.add(record)
    session.flush()
    if session.get(Tag, "missing") is None:
        session.add(Tag(name="missing"))
    session.add(AssetTag(asset_id=record.id, tag_name="missing", origin="automatic"))
    session.commit()
    return content, record


def _spec(path: Path) -> SeedAssetSpec:
    stat = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "info_name": path.name,
        "tags": ["input"],
        "fname": path.name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def _stored_hash(path: Path) -> str:
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    return to_stored_hash(digest)


def test_single_hash_match_recovers(session, temp_dir: Path):
    path = temp_dir / "restored.bin"
    path.write_bytes(b"restored bytes")
    content, record = _missing_content(session, path, _stored_hash(path))

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 0
    assert session.get(AssetContent, content.id).is_missing is False
    assert session.get(AssetTag, {"asset_id": record.id, "tag_name": "missing"}) is None


def test_ambiguous_hash_match_recovers_nothing(session, temp_dir: Path):
    path = temp_dir / "ambiguous.bin"
    path.write_bytes(b"same bytes")
    digest = _stored_hash(path)
    first, _ = _missing_content(session, path, digest)
    second, _ = _missing_content(session, path, digest)

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 1
    assert session.get(AssetContent, first.id).is_missing is True
    assert session.get(AssetContent, second.id).is_missing is True
    assert len(session.scalars(select(AssetContent)).all()) == 3


def test_no_hash_match_creates_fresh_rows(session, temp_dir: Path):
    path = temp_dir / "different.bin"
    path.write_bytes(b"current bytes")
    missing, _ = _missing_content(session, path, "old")

    with patch("app.assets.scanner.mode.hashing_enabled", return_value=True):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 1
    assert session.get(AssetContent, missing.id).is_missing is True
    assert len(session.scalars(select(Asset)).all()) == 2


def test_off_mode_no_recovery(session, temp_dir: Path):
    path = temp_dir / "off.bin"
    path.write_bytes(b"bytes")
    missing, _ = _missing_content(session, path, _stored_hash(path))

    with (
        patch("app.assets.scanner.mode.hashing_enabled", return_value=False),
        patch("app.assets.scanner_changes.snapshot_hash") as hash_mock,
    ):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    hash_mock.assert_not_called()
    assert created == 1
    assert session.get(AssetContent, missing.id).is_missing is True


def test_unstable_hash_requeues(session, temp_dir: Path):
    path = temp_dir / "unstable.bin"
    path.write_bytes(b"bytes")
    missing, _ = _missing_content(session, path, "old")

    with (
        patch("app.assets.scanner.mode.hashing_enabled", return_value=True),
        patch("app.assets.scanner_changes.snapshot_hash", return_value=None),
    ):
        created = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert created == 0
    assert pending_recovery_count() == 1
    assert session.get(AssetContent, missing.id).is_missing is True
