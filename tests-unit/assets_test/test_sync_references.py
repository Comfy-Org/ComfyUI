"""Tests for sync_references_with_filesystem in scanner.py."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import (
    Asset,
    AssetReference,
    AssetReferenceTag,
    Base,
    Tag,
)
from app.assets.database.queries.asset_reference import (
    bulk_insert_references_ignore_conflicts,
    get_references_for_prefixes,
    get_unenriched_references,
    restore_references_by_paths,
)
from app.assets.scanner import sync_references_with_filesystem
from app.assets.services.file_utils import get_mtime_ns


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(db_engine):
    with Session(db_engine) as sess:
        yield sess


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_file(temp_dir: Path, name: str, content: bytes = b"\x00" * 100) -> str:
    """Create a file and return its absolute path (no symlink resolution)."""
    p = temp_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return os.path.abspath(str(p))


def _stat_mtime_ns(path: str) -> int:
    return get_mtime_ns(os.stat(path, follow_symlinks=True))


def _make_asset(
    session: Session,
    asset_id: str,
    file_path: str,
    ref_id: str,
    *,
    asset_hash: str | None = None,
    size_bytes: int = 100,
    mtime_ns: int | None = None,
    needs_verify: bool = False,
    is_missing: bool = False,
) -> tuple[Asset, AssetReference]:
    """Insert an Asset + AssetReference and flush."""
    asset = session.get(Asset, asset_id)
    if asset is None:
        asset = Asset(id=asset_id, hash=asset_hash, size_bytes=size_bytes)
        session.add(asset)
        session.flush()

    ref = AssetReference(
        id=ref_id,
        asset_id=asset_id,
        name=f"test-{ref_id}",
        owner_id="system",
        file_path=file_path,
        mtime_ns=mtime_ns,
        needs_verify=needs_verify,
        is_missing=is_missing,
    )
    session.add(ref)
    session.flush()
    return asset, ref


def _ensure_missing_tag(session: Session):
    """Ensure the 'missing' tag exists."""
    if not session.get(Tag, "missing"):
        session.add(Tag(name="missing"))
        session.flush()


class _VerifyCase:
    def __init__(self, id, stat_unchanged, needs_verify_before, expect_needs_verify):
        self.id = id
        self.stat_unchanged = stat_unchanged
        self.needs_verify_before = needs_verify_before
        self.expect_needs_verify = expect_needs_verify


VERIFY_CASES = [
    _VerifyCase(
        id="unchanged_clears_verify",
        stat_unchanged=True,
        needs_verify_before=True,
        expect_needs_verify=False,
    ),
    _VerifyCase(
        id="unchanged_keeps_clear",
        stat_unchanged=True,
        needs_verify_before=False,
        expect_needs_verify=False,
    ),
    _VerifyCase(
        id="changed_sets_verify",
        stat_unchanged=False,
        needs_verify_before=False,
        expect_needs_verify=True,
    ),
    _VerifyCase(
        id="changed_keeps_verify",
        stat_unchanged=False,
        needs_verify_before=True,
        expect_needs_verify=True,
    ),
]


@pytest.mark.parametrize("case", VERIFY_CASES, ids=lambda c: c.id)
def test_needs_verify_toggling(session, temp_dir, case):
    """needs_verify is set/cleared based on mtime+size match."""
    fp = _create_file(temp_dir, "model.bin")
    real_mtime = _stat_mtime_ns(fp)

    mtime_for_db = real_mtime if case.stat_unchanged else real_mtime + 1
    _make_asset(
        session, "a1", fp, "r1",
        asset_hash="blake3:abc",
        mtime_ns=mtime_for_db,
        needs_verify=case.needs_verify_before,
    )
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    ref = session.get(AssetReference, "r1")
    assert ref.needs_verify is case.expect_needs_verify


class _MissingCase:
    def __init__(self, id, file_exists, expect_is_missing):
        self.id = id
        self.file_exists = file_exists
        self.expect_is_missing = expect_is_missing


MISSING_CASES = [
    _MissingCase(id="existing_file_not_missing", file_exists=True, expect_is_missing=False),
    _MissingCase(id="missing_file_marked_missing", file_exists=False, expect_is_missing=True),
]


@pytest.mark.parametrize("case", MISSING_CASES, ids=lambda c: c.id)
def test_is_missing_flag(session, temp_dir, case):
    """is_missing reflects whether the file exists on disk."""
    if case.file_exists:
        fp = _create_file(temp_dir, "model.bin")
        mtime = _stat_mtime_ns(fp)
    else:
        fp = str(temp_dir / "gone.bin")
        mtime = 999

    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    ref = session.get(AssetReference, "r1")
    assert ref.is_missing is case.expect_is_missing


def test_seed_asset_all_missing_deletes_asset(session, temp_dir):
    """Seed asset with all refs missing gets deleted entirely."""
    fp = str(temp_dir / "gone.bin")
    _make_asset(session, "seed1", fp, "r1", asset_hash=None, mtime_ns=999)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    assert session.get(Asset, "seed1") is None
    assert session.get(AssetReference, "r1") is None


def test_seed_asset_some_exist_returns_survivors(session, temp_dir):
    """Seed asset with at least one existing ref survives and is returned."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "seed1", fp, "r1", asset_hash=None, mtime_ns=mtime)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        survivors = sync_references_with_filesystem(
            session, "models", collect_existing_paths=True,
        )
        session.commit()

    assert session.get(Asset, "seed1") is not None
    assert os.path.abspath(fp) in survivors


def test_hashed_asset_prunes_missing_refs_when_one_is_ok(session, temp_dir):
    """Hashed asset with one stat-unchanged ref deletes missing refs."""
    fp_ok = _create_file(temp_dir, "good.bin")
    fp_gone = str(temp_dir / "gone.bin")
    mtime = _stat_mtime_ns(fp_ok)

    _make_asset(session, "h1", fp_ok, "r_ok", asset_hash="blake3:aaa", mtime_ns=mtime)
    # Second ref on same asset, file missing
    ref_gone = AssetReference(
        id="r_gone", asset_id="h1", name="gone",
        owner_id="system", file_path=fp_gone, mtime_ns=999,
    )
    session.add(ref_gone)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    assert session.get(AssetReference, "r_ok") is not None
    assert session.get(AssetReference, "r_gone") is None


def test_hashed_asset_all_missing_keeps_refs(session, temp_dir):
    """Hashed asset with all refs missing keeps refs (no pruning)."""
    fp = str(temp_dir / "gone.bin")
    _make_asset(session, "h1", fp, "r1", asset_hash="blake3:aaa", mtime_ns=999)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    assert session.get(AssetReference, "r1") is not None
    ref = session.get(AssetReference, "r1")
    assert ref.is_missing is True


def test_missing_tag_added_when_all_refs_gone(session, temp_dir):
    """Missing tag is added to hashed asset when all refs are missing."""
    _ensure_missing_tag(session)
    fp = str(temp_dir / "gone.bin")
    _make_asset(session, "h1", fp, "r1", asset_hash="blake3:aaa", mtime_ns=999)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(
            session, "models", update_missing_tags=True,
        )
        session.commit()

    session.expire_all()
    tag_link = session.get(AssetReferenceTag, ("r1", "missing"))
    assert tag_link is not None


def test_missing_tag_removed_when_ref_ok(session, temp_dir):
    """Missing tag is removed from hashed asset when a ref is stat-unchanged."""
    _ensure_missing_tag(session)
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "h1", fp, "r1", asset_hash="blake3:aaa", mtime_ns=mtime)
    # Pre-add a stale missing tag
    session.add(AssetReferenceTag(
        asset_reference_id="r1", tag_name="missing", origin="automatic",
    ))
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(
            session, "models", update_missing_tags=True,
        )
        session.commit()

    session.expire_all()
    tag_link = session.get(AssetReferenceTag, ("r1", "missing"))
    assert tag_link is None


def test_missing_tags_not_touched_when_flag_false(session, temp_dir):
    """Missing tags are not modified when update_missing_tags=False."""
    _ensure_missing_tag(session)
    fp = str(temp_dir / "gone.bin")
    _make_asset(session, "h1", fp, "r1", asset_hash="blake3:aaa", mtime_ns=999)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(
            session, "models", update_missing_tags=False,
        )
        session.commit()

    tag_link = session.get(AssetReferenceTag, ("r1", "missing"))
    assert tag_link is None  # tag was never added


def test_returns_none_when_collect_false(session, temp_dir):
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        result = sync_references_with_filesystem(
            session, "models", collect_existing_paths=False,
        )

    assert result is None


def test_returns_empty_set_for_no_prefixes(session):
    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[]):
        result = sync_references_with_filesystem(
            session, "models", collect_existing_paths=True,
        )

    assert result == set()


def test_no_references_is_noop(session, temp_dir):
    """No crash and no side effects when there are no references."""
    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        survivors = sync_references_with_filesystem(
            session, "models", collect_existing_paths=True,
        )
        session.commit()

    assert survivors == set()


# ---------------------------------------------------------------------------
# Soft-delete persistence across scanner operations
# ---------------------------------------------------------------------------

def _soft_delete_ref(session: Session, ref_id: str) -> None:
    """Mark a reference as soft-deleted (mimics the API DELETE behaviour)."""
    ref = session.get(AssetReference, ref_id)
    ref.deleted_at = datetime(2025, 1, 1)
    session.flush()


def test_soft_deleted_ref_excluded_from_get_references_for_prefixes(session, temp_dir):
    """get_references_for_prefixes skips soft-deleted references."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    _soft_delete_ref(session, "r1")
    session.commit()

    rows = get_references_for_prefixes(session, [str(temp_dir)], include_missing=True)
    assert len(rows) == 0


def test_sync_does_not_resurrect_soft_deleted_ref(session, temp_dir):
    """Scanner sync leaves soft-deleted refs untouched even when file exists on disk."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    _soft_delete_ref(session, "r1")
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    ref = session.get(AssetReference, "r1")
    assert ref.deleted_at is not None, "soft-deleted ref must stay deleted after sync"


def test_bulk_insert_does_not_overwrite_soft_deleted_ref(session, temp_dir):
    """bulk_insert_references_ignore_conflicts cannot replace a soft-deleted row."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    _soft_delete_ref(session, "r1")
    session.commit()

    now = datetime.now(tz=None)
    bulk_insert_references_ignore_conflicts(session, [
        {
            "id": "r_new",
            "asset_id": "a1",
            "file_path": fp,
            "name": "model.bin",
            "owner_id": "",
            "mtime_ns": mtime,
            "preview_id": None,
            "user_metadata": None,
            "created_at": now,
            "updated_at": now,
            "last_access_time": now,
        }
    ])
    session.commit()

    session.expire_all()
    # Original row is still the soft-deleted one
    ref = session.get(AssetReference, "r1")
    assert ref is not None
    assert ref.deleted_at is not None
    # The new row was not inserted (conflict on file_path)
    assert session.get(AssetReference, "r_new") is None


def test_restore_references_by_paths_skips_soft_deleted(session, temp_dir):
    """restore_references_by_paths does not clear is_missing on soft-deleted refs."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(
        session, "a1", fp, "r1",
        asset_hash="blake3:abc", mtime_ns=mtime, is_missing=True,
    )
    _soft_delete_ref(session, "r1")
    session.commit()

    restored = restore_references_by_paths(session, [fp])
    session.commit()

    assert restored == 0
    session.expire_all()
    ref = session.get(AssetReference, "r1")
    assert ref.is_missing is True, "is_missing must not be cleared on soft-deleted ref"
    assert ref.deleted_at is not None


def test_get_unenriched_references_excludes_soft_deleted(session, temp_dir):
    """Enrichment queries do not pick up soft-deleted references."""
    fp = _create_file(temp_dir, "model.bin")
    mtime = _stat_mtime_ns(fp)
    _make_asset(session, "a1", fp, "r1", asset_hash="blake3:abc", mtime_ns=mtime)
    _soft_delete_ref(session, "r1")
    session.commit()

    rows = get_unenriched_references(session, [str(temp_dir)], max_level=2)
    assert len(rows) == 0


def test_sync_ignores_soft_deleted_seed_asset(session, temp_dir):
    """Soft-deleted seed ref is not garbage-collected even when file is missing."""
    fp = str(temp_dir / "gone.bin")  # file does not exist
    _make_asset(session, "seed1", fp, "r1", asset_hash=None, mtime_ns=999)
    _soft_delete_ref(session, "r1")
    session.commit()

    with patch("app.assets.scanner.get_prefixes_for_root", return_value=[str(temp_dir)]):
        sync_references_with_filesystem(session, "models")
        session.commit()

    session.expire_all()
    # Asset and ref must still exist — scanner did not see the soft-deleted row
    assert session.get(Asset, "seed1") is not None
    assert session.get(AssetReference, "r1") is not None
