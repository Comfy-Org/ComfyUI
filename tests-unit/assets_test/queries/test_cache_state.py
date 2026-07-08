"""Tests for cache_state (AssetReference file path) query functions."""
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import (
    list_references_by_asset_id,
    upsert_reference,
    get_unreferenced_unhashed_asset_ids,
    delete_assets_by_ids,
    get_references_for_prefixes,
    bulk_update_needs_verify,
    delete_references_by_ids,
    delete_orphaned_seed_asset,
    bulk_insert_references_ignore_conflicts,
    get_references_by_paths_and_asset_ids,
    mark_references_missing_outside_prefixes,
    restore_references_by_paths,
)
from app.assets.helpers import select_best_live_path, get_utc_now


def _make_asset(session: Session, hash_val: str | None = None, size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size)
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    file_path: str,
    name: str = "test",
    mtime_ns: int | None = None,
    needs_verify: bool = False,
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        asset_id=asset.id,
        file_path=file_path,
        name=name,
        mtime_ns=mtime_ns,
        needs_verify=needs_verify,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()
    return ref


class TestListReferencesByAssetId:
    def test_returns_empty_for_no_references(self, session: Session):
        asset = _make_asset(session, "hash1")
        refs = list_references_by_asset_id(session, asset_id=asset.id)
        assert list(refs) == []

    def test_returns_references_for_asset(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "/path/a.bin", name="a")
        _make_reference(session, asset, "/path/b.bin", name="b")
        session.commit()

        refs = list_references_by_asset_id(session, asset_id=asset.id)
        paths = [r.file_path for r in refs]
        assert set(paths) == {"/path/a.bin", "/path/b.bin"}

    def test_does_not_return_other_assets_references(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")
        _make_reference(session, asset1, "/path/asset1.bin", name="a1")
        _make_reference(session, asset2, "/path/asset2.bin", name="a2")
        session.commit()

        refs = list_references_by_asset_id(session, asset_id=asset1.id)
        paths = [r.file_path for r in refs]
        assert paths == ["/path/asset1.bin"]


class TestSelectBestLivePath:
    def test_returns_empty_for_empty_list(self):
        result = select_best_live_path([])
        assert result == ""

    def test_returns_empty_when_no_files_exist(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref = _make_reference(session, asset, "/nonexistent/path.bin")
        session.commit()

        result = select_best_live_path([ref])
        assert result == ""

    def test_prefers_verified_path(self, session: Session, tmp_path):
        """needs_verify=False should be preferred."""
        asset = _make_asset(session, "hash1")

        verified_file = tmp_path / "verified.bin"
        verified_file.write_bytes(b"data")

        unverified_file = tmp_path / "unverified.bin"
        unverified_file.write_bytes(b"data")

        ref_verified = _make_reference(
            session, asset, str(verified_file), name="verified", needs_verify=False
        )
        ref_unverified = _make_reference(
            session, asset, str(unverified_file), name="unverified", needs_verify=True
        )
        session.commit()

        refs = [ref_unverified, ref_verified]
        result = select_best_live_path(refs)
        assert result == str(verified_file)

    def test_falls_back_to_existing_unverified(self, session: Session, tmp_path):
        """If all references need verification, return first existing path."""
        asset = _make_asset(session, "hash1")

        existing_file = tmp_path / "exists.bin"
        existing_file.write_bytes(b"data")

        ref = _make_reference(session, asset, str(existing_file), needs_verify=True)
        session.commit()

        result = select_best_live_path([ref])
        assert result == str(existing_file)


class TestSelectBestLivePathWithMocking:
    def test_handles_missing_file_path_attr(self):
        """Gracefully handle references with None file_path."""

        class MockRef:
            file_path = None
            needs_verify = False

        result = select_best_live_path([MockRef()])
        assert result == ""


class TestUpsertReference:
    @pytest.mark.parametrize(
        "initial_mtime,second_mtime,expect_created,expect_updated,final_mtime",
        [
            # New reference creation
            (None, 12345, True, False, 12345),
            # Existing reference, same mtime - no update
            (100, 100, False, False, 100),
            # Existing reference, different mtime - update
            (100, 200, False, True, 200),
        ],
        ids=["new_reference", "existing_no_change", "existing_update_mtime"],
    )
    def test_upsert_scenarios(
        self, session: Session, initial_mtime, second_mtime, expect_created, expect_updated, final_mtime
    ):
        asset = _make_asset(session, "hash1")
        file_path = f"/path_{initial_mtime}_{second_mtime}.bin"
        name = f"file_{initial_mtime}_{second_mtime}"

        # Create initial reference if needed
        if initial_mtime is not None:
            upsert_reference(session, asset_id=asset.id, file_path=file_path, name=name, mtime_ns=initial_mtime)
            session.commit()

        # The upsert call we're testing
        created, updated = upsert_reference(
            session, asset_id=asset.id, file_path=file_path, name=name, mtime_ns=second_mtime
        )
        session.commit()

        assert created is expect_created
        assert updated is expect_updated
        ref = session.query(AssetReference).filter_by(file_path=file_path).one()
        assert ref.mtime_ns == final_mtime

    def test_upsert_restores_missing_reference(self, session: Session):
        """Upserting a reference that was marked missing should restore it."""
        asset = _make_asset(session, "hash1")
        file_path = "/restored/file.bin"

        ref = _make_reference(session, asset, file_path, mtime_ns=100)
        ref.is_missing = True
        session.commit()

        created, updated = upsert_reference(
            session, asset_id=asset.id, file_path=file_path, name="restored", mtime_ns=100
        )
        session.commit()

        assert created is False
        assert updated is True
        restored_ref = session.query(AssetReference).filter_by(file_path=file_path).one()
        assert restored_ref.is_missing is False


class TestRestoreReferencesByPaths:
    def test_restores_missing_references(self, session: Session):
        asset = _make_asset(session, "hash1")
        missing_path = "/missing/file.bin"
        active_path = "/active/file.bin"

        missing_ref = _make_reference(session, asset, missing_path, name="missing")
        missing_ref.is_missing = True
        _make_reference(session, asset, active_path, name="active")
        session.commit()

        restored = restore_references_by_paths(session, [missing_path])
        session.commit()

        assert restored == 1
        ref = session.query(AssetReference).filter_by(file_path=missing_path).one()
        assert ref.is_missing is False

    def test_empty_list_restores_nothing(self, session: Session):
        restored = restore_references_by_paths(session, [])
        assert restored == 0


class TestMarkReferencesMissingOutsidePrefixes:
    def test_marks_references_missing_outside_prefixes(self, session: Session, tmp_path):
        asset = _make_asset(session, "hash1")
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        valid_path = str(valid_dir / "file.bin")
        invalid_path = str(invalid_dir / "file.bin")

        _make_reference(session, asset, valid_path, name="valid")
        _make_reference(session, asset, invalid_path, name="invalid")
        session.commit()

        marked = mark_references_missing_outside_prefixes(session, [str(valid_dir)])
        session.commit()

        assert marked == 1
        all_refs = session.query(AssetReference).all()
        assert len(all_refs) == 2

        valid_ref = next(r for r in all_refs if r.file_path == valid_path)
        invalid_ref = next(r for r in all_refs if r.file_path == invalid_path)
        assert valid_ref.is_missing is False
        assert invalid_ref.is_missing is True

    def test_empty_prefixes_marks_nothing(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "/some/path.bin")
        session.commit()

        marked = mark_references_missing_outside_prefixes(session, [])

        assert marked == 0


class TestGetUnreferencedUnhashedAssetIds:
    def test_returns_unreferenced_unhashed_assets(self, session: Session):
        # Unhashed asset (hash=None) with no references (no file_path)
        no_refs = _make_asset(session, hash_val=None)
        # Unhashed asset with active reference (not unreferenced)
        with_active_ref = _make_asset(session, hash_val=None)
        _make_reference(session, with_active_ref, "/has/ref.bin", name="has_ref")
        # Unhashed asset with only missing reference (should be unreferenced)
        with_missing_ref = _make_asset(session, hash_val=None)
        missing_ref = _make_reference(session, with_missing_ref, "/missing/ref.bin", name="missing_ref")
        missing_ref.is_missing = True
        # Regular asset (hash not None) - should not be returned
        _make_asset(session, hash_val="blake3:regular")
        session.commit()

        unreferenced = get_unreferenced_unhashed_asset_ids(session)

        assert no_refs.id in unreferenced
        assert with_missing_ref.id in unreferenced
        assert with_active_ref.id not in unreferenced


class TestDeleteAssetsByIds:
    def test_deletes_assets_and_references(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "/test/path.bin", name="test")
        session.commit()

        deleted = delete_assets_by_ids(session, [asset.id])
        session.commit()

        assert deleted == 1
        assert session.query(Asset).count() == 0
        assert session.query(AssetReference).count() == 0

    def test_empty_list_deletes_nothing(self, session: Session):
        _make_asset(session, "hash1")
        session.commit()

        deleted = delete_assets_by_ids(session, [])

        assert deleted == 0
        assert session.query(Asset).count() == 1


class TestGetReferencesForPrefixes:
    def test_returns_references_matching_prefix(self, session: Session, tmp_path):
        asset = _make_asset(session, "hash1")
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        path1 = str(dir1 / "file.bin")
        path2 = str(dir2 / "file.bin")

        _make_reference(session, asset, path1, name="file1", mtime_ns=100)
        _make_reference(session, asset, path2, name="file2", mtime_ns=200)
        session.commit()

        rows = get_references_for_prefixes(session, [str(dir1)])

        assert len(rows) == 1
        assert rows[0].file_path == path1

    def test_empty_prefixes_returns_empty(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "/some/path.bin")
        session.commit()

        rows = get_references_for_prefixes(session, [])

        assert rows == []


class TestBulkSetNeedsVerify:
    def test_sets_needs_verify_flag(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1 = _make_reference(session, asset, "/path1.bin", needs_verify=False)
        ref2 = _make_reference(session, asset, "/path2.bin", needs_verify=False)
        session.commit()

        updated = bulk_update_needs_verify(session, [ref1.id, ref2.id], True)
        session.commit()

        assert updated == 2
        session.refresh(ref1)
        session.refresh(ref2)
        assert ref1.needs_verify is True
        assert ref2.needs_verify is True

    def test_empty_list_updates_nothing(self, session: Session):
        updated = bulk_update_needs_verify(session, [], True)
        assert updated == 0


class TestDeleteReferencesByIds:
    def test_deletes_references_by_id(self, session: Session):
        asset = _make_asset(session, "hash1")
        ref1 = _make_reference(session, asset, "/path1.bin")
        _make_reference(session, asset, "/path2.bin")
        session.commit()

        deleted = delete_references_by_ids(session, [ref1.id])
        session.commit()

        assert deleted == 1
        assert session.query(AssetReference).count() == 1

    def test_empty_list_deletes_nothing(self, session: Session):
        deleted = delete_references_by_ids(session, [])
        assert deleted == 0


class TestDeleteOrphanedSeedAsset:
    @pytest.mark.parametrize(
        "create_asset,expected_deleted,expected_count",
        [
            (True, True, 0),   # Existing asset gets deleted
            (False, False, 0),  # Nonexistent returns False
        ],
        ids=["deletes_existing", "nonexistent_returns_false"],
    )
    def test_delete_orphaned_seed_asset(
        self, session: Session, create_asset, expected_deleted, expected_count
    ):
        asset_id = "nonexistent-id"
        if create_asset:
            asset = _make_asset(session, hash_val=None)
            asset_id = asset.id
            _make_reference(session, asset, "/test/path.bin", name="test")
            session.commit()

        deleted = delete_orphaned_seed_asset(session, asset_id)
        if create_asset:
            session.commit()

        assert deleted is expected_deleted
        assert session.query(Asset).count() == expected_count


class TestBulkInsertReferencesIgnoreConflicts:
    def test_inserts_multiple_references(self, session: Session):
        asset = _make_asset(session, "hash1")
        now = get_utc_now()
        rows = [
            {
                "asset_id": asset.id,
                "file_path": "/bulk1.bin",
                "name": "bulk1",
                "mtime_ns": 100,
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
            {
                "asset_id": asset.id,
                "file_path": "/bulk2.bin",
                "name": "bulk2",
                "mtime_ns": 200,
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
        ]
        bulk_insert_references_ignore_conflicts(session, rows)
        session.commit()

        assert session.query(AssetReference).count() == 2

    def test_ignores_conflicts(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_reference(session, asset, "/existing.bin", mtime_ns=100)
        session.commit()

        now = get_utc_now()
        rows = [
            {
                "asset_id": asset.id,
                "file_path": "/existing.bin",
                "name": "existing",
                "mtime_ns": 999,
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
            {
                "asset_id": asset.id,
                "file_path": "/new.bin",
                "name": "new",
                "mtime_ns": 200,
                "created_at": now,
                "updated_at": now,
                "last_access_time": now,
            },
        ]
        bulk_insert_references_ignore_conflicts(session, rows)
        session.commit()

        assert session.query(AssetReference).count() == 2
        existing = session.query(AssetReference).filter_by(file_path="/existing.bin").one()
        assert existing.mtime_ns == 100  # Original value preserved

    def test_empty_list_is_noop(self, session: Session):
        bulk_insert_references_ignore_conflicts(session, [])
        assert session.query(AssetReference).count() == 0


class TestGetReferencesByPathsAndAssetIds:
    def test_returns_matching_paths(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")

        _make_reference(session, asset1, "/path1.bin")
        _make_reference(session, asset2, "/path2.bin")
        session.commit()

        path_to_asset = {
            "/path1.bin": asset1.id,
            "/path2.bin": asset2.id,
        }
        winners = get_references_by_paths_and_asset_ids(session, path_to_asset)

        assert winners == {"/path1.bin", "/path2.bin"}

    def test_excludes_non_matching_asset_ids(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")

        _make_reference(session, asset1, "/path1.bin")
        session.commit()

        # Path exists but with different asset_id
        path_to_asset = {"/path1.bin": asset2.id}
        winners = get_references_by_paths_and_asset_ids(session, path_to_asset)

        assert winners == set()

    def test_empty_dict_returns_empty(self, session: Session):
        winners = get_references_by_paths_and_asset_ids(session, {})
        assert winners == set()
