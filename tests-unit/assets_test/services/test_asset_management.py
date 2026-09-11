"""Tests for asset_management services."""
import errno
import os

import folder_paths
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import ensure_tags_exist, add_tags_to_reference
from app.assets.helpers import get_utc_now
from app.assets.services import (
    AssetFileDeleteForbiddenError,
    delete_asset_reference_with_file,
    get_asset_detail,
    update_asset_metadata,
    delete_asset_reference,
    set_asset_preview,
)
from app.assets.services import asset_management
from app.assets.services.asset_management import resolve_hash_to_path


def _make_asset(session: Session, hash_val: str = "blake3:test", size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size, mime_type="application/octet-stream")
    session.add(asset)
    session.flush()
    return asset


def _make_reference(
    session: Session,
    asset: Asset,
    name: str = "test",
    owner_id: str = "",
) -> AssetReference:
    now = get_utc_now()
    ref = AssetReference(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(ref)
    session.flush()
    return ref


class TestGetAssetDetail:
    def test_returns_none_for_nonexistent(self, mock_create_session):
        result = get_asset_detail(reference_id="nonexistent")
        assert result is None

    def test_returns_asset_with_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, name="test.bin")
        ensure_tags_exist(session, ["alpha", "beta"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["alpha", "beta"])
        session.commit()

        result = get_asset_detail(reference_id=ref.id)

        assert result is not None
        assert result.ref.id == ref.id
        assert result.asset.hash == asset.hash
        assert set(result.tags) == {"alpha", "beta"}

    def test_respects_owner_visibility(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        # Wrong owner cannot see
        result = get_asset_detail(reference_id=ref.id, owner_id="user2")
        assert result is None

        # Correct owner can see
        result = get_asset_detail(reference_id=ref.id, owner_id="user1")
        assert result is not None


class TestUpdateAssetMetadata:
    def test_updates_name(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, name="old_name.bin")
        ref_id = ref.id
        session.commit()

        update_asset_metadata(
            reference_id=ref_id,
            name="new_name.bin",
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.name == "new_name.bin"

    def test_updates_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ensure_tags_exist(session, ["old"])
        add_tags_to_reference(session, reference_id=ref.id, tags=["old"])
        session.commit()

        result = update_asset_metadata(
            reference_id=ref.id,
            tags=["new1", "new2"],
        )

        assert set(result.tags) == {"new1", "new2"}
        assert "old" not in result.tags

    def test_updates_user_metadata(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ref_id = ref.id
        session.commit()

        update_asset_metadata(
            reference_id=ref_id,
            user_metadata={"key": "value", "num": 42},
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.user_metadata["key"] == "value"
        assert updated_ref.user_metadata["num"] == 42

    def test_raises_for_nonexistent(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            update_asset_metadata(reference_id="nonexistent", name="fail")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            update_asset_metadata(
                reference_id=ref.id,
                name="new",
                owner_id="user2",
            )


class TestDeleteAssetReference:
    def test_soft_deletes_reference(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        ref_id = ref.id
        session.commit()

        result = delete_asset_reference(
            reference_id=ref_id,
            owner_id="",
            delete_content_if_orphan=False,
        )

        assert result is True
        # Row still exists but is marked as soft-deleted
        session.expire_all()
        row = session.get(AssetReference, ref_id)
        assert row is not None
        assert row.deleted_at is not None

    def test_returns_false_for_nonexistent(self, mock_create_session):
        result = delete_asset_reference(
            reference_id="nonexistent",
            owner_id="",
        )
        assert result is False

    def test_returns_false_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        ref_id = ref.id
        session.commit()

        result = delete_asset_reference(
            reference_id=ref_id,
            owner_id="user2",
        )

        assert result is False
        assert session.get(AssetReference, ref_id) is not None

    def test_keeps_asset_if_other_references_exist(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref1 = _make_reference(session, asset, name="ref1")
        _make_reference(session, asset, name="ref2")  # Second ref keeps asset alive
        asset_id = asset.id
        session.commit()

        delete_asset_reference(
            reference_id=ref1.id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Asset should still exist
        assert session.get(Asset, asset_id) is not None

    def test_deletes_orphaned_asset(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset)
        asset_id = asset.id
        ref_id = ref.id
        session.commit()

        delete_asset_reference(
            reference_id=ref_id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Both ref and asset should be gone
        assert session.get(AssetReference, ref_id) is None
        assert session.get(Asset, asset_id) is None


class TestDeleteAssetReferenceWithFile:
    def test_deletes_only_selected_reference_file(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        retained_file = temp_dir / "retained.bin"
        selected_file.write_bytes(b"same-content")
        retained_file.write_bytes(b"same-content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        retained_ref = _make_reference(session, asset, name=retained_file.name)
        retained_ref.file_path = str(retained_file)
        asset_id = asset.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref.id,
            owner_id="",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
        )

        assert result is True
        assert not selected_file.exists()
        assert retained_file.exists()
        assert session.get(Asset, asset_id) is not None
        assert session.get(AssetReference, retained_ref.id) is not None

    def test_rejects_ownerless_reference_without_explicit_permission(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, owner_id="")
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        with pytest.raises(AssetFileDeleteForbiddenError, match="owning user"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="another-user",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        assert selected_file.exists()
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_allows_ownerless_reference_in_single_user_mode(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, owner_id="")
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id="default",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
            allow_ownerless=True,
        )

        assert result is True
        assert not selected_file.exists()
        assert session.get(AssetReference, selected_ref_id) is None

    def test_uses_normalized_owner_for_deletion(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, owner_id="user1")
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        owner_ids = []
        real_delete = asset_management.delete_reference_by_id

        def capture_owner_id(session, reference_id, owner_id):
            owner_ids.append(owner_id)
            return real_delete(session, reference_id, owner_id)

        monkeypatch.setattr(
            asset_management, "delete_reference_by_id", capture_owner_id
        )

        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id=" user1 ",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
        )

        assert result is True
        assert owner_ids == ["user1"]

    def test_preserves_soft_deleted_shared_reference(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        retained_file = temp_dir / "retained.bin"
        selected_file.write_bytes(b"same-content")
        retained_file.write_bytes(b"same-content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        retained_ref = _make_reference(session, asset, name=retained_file.name)
        retained_ref.file_path = str(retained_file)
        retained_ref.deleted_at = get_utc_now()
        asset_id = asset.id
        retained_ref_id = retained_ref.id
        session.commit()

        result = delete_asset_reference_with_file(
            reference_id=selected_ref.id,
            owner_id="",
            staging_directory=str(temp_dir / "staging"),
            expected_file_path=str(selected_file),
            allowed_directories=[str(temp_dir)],
        )

        assert result is True
        assert not selected_file.exists()
        assert retained_file.exists()
        assert session.get(Asset, asset_id) is not None
        assert session.get(AssetReference, retained_ref_id) is not None

    def test_restores_file_when_commit_fails(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "selected.bin"
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        def fail_commit(_session):
            raise RuntimeError("commit failed")

        monkeypatch.setattr(Session, "commit", fail_commit)

        with pytest.raises(RuntimeError, match="commit failed"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        session.expire_all()
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_retains_final_cleanup_failure_in_managed_temp(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        staging_directory = temp_dir / "staging"
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        def fail_remove(_path):
            raise PermissionError("file is busy")

        monkeypatch.setattr("app.assets.services.asset_management.os.remove", fail_remove)
        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id="",
            staging_directory=str(staging_directory),
            expected_file_path=str(selected_file),
            allowed_directories=[str(selected_file.parent)],
        )

        session.expire_all()
        assert result is True
        assert not selected_file.exists()
        queued_files = list(staging_directory.glob(".comfy-delete-*.tmp"))
        assert len(queued_files) == 1
        assert queued_files[0].read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is None

    def test_stages_cross_device_file_in_managed_temp(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        staging_directory = temp_dir / "staging"
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        copy_calls = []
        real_copy = asset_management._copy_file_with_fsync

        def fail_cross_device_replace(_source, _destination):
            raise OSError(errno.EXDEV, "Cross-device link")

        def capture_copy(source, destination):
            copy_calls.append((os.fspath(source), os.fspath(destination)))
            return real_copy(source, destination)

        monkeypatch.setattr(
            asset_management.os, "replace", fail_cross_device_replace
        )
        monkeypatch.setattr(
            asset_management, "_copy_file_with_fsync", capture_copy
        )

        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id="",
            staging_directory=str(staging_directory),
            expected_file_path=str(selected_file),
            allowed_directories=[str(selected_file.parent)],
        )

        session.expire_all()
        assert result is True
        assert len(copy_calls) == 1
        assert copy_calls[0][0] == str(selected_file)
        assert os.path.dirname(copy_calls[0][1]) == str(staging_directory)
        assert not selected_file.exists()
        assert list(staging_directory.glob(".comfy-delete-*.tmp")) == []
        assert session.get(AssetReference, selected_ref_id) is None

    def test_retains_cross_device_cleanup_failure_in_managed_temp(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        staging_directory = temp_dir / "staging"
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        real_copy = asset_management._copy_file_with_fsync
        real_remove = os.remove
        copy_calls = []

        def fail_cross_device_replace(_source, _destination):
            raise OSError(errno.EXDEV, "Cross-device link")

        def capture_copy(source, destination):
            copy_calls.append((os.fspath(source), os.fspath(destination)))
            return real_copy(source, destination)

        def fail_final_cleanup(path):
            if os.fspath(path) == os.fspath(selected_file):
                return real_remove(path)
            raise PermissionError("file is busy")

        monkeypatch.setattr(
            asset_management.os, "replace", fail_cross_device_replace
        )
        monkeypatch.setattr(
            asset_management, "_copy_file_with_fsync", capture_copy
        )
        monkeypatch.setattr(asset_management.os, "remove", fail_final_cleanup)

        result = delete_asset_reference_with_file(
            reference_id=selected_ref_id,
            owner_id="",
            staging_directory=str(staging_directory),
            expected_file_path=str(selected_file),
            allowed_directories=[str(selected_file.parent)],
        )

        session.expire_all()
        queued_files = list(staging_directory.glob(".comfy-delete-*.tmp"))
        assert result is True
        assert len(copy_calls) == 1
        assert not selected_file.exists()
        assert len(queued_files) == 1
        assert queued_files[0].read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is None

    def test_cleans_cross_device_copy_when_source_unlink_fails(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        staging_directory = temp_dir / "staging"
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        real_remove = os.remove

        def fail_cross_device_replace(_source, _destination):
            raise OSError(errno.EXDEV, "Cross-device link")

        def fail_source_unlink(path):
            if os.fspath(path) == os.fspath(selected_file):
                raise PermissionError(errno.EACCES, "Permission denied")
            return real_remove(path)

        monkeypatch.setattr(
            asset_management.os, "replace", fail_cross_device_replace
        )
        monkeypatch.setattr(asset_management.os, "remove", fail_source_unlink)

        with pytest.raises(PermissionError, match="Permission denied"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(staging_directory),
                expected_file_path=str(selected_file),
                allowed_directories=[str(selected_file.parent)],
            )

        session.expire_all()
        assert selected_file.read_bytes() == b"content"
        assert list(staging_directory.glob(".comfy-delete-*.tmp")) == []
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_restores_cross_device_stage_when_commit_fails(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        staging_directory = temp_dir / "staging"
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        real_replace = os.replace
        real_copy = asset_management._copy_file_with_fsync
        replace_calls = 0
        copy_calls = []

        def fail_first_replace(source, destination):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 1:
                raise OSError(errno.EXDEV, "Cross-device link")
            return real_replace(source, destination)

        def capture_copy(source, destination):
            copy_calls.append((os.fspath(source), os.fspath(destination)))
            return real_copy(source, destination)

        def fail_commit(_session):
            raise RuntimeError("commit failed")

        monkeypatch.setattr(asset_management.os, "replace", fail_first_replace)
        monkeypatch.setattr(
            asset_management, "_copy_file_with_fsync", capture_copy
        )
        monkeypatch.setattr(Session, "commit", fail_commit)

        with pytest.raises(RuntimeError, match="commit failed"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(staging_directory),
                expected_file_path=str(selected_file),
                allowed_directories=[str(selected_file.parent)],
            )

        session.expire_all()
        assert len(copy_calls) == 2
        assert copy_calls[0][0] == str(selected_file)
        assert copy_calls[1][1].startswith(
            str(selected_file.parent / ".comfy-restore-")
        )
        assert selected_file.read_bytes() == b"content"
        assert list(staging_directory.glob(".comfy-delete-*.tmp")) == []
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_does_not_fallback_for_non_cross_device_staging_error(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        copy_calls = []

        def fail_replace(_source, _destination):
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(asset_management.os, "replace", fail_replace)
        monkeypatch.setattr(
            asset_management,
            "_copy_file_with_fsync",
            lambda *args: copy_calls.append(args),
        )

        with pytest.raises(PermissionError, match="Permission denied"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(selected_file.parent)],
            )

        session.expire_all()
        assert copy_calls == []
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_rechecks_containment_before_cross_device_fallback(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")
        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()
        containment_results = iter([True, True, False])
        copy_calls = []

        def fail_cross_device_replace(_source, _destination):
            raise OSError(errno.EXDEV, "Cross-device link")

        def capture_copy(*args):
            copy_calls.append(args)

        monkeypatch.setattr(
            folder_paths,
            "is_within_directory",
            lambda *_args: next(containment_results),
        )
        monkeypatch.setattr(
            asset_management.os,
            "replace",
            fail_cross_device_replace,
        )
        monkeypatch.setattr(
            asset_management,
            "_copy_file_with_fsync",
            capture_copy,
        )

        with pytest.raises(AssetFileDeleteForbiddenError, match="moved outside"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(selected_file.parent)],
            )

        session.expire_all()
        assert copy_calls == []
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_rejects_a_source_path_changed_after_authorization(
        self, mock_create_session, session: Session, temp_dir
    ):
        selected_file = temp_dir / "selected.bin"
        replacement_file = temp_dir / "replacement.bin"
        selected_file.write_bytes(b"selected")
        replacement_file.write_bytes(b"replacement")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(replacement_file)
        selected_ref_id = selected_ref.id
        session.commit()

        with pytest.raises(PermissionError, match="source path changed"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir)],
            )

        assert selected_file.exists()
        assert replacement_file.exists()
        assert session.get(AssetReference, selected_ref_id) is not None

    def test_rechecks_containment_immediately_before_staging(
        self, mock_create_session, session: Session, temp_dir, monkeypatch
    ):
        selected_file = temp_dir / "managed" / "selected.bin"
        selected_file.parent.mkdir()
        selected_file.write_bytes(b"content")

        asset = _make_asset(session)
        selected_ref = _make_reference(session, asset, name=selected_file.name)
        selected_ref.file_path = str(selected_file)
        selected_ref_id = selected_ref.id
        session.commit()

        containment_results = iter([True, False])
        monkeypatch.setattr(
            folder_paths,
            "is_within_directory",
            lambda *_args: next(containment_results),
        )

        with pytest.raises(PermissionError, match="moved outside"):
            delete_asset_reference_with_file(
                reference_id=selected_ref_id,
                owner_id="",
                staging_directory=str(temp_dir / "staging"),
                expected_file_path=str(selected_file),
                allowed_directories=[str(temp_dir / "managed")],
            )

        session.expire_all()
        assert selected_file.read_bytes() == b"content"
        assert session.get(AssetReference, selected_ref_id) is not None


class TestSetAssetPreview:
    def test_sets_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session, hash_val="blake3:main")
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        ref_id = ref.id
        preview_ref_id = preview_ref.id
        session.commit()

        set_asset_preview(
            reference_id=ref_id,
            preview_reference_id=preview_ref_id,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.preview_id == preview_ref_id

    def test_clears_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        ref = _make_reference(session, asset)
        preview_ref = _make_reference(session, preview_asset, name="preview.png")
        ref.preview_id = preview_ref.id
        ref_id = ref.id
        session.commit()

        set_asset_preview(
            reference_id=ref_id,
            preview_reference_id=None,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_ref = session.get(AssetReference, ref_id)
        assert updated_ref.preview_id is None

    def test_raises_for_nonexistent_ref(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            set_asset_preview(reference_id="nonexistent")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        ref = _make_reference(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            set_asset_preview(
                reference_id=ref.id,
                preview_reference_id=None,
                owner_id="user2",
            )


class TestResolveHashToPath:
    def test_returns_none_for_unknown_hash(self, mock_create_session):
        result = resolve_hash_to_path("blake3:" + "a" * 64)
        assert result is None

    @pytest.mark.parametrize(
        "ref_owner, query_owner, expect_found",
        [
            ("user1", "user1", True),
            ("user1", "user2", False),
            ("", "anyone", True),
            ("", "", True),
        ],
        ids=[
            "owner_sees_own_ref",
            "other_owner_blocked",
            "ownerless_visible_to_anyone",
            "ownerless_visible_to_empty",
        ],
    )
    def test_owner_visibility(
        self, ref_owner, query_owner, expect_found,
        mock_create_session, session: Session, temp_dir,
    ):
        f = temp_dir / "file.bin"
        f.write_bytes(b"data")
        asset = _make_asset(session, hash_val="blake3:" + "b" * 64)
        ref = _make_reference(session, asset, name="file.bin", owner_id=ref_owner)
        ref.file_path = str(f)
        session.commit()

        result = resolve_hash_to_path(asset.hash, owner_id=query_owner)
        if expect_found:
            assert result is not None
            assert result.abs_path == str(f)
        else:
            assert result is None
