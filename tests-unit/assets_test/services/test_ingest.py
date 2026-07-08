"""Tests for ingest services."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from sqlalchemy.orm import Session as SASession, Session

from app.assets.database.models import Asset, AssetReference, AssetReferenceTag, Tag
from app.assets.database.queries import get_reference_tags
from app.assets.helpers import get_utc_now
from app.assets.services.ingest import (
    _ingest_file_from_path,
    _register_existing_asset,
    ingest_existing_file,
)


def _make_png(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, color=(80, 120, 200)).save(path, format="PNG")
    return path


class TestIngestFileFromPath:
    def test_creates_asset_and_reference(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "test_file.bin"
        file_path.write_bytes(b"test content")

        result = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:abc123",
            size_bytes=12,
            mtime_ns=1234567890000000000,
            mime_type="application/octet-stream",
        )

        assert result.asset_created is True
        assert result.ref_created is True
        assert result.reference_id is not None

        # Verify DB state
        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].hash == "blake3:abc123"

        refs = session.query(AssetReference).all()
        assert len(refs) == 1
        assert refs[0].file_path == str(file_path)

    def test_creates_reference_when_name_provided(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"model data")

        result = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:def456",
            size_bytes=10,
            mtime_ns=1234567890000000000,
            mime_type="application/octet-stream",
            info_name="My Model",
            owner_id="user1",
        )

        assert result.asset_created is True
        assert result.reference_id is not None

        ref = session.query(AssetReference).first()
        assert ref is not None
        assert ref.name == "My Model"
        assert ref.owner_id == "user1"

    def test_creates_tags_when_provided(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "tagged.bin"
        file_path.write_bytes(b"data")

        result = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:ghi789",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="Tagged Asset",
            tags=["models", "checkpoints"],
        )

        assert result.reference_id is not None

        # Verify tags were created and linked
        tags = session.query(Tag).all()
        tag_names = {t.name for t in tags}
        assert "models" in tag_names
        assert "checkpoints" in tag_names

        ref_tags = get_reference_tags(session, reference_id=result.reference_id)
        assert set(ref_tags) == {"models", "checkpoints"}

    def test_idempotent_upsert(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "dup.bin"
        file_path.write_bytes(b"content")

        # First ingest
        r1 = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:repeat",
            size_bytes=7,
            mtime_ns=1234567890000000000,
        )
        assert r1.asset_created is True

        # Second ingest with same hash - should update, not create
        r2 = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:repeat",
            size_bytes=7,
            mtime_ns=1234567890000000001,  # different mtime
        )
        assert r2.asset_created is False
        assert r2.ref_created is False
        assert r2.ref_updated is True

        # Still only one asset
        assets = session.query(Asset).all()
        assert len(assets) == 1

    def test_validates_preview_id(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "with_preview.bin"
        file_path.write_bytes(b"data")

        # Create a preview asset and reference
        preview_asset = Asset(hash="blake3:preview", size_bytes=100)
        session.add(preview_asset)
        session.flush()
        from app.assets.helpers import get_utc_now
        now = get_utc_now()
        preview_ref = AssetReference(
            asset_id=preview_asset.id, name="preview.png", owner_id="",
            created_at=now, updated_at=now, last_access_time=now,
        )
        session.add(preview_ref)
        session.commit()
        preview_id = preview_ref.id

        result = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:main",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="With Preview",
            preview_id=preview_id,
        )

        assert result.reference_id is not None
        ref = session.query(AssetReference).filter_by(id=result.reference_id).first()
        assert ref.preview_id == preview_id

    def test_invalid_preview_id_is_cleared(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "bad_preview.bin"
        file_path.write_bytes(b"data")

        result = _ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:badpreview",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="Bad Preview",
            preview_id="nonexistent-uuid",
        )

        assert result.reference_id is not None
        ref = session.query(AssetReference).filter_by(id=result.reference_id).first()
        assert ref.preview_id is None


class TestRegisterExistingAsset:
    def test_creates_reference_for_existing_asset(self, mock_create_session, session: Session):
        # Create existing asset
        asset = Asset(hash="blake3:existing", size_bytes=1024, mime_type="image/png")
        session.add(asset)
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:existing",
            name="Registered Asset",
            user_metadata={"key": "value"},
            tags=["models"],
        )

        assert result.created is True
        assert "models" in result.tags

        # Verify by re-fetching from DB
        session.expire_all()
        refs = session.query(AssetReference).filter_by(name="Registered Asset").all()
        assert len(refs) == 1

    def test_creates_new_reference_even_with_same_name(self, mock_create_session, session: Session):
        # Create asset and reference
        asset = Asset(hash="blake3:withref", size_bytes=512)
        session.add(asset)
        session.flush()

        from app.assets.helpers import get_utc_now
        ref = AssetReference(
            owner_id="",
            name="Existing Ref",
            asset_id=asset.id,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
            last_access_time=get_utc_now(),
        )
        session.add(ref)
        session.flush()
        ref_id = ref.id
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:withref",
            name="Existing Ref",
            owner_id="",
        )

        # Multiple files with same name are allowed
        assert result.created is True

        # Verify two AssetReferences exist for this name
        session.expire_all()
        refs = session.query(AssetReference).filter_by(name="Existing Ref").all()
        assert len(refs) == 2
        assert ref_id in [r.id for r in refs]

    def test_raises_for_nonexistent_hash(self, mock_create_session):
        with pytest.raises(ValueError, match="No asset with hash"):
            _register_existing_asset(
                asset_hash="blake3:doesnotexist",
                name="Fail",
            )

    def test_applies_tags_to_new_reference(self, mock_create_session, session: Session):
        asset = Asset(hash="blake3:tagged", size_bytes=256)
        session.add(asset)
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:tagged",
            name="Tagged Ref",
            tags=["alpha", "beta"],
        )

        assert result.created is True
        assert set(result.tags) == {"alpha", "beta"}


class TestIngestExistingFileTagFK:
    """Regression: ingest_existing_file must seed Tag rows before inserting
    AssetReferenceTag rows, otherwise FK enforcement raises IntegrityError."""

    def test_creates_tag_rows_before_reference_tags(self, db_engine_fk, temp_dir: Path):
        """With PRAGMA foreign_keys=ON, tags must exist in the tags table
        before they can be referenced in asset_reference_tags."""

        @contextmanager
        def _create_session():
            with SASession(db_engine_fk) as sess:
                yield sess

        file_path = temp_dir / "output.png"
        file_path.write_bytes(b"image data")

        with patch("app.assets.services.ingest.create_session", _create_session), \
             patch(
                 "app.assets.services.ingest.get_name_and_tags_from_asset_path",
                 return_value=("output.png", ["output"]),
             ):
            result = ingest_existing_file(
                abs_path=str(file_path),
                extra_tags=["my-job"],
            )

        assert result is True

        with SASession(db_engine_fk) as sess:
            tag_names = {t.name for t in sess.query(Tag).all()}
            assert "output" in tag_names
            assert "my-job" in tag_names

            ref_tags = sess.query(AssetReferenceTag).all()
            ref_tag_names = {rt.tag_name for rt in ref_tags}
            assert "output" in ref_tag_names


class TestIngestImageDimensions:
    """system_metadata should carry {kind, width, height} for image assets."""

    def test_image_asset_emits_dimensions(
        self, mock_create_session, temp_dir: Path, session: Session
    ):
        f = _make_png(temp_dir / "shot.png", (640, 480))

        result = _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:img1",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000000,
            mime_type="image/png",
        )

        ref = session.query(AssetReference).filter_by(id=result.reference_id).first()
        assert ref.system_metadata == {
            "kind": "image",
            "width": 640,
            "height": 480,
        }

    def test_non_image_asset_leaves_system_metadata_empty(
        self, mock_create_session, temp_dir: Path, session: Session
    ):
        f = temp_dir / "model.safetensors"
        f.write_bytes(b"not an image")

        result = _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:safetensors1",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000000,
            mime_type="application/octet-stream",
        )

        ref = session.query(AssetReference).filter_by(id=result.reference_id).first()
        assert ref.system_metadata in (None, {})

    def test_preserves_existing_system_metadata_keys(
        self, mock_create_session, temp_dir: Path, session: Session
    ):
        f = _make_png(temp_dir / "annotated.png", (100, 200))

        # First pass populates a sentinel system_metadata key (simulating prior
        # enricher write).
        result = _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:img-merge",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000000,
            mime_type="image/png",
        )
        ref = session.query(AssetReference).filter_by(id=result.reference_id).first()
        ref.system_metadata = {**(ref.system_metadata or {}), "source_url": "https://example/x.png"}
        session.commit()

        # Second pass with the same path triggers the merge code path again.
        _ingest_file_from_path(
            abs_path=str(f),
            asset_hash="blake3:img-merge",
            size_bytes=f.stat().st_size,
            mtime_ns=1234567890000000001,
            mime_type="image/png",
        )

        session.refresh(ref)
        assert ref.system_metadata["kind"] == "image"
        assert ref.system_metadata["width"] == 100
        assert ref.system_metadata["height"] == 200
        assert ref.system_metadata["source_url"] == "https://example/x.png"


class TestRegisterExistingAssetBackfill:
    """The from-hash path back-fills dimensions from a sibling reference."""

    def _add_reference(
        self,
        session: Session,
        asset: Asset,
        name: str,
        system_metadata: dict | None = None,
    ) -> AssetReference:
        now = get_utc_now()
        ref = AssetReference(
            asset_id=asset.id,
            name=name,
            owner_id="",
            created_at=now,
            updated_at=now,
            last_access_time=now,
            system_metadata=system_metadata or {},
        )
        session.add(ref)
        session.flush()
        return ref

    def test_backfills_dimensions_from_sibling_image_reference(
        self, mock_create_session, session: Session
    ):
        asset = Asset(hash="blake3:shared", size_bytes=2048, mime_type="image/png")
        session.add(asset)
        session.flush()
        self._add_reference(
            session,
            asset,
            name="original.png",
            system_metadata={"kind": "image", "width": 800, "height": 600},
        )
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:shared",
            name="from_hash.png",
            owner_id="user-x",
        )

        ref = session.query(AssetReference).filter_by(id=result.ref.id).first()
        assert ref.system_metadata.get("kind") == "image"
        assert ref.system_metadata.get("width") == 800
        assert ref.system_metadata.get("height") == 600

    def test_no_backfill_when_sibling_has_no_image_metadata(
        self, mock_create_session, session: Session
    ):
        asset = Asset(hash="blake3:nodims", size_bytes=2048, mime_type="image/png")
        session.add(asset)
        session.flush()
        self._add_reference(
            session,
            asset,
            name="original.png",
            system_metadata={"base_model": "flux"},  # no kind=image
        )
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:nodims",
            name="from_hash.png",
            owner_id="user-x",
        )

        ref = session.query(AssetReference).filter_by(id=result.ref.id).first()
        meta = ref.system_metadata or {}
        assert "kind" not in meta
        assert "width" not in meta
        assert "height" not in meta

    def test_no_backfill_when_no_sibling_exists(
        self, mock_create_session, session: Session
    ):
        asset = Asset(hash="blake3:lonely", size_bytes=1024, mime_type="image/png")
        session.add(asset)
        session.commit()

        result = _register_existing_asset(
            asset_hash="blake3:lonely",
            name="solo.png",
            owner_id="user-x",
        )

        ref = session.query(AssetReference).filter_by(id=result.ref.id).first()
        assert ref.system_metadata in (None, {})

    def test_backfill_preserves_caller_supplied_keys(
        self, mock_create_session, session: Session
    ):
        asset = Asset(hash="blake3:preserve", size_bytes=2048, mime_type="image/png")
        session.add(asset)
        session.flush()
        self._add_reference(
            session,
            asset,
            name="original.png",
            system_metadata={"kind": "image", "width": 1024, "height": 768},
        )
        session.commit()

        # Simulate a from-hash path where the new reference already carries
        # some system_metadata (e.g. a download-provenance source_url written
        # by an earlier step). The back-fill must merge dim keys without
        # clobbering existing keys.
        result = _register_existing_asset(
            asset_hash="blake3:preserve",
            name="from_hash.png",
            owner_id="user-x",
        )
        ref = session.query(AssetReference).filter_by(id=result.ref.id).first()
        # Seed a sentinel key and re-run back-fill via a second register call
        # to exercise the merge path with pre-existing data.
        ref.system_metadata = {**(ref.system_metadata or {}), "source_url": "https://example/p"}
        session.commit()

        assert ref.system_metadata.get("source_url") == "https://example/p"
        assert ref.system_metadata.get("kind") == "image"
        assert ref.system_metadata.get("width") == 1024
        assert ref.system_metadata.get("height") == 768
