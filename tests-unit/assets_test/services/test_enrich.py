"""Tests for asset enrichment (mime_type and hash population)."""
import os
from pathlib import Path

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.services.file_utils import get_mtime_ns
from app.assets.scanner import (
    ENRICHMENT_HASHED,
    ENRICHMENT_METADATA,
    ENRICHMENT_STUB,
    enrich_asset,
)


def _create_stub_asset(
    session: Session,
    file_path: str,
    asset_id: str = "test-asset-id",
    reference_id: str = "test-ref-id",
    name: str | None = None,
) -> tuple[Asset, AssetReference]:
    """Create a stub asset with reference for testing enrichment."""
    # Use the real file's mtime so the optimistic guard in enrich_asset passes
    try:
        stat_result = os.stat(file_path, follow_symlinks=True)
        mtime_ns = get_mtime_ns(stat_result)
    except OSError:
        mtime_ns = 1234567890000000000

    asset = Asset(
        id=asset_id,
        hash=None,
        size_bytes=100,
        mime_type=None,
    )
    session.add(asset)
    session.flush()

    ref = AssetReference(
        id=reference_id,
        asset_id=asset_id,
        name=name or f"test-asset-{asset_id}",
        owner_id="system",
        file_path=file_path,
        mtime_ns=mtime_ns,
        enrichment_level=ENRICHMENT_STUB,
    )
    session.add(ref)
    session.flush()

    return asset, ref


class TestEnrichAsset:
    def test_extracts_mime_type_and_updates_asset(
        self, db_engine, temp_dir: Path, session: Session
    ):
        """Verify mime_type is written to the Asset table during enrichment."""
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"\x00" * 100)

        asset, ref = _create_stub_asset(
            session, str(file_path), "asset-1", "ref-1"
        )
        session.commit()

        new_level = enrich_asset(
            session,
            file_path=str(file_path),
            reference_id=ref.id,
            asset_id=asset.id,
            extract_metadata=True,
            compute_hash=False,
        )

        assert new_level == ENRICHMENT_METADATA

        session.expire_all()
        updated_asset = session.get(Asset, "asset-1")
        assert updated_asset is not None
        assert updated_asset.mime_type == "application/safetensors"

    def test_computes_hash_and_updates_asset(
        self, db_engine, temp_dir: Path, session: Session
    ):
        """Verify hash is written to the Asset table during enrichment."""
        file_path = temp_dir / "data.bin"
        file_path.write_bytes(b"test content for hashing")

        asset, ref = _create_stub_asset(
            session, str(file_path), "asset-2", "ref-2"
        )
        session.commit()

        new_level = enrich_asset(
            session,
            file_path=str(file_path),
            reference_id=ref.id,
            asset_id=asset.id,
            extract_metadata=True,
            compute_hash=True,
        )

        assert new_level == ENRICHMENT_HASHED

        session.expire_all()
        updated_asset = session.get(Asset, "asset-2")
        assert updated_asset is not None
        assert updated_asset.hash is not None
        assert updated_asset.hash.startswith("blake3:")

    def test_enrichment_updates_both_mime_and_hash(
        self, db_engine, temp_dir: Path, session: Session
    ):
        """Verify both mime_type and hash are set when full enrichment runs."""
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"\x00" * 50)

        asset, ref = _create_stub_asset(
            session, str(file_path), "asset-3", "ref-3"
        )
        session.commit()

        enrich_asset(
            session,
            file_path=str(file_path),
            reference_id=ref.id,
            asset_id=asset.id,
            extract_metadata=True,
            compute_hash=True,
        )

        session.expire_all()
        updated_asset = session.get(Asset, "asset-3")
        assert updated_asset is not None
        assert updated_asset.mime_type == "application/safetensors"
        assert updated_asset.hash is not None
        assert updated_asset.hash.startswith("blake3:")

    def test_missing_file_returns_stub_level(
        self, db_engine, temp_dir: Path, session: Session
    ):
        """Verify missing files don't cause errors and return STUB level."""
        file_path = temp_dir / "nonexistent.bin"

        asset, ref = _create_stub_asset(
            session, str(file_path), "asset-4", "ref-4"
        )
        session.commit()

        new_level = enrich_asset(
            session,
            file_path=str(file_path),
            reference_id=ref.id,
            asset_id=asset.id,
            extract_metadata=True,
            compute_hash=True,
        )

        assert new_level == ENRICHMENT_STUB

        session.expire_all()
        updated_asset = session.get(Asset, "asset-4")
        assert updated_asset.mime_type is None
        assert updated_asset.hash is None

    def test_duplicate_hash_merges_into_existing_asset(
        self, db_engine, temp_dir: Path, session: Session
    ):
        """Verify duplicate files merge into existing asset instead of failing."""
        file_path_1 = temp_dir / "file1.bin"
        file_path_2 = temp_dir / "file2.bin"
        content = b"identical content"
        file_path_1.write_bytes(content)
        file_path_2.write_bytes(content)

        asset1, ref1 = _create_stub_asset(
            session, str(file_path_1), "asset-dup-1", "ref-dup-1"
        )
        asset2, ref2 = _create_stub_asset(
            session, str(file_path_2), "asset-dup-2", "ref-dup-2"
        )
        session.commit()

        enrich_asset(
            session,
            file_path=str(file_path_1),
            reference_id=ref1.id,
            asset_id=asset1.id,
            extract_metadata=True,
            compute_hash=True,
        )

        enrich_asset(
            session,
            file_path=str(file_path_2),
            reference_id=ref2.id,
            asset_id=asset2.id,
            extract_metadata=True,
            compute_hash=True,
        )

        session.expire_all()

        updated_asset1 = session.get(Asset, "asset-dup-1")
        assert updated_asset1 is not None
        assert updated_asset1.hash is not None

        updated_asset2 = session.get(Asset, "asset-dup-2")
        assert updated_asset2 is None

        updated_ref2 = session.get(AssetReference, "ref-dup-2")
        assert updated_ref2 is not None
        assert updated_ref2.asset_id == "asset-dup-1"
