"""Tests for bulk ingest services."""

from pathlib import Path

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.services.bulk_ingest import SeedAssetSpec, batch_insert_seed_assets


class TestBatchInsertSeedAssets:
    def test_populates_mime_type_for_model_files(self, session: Session, temp_dir: Path):
        """Verify mime_type is stored in the Asset table for model files."""
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"fake safetensors content")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 24,
                "mtime_ns": 1234567890000000000,
                "info_name": "Test Model",
                "tags": ["models"],
                "fname": "model.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1

        # Verify Asset has mime_type populated
        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].mime_type == "application/safetensors"

    def test_mime_type_none_when_not_provided(self, session: Session, temp_dir: Path):
        """Verify mime_type is None when not provided in spec."""
        file_path = temp_dir / "unknown.bin"
        file_path.write_bytes(b"binary data")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 11,
                "mtime_ns": 1234567890000000000,
                "info_name": "Unknown File",
                "tags": [],
                "fname": "unknown.bin",
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1

        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].mime_type is None

    def test_various_model_mime_types(self, session: Session, temp_dir: Path):
        """Verify various model file types get correct mime_type."""
        test_cases = [
            ("model.safetensors", "application/safetensors"),
            ("model.pt", "application/pytorch"),
            ("model.ckpt", "application/pickle"),
            ("model.gguf", "application/gguf"),
        ]

        specs: list[SeedAssetSpec] = []
        for filename, mime_type in test_cases:
            file_path = temp_dir / filename
            file_path.write_bytes(b"content")
            specs.append(
                {
                    "abs_path": str(file_path),
                    "size_bytes": 7,
                    "mtime_ns": 1234567890000000000,
                    "info_name": filename,
                    "tags": [],
                    "fname": filename,
                    "metadata": None,
                    "hash": None,
                    "mime_type": mime_type,
                }
            )

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == len(test_cases)

        for filename, expected_mime in test_cases:
            ref = session.query(AssetReference).filter_by(name=filename).first()
            assert ref is not None
            asset = session.query(Asset).filter_by(id=ref.asset_id).first()
            assert asset.mime_type == expected_mime, f"Expected {expected_mime} for {filename}, got {asset.mime_type}"


class TestMetadataExtraction:
    def test_extracts_mime_type_for_model_files(self, temp_dir: Path):
        """Verify metadata extraction returns correct mime_type for model files."""
        from app.assets.services.metadata_extract import extract_file_metadata

        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"fake safetensors content")

        meta = extract_file_metadata(str(file_path))

        assert meta.content_type == "application/safetensors"

    def test_mime_type_for_various_model_formats(self, temp_dir: Path):
        """Verify various model file types get correct mime_type from metadata."""
        from app.assets.services.metadata_extract import extract_file_metadata

        test_cases = [
            ("model.safetensors", "application/safetensors"),
            ("model.sft", "application/safetensors"),
            ("model.pt", "application/pytorch"),
            ("model.pth", "application/pytorch"),
            ("model.ckpt", "application/pickle"),
            ("model.pkl", "application/pickle"),
            ("model.gguf", "application/gguf"),
        ]

        for filename, expected_mime in test_cases:
            file_path = temp_dir / filename
            file_path.write_bytes(b"content")

            meta = extract_file_metadata(str(file_path))

            assert meta.content_type == expected_mime, f"Expected {expected_mime} for {filename}, got {meta.content_type}"
