"""Tests for bulk ingest services."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetReference
from app.assets.database.queries import get_reference_tags
from app.assets.scanner import build_asset_specs
from app.assets.services.bulk_ingest import SeedAssetSpec, batch_insert_seed_assets


class TestBatchInsertSeedAssets:
    def test_locked_file_logs_typed_error_without_path(self, caplog):
        locked_path = "/private/user/models/locked.safetensors"

        with (
            patch(
                "app.assets.scanner.os.stat", side_effect=PermissionError(locked_path)
            ),
            caplog.at_level(logging.WARNING),
        ):
            specs, tags, skipped = build_asset_specs([locked_path], set())

        assert specs == []
        assert tags == set()
        assert skipped == 0
        assert caplog.messages == [
            "Asset scan error: phase=discovery_stat error_type=permission_denied"
        ]
        assert locked_path not in caplog.text

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

    def test_duplicate_paths_merge_tags_before_insert(
        self, session: Session, temp_dir: Path
    ):
        """Overlapping model-folder registrations can emit the same path twice."""
        file_path = temp_dir / "shared.safetensors"
        file_path.write_bytes(b"shared model")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 12,
                "mtime_ns": 1234567890000000000,
                "info_name": "Shared Model",
                "tags": ["models", "model_type:checkpoints"],
                "fname": "shared.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            },
            {
                "abs_path": str(file_path),
                "size_bytes": 12,
                "mtime_ns": 1234567890000000000,
                "info_name": "Shared Model",
                "tags": ["models", "model_type:diffusion_models"],
                "fname": "shared.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            },
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1
        assert result.won_paths == 1
        refs = session.query(AssetReference).all()
        assert len(refs) == 1
        assert set(get_reference_tags(session, reference_id=refs[0].id)) == {
            "models",
            "model_type:checkpoints",
            "model_type:diffusion_models",
        }

    def test_duplicate_paths_are_merged_after_abspath_normalization(
        self, session: Session, temp_dir: Path, monkeypatch
    ):
        """The scanner may emit equivalent paths with different spelling."""
        file_path = temp_dir / "same-file.safetensors"
        file_path.write_bytes(b"shared model")
        monkeypatch.chdir(temp_dir)
        relative_path = file_path.name
        absolute_path = os.path.abspath(relative_path)

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": relative_path,
                "size_bytes": 12,
                "mtime_ns": 1234567890000000000,
                "info_name": "Shared Model",
                "tags": ["models", "model_type:checkpoints"],
                "fname": "same-file.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            },
            {
                "abs_path": absolute_path,
                "size_bytes": 12,
                "mtime_ns": 1234567890000000000,
                "info_name": "Shared Model",
                "tags": ["models", "model_type:diffusion_models"],
                "fname": "same-file.safetensors",
                "metadata": None,
                "hash": None,
                "mime_type": "application/safetensors",
            },
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1
        assert result.won_paths == 1
        refs = session.query(AssetReference).all()
        assert len(refs) == 1
        assert refs[0].file_path == absolute_path
        # loader_path is persisted from the spec's fname (compute_loader_path).
        assert refs[0].loader_path == "same-file.safetensors"
        assert set(get_reference_tags(session, reference_id=refs[0].id)) == {
            "models",
            "model_type:checkpoints",
            "model_type:diffusion_models",
        }

    def test_scanner_duplicate_shared_model_paths_keep_all_model_type_tags(
        self, session: Session, temp_dir: Path
    ):
        """Shared extra model roots make scanner collection emit duplicate paths."""
        shared_root = temp_dir / "shared"
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        temp_root = temp_dir / "temp"
        for directory in (shared_root, input_dir, output_dir, temp_root):
            directory.mkdir()
        file_path = shared_root / "dual_use_model.safetensors"
        file_path.write_bytes(b"shared model")

        with (
            patch("app.assets.services.path_utils.folder_paths") as mock_fp,
            patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[
                    ("checkpoints", [str(shared_root)], {".safetensors"}),
                    ("diffusion_models", [str(shared_root)], {".safetensors"}),
                ],
            ),
        ):
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_root)

            specs, tag_pool, skipped = build_asset_specs(
                paths=[str(file_path), str(file_path)],
                existing_paths=set(),
                enable_metadata_extraction=False,
                compute_hashes=False,
            )

        assert skipped == 0
        assert len(specs) == 2
        assert tag_pool == {
            "models",
            "model_type:checkpoints",
            "model_type:diffusion_models",
        }

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1
        assert result.won_paths == 1
        refs = session.query(AssetReference).all()
        assert len(refs) == 1
        assert set(get_reference_tags(session, reference_id=refs[0].id)) == {
            "models",
            "model_type:checkpoints",
            "model_type:diffusion_models",
        }

    def test_loader_path_persisted_as_null_when_fname_is_none(
        self, session: Session, temp_dir: Path
    ):
        """A file with no in-root loader path (fname=None, e.g. an orphan under
        models_root) persists loader_path as NULL rather than a synthesized value."""
        file_path = temp_dir / "orphan.bin"
        file_path.write_bytes(b"x")

        specs: list[SeedAssetSpec] = [
            {
                "abs_path": str(file_path),
                "size_bytes": 1,
                "mtime_ns": 1234567890000000000,
                "info_name": "orphan.bin",
                "tags": [],
                "fname": None,
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
        ]

        result = batch_insert_seed_assets(session, specs=specs, owner_id="")

        assert result.inserted_refs == 1
        refs = session.query(AssetReference).all()
        assert len(refs) == 1
        assert refs[0].file_path == str(file_path)
        assert refs[0].loader_path is None


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
