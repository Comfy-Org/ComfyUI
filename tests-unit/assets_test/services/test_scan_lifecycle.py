from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select

from app.assets.database.models import AssetContent
from app.assets.scanner import build_asset_specs, seed_asset_specs, sync_prefixes_with_filesystem


def _scan(session, root: Path) -> int:
    paths = [str(path) for path in root.iterdir()]
    specs, _, _ = build_asset_specs(paths, set(), enable_metadata_extraction=False)
    return seed_asset_specs(session, specs)


def test_e2e_scan_seed_detect_prune(session, temp_dir: Path):
    root = temp_dir / "input"
    root.mkdir()
    removed = root / "removed.bin"
    edited = root / "edited.bin"
    removed.write_bytes(b"removed")
    edited.write_bytes(b"old")
    with patch("folder_paths.get_input_directory", return_value=str(root)):
        assert _scan(session, root) == 2
        removed.unlink()
        edited.write_bytes(b"replacement")
        (root / "partial.part").write_bytes(b"partial")
        with patch("app.assets.scanner.mode.hashing_enabled", return_value=False):
            sync_prefixes_with_filesystem(session, [str(root)])
            _scan(session, root)
    session.commit()
    contents = list(session.scalars(select(AssetContent)))
    assert len(contents) == 3
    assert len([content for content in contents if content.is_missing]) == 2


def test_second_scan_idempotent(session, temp_dir: Path):
    root = temp_dir / "input"
    root.mkdir()
    (root / "stable.bin").write_bytes(b"stable")
    with patch("folder_paths.get_input_directory", return_value=str(root)):
        assert _scan(session, root) == 1
        assert _scan(session, root) == 0
