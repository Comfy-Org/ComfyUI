from unittest.mock import patch

from sqlalchemy import select

from app.assets.database.models import AssetContent
from app.assets.scanner import build_asset_specs, seed_asset_specs


def test_drifting_file_never_reaches_seed(session, temp_dir):
    path = temp_dir / "drifting.bin"
    path.write_bytes(b"still downloading")

    with (
        patch("folder_paths.get_input_directory", return_value=str(temp_dir)),
        patch("app.assets.scanner._two_stat_admit", return_value=([], [str(path)])),
    ):
        specs, _, _ = build_asset_specs([str(path)], set(), enable_metadata_extraction=False)
        seed_asset_specs(session, specs)

    assert list(session.scalars(select(AssetContent))) == []
