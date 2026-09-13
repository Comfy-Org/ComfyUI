"""Tests for how _build_asset_response derives the response `loader_path`.

Guards the persist-and-read contract: the response reads the stored
`loader_path` verbatim, with no read-time recomputation. Like tags, the
value is a seed-time derivative healed by the scan lifecycle.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from app.assets.api.routes import _build_asset_response
from app.assets.services.schemas import AssetDetailResult, ReferenceData

_TS = datetime(2024, 1, 1, 0, 0, 0)


def _make_result(
    *, file_path: str | None, loader_path: str | None
) -> AssetDetailResult:
    ref = ReferenceData(
        id="ref-1",
        name="model.safetensors",
        file_path=file_path,
        loader_path=loader_path,
        user_metadata=None,
        preview_id=None,
        created_at=_TS,
        updated_at=_TS,
        last_access_time=_TS,
    )
    return AssetDetailResult(ref=ref, asset=None, tags=[])


def test_uses_persisted_loader_path_without_recomputing():
    """A stored loader_path is returned verbatim, not re-derived from file_path.

    The sentinel value could never be produced by compute_loader_path for this
    file_path, so seeing it in the response proves the stored column is read.
    """
    result = _make_result(
        file_path="/unmatched/root/model.safetensors",
        loader_path="SENTINEL/stored.safetensors",
    )

    resp = _build_asset_response(result, {})

    assert resp.loader_path == "SENTINEL/stored.safetensors"


def test_null_stored_loader_path_is_served_as_null(tmp_path: Path):
    """No read-time recomputation: a NULL column is served as null even when
    the path would resolve."""
    models = tmp_path / "models"
    ckpt = models / "checkpoints"
    ckpt.mkdir(parents=True)
    f = ckpt / "bar.safetensors"
    f.touch()

    with patch("app.assets.services.path_utils.folder_paths") as mock_fp, patch(
        "app.assets.services.path_utils.get_comfy_models_folders",
        return_value=[("checkpoints", [str(ckpt)], {".safetensors"})],
    ):
        mock_fp.get_input_directory.return_value = str(tmp_path / "in")
        mock_fp.get_output_directory.return_value = str(tmp_path / "out")
        mock_fp.get_temp_directory.return_value = str(tmp_path / "tmp")
        mock_fp.models_dir = str(models)

        result = _make_result(file_path=str(f), loader_path=None)
        resp = _build_asset_response(result, {})

        assert resp.loader_path is None
        assert resp.display_name == "checkpoints/bar.safetensors"


def test_all_path_fields_null_without_file_path():
    """API-created / hash-only references (no file_path) expose no paths."""
    result = _make_result(file_path=None, loader_path=None)

    resp = _build_asset_response(result, {})

    assert resp.loader_path is None
    assert resp.display_name is None
