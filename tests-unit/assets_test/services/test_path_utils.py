"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import get_asset_category_and_relative_path


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_dir = root_path / "models" / "checkpoints"
        for d in (input_dir, output_dir, temp_dir, models_dir):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[("checkpoints", [str(models_dir)])],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models": models_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")
