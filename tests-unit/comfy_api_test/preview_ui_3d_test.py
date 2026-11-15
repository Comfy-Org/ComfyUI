import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import folder_paths
from comfy_api.latest import IO, UI, Types
from comfy_extras.nodes_load_3d import Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud
from comfy_extras.nodes_save_3d import Save3DAdvanced, SaveGaussianSplat, SaveGLB, SavePointCloud


def test_saved_3d_models_reports_files_and_viewer_state():
    saved = UI.SavedResult("model_00001.glb", "3d", IO.FolderType.output)
    ui = UI.Saved3DModels([saved], {"fov": 35}, [{"scale": 1}])

    assert ui.as_dict() == {
        "3d": [{"filename": "model_00001.glb", "subfolder": "3d", "type": "output"}],
        "camera_info": [{"fov": 35}],
        "model_3d_info": [{"scale": 1}],
    }


def test_saved_3d_models_defaults_to_no_viewer_state():
    ui = UI.Saved3DModels([UI.SavedResult("m.glb", "", IO.FolderType.output)])

    assert ui.as_dict()["camera_info"] == [None]
    assert ui.as_dict()["model_3d_info"] == []


def test_preview_ui_3d_advanced_splits_the_path_into_a_saved_result():
    ui = UI.PreviewUI3DAdvanced("3d/model.glb", {"fov": 35}, [])

    assert ui.as_dict()["3d"] == [{"filename": "model.glb", "subfolder": "3d", "type": "output"}]
    assert ui.as_dict()["camera_info"] == [{"fov": 35}]


@pytest.mark.parametrize("folder_type", [IO.FolderType.temp, "temp"])
def test_preview_ui_3d_advanced_uses_the_folder_type(folder_type):
    ui = UI.PreviewUI3DAdvanced("preview.glb", None, [], folder_type=folder_type)

    assert ui.as_dict()["3d"] == [{"filename": "preview.glb", "subfolder": "", "type": "temp"}]


@pytest.mark.parametrize(
    ("node_cls", "file_format", "prefix"),
    [
        (Preview3DAdvanced, "glb", "preview3d_advanced_"),
        (Preview3DAdvanced, "obj", "preview3d_advanced_"),
        (PreviewGaussianSplat, "spz", "preview_splat_"),
        (PreviewPointCloud, "ply", "preview_pointcloud_"),
    ],
)
def test_preview_nodes_report_the_temp_file_they_wrote(tmp_path, node_cls, file_format, prefix):
    model = Types.File3D(BytesIO(b"model-bytes"), file_format)

    with patch.object(folder_paths, "get_temp_directory", return_value=str(tmp_path)):
        output = node_cls.execute(model, viewport_state={}, width=1, height=1)

    (item,) = output.ui.as_dict()["3d"]
    assert item["type"] == "temp"
    assert item["subfolder"] == ""
    assert item["filename"].startswith(prefix) and item["filename"].endswith(f".{file_format}")
    assert os.path.isfile(os.path.join(tmp_path, item["filename"]))


@pytest.mark.parametrize(
    ("node_cls", "file_format"),
    [(Save3DAdvanced, "glb"), (SaveGaussianSplat, "spz"), (SavePointCloud, "ply")],
)
def test_save_nodes_report_the_output_file_they_wrote(tmp_path, node_cls, file_format):
    model = Types.File3D(BytesIO(b"model-bytes"), file_format)

    with patch.object(folder_paths, "get_output_directory", return_value=str(tmp_path)):
        output = node_cls.execute(model, viewport_state={}, width=1, height=1, filename_prefix="3d/ComfyUI")

    ui = output.ui.as_dict()
    (item,) = ui["3d"]
    assert item["type"] == "output"
    assert item["subfolder"] == "3d"
    assert item["filename"].startswith("ComfyUI_") and item["filename"].endswith(f".{file_format}")
    assert set(ui) == {"3d", "camera_info", "model_3d_info"}
    assert os.path.isfile(os.path.join(tmp_path, "3d", item["filename"]))


def test_save_glb_reports_a_file3d_input_in_the_same_shape(tmp_path):
    model = Types.File3D(BytesIO(b"model-bytes"), "glb")

    with (
        patch.object(folder_paths, "get_output_directory", return_value=str(tmp_path)),
        patch.object(SaveGLB, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None)),
    ):
        output = SaveGLB.execute(model, filename_prefix="3d/ComfyUI")

    ui = output.ui.as_dict()
    (item,) = ui["3d"]
    assert item == {"filename": item["filename"], "subfolder": "3d", "type": "output"}
    assert item["filename"].endswith("_.glb")
    assert ui["camera_info"] == [None]
    assert os.path.isfile(os.path.join(tmp_path, "3d", item["filename"]))
