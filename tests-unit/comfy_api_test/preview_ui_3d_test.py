import os
from io import BytesIO
from unittest.mock import patch

import pytest

import folder_paths
from comfy_api.latest import IO, UI, Types
from comfy_extras.nodes_load_3d import Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud
from comfy_extras.nodes_save_3d import Save3DAdvanced, SaveGaussianSplat, SavePointCloud


def test_preview_ui_3d_advanced_keeps_bare_path_by_default():
    ui = UI.PreviewUI3DAdvanced("3d/model.glb", {"fov": 35}, [])

    assert ui.as_dict() == {"result": ["3d/model.glb", {"fov": 35}, []]}


def test_preview_ui_3d_advanced_reports_saved_result_as_standard_output_item():
    saved = UI.SavedResult("model_00001.glb", "3d", IO.FolderType.output)
    ui = UI.PreviewUI3DAdvanced("3d/model_00001.glb", None, [], saved_result=saved)

    assert ui.as_dict() == {
        "result": ["3d/model_00001.glb", None, []],
        "3d": [{"filename": "model_00001.glb", "subfolder": "3d", "type": "output"}],
    }


@pytest.mark.parametrize("folder_type", [IO.FolderType.temp, "temp"])
def test_preview_ui_3d_advanced_annotates_folder_type(folder_type):
    ui = UI.PreviewUI3DAdvanced("preview.glb", None, [], folder_type=folder_type)

    result = ui.as_dict()["result"]

    assert result[0] == "preview.glb [temp]"
    name, base_dir = folder_paths.annotated_filepath(result[0])
    assert name == "preview.glb"
    assert base_dir == folder_paths.get_temp_directory()


@pytest.mark.parametrize(
    ("node_cls", "file_format", "prefix"),
    [
        (Preview3DAdvanced, "glb", "preview3d_advanced_"),
        (Preview3DAdvanced, "obj", "preview3d_advanced_"),
        (PreviewGaussianSplat, "spz", "preview_splat_"),
        (PreviewPointCloud, "ply", "preview_pointcloud_"),
    ],
)
def test_preview_nodes_report_where_they_wrote_the_file(tmp_path, node_cls, file_format, prefix):
    model = Types.File3D(BytesIO(b"model-bytes"), file_format)

    with patch.object(folder_paths, "get_temp_directory", return_value=str(tmp_path)):
        output = node_cls.execute(model, viewport_state={}, width=1, height=1)
        reported = output.ui.as_dict()["result"][0]
        name, base_dir = folder_paths.annotated_filepath(reported)

    assert reported.endswith(f".{file_format} [temp]")
    assert name.startswith(prefix)
    assert base_dir == str(tmp_path)
    assert os.path.isfile(os.path.join(tmp_path, name))


@pytest.mark.parametrize(
    ("node_cls", "file_format"),
    [(Save3DAdvanced, "glb"), (SaveGaussianSplat, "spz"), (SavePointCloud, "ply")],
)
def test_save_nodes_report_the_saved_file_as_a_3d_output_item(tmp_path, node_cls, file_format):
    model = Types.File3D(BytesIO(b"model-bytes"), file_format)

    with patch.object(folder_paths, "get_output_directory", return_value=str(tmp_path)):
        output = node_cls.execute(model, viewport_state={}, width=1, height=1, filename_prefix="3d/ComfyUI")

    ui = output.ui.as_dict()
    (item,) = ui["3d"]
    assert item["subfolder"] == "3d"
    assert item["type"] == "output"
    assert item["filename"].startswith("ComfyUI_") and item["filename"].endswith(f".{file_format}")
    assert ui["result"][0] == f"3d/{item['filename']}"
    assert os.path.isfile(os.path.join(tmp_path, "3d", item["filename"]))
