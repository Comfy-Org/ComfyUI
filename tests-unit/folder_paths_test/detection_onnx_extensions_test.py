import os
import tempfile

import folder_paths


def test_detection_folder_allows_onnx_extension():
    _, extensions = folder_paths.folder_names_and_paths["detection"]
    assert ".onnx" in extensions


def test_detection_folder_lists_onnx_files():
    with tempfile.TemporaryDirectory() as detection_dir:
        onnx_path = os.path.join(detection_dir, "vitpose-l-wholebody.onnx")
        with open(onnx_path, "wb") as onnx_file:
            onnx_file.write(b"onnx")

        folder_paths.add_model_folder_path("detection", detection_dir, is_default=True)
        folder_paths.filename_list_cache.pop("detection", None)

        filenames = folder_paths.get_filename_list("detection")
        assert "vitpose-l-wholebody.onnx" in filenames
        assert folder_paths.get_full_path("detection", "vitpose-l-wholebody.onnx") == onnx_path
