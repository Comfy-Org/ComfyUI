import pytest
import os
import tempfile
from folder_paths import get_input_subfolders, set_input_directory

@pytest.fixture(scope="module")
def mock_folder_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested folder structure
        folders = [
            "folder1",
            "folder1/subfolder1",
            "folder1/subfolder2",
            "folder2",
            "folder2/deep",
            "folder2/deep/nested",
            "empty_folder"
        ]

        # Create the folders
        for folder in folders:
            os.makedirs(os.path.join(temp_dir, folder))

        # Add some files to test they're not included
        with open(os.path.join(temp_dir, "root_file.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "folder1", "test.txt"), "w") as f:
            f.write("test")

        set_input_directory(temp_dir)
        yield temp_dir


def test_gets_all_folders(mock_folder_structure):
    folders = get_input_subfolders()
    expected = ["folder1", "folder1/subfolder1", "folder1/subfolder2",
                "folder2", "folder2/deep", "folder2/deep/nested", "empty_folder"]
    assert sorted(folders) == sorted(expected)


def test_handles_nonexistent_input_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent = os.path.join(temp_dir, "nonexistent")
        set_input_directory(nonexistent)
        assert get_input_subfolders() == []


def test_empty_input_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        set_input_directory(temp_dir)
        assert get_input_subfolders() == []  # Empty since we don't include root


def test_supported_pt_extensions():
    from folder_paths import supported_pt_extensions
    assert ".onnx" in supported_pt_extensions
