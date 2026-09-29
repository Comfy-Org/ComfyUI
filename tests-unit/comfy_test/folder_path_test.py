### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import errno
import pytest
import os
import shutil
import tempfile
from unittest.mock import patch
from importlib import reload

import folder_paths
import comfy.cli_args


@pytest.fixture()
def clear_folder_paths():
    # Reload the module after each test to ensure isolation
    yield
    reload(folder_paths)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def set_base_dir(monkeypatch):
    def _set_base_dir(base_dir):
        monkeypatch.setattr(comfy.cli_args.args, "base_directory", base_dir)
        reload(folder_paths)
    yield _set_base_dir
    monkeypatch.undo()
    reload(folder_paths)


def test_get_directory_by_type(clear_folder_paths):
    test_dir = "/test/dir"
    folder_paths.set_output_directory(test_dir)
    assert folder_paths.get_directory_by_type("output") == test_dir
    assert folder_paths.get_directory_by_type("invalid") is None

def test_annotated_filepath():
    assert folder_paths.annotated_filepath("test.txt") == ("test.txt", None)
    assert folder_paths.annotated_filepath("test.txt [output]") == ("test.txt", folder_paths.get_output_directory())
    assert folder_paths.annotated_filepath("test.txt [input]") == ("test.txt", folder_paths.get_input_directory())
    assert folder_paths.annotated_filepath("test.txt [temp]") == ("test.txt", folder_paths.get_temp_directory())

def test_get_annotated_filepath():
    default_dir = "/default/dir"
    # get_annotated_filepath now normalizes with os.path.abspath (part of the
    # GHSA-779p traversal hardening), so compare against the normalized form —
    # on Windows abspath also prepends the current drive letter.
    assert folder_paths.get_annotated_filepath("test.txt", default_dir) == os.path.abspath(os.path.join(default_dir, "test.txt"))
    assert folder_paths.get_annotated_filepath("test.txt [output]") == os.path.abspath(os.path.join(folder_paths.get_output_directory(), "test.txt"))

def test_add_model_folder_path_append(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_insert(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_re_add_existing_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/old_default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/old_default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/test/path", "/old_default/path"]


def test_add_model_folder_path_re_add_existing_non_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_recursive_search(temp_dir):
    os.makedirs(os.path.join(temp_dir, "subdir"))
    open(os.path.join(temp_dir, "file1.txt"), "w").close()
    open(os.path.join(temp_dir, "subdir", "file2.txt"), "w").close()

    files, dirs = folder_paths.recursive_search(temp_dir)
    assert set(files) == {"file1.txt", os.path.join("subdir", "file2.txt")}
    assert len(dirs) == 2  # temp_dir and subdir

def test_get_filename_list_survives_a_listed_folder_vanishing(temp_dir, clear_folder_paths):
    kept = os.path.join(temp_dir, "kept")
    gone = os.path.join(temp_dir, "gone")
    for path, name in ((kept, "a.safetensors"), (gone, "b.safetensors")):
        os.makedirs(path)
        open(os.path.join(path, name), "w").close()
    folder_paths.folder_names_and_paths["test_folder"] = ([kept, gone], {".safetensors"})
    assert folder_paths.get_filename_list("test_folder") == ["a.safetensors", "b.safetensors"]

    shutil.rmtree(gone)

    assert folder_paths.get_filename_list("test_folder") == ["a.safetensors"]


def test_recursive_search_skips_a_root_whose_mtime_raises_oserror(temp_dir, monkeypatch):
    """Not only FileNotFoundError: e.g. WinError 433 (device gone)."""
    open(os.path.join(temp_dir, "file.txt"), "w").close()
    real_getmtime = os.path.getmtime

    def getmtime(path):
        if path == temp_dir:
            raise OSError(errno.EINVAL, "A device which does not exist was specified", path)
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", getmtime)
    files, dirs = folder_paths.recursive_search(temp_dir)
    assert files == ["file.txt"]
    assert temp_dir not in dirs


def test_recursive_search_still_raises_a_subdirectory_oserror(temp_dir, monkeypatch):
    """Unchanged from before: e.g. a link loop (ELOOP, WinError 1921) fails the listing."""
    subdir = os.path.join(temp_dir, "subdir")
    os.makedirs(subdir)
    real_getmtime = os.path.getmtime

    def getmtime(path):
        if path == subdir:
            raise OSError(errno.ELOOP, "Too many levels of symbolic links", path)
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", getmtime)
    with pytest.raises(OSError):
        folder_paths.recursive_search(temp_dir)


def test_filter_files_extensions():
    files = ["file1.txt", "file2.jpg", "file3.png", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".txt"]) == ["file1.txt", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".jpg", ".png"]) == ["file2.jpg", "file3.png"]
    assert folder_paths.filter_files_extensions(files, []) == files

@patch("folder_paths.recursive_search")
@patch("folder_paths.folder_names_and_paths")
def test_get_filename_list(mock_folder_names_and_paths, mock_recursive_search):
    mock_folder_names_and_paths.__getitem__.return_value = (["/test/path"], {".txt"})
    mock_recursive_search.return_value = (["file1.txt", "file2.jpg"], {})
    assert folder_paths.get_filename_list("test_folder") == ["file1.txt"]

def test_get_save_image_path(temp_dir):
    with patch("folder_paths.output_directory", temp_dir):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("test", temp_dir, 100, 100)
        assert os.path.samefile(full_output_folder, temp_dir)
        assert filename == "test"
        assert counter == 1
        assert subfolder == ""
        assert filename_prefix == "test"


def test_base_path_changes(set_base_dir):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir(test_dir)

    assert folder_paths.base_path == test_dir
    assert folder_paths.models_dir == os.path.join(test_dir, "models")
    assert folder_paths.input_directory == os.path.join(test_dir, "input")
    assert folder_paths.output_directory == os.path.join(test_dir, "output")
    assert folder_paths.temp_directory == os.path.join(test_dir, "temp")
    assert folder_paths.user_directory == os.path.join(test_dir, "user")

    assert os.path.join(test_dir, "custom_nodes") in folder_paths.get_folder_paths("custom_nodes")

    for name in ["checkpoints", "loras", "vae", "configs", "embeddings", "controlnet", "classifiers"]:
        assert folder_paths.get_folder_paths(name)[0] == os.path.join(test_dir, "models", name)


def test_base_path_change_clears_old(set_base_dir):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir(test_dir)

    assert len(folder_paths.get_folder_paths("custom_nodes")) == 1

    single_model_paths = [
        "checkpoints",
        "loras",
        "vae",
        "configs",
        "clip_vision",
        "style_models",
        "diffusers",
        "vae_approx",
        "gligen",
        "upscale_models",
        "embeddings",
        "hypernetworks",
        "photomaker",
        "classifiers",
    ]
    for name in single_model_paths:
        assert len(folder_paths.get_folder_paths(name)) == 1

    for name in ["controlnet", "diffusion_models", "text_encoders"]:
        assert len(folder_paths.get_folder_paths(name)) == 2


def test_models_directory_cli_and_getters(temp_dir, monkeypatch):
    try:
        monkeypatch.setattr(comfy.cli_args.args, "models_directory", temp_dir)
        reload(folder_paths)

        assert folder_paths.models_dir == os.path.abspath(temp_dir)

        with pytest.raises(Exception):
            comfy.cli_args.is_valid_directory(os.path.join(temp_dir, "non_existent_folder_path"))
    finally:
        monkeypatch.undo()
        reload(folder_paths)
