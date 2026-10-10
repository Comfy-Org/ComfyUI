import os
import shutil
import tempfile

import pytest

import folder_paths


@pytest.fixture
def model_folder():
    """Register a temporary model category with a tracked subfolder."""
    folder_name = "cache_invalidation_test_cat"
    with tempfile.TemporaryDirectory() as base:
        category = os.path.join(base, "category")
        subfolder = os.path.join(category, "sub")
        os.makedirs(subfolder)
        open(os.path.join(category, "a.safetensors"), "w").close()
        open(os.path.join(subfolder, "b.safetensors"), "w").close()

        folder_paths.folder_names_and_paths[folder_name] = (
            [category],
            {".safetensors"},
        )
        try:
            yield folder_name, category, subfolder
        finally:
            folder_paths.folder_names_and_paths.pop(folder_name, None)
            folder_paths.filename_list_cache.pop(folder_name, None)
            folder_paths.cache_helper.clear()


def test_rebuilds_when_tracked_subfolder_deleted(model_folder):
    folder_name, _category, subfolder = model_folder

    # Populate the filename cache, which records the mtime of every subfolder.
    initial = folder_paths.get_filename_list(folder_name)
    assert sorted(initial) == ["a.safetensors", "sub/b.safetensors"]

    # Remove a tracked subfolder at runtime (e.g. user deletes a model folder).
    shutil.rmtree(subfolder)

    # The cache must be treated as stale and rebuilt, not crash with
    # FileNotFoundError when probing the deleted folder's mtime.
    refreshed = folder_paths.get_filename_list(folder_name)
    assert refreshed == ["a.safetensors"]
