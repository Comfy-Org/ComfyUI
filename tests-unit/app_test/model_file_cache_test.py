import os

from app.model_manager import ModelFileManager


class TestModelFileListCache:
    def test_cache_is_reused_when_unchanged(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "m.safetensors").write_bytes(b"x")

        mgr = ModelFileManager()
        out = mgr.recursive_search_models_(str(tmp_path), 0)
        mgr.set_cache(str(tmp_path), out)

        # nothing changed, so the cached result must be returned instead of None
        assert mgr.cache_model_file_list_(str(tmp_path)) is out

    def test_cache_invalidated_when_top_folder_changes(self, tmp_path):
        mgr = ModelFileManager()
        out = mgr.recursive_search_models_(str(tmp_path), 0)
        mgr.set_cache(str(tmp_path), out)
        assert mgr.cache_model_file_list_(str(tmp_path)) is out

        # a file added directly under the top folder changes its mtime, which must
        # invalidate the cache (the top folder is tracked, not just subfolders)
        bumped = os.path.getmtime(str(tmp_path)) + 100
        os.utime(str(tmp_path), (bumped, bumped))
        assert mgr.cache_model_file_list_(str(tmp_path)) is None
