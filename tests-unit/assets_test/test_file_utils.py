import os
import sys

import pytest

from app.assets.services.file_utils import is_visible, list_files_recursively


class TestIsVisible:
    def test_visible_file(self):
        assert is_visible("file.txt") is True

    def test_hidden_file(self):
        assert is_visible(".hidden") is False

    def test_hidden_directory(self):
        assert is_visible(".git") is False

    def test_visible_directory(self):
        assert is_visible("src") is True

    def test_dotdot_is_hidden(self):
        assert is_visible("..") is False

    def test_dot_is_hidden(self):
        assert is_visible(".") is False


class TestListFilesRecursively:
    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / "visible.txt").write_text("a")
        (tmp_path / ".hidden").write_text("b")

        result = list_files_recursively(str(tmp_path))

        assert len(result) == 1
        assert result[0].endswith("visible.txt")

    def test_skips_hidden_directories(self, tmp_path):
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("a")

        visible_dir = tmp_path / "visible_dir"
        visible_dir.mkdir()
        (visible_dir / "file.txt").write_text("b")

        result = list_files_recursively(str(tmp_path))

        assert len(result) == 1
        assert "visible_dir" in result[0]
        assert ".hidden_dir" not in result[0]

    def test_empty_directory(self, tmp_path):
        result = list_files_recursively(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self, tmp_path):
        result = list_files_recursively(str(tmp_path / "nonexistent"))
        assert result == []

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_follows_symlinked_directories(self, tmp_path):
        target = tmp_path / "real_dir"
        target.mkdir()
        (target / "model.safetensors").write_text("data")

        root = tmp_path / "root"
        root.mkdir()
        (root / "link").symlink_to(target)

        result = list_files_recursively(str(root))

        assert len(result) == 1
        assert result[0].endswith("model.safetensors")
        assert "link" in result[0]

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_follows_symlinked_files(self, tmp_path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")

        root = tmp_path / "root"
        root.mkdir()
        (root / "link.txt").symlink_to(real_file)

        result = list_files_recursively(str(root))

        assert len(result) == 1
        assert result[0].endswith("link.txt")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_circular_symlinks_do_not_loop(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        (dir_a / "file.txt").write_text("a")
        # a/b -> a  (circular)
        (dir_a / "b").symlink_to(dir_a)

        result = list_files_recursively(str(dir_a))

        assert len(result) == 1
        assert result[0].endswith("file.txt")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_mutual_circular_symlinks(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "file_a.txt").write_text("a")
        (dir_b / "file_b.txt").write_text("b")
        # a/link_b -> b and b/link_a -> a
        (dir_a / "link_b").symlink_to(dir_b)
        (dir_b / "link_a").symlink_to(dir_a)

        result = list_files_recursively(str(dir_a))
        basenames = sorted(os.path.basename(p) for p in result)

        assert "file_a.txt" in basenames
        assert "file_b.txt" in basenames
