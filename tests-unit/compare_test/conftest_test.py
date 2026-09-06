import os
import tempfile

import pytest

from tests.compare.conftest import gather_file_basenames


class TestGatherFileBasenames:
    def test_returns_png_basenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.png"), "w").close()
            open(os.path.join(tmpdir, "b.txt"), "w").close()
            open(os.path.join(tmpdir, "c.png"), "w").close()
            assert sorted(gather_file_basenames(tmpdir)) == ["a.png", "c.png"]

    def test_missing_directory_returns_empty_list(self):
        assert gather_file_basenames("/nonexistent/path/12345") == []

    def test_non_string_input_raises(self):
        with pytest.raises(AssertionError):
            gather_file_basenames(123)  # type: ignore[arg-type]
