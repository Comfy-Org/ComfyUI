import os
from pathlib import Path

import pytest

from app.assets.helpers import path_prefix_matcher

from .path_prefix_cases import anchor_case_paths, prefix_case_paths


def _is_relative_to_any(path: str, prefixes: list[str]) -> bool:
    candidate = Path(os.path.abspath(path))
    return any(candidate.is_relative_to(os.path.abspath(prefix)) for prefix in prefixes)


def test_matches_path_is_relative_to(tmp_path):
    root = os.path.abspath(str(tmp_path / "root"))
    other = os.path.abspath(str(tmp_path / "models" / "checkpoints"))
    for prefixes in ([root], [other, root], [root + os.sep]):
        matches = path_prefix_matcher(prefixes)
        for path, _ in prefix_case_paths(root):
            assert matches(path) == _is_relative_to_any(path, prefixes), (path, prefixes)


def test_no_prefixes_matches_nothing(tmp_path):
    assert path_prefix_matcher([])(str(tmp_path)) is False


def test_filesystem_root_contains_everything(tmp_path):
    path = os.path.abspath(str(tmp_path))
    assert path_prefix_matcher([Path(path).anchor])(path) is True


@pytest.mark.skipif(os.sep != "/", reason="POSIX anchors; Windows drives and UNC shares normalize differently")
def test_matches_path_is_relative_to_across_posix_anchors():
    for path, prefix, expected in anchor_case_paths():
        assert _is_relative_to_any(path, [prefix]) is expected, (path, prefix)
        assert path_prefix_matcher([prefix])(path) is expected, (path, prefix)
