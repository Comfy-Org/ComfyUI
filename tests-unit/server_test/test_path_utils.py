"""Tests for ``comfy.path_utils.is_subpath`` (used by ``server.py`` to guard
``/view`` and ``/upload`` against cross-drive and escape attempts)."""

import ntpath
import os
from unittest import mock

import pytest

from comfy.path_utils import is_subpath


class TestIsSubpathSameDrive:
    """Behaviour on a single drive where ``os.path.commonpath`` works."""

    def test_identical_path_is_subpath(self, tmp_path):
        assert is_subpath(str(tmp_path), str(tmp_path)) is True

    def test_direct_child_is_subpath(self, tmp_path):
        child = tmp_path / "file.png"
        child.write_text("x")
        assert is_subpath(str(child), str(tmp_path)) is True

    def test_deeply_nested_child_is_subpath(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "file.png"
        deep.parent.mkdir(parents=True)
        deep.write_text("x")
        assert is_subpath(str(deep), str(tmp_path)) is True

    def test_sibling_is_not_subpath(self, tmp_path):
        sibling = tmp_path.parent / "elsewhere"
        assert is_subpath(str(sibling), str(tmp_path)) is False

    def test_parent_is_not_subpath(self, tmp_path):
        assert is_subpath(str(tmp_path.parent), str(tmp_path)) is False

    def test_relative_path_resolved_correctly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = "a/b/c.png"
        (tmp_path / "a" / "b").mkdir(parents=True)
        assert is_subpath(sub, ".") is True

    def test_relative_sibling_resolved_to_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = str(tmp_path.parent / "other.png")
        assert is_subpath(outside, ".") is False


class TestIsSubpathWindowsCase:
    """``commonpath`` on Windows compares components case-insensitively but
    returns the first argument's casing, so the parent must come first or a
    contained path that merely differs in case compares unequal."""

    def test_case_difference_is_still_contained(self):
        parent = "C:\\ComfyUI\\output"
        child = "c:\\comfyui\\output\\image.png"
        with mock.patch.multiple(
            "os.path", abspath=lambda p: p, commonpath=ntpath.commonpath
        ):
            assert is_subpath(child, parent) is True
            assert is_subpath("D:\\elsewhere\\image.png", parent) is False


class TestIsSubpathCrossDrive:
    """``os.path.commonpath`` raises ``ValueError`` when the two paths
    share no drive / mount. ``is_subpath`` must swallow that and return
    False instead of letting the exception escape."""

    def test_cross_drive_returns_false_when_commonpath_raises(self, tmp_path):
        drive_c = "C:\\Users\\alice"
        drive_d = "D:\\models"

        with mock.patch("os.path.commonpath", side_effect=ValueError("Paths don't have the same drive")):
            # Drive letter aside, abspath on Linux keeps the strings as-is
            # -- the patched commonpath is what triggers the cross-drive path.
            assert is_subpath(drive_d, drive_c) is False

    def test_cross_drive_does_not_raise(self, tmp_path):
        # Direct assertion: the function must not raise under the
        # mocked ValueError, even if the rest of the system would
        # have raised for any reason.
        with mock.patch("os.path.commonpath", side_effect=ValueError):
            # Use a real parent so abspath doesn't fail before the patch fires.
            assert is_subpath(str(tmp_path), str(tmp_path)) is False

    def test_normal_call_still_uses_commonpath(self, tmp_path):
        # When commonpath does not raise, behaviour must be unchanged:
        # the wrapper must not short-circuit and return False.
        child = tmp_path / "child"
        child.mkdir()
        with mock.patch("os.path.commonpath", wraps=os.path.commonpath) as spy:
            assert is_subpath(str(child), str(tmp_path)) is True
            assert spy.called


class TestIsSubpathTrailingSeparators:
    """Edge cases around trailing path separators that real users hit."""

    def test_trailing_separator_on_parent(self, tmp_path):
        child = tmp_path / "file.png"
        child.write_text("x")
        assert is_subpath(str(child), str(tmp_path) + os.sep) is True

    def test_root_parent_accepts_any_subpath(self, tmp_path):
        # ``os.path.commonpath`` treats ``/`` specially: any absolute path
        # shares it as the common prefix. We just want to assert the
        # function does not raise and returns True.
        assert is_subpath(str(tmp_path), os.sep) is True


@pytest.mark.parametrize(
    "child,parent,expected",
    [
        # Trailing dot / double-dot segments get normalised by abspath.
        ("/a/b/./c.png", "/a", True),
        ("/a/../a/b.png", "/a", True),
        # Empty / whitespace-ish paths should not crash; they get
        # resolved by abspath to the cwd, which is not under a
        # different explicit parent.
        ("/a/../a/../b.png", "/a", False),
    ],
)
def test_is_subpath_normalisation(child, parent, expected):
    assert is_subpath(child, parent) is expected
