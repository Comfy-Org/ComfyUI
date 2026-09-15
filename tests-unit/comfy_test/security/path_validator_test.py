import os
import sys
import tempfile
from pathlib import Path

import pytest

from comfy.security.path_validator import resolve_safe_path


@pytest.fixture
def base():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp).resolve()


def test_simple_filename_inside_base(base):
    result = resolve_safe_path(base, "foo.png")
    assert result == base / "foo.png"


def test_subfolder_plus_filename(base):
    result = resolve_safe_path(base, "subdir", "foo.png")
    assert result == base / "subdir" / "foo.png"


def test_empty_subfolder(base):
    result = resolve_safe_path(base, "", "foo.png")
    assert result == base / "foo.png"


def test_dotdot_segment_rejected(base):
    assert resolve_safe_path(base, "..", "etc") is None
    assert resolve_safe_path(base, "../etc") is None
    assert resolve_safe_path(base, "sub/../..", "etc") is None


def test_absolute_user_path_rejected(base):
    if sys.platform == "win32":
        assert resolve_safe_path(base, "C:\\Windows\\System32") is None
    else:
        assert resolve_safe_path(base, "/etc/passwd") is None


def test_null_byte_rejected(base):
    assert resolve_safe_path(base, "foo\x00.png") is None
    assert resolve_safe_path(base, "sub\x00", "foo.png") is None


def test_dotdot_inside_base_is_allowed(base):
    (base / "sub").mkdir()
    result = resolve_safe_path(base, "sub", "..", "foo.png")
    assert result == base / "foo.png"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink permissions on Windows CI")
def test_symlink_escape_rejected(base):
    outside = Path(tempfile.mkdtemp())
    try:
        link = base / "escape"
        link.symlink_to(outside)
        assert resolve_safe_path(base, "escape", "secret") is None
    finally:
        # cleanup outside dir
        try:
            outside.rmdir()
        except OSError:
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="symlink permissions on Windows CI")
def test_symlink_inside_base_is_allowed(base):
    target = base / "real"
    target.mkdir()
    link = base / "alias"
    link.symlink_to(target)
    result = resolve_safe_path(base, "alias", "foo.png")
    assert result == (target / "foo.png").resolve()


def test_base_canonicalizes_via_resolve(base):
    # Pass an unresolved base (with trailing slash) and verify containment still works.
    unresolved = str(base) + os.sep
    assert resolve_safe_path(unresolved, "foo.png") == base / "foo.png"


def test_dot_segments_are_normalized(base):
    assert resolve_safe_path(base, ".", "foo.png") == base / "foo.png"
    assert resolve_safe_path(base, "./sub/./foo.png") == base / "sub" / "foo.png"


def test_path_objects_accepted(base):
    assert resolve_safe_path(base, Path("foo.png")) == base / "foo.png"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path separators")
def test_windows_backslash_traversal_rejected(base):
    assert resolve_safe_path(base, "..\\etc") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific UNC paths")
def test_windows_unc_path_rejected(base):
    assert resolve_safe_path(base, "\\\\server\\share\\file") is None


def test_returns_path_instance(base):
    result = resolve_safe_path(base, "foo.png")
    assert isinstance(result, Path)
    assert result.is_absolute()
