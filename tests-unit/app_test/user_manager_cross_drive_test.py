"""Regression tests for #15820 — a userdata path on another Windows drive.

`get_request_user_filepath()` joins the requested file onto the user root and
then compares them with `os.path.commonpath()`, which raises

    ValueError: Paths don't have the same drive

when the two paths sit on different Windows drives. Requesting
`/userdata/C:%5CWindows%5Ctemp%5Ctest.txt` from an install whose user
directory is on `D:` therefore produced an unhandled 500 rather than the
intended rejection. Measured on the pre-fix build, user directory on `D:`:

    'C:\\\\Windows\\\\temp\\\\test.txt'      -> ValueError: Paths don't have the same drive
    'C:%5CWindows%5Ctemp%5Ctest.txt'  -> ValueError: Paths don't have the same drive
    '../escape.json'                  -> None      (rejected, as it should be)
    'sub/ok.json'                     -> <path inside the user directory>

The cross-drive case is Windows-only. To keep the guard covered on POSIX CI as
well, the tests below drive the same ValueError out of `os.path.commonpath()`
by call position rather than by comparing drive letters: on POSIX a leading
`Z:` is an ordinary filename character, so the join lands inside the user
directory and a drive-letter comparison would never fire. One test also does it
for real on Windows when a second drive exists.
"""

import os
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import folder_paths
from app.user_manager import UserManager


@pytest.fixture
def user_manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        original = folder_paths.get_user_directory()
        folder_paths.set_user_directory(temp_dir)
        with patch("app.user_manager.args") as mock_args:
            mock_args.multi_user = False
            manager = UserManager()
            manager.users = {"default": "default"}
            yield manager
        folder_paths.set_user_directory(original)


@pytest.fixture
def request_():
    request = MagicMock()
    request.headers = {}
    return request


@contextmanager
def commonpath_raising_on_the_user_root_check():
    """Raise a drive mismatch from the `(user_root, path)` comparison only.

    Deciding when to raise by comparing drive letters would fire on Windows
    only: on POSIX a leading `Z:` is an ordinary filename character, so the
    join lands inside the user directory and nothing raises. Keying on the call
    instead keeps the guard covered on every platform.

    The earlier `(root_dir, user_root)` comparison in the same function keeps
    working: both of those paths are derived from the user directory, so a real
    drive mismatch cannot occur there.
    """
    real_commonpath = os.path.commonpath
    calls = []

    def fake_commonpath(paths):
        calls.append(paths)
        if len(calls) == 1:
            return real_commonpath(paths)
        raise ValueError("Paths don't have the same drive")

    with patch("os.path.commonpath", side_effect=fake_commonpath):
        yield calls


def test_cross_drive_path_is_rejected_not_raised(user_manager, request_):
    """A drive mismatch must return None, the same as any other escape."""
    with commonpath_raising_on_the_user_root_check() as calls:
        result = user_manager.get_request_user_filepath(
            request_, "Z:\\Windows\\temp\\test.txt", create_dir=False
        )

    assert len(calls) == 2, "the containment check under test was never reached"
    assert result is None


def test_url_encoded_cross_drive_path_is_rejected(user_manager, request_):
    """The reported request shape: the drive letter arrives percent-encoded."""
    with commonpath_raising_on_the_user_root_check() as calls:
        result = user_manager.get_request_user_filepath(
            request_, "Z:%5CWindows%5Ctemp%5Ctest.txt", create_dir=False
        )

    assert len(calls) == 2, "the containment check under test was never reached"
    assert result is None


def test_ordinary_paths_are_unaffected(user_manager, request_):
    """The guard must not change the two outcomes that already worked."""
    inside = user_manager.get_request_user_filepath(
        request_, "sub/workflow.json", create_dir=False
    )
    assert inside is not None
    assert inside.endswith(os.path.join("sub", "workflow.json"))

    assert (
        user_manager.get_request_user_filepath(request_, "../escape.json", create_dir=False)
        is None
    )
    assert user_manager.get_request_user_filepath(request_, None, create_dir=False) is not None


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows-only")
def test_real_cross_drive_path_on_windows(user_manager, request_):
    """Same case again without patching, when a second real drive exists."""
    user_drive = os.path.splitdrive(folder_paths.get_user_directory())[0].upper()
    other = next(
        (
            f"{letter}:"
            for letter in "CDEFGH"
            if f"{letter}:" != user_drive and os.path.exists(f"{letter}:\\")
        ),
        None,
    )
    if other is None:
        pytest.skip("no second drive available on this machine")

    assert (
        user_manager.get_request_user_filepath(
            request_, f"{other}\\Windows\\temp\\test.txt", create_dir=False
        )
        is None
    )
