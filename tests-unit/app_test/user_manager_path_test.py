"""Tests for the path containment check in UserManager.get_request_user_filepath().

The `file` parameter comes straight from the /userdata routes, so a value the
containment check cannot handle must return None (the routes answer 403) rather
than raise out of the request handler.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

import folder_paths
from app.user_manager import UserManager


@pytest.fixture
def user_manager():
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = folder_paths.get_user_directory()
        folder_paths.set_user_directory(temp_dir)
        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = False
            yield UserManager()
        folder_paths.set_user_directory(original_dir)


@pytest.fixture
def mock_request():
    request = MagicMock()
    request.headers = {}
    return request


def symlink(link, target):
    try:
        os.symlink(target, link, target_is_directory=os.path.isdir(target))
    except OSError as e:
        pytest.skip(f"cannot create a symlink here: {e}")


def test_parent_traversal_is_rejected(user_manager, mock_request):
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = False
        path = user_manager.get_request_user_filepath(mock_request, "../../evil.json", create_dir=False)
    assert path is None


@pytest.mark.skipif(os.name != "nt", reason="only Windows paths carry a drive letter")
def test_other_drive_is_rejected(user_manager, mock_request):
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = False
        path = user_manager.get_request_user_filepath(mock_request, r"Z:\evil.json", create_dir=False)
    assert path is None


def test_symlinked_user_root_is_rejected(user_manager, mock_request):
    with tempfile.TemporaryDirectory() as outside:
        symlink(os.path.join(folder_paths.get_user_directory(), "default"), outside)
        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = False
            assert user_manager.get_request_user_filepath(mock_request, "comfy.settings.json", create_dir=False) is None
            with pytest.raises(web.HTTPForbidden):
                user_manager.settings.get_settings(mock_request)
            with pytest.raises(web.HTTPForbidden):
                user_manager.settings.save_settings(mock_request, {})


def test_symlinked_settings_file_is_rejected(user_manager, mock_request):
    user_root = os.path.join(folder_paths.get_user_directory(), "default")
    os.makedirs(user_root)
    with tempfile.TemporaryDirectory() as outside:
        target = os.path.join(outside, "comfy.settings.json")
        with open(target, "w") as f:
            f.write("{}")
        symlink(os.path.join(user_root, "comfy.settings.json"), target)
        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = False
            assert user_manager.get_request_user_filepath(mock_request, "comfy.settings.json", create_dir=False) is None
            with pytest.raises(web.HTTPForbidden):
                user_manager.settings.get_settings(mock_request)
