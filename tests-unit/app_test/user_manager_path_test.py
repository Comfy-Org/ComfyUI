"""Tests for the path containment check in UserManager.get_request_user_filepath().

The `file` parameter comes straight from the /userdata routes, so a value the
containment check cannot handle must return None (the routes answer 403) rather
than raise out of the request handler.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

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
