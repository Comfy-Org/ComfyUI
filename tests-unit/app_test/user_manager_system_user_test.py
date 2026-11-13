"""Tests for System User Protection in user_manager.py

Tests cover:
- get_request_user_id(): 1st defense layer - blocks System Users from HTTP headers
- get_request_user_filepath(): 2nd defense layer - structural blocking via get_public_user_directory()
- add_user(): 3rd defense layer - prevents creation of System User names
- Defense layers integration tests
"""

import os
import pytest
from unittest.mock import MagicMock, patch
import tempfile

import folder_paths
from app.user_manager import UserManager


@pytest.fixture
def mock_user_directory():
    """Create a temporary user directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = folder_paths.get_user_directory()
        folder_paths.set_user_directory(temp_dir)
        yield temp_dir
        folder_paths.set_user_directory(original_dir)


@pytest.fixture
def user_manager(mock_user_directory):
    """Create a UserManager instance for testing."""
    with patch('app.user_manager.args') as mock_args:
        mock_args.multi_user = True
        manager = UserManager()
        # Add a default user for testing
        manager.users = {"default": "default", "test_user_123": "Test User"}
        yield manager


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.headers = {}
    return request


class TestGetRequestUserId:
    """Tests for get_request_user_id() - 1st defense layer.

    Verifies:
    - System Users (__ prefix) in HTTP header are rejected with KeyError
    - Public Users pass through successfully
    """

    def test_system_user_raises_error(self, user_manager, mock_request):
        """Test System User in header raises KeyError."""
        mock_request.headers = {"comfy-user": "__system"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            with pytest.raises(KeyError, match="Unknown user"):
                user_manager.get_request_user_id(mock_request)

    def test_system_user_cache_raises_error(self, user_manager, mock_request):
        """Test System User cache raises KeyError."""
        mock_request.headers = {"comfy-user": "__cache"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            with pytest.raises(KeyError, match="Unknown user"):
                user_manager.get_request_user_id(mock_request)

    def test_normal_user_works(self, user_manager, mock_request):
        """Test normal user access works."""
        mock_request.headers = {"comfy-user": "default"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            user_id = user_manager.get_request_user_id(mock_request)
            assert user_id == "default"

    def test_unknown_user_raises_error(self, user_manager, mock_request):
        """Test unknown user raises KeyError."""
        mock_request.headers = {"comfy-user": "unknown_user"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            with pytest.raises(KeyError, match="Unknown user"):
                user_manager.get_request_user_id(mock_request)


class TestGetRequestUserFilepath:
    """Tests for get_request_user_filepath() - 2nd defense layer.

    Verifies:
    - Returns None when get_public_user_directory() returns None (System User)
    - Acts as backup defense if 1st layer is bypassed
    """

    def test_system_user_returns_none(self, user_manager, mock_request, mock_user_directory):
        """Test System User returns None (structural blocking)."""
        # First, we need to mock get_request_user_id to return System User
        # But actually, get_request_user_id will raise KeyError first
        # So we test via get_public_user_directory returning None
        mock_request.headers = {"comfy-user": "default"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            # Patch get_public_user_directory to return None for testing
            with patch.object(folder_paths, 'get_public_user_directory', return_value=None):
                result = user_manager.get_request_user_filepath(mock_request, "test.txt")
                assert result is None

    def test_normal_user_gets_path(self, user_manager, mock_request, mock_user_directory):
        """Test normal user gets valid filepath."""
        mock_request.headers = {"comfy-user": "default"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            path = user_manager.get_request_user_filepath(mock_request, "test.txt")
            assert path is not None
            assert "default" in path
            assert path.endswith("test.txt")

    def test_traversal_returns_none(self, user_manager, mock_request, mock_user_directory):
        """Test dot-dot traversal outside user directory returns None."""
        mock_request.headers = {"comfy-user": "default"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            result = user_manager.get_request_user_filepath(mock_request, "../escape.txt", create_dir=False)
            assert result is None

    def test_symlink_escape_returns_none(self, user_manager, mock_request, mock_user_directory):
        """Test symlink inside user directory pointing outside returns None."""
        mock_request.headers = {"comfy-user": "default"}
        user_dir = os.path.join(mock_user_directory, "default")
        os.makedirs(user_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as outside_dir:
            link_path = os.path.join(user_dir, "escape_link")
            try:
                os.symlink(outside_dir, link_path)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks not supported on this platform/filesystem")

            with patch('app.user_manager.args') as mock_args:
                mock_args.multi_user = True
                result = user_manager.get_request_user_filepath(
                    mock_request, "escape_link/secret.txt", create_dir=False
                )
                assert result is None

    def test_symlink_user_root_returns_none(self, user_manager, mock_request, mock_user_directory):
        """Test symlinked user root pointing outside user directory returns None."""
        mock_request.headers = {"comfy-user": "default"}
        user_dir = os.path.join(mock_user_directory, "default")

        with tempfile.TemporaryDirectory() as outside_dir:
            try:
                os.symlink(outside_dir, user_dir)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks not supported on this platform/filesystem")

            with patch('app.user_manager.args') as mock_args:
                mock_args.multi_user = True
                result = user_manager.get_request_user_filepath(
                    mock_request, "secret.txt", create_dir=False
                )
                assert result is None


class TestAddUser:
    """Tests for add_user() - 3rd defense layer (creation-time blocking).

    Verifies:
    - System User name (__ prefix) creation is rejected with ValueError
    - Sanitized usernames that become System User are also rejected
    """

    def test_system_user_prefix_name_raises(self, user_manager):
        """Test System User prefix in name raises ValueError."""
        with pytest.raises(ValueError, match="System User prefix not allowed"):
            user_manager.add_user("__system")

    def test_system_user_prefix_cache_raises(self, user_manager):
        """Test System User cache prefix raises ValueError."""
        with pytest.raises(ValueError, match="System User prefix not allowed"):
            user_manager.add_user("__cache")

    def test_sanitized_system_user_prefix_raises(self, user_manager):
        """Test sanitized name becoming System User prefix raises ValueError (bypass prevention)."""
        # "__test" directly starts with System User prefix
        with pytest.raises(ValueError, match="System User prefix not allowed"):
            user_manager.add_user("__test")

    def test_normal_user_creation(self, user_manager, mock_user_directory):
        """Test normal user creation works."""
        user_id = user_manager.add_user("Normal User")
        assert user_id is not None
        assert not user_id.startswith("__")
        assert "Normal-User" in user_id or "Normal_User" in user_id

    def test_empty_name_raises(self, user_manager):
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError, match="username not provided"):
            user_manager.add_user("")

    def test_whitespace_only_raises(self, user_manager):
        """Test whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="username not provided"):
            user_manager.add_user("   ")


class TestDefenseLayers:
    """Integration tests for all three defense layers.

    Verifies:
    - Each defense layer blocks System Users independently
    - System User bypass is impossible through any layer
    """

    def test_layer1_get_request_user_id(self, user_manager, mock_request):
        """Test 1st defense layer blocks System Users."""
        mock_request.headers = {"comfy-user": "__system"}

        with patch('app.user_manager.args') as mock_args:
            mock_args.multi_user = True
            with pytest.raises(KeyError):
                user_manager.get_request_user_id(mock_request)

    def test_layer2_get_public_user_directory(self):
        """Test 2nd defense layer blocks System Users."""
        result = folder_paths.get_public_user_directory("__system")
        assert result is None

    def test_layer3_add_user(self, user_manager):
        """Test 3rd defense layer blocks System User creation."""
        with pytest.raises(ValueError):
            user_manager.add_user("__system")
