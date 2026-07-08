import argparse
import pytest
from requests.exceptions import HTTPError
from unittest.mock import patch, mock_open

from app.frontend_management import (
    FrontendManager,
    FrontEndProvider,
    Release,
)
from comfy.cli_args import DEFAULT_VERSION_STRING


@pytest.fixture
def mock_releases():
    return [
        Release(
            id=1,
            tag_name="1.0.0",
            name="Release 1.0.0",
            prerelease=False,
            created_at="2022-01-01T00:00:00Z",
            published_at="2022-01-01T00:00:00Z",
            body="Release notes for 1.0.0",
            assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
        ),
        Release(
            id=2,
            tag_name="2.0.0",
            name="Release 2.0.0",
            prerelease=False,
            created_at="2022-02-01T00:00:00Z",
            published_at="2022-02-01T00:00:00Z",
            body="Release notes for 2.0.0",
            assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
        ),
    ]


@pytest.fixture
def mock_provider(mock_releases):
    provider = FrontEndProvider(
        owner="test-owner",
        repo="test-repo",
    )
    provider.all_releases = mock_releases
    provider.latest_release = mock_releases[1]
    FrontendManager.PROVIDERS = [provider]
    return provider


@pytest.fixture(autouse=True)
def clear_cache():
    import utils.install_util
    import app.frontend_management

    utils.install_util.PACKAGE_VERSIONS = {}
    app.frontend_management.COMFY_PACKAGE_VERSIONS = []


def test_get_release(mock_provider, mock_releases):
    version = "1.0.0"
    release = mock_provider.get_release(version)
    assert release == mock_releases[0]


def test_get_release_latest(mock_provider, mock_releases):
    version = "latest"
    release = mock_provider.get_release(version)
    assert release == mock_releases[1]


def test_get_release_invalid_version(mock_provider):
    version = "invalid"
    with pytest.raises(ValueError):
        mock_provider.get_release(version)


def test_init_frontend_default():
    version_string = DEFAULT_VERSION_STRING
    frontend_path = FrontendManager.init_frontend(version_string)
    assert frontend_path == FrontendManager.default_frontend_path()


def test_init_frontend_invalid_version():
    version_string = "test-owner/test-repo@1.100.99"
    with pytest.raises(HTTPError):
        FrontendManager.init_frontend_unsafe(version_string)


def test_init_frontend_invalid_provider():
    version_string = "invalid/invalid@latest"
    with pytest.raises(HTTPError):
        FrontendManager.init_frontend_unsafe(version_string)


@pytest.fixture
def mock_os_functions():
    with (
        patch("app.frontend_management.os.makedirs") as mock_makedirs,
        patch("app.frontend_management.os.listdir") as mock_listdir,
        patch("app.frontend_management.os.rmdir") as mock_rmdir,
    ):
        mock_listdir.return_value = []  # Simulate empty directory
        yield mock_makedirs, mock_listdir, mock_rmdir


@pytest.fixture
def mock_download():
    with patch("app.frontend_management.download_release_asset_zip") as mock:
        mock.side_effect = Exception("Download failed")  # Simulate download failure
        yield mock


def test_finally_block(mock_os_functions, mock_download, mock_provider):
    # Arrange
    mock_makedirs, mock_listdir, mock_rmdir = mock_os_functions
    version_string = "test-owner/test-repo@1.0.0"

    # Act & Assert
    with pytest.raises(Exception):
        FrontendManager.init_frontend_unsafe(version_string, mock_provider)

    # Assert
    mock_makedirs.assert_called_once()
    mock_download.assert_called_once()
    mock_listdir.assert_called_once()
    mock_rmdir.assert_called_once()


def test_parse_version_string():
    version_string = "owner/repo@1.0.0"
    repo_owner, repo_name, version = FrontendManager.parse_version_string(
        version_string
    )
    assert repo_owner == "owner"
    assert repo_name == "repo"
    assert version == "1.0.0"


def test_parse_version_string_invalid():
    version_string = "invalid"
    with pytest.raises(argparse.ArgumentTypeError):
        FrontendManager.parse_version_string(version_string)


def test_init_frontend_default_with_mocks():
    # Arrange
    version_string = DEFAULT_VERSION_STRING

    # Act
    with (
        patch("app.frontend_management.check_comfy_packages_versions") as mock_check,
        patch.object(
            FrontendManager, "default_frontend_path", return_value="/mocked/path"
        ),
    ):
        frontend_path = FrontendManager.init_frontend(version_string)

    # Assert
    assert frontend_path == "/mocked/path"
    mock_check.assert_called_once()


def test_init_frontend_fallback_on_error():
    # Arrange
    version_string = "test-owner/test-repo@1.0.0"

    # Act
    with (
        patch.object(
            FrontendManager, "init_frontend_unsafe", side_effect=Exception("Test error")
        ),
        patch("app.frontend_management.check_comfy_packages_versions") as mock_check,
        patch.object(
            FrontendManager, "default_frontend_path", return_value="/default/path"
        ),
    ):
        frontend_path = FrontendManager.init_frontend(version_string)

    # Assert
    assert frontend_path == "/default/path"
    mock_check.assert_called_once()


def test_get_frontend_version():
    # Arrange
    expected_version = "1.25.0"
    mock_requirements_content = """torch
torchsde
comfyui-frontend-package==1.25.0
other-package==1.0.0
numpy"""

    # Act
    with patch("builtins.open", mock_open(read_data=mock_requirements_content)):
        version = FrontendManager.get_required_frontend_version()

    # Assert
    assert version == expected_version


def test_get_frontend_version_invalid_semver():
    # Arrange
    mock_requirements_content = """torch
torchsde
comfyui-frontend-package==1.29.3.75
other-package==1.0.0
numpy"""

    # Act
    with patch("builtins.open", mock_open(read_data=mock_requirements_content)):
        version = FrontendManager.get_required_frontend_version()

    # Assert
    assert version is None


def test_get_templates_version():
    # Arrange
    expected_version = "0.1.41"
    mock_requirements_content = """torch
torchsde
comfyui-frontend-package==1.25.0
comfyui-workflow-templates==0.1.41
other-package==1.0.0
numpy"""

    # Act
    with patch("builtins.open", mock_open(read_data=mock_requirements_content)):
        version = FrontendManager.get_required_templates_version()

    # Assert
    assert version == expected_version


def test_get_templates_version_not_found():
    # Arrange
    mock_requirements_content = """torch
torchsde
comfyui-frontend-package==1.25.0
other-package==1.0.0
numpy"""

    # Act
    with patch("builtins.open", mock_open(read_data=mock_requirements_content)):
        version = FrontendManager.get_required_templates_version()

    # Assert
    assert version is None


def test_get_templates_version_invalid_semver():
    # Arrange
    mock_requirements_content = """torch
torchsde
comfyui-workflow-templates==1.0.0.beta
other-package==1.0.0
numpy"""

    # Act
    with patch("builtins.open", mock_open(read_data=mock_requirements_content)):
        version = FrontendManager.get_required_templates_version()

    # Assert
    assert version is None


def test_get_installed_templates_version():
    # Arrange
    expected_version = "0.1.40"

    # Act
    with patch("app.frontend_management.version", return_value=expected_version):
        version = FrontendManager.get_installed_templates_version()

    # Assert
    assert version == expected_version


def test_get_installed_templates_version_not_installed():
    # Act
    with patch(
        "app.frontend_management.version", side_effect=Exception("Package not found")
    ):
        version = FrontendManager.get_installed_templates_version()

    # Assert
    assert version is None
