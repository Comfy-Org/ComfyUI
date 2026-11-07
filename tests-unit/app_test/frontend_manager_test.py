import argparse
import errno
import io
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import Mock, patch, mock_open

import pytest
from requests.exceptions import HTTPError

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
            tag_name="v2.0.0",
            name="Release 2.0.0",
            prerelease=False,
            created_at="2022-02-01T00:00:00Z",
            published_at="2022-02-01T00:00:00Z",
            body="Release notes for 2.0.0",
            assets=[{"name": "dist.zip", "url": "https://example.com/dist.zip"}],
        ),
    ]


@pytest.fixture
def mock_github(mock_releases):
    """Serve release lookups by tag from mock_releases and record the requested URLs."""
    releases_by_tag = {release["tag_name"]: release for release in mock_releases}
    requested_urls = []

    def fake_get(url, timeout=None):
        requested_urls.append(url)
        release = releases_by_tag.get(url.rsplit("/", 1)[-1])
        return SimpleNamespace(
            status_code=200 if release else 404,
            json=lambda: release,
            raise_for_status=lambda: None,
        )

    with patch("app.frontend_management.requests.get", side_effect=fake_get):
        yield requested_urls


@pytest.fixture
def mock_provider(mock_releases, mock_github):
    provider = FrontEndProvider(
        owner="test-owner",
        repo="test-repo",
    )
    provider.latest_release = mock_releases[1]
    FrontendManager.PROVIDERS = [provider]
    return provider


@pytest.fixture(autouse=True)
def clear_cache():
    import utils.install_util
    import app.frontend_management

    utils.install_util.PACKAGE_VERSIONS = {}
    app.frontend_management.COMFY_PACKAGE_VERSIONS = []


def test_get_release(mock_provider, mock_releases, mock_github):
    release = mock_provider.get_release("2.0.0")

    assert release == mock_releases[1]
    assert mock_github == ["https://api.github.com/repos/test-owner/test-repo/releases/tags/v2.0.0"]


def test_get_release_without_v_prefix_tag(mock_provider, mock_releases, mock_github):
    release = mock_provider.get_release("1.0.0")

    assert release == mock_releases[0]
    assert mock_github == [
        "https://api.github.com/repos/test-owner/test-repo/releases/tags/v1.0.0",
        "https://api.github.com/repos/test-owner/test-repo/releases/tags/1.0.0",
    ]


def test_get_release_rate_limited(mock_provider):
    response = Mock(status_code=403)
    response.raise_for_status.side_effect = HTTPError("403 rate limit exceeded")
    with patch("app.frontend_management.requests.get", return_value=response) as mock_get:
        with pytest.raises(HTTPError):
            mock_provider.get_release("2.0.0")
    mock_get.assert_called_once()


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


def test_init_frontend_invalid_version(mock_github):
    version_string = "test-owner/test-repo@1.100.99"
    with pytest.raises(ValueError):
        FrontendManager.init_frontend_unsafe(version_string)


def test_init_frontend_invalid_provider():
    version_string = "invalid/invalid@latest"
    with pytest.raises(HTTPError):
        FrontendManager.init_frontend_unsafe(version_string)


def test_failed_download_cleanup(tmp_path, mock_provider):
    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch("app.frontend_management.download_release_asset_zip", side_effect=Exception("Download failed")) as download,
    ):
        with pytest.raises(Exception, match="Download failed"):
            FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0", mock_provider)

    download.assert_called_once()
    assert list((tmp_path / "test-owner_test-repo").iterdir()) == []


@pytest.mark.parametrize("partial", [False, True])
def test_interrupted_frontend_download_cleanup(tmp_path, mock_provider, partial):
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"
    interruption = KeyboardInterrupt("Download interrupted")

    def interrupted_download(release, destination_path):
        """Interrupt before or after writing a partial frontend asset."""
        if partial:
            (Path(destination_path) / "asset.js").write_text("partial")
        raise interruption

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch("app.frontend_management.download_release_asset_zip", side_effect=interrupted_download),
    ):
        with pytest.raises(KeyboardInterrupt) as error:
            FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0", mock_provider)

    assert error.value is interruption
    assert not cached.exists()
    assert list(cached.parent.iterdir()) == []


@pytest.mark.parametrize("version", ["1.0.0", "v1.0.0", "latest", "prerelease"])
def test_failed_frontend_extraction_is_retried(tmp_path, mock_provider, version):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("assets/app.js", "console.log('frontend')")
        zip_file.writestr("index.html", "<html>frontend</html>")
    valid_zip = archive.getvalue()
    corrupt_zip = valid_zip.replace(b"<html>frontend</html>", b"<html>corrupt!</html>")
    release = mock_provider.get_release("1.0.0")
    response = Mock(content=corrupt_zip)

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch.object(FrontEndProvider, "get_release", return_value=release),
        patch("app.frontend_management.requests.get", return_value=response) as download,
        patch.object(FrontendManager, "default_frontend_path", return_value="/default/path"),
    ):
        version_string = f"test-owner/test-repo@{version}"
        assert FrontendManager.init_frontend(version_string) == "/default/path"
        response.content = valid_zip
        frontend_path = FrontendManager.init_frontend(version_string)
        assert FrontendManager.init_frontend(version_string) == frontend_path

    assert download.call_count == 2
    assert (tmp_path / "test-owner_test-repo" / "1.0.0" / "index.html").read_text() == "<html>frontend</html>"
    assert frontend_path == str(tmp_path / "test-owner_test-repo" / "1.0.0")


def test_frontend_is_published_after_extraction(tmp_path, mock_provider):
    """Keep staged assets hidden until the completed frontend is published."""
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"
    staging_paths = []

    def download_frontend(release, destination_path):
        """Write frontend assets while checking that the cache is still hidden."""
        staging = Path(destination_path)
        staging_paths.append(staging)
        assert staging.is_dir()
        assert staging.parent.parent == cached.parent
        assert not cached.exists()
        (staging / "asset.js").write_text("frontend asset")
        assert not cached.exists()
        (staging / "index.html").write_text("complete frontend")

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch("app.frontend_management.download_release_asset_zip", side_effect=download_frontend),
    ):
        assert FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0", mock_provider) == str(cached)

    assert (cached / "index.html").read_text() == "complete frontend"
    assert (cached / "asset.js").read_text() == "frontend asset"
    assert all(not path.exists() for path in staging_paths)
    assert list(cached.parent.iterdir()) == [cached]


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes follow umask")
@pytest.mark.parametrize("umask", [0o022, 0o002, 0o027, 0o077])
def test_frontend_publication_preserves_directory_mode(tmp_path, umask):
    """Published frontends use ordinary mkdir permissions under the caller's umask."""
    script = dedent("""
        import os
        import stat
        import sys
        from pathlib import Path
        from unittest.mock import patch
        from app.frontend_management import FrontendManager, FrontEndProvider

        os.umask(int(sys.argv[2]))
        root = Path(sys.argv[1])
        ordinary = root / "ordinary"
        ordinary.mkdir()

        def download_frontend(release, destination_path):
            (Path(destination_path) / "index.html").write_text("complete frontend")

        with (
            patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(root)),
            patch.object(FrontEndProvider, "get_release", return_value={"tag_name": "v1.0.0"}),
            patch("app.frontend_management.download_release_asset_zip", side_effect=download_frontend),
        ):
            cached = Path(FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0"))

        published_mode = stat.S_IMODE(cached.stat().st_mode)
        ordinary_mode = stat.S_IMODE(ordinary.stat().st_mode)
        assert published_mode == ordinary_mode, (oct(published_mode), oct(ordinary_mode))
        assert (cached / "index.html").read_text() == "complete frontend"
        assert list(cached.parent.iterdir()) == [cached]
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path), str(umask)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", [None, RuntimeError("Download failed"), KeyboardInterrupt("Download interrupted")])
def test_concurrent_frontend_initializers(tmp_path, mock_provider, failure):
    """Preserve a concurrent winner when the first extraction completes or fails."""
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"
    extracting = Event()
    finish_extraction = Event()

    def download_frontend(release, destination_path):
        """Pause the first extraction until another initializer has published."""
        staging = Path(destination_path)
        if not extracting.is_set():
            (staging / "asset.js").write_text("first frontend asset")
            extracting.set()
            assert finish_extraction.wait(timeout=10)
            if failure is not None:
                raise failure
            (staging / "index.html").write_text("first frontend")
        else:
            (staging / "asset.js").write_text("second frontend asset")
            (staging / "index.html").write_text("second frontend")

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch("app.frontend_management.download_release_asset_zip", side_effect=download_frontend) as download,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        first = executor.submit(FrontendManager.init_frontend_unsafe, "test-owner/test-repo@1.0.0", mock_provider)
        try:
            assert extracting.wait(timeout=10)
            assert not cached.exists()
            assert FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0", mock_provider) == str(cached)
            assert (cached / "index.html").read_text() == "second frontend"
        finally:
            finish_extraction.set()
        if failure is None:
            assert first.result(timeout=10) == str(cached)
        else:
            with pytest.raises(type(failure)) as error:
                first.result(timeout=10)
            assert error.value is failure

    assert download.call_count == 2
    assert (cached / "index.html").read_text() == "second frontend"
    assert (cached / "asset.js").read_text() == "second frontend asset"
    assert list(cached.parent.iterdir()) == [cached]


@pytest.mark.parametrize("version", ["1.0.0", "latest"])
def test_frontend_preserves_completed_cache(tmp_path, mock_provider, version):
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"
    cached.mkdir(parents=True)
    (cached / "index.html").write_text("existing frontend")

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch.object(mock_provider, "get_release", return_value=mock_provider.get_release("1.0.0")),
        patch("app.frontend_management.download_release_asset_zip") as download,
    ):
        assert FrontendManager.init_frontend_unsafe(f"test-owner/test-repo@{version}", mock_provider) == str(cached)

    download.assert_not_called()
    assert (cached / "index.html").read_text() == "existing frontend"
    assert list(cached.parent.iterdir()) == [cached]


@pytest.mark.parametrize("error_number", [errno.EEXIST, errno.ENOTEMPTY])
def test_frontend_reuses_concurrently_published_cache(tmp_path, mock_provider, error_number):
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"

    def publish_other_frontend(source, destination):
        """Simulate platform-specific rename errors after another publisher wins."""
        assert destination == str(cached)
        cached.mkdir()
        (cached / "index.html").write_text("existing frontend")
        raise OSError(error_number, "Destination already exists")

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch("app.frontend_management.download_release_asset_zip"),
        patch("app.frontend_management.os.rename", side_effect=publish_other_frontend),
    ):
        assert FrontendManager.init_frontend_unsafe("test-owner/test-repo@1.0.0", mock_provider) == str(cached)

    assert (cached / "index.html").read_text() == "existing frontend"
    assert list(cached.parent.iterdir()) == [cached]


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.ENOTEMPTY])
def test_frontend_publication_error_falls_back(tmp_path, mock_provider, error_number):
    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch.object(FrontEndProvider, "get_release", return_value=mock_provider.get_release("1.0.0")),
        patch("app.frontend_management.download_release_asset_zip"),
        patch("app.frontend_management.os.rename", side_effect=OSError(error_number, "Cannot publish frontend")),
        patch.object(FrontendManager, "default_frontend_path", return_value="/default/path"),
    ):
        assert FrontendManager.init_frontend("test-owner/test-repo@1.0.0") == "/default/path"

    assert list((tmp_path / "test-owner_test-repo").iterdir()) == []


@pytest.mark.parametrize("version", ["1.0.0", "v1.0.0"])
def test_init_frontend_uses_cached_version(tmp_path, version):
    cached = tmp_path / "test-owner_test-repo" / "1.0.0"
    cached.mkdir(parents=True)

    with (
        patch.object(FrontendManager, "CUSTOM_FRONTENDS_ROOT", str(tmp_path)),
        patch.object(FrontEndProvider, "get_release", side_effect=HTTPError("403 rate limit exceeded")),
        patch.object(FrontendManager, "default_frontend_path", return_value="/default/path"),
    ):
        frontend_path = FrontendManager.init_frontend(f"test-owner/test-repo@{version}")

    assert frontend_path == str(cached)


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


def test_template_asset_map_skips_missing_assets():
    template_entries = [
        SimpleNamespace(
            template_id="example",
            assets=[
                SimpleNamespace(filename="example.json"),
                SimpleNamespace(filename="preview.webp"),
            ],
        )
    ]

    def get_asset_path(template_id, filename):
        if filename == "preview.webp":
            raise FileNotFoundError
        return f"/templates/{filename}"

    with (
        patch(
            "comfyui_workflow_templates.iter_templates",
            return_value=template_entries,
        ),
        patch(
            "comfyui_workflow_templates.get_asset_path",
            side_effect=get_asset_path,
        ),
    ):
        asset_map = FrontendManager.template_asset_map()

    assert asset_map == {"example.json": "/templates/example.json"}
