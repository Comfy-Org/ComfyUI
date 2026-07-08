import argparse
import logging
import os
import re
import sys
import tempfile
import zipfile
import importlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, TypedDict, Optional
from aiohttp import web
from importlib.metadata import version

import requests
from typing_extensions import NotRequired

from utils.install_util import get_missing_requirements_message, get_required_packages_versions

from comfy.cli_args import DEFAULT_VERSION_STRING
import app.logger


def frontend_install_warning_message():
    return f"""
{get_missing_requirements_message()}

The ComfyUI frontend is shipped in a pip package so it needs to be updated separately from the ComfyUI code.
""".strip()

def parse_version(version: str) -> tuple[int, int, int]:
        return tuple(map(int, version.split(".")))

def is_valid_version(version: str) -> bool:
    """Validate if a string is a valid semantic version (X.Y.Z format)."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)$"
    return bool(re.match(pattern, version))

def get_required_frontend_version():
    return get_required_packages_versions().get("comfyui-frontend-package", None)


COMFY_PACKAGE_VERSIONS = []
def get_comfy_package_versions():
    """List installed/required versions for every comfy* package in requirements.txt."""
    if COMFY_PACKAGE_VERSIONS:
        return COMFY_PACKAGE_VERSIONS.copy()
    out = COMFY_PACKAGE_VERSIONS
    for name, required in (get_required_packages_versions() or {}).items():
        if not name.startswith("comfy"):
            continue
        try:
            installed = version(name)
        except Exception:
            installed = None
        out.append({"name": name, "installed": installed, "required": required})
    return out.copy()


def check_comfy_packages_versions():
    """Warn for every comfy* package whose installed version is below requirements.txt."""
    from packaging.version import InvalidVersion, parse as parse_pep440
    outdated_packages = []

    for pkg in get_comfy_package_versions():
        installed_str = pkg["installed"]
        required_str = pkg["required"]
        if not installed_str or not required_str:
            continue
        try:
            outdated = parse_pep440(installed_str) < parse_pep440(required_str)
        except InvalidVersion as e:
            logging.error(f"Failed to check {pkg['name']} version: {e}")
            continue
        if outdated:
            outdated_packages.append((pkg["name"], installed_str, required_str))
        else:
            logging.info("{} version: {}".format(pkg["name"], installed_str))

    if outdated_packages:
        package_warnings = "\n".join(
            f"Installed {name} version {installed} is lower than the recommended version {required}."
            for name, installed, required in outdated_packages
        )
        app.logger.log_startup_warning(
            f"""
________________________________________________________________________
WARNING WARNING WARNING WARNING WARNING

{package_warnings}

{get_missing_requirements_message()}
________________________________________________________________________
""".strip()
        )


REQUEST_TIMEOUT = 10  # seconds


class Asset(TypedDict):
    url: str


class Release(TypedDict):
    id: int
    tag_name: str
    name: str
    prerelease: bool
    created_at: str
    published_at: str
    body: str
    assets: NotRequired[list[Asset]]


@dataclass
class FrontEndProvider:
    owner: str
    repo: str

    @property
    def folder_name(self) -> str:
        return f"{self.owner}_{self.repo}"

    @property
    def release_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"

    @cached_property
    def all_releases(self) -> list[Release]:
        releases = []
        api_url = self.release_url
        while api_url:
            response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises an HTTPError if the response was an error
            releases.extend(response.json())
            # GitHub uses the Link header to provide pagination links. Check if it exists and update api_url accordingly.
            if "next" in response.links:
                api_url = response.links["next"]["url"]
            else:
                api_url = None
        return releases

    @cached_property
    def latest_release(self) -> Release:
        latest_release_url = f"{self.release_url}/latest"
        response = requests.get(latest_release_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        return response.json()

    @cached_property
    def latest_prerelease(self) -> Release:
        """Get the latest pre-release version - even if it's older than the latest release"""
        release = [release for release in self.all_releases if release["prerelease"]]

        if not release:
            raise ValueError("No pre-releases found")

        # GitHub returns releases in reverse chronological order, so first is latest
        return release[0]

    def get_release(self, version: str) -> Release:
        if version == "latest":
            return self.latest_release
        elif version == "prerelease":
            return self.latest_prerelease
        else:
            for release in self.all_releases:
                if release["tag_name"] in [version, f"v{version}"]:
                    return release
            raise ValueError(f"Version {version} not found in releases")


def download_release_asset_zip(release: Release, destination_path: str) -> None:
    """Download dist.zip from github release."""
    asset_url = None
    for asset in release.get("assets", []):
        if asset["name"] == "dist.zip":
            asset_url = asset["url"]
            break

    if not asset_url:
        raise ValueError("dist.zip not found in the release assets")

    # Use a temporary file to download the zip content
    with tempfile.TemporaryFile() as tmp_file:
        headers = {"Accept": "application/octet-stream"}
        response = requests.get(
            asset_url, headers=headers, allow_redirects=True, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()  # Ensure we got a successful response

        # Write the content to the temporary file
        tmp_file.write(response.content)

        # Go back to the beginning of the temporary file
        tmp_file.seek(0)

        # Extract the zip file content to the destination path
        with zipfile.ZipFile(tmp_file, "r") as zip_ref:
            zip_ref.extractall(destination_path)


class FrontendManager:
    CUSTOM_FRONTENDS_ROOT = str(Path(__file__).parents[1] / "web_custom_versions")

    @classmethod
    def get_required_frontend_version(cls) -> str:
        """Get the required frontend package version."""
        return get_required_frontend_version()

    @classmethod
    def get_installed_templates_version(cls) -> str:
        """Get the currently installed workflow templates package version."""
        try:
            templates_version_str = version("comfyui-workflow-templates")
            return templates_version_str
        except Exception:
            return None

    @classmethod
    def get_required_templates_version(cls) -> str:
        return get_required_packages_versions().get("comfyui-workflow-templates", None)

    @classmethod
    def get_comfy_package_versions(cls):
        """List installed/required versions for every comfy* package in requirements.txt."""
        return get_comfy_package_versions()

    @classmethod
    def default_frontend_path(cls) -> str:
        try:
            import comfyui_frontend_package

            return str(importlib.resources.files(comfyui_frontend_package) / "static")
        except ImportError:
            logging.error(
                f"""
********** ERROR ***********

comfyui-frontend-package is not installed.

{frontend_install_warning_message()}

********** ERROR ***********
""".strip()
            )
            sys.exit(-1)

    @classmethod
    def template_asset_map(cls) -> Optional[Dict[str, str]]:
        """Return a mapping of template asset names to their absolute paths."""
        try:
            from comfyui_workflow_templates import (
                get_asset_path,
                iter_templates,
            )
        except ImportError:
            logging.error(
                f"""
********** ERROR ***********

comfyui-workflow-templates is not installed.

{frontend_install_warning_message()}

********** ERROR ***********
""".strip()
            )
            return None

        try:
            template_entries = list(iter_templates())
        except Exception as exc:
            logging.error(f"Failed to enumerate workflow templates: {exc}")
            return None

        asset_map: Dict[str, str] = {}
        try:
            for entry in template_entries:
                for asset in entry.assets:
                    asset_map[asset.filename] = get_asset_path(
                        entry.template_id, asset.filename
                    )
        except Exception as exc:
            logging.error(f"Failed to resolve template asset paths: {exc}")
            return None

        if not asset_map:
            logging.error("No workflow template assets found. Did the packages install correctly?")
            return None

        return asset_map


    @classmethod
    def legacy_templates_path(cls) -> Optional[str]:
        """Return the legacy templates directory shipped inside the meta package."""
        try:
            import comfyui_workflow_templates

            return str(
                importlib.resources.files(comfyui_workflow_templates) / "templates"
            )
        except ImportError:
            logging.error(
                f"""
********** ERROR ***********

comfyui-workflow-templates is not installed.

{frontend_install_warning_message()}

********** ERROR ***********
""".strip()
            )
            return None

    @classmethod
    def embedded_docs_path(cls) -> str:
        """Get the path to embedded documentation"""
        try:
            import comfyui_embedded_docs

            return str(
                importlib.resources.files(comfyui_embedded_docs) / "docs"
            )
        except ImportError:
            logging.info("comfyui-embedded-docs package not found")
            return None

    @classmethod
    def parse_version_string(cls, value: str) -> tuple[str, str, str]:
        """
        Args:
            value (str): The version string to parse.

        Returns:
            tuple[str, str]: A tuple containing provider name and version.

        Raises:
            argparse.ArgumentTypeError: If the version string is invalid.
        """
        VERSION_PATTERN = r"^([a-zA-Z0-9][a-zA-Z0-9-]{0,38})/([a-zA-Z0-9_.-]+)@(v?\d+\.\d+\.\d+[-._a-zA-Z0-9]*|latest|prerelease)$"
        match_result = re.match(VERSION_PATTERN, value)
        if match_result is None:
            raise argparse.ArgumentTypeError(f"Invalid version string: {value}")

        return match_result.group(1), match_result.group(2), match_result.group(3)

    @classmethod
    def init_frontend_unsafe(
        cls, version_string: str, provider: Optional[FrontEndProvider] = None
    ) -> str:
        """
        Initializes the frontend for the specified version.

        Args:
            version_string (str): The version string.
            provider (FrontEndProvider, optional): The provider to use. Defaults to None.

        Returns:
            str: The path to the initialized frontend.

        Raises:
            Exception: If there is an error during the initialization process.
            main error source might be request timeout or invalid URL.
        """
        if version_string == DEFAULT_VERSION_STRING:
            check_comfy_packages_versions()
            return cls.default_frontend_path()

        repo_owner, repo_name, version = cls.parse_version_string(version_string)

        if version.startswith("v"):
            expected_path = str(
                Path(cls.CUSTOM_FRONTENDS_ROOT)
                / f"{repo_owner}_{repo_name}"
                / version.lstrip("v")
            )
            if os.path.exists(expected_path):
                logging.info(
                    f"Using existing copy of specific frontend version tag: {repo_owner}/{repo_name}@{version}"
                )
                return expected_path

        logging.info(
            f"Initializing frontend: {repo_owner}/{repo_name}@{version}, requesting version details from GitHub..."
        )

        provider = provider or FrontEndProvider(repo_owner, repo_name)
        release = provider.get_release(version)

        semantic_version = release["tag_name"].lstrip("v")
        web_root = str(
            Path(cls.CUSTOM_FRONTENDS_ROOT) / provider.folder_name / semantic_version
        )
        if not os.path.exists(web_root):
            try:
                os.makedirs(web_root, exist_ok=True)
                logging.info(
                    "Downloading frontend(%s) version(%s) to (%s)",
                    provider.folder_name,
                    semantic_version,
                    web_root,
                )
                logging.debug(release)
                download_release_asset_zip(release, destination_path=web_root)
            finally:
                # Clean up the directory if it is empty, i.e. the download failed
                if not os.listdir(web_root):
                    os.rmdir(web_root)

        return web_root

    @classmethod
    def init_frontend(cls, version_string: str) -> str:
        """
        Initializes the frontend with the specified version string.

        Args:
            version_string (str): The version string to initialize the frontend with.

        Returns:
            str: The path of the initialized frontend.
        """
        try:
            return cls.init_frontend_unsafe(version_string)
        except Exception as e:
            logging.error("Failed to initialize frontend: %s", e)
            logging.info("Falling back to the default frontend.")
            check_comfy_packages_versions()
            return cls.default_frontend_path()
    @classmethod
    def template_asset_handler(cls):
        assets = cls.template_asset_map()
        if not assets:
            return None

        async def serve_template(request: web.Request) -> web.StreamResponse:
            rel_path = request.match_info.get("path", "")
            target = assets.get(rel_path)
            if target is None:
                raise web.HTTPNotFound()
            return web.FileResponse(target)

        return serve_template
