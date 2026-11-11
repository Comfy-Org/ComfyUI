import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/stable-release.yml"


@pytest.fixture
def steps():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["package_comfy_windows"]["steps"]


@pytest.fixture
def version_check(steps):
    return next(step for step in steps if step.get("name") == "Check release version")


def test_version_is_checked_before_restoring_build_dependencies(steps):
    names = [step.get("name") for step in steps]
    assert "Check release version" in names
    check_index = names.index("Check release version")
    checkout_index = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/checkout@"))
    cache_index = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/cache/restore@"))
    assert checkout_index < check_index < cache_index
    assert "if" not in steps[check_index]
    assert steps[check_index]["shell"] == "python"
    assert steps[check_index]["env"]["RELEASE_TAG"] == "${{ inputs.git_tag }}"
    assert "${{" not in steps[check_index]["run"]


@pytest.mark.parametrize(
    "tag, version, matches",
    [
        ("v0.45.0", "0.45.0", True),
        ("v0.14.1", "0.14.1", True),
        ("v0.14.2", "0.14.1", False),
        ("v0.18.3", "0.18.2", False),
        ("v0.20.2", "0.20.1", False),
    ],
)
def test_release_version(version_check, tmp_path, tag, version, matches):
    (tmp_path / "comfyui_version.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", version_check["run"]],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_TAG": tag},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == matches, result.stderr
    if not matches:
        assert tag in result.stderr
        assert version in result.stderr


def test_version_check_does_not_execute_checked_out_code(version_check, tmp_path):
    (tmp_path / "comfyui_version.py").write_text(
        "__version__ = '0.45.0'\nraise RuntimeError('must not execute')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", version_check["run"]],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_TAG": "v0.45.0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
