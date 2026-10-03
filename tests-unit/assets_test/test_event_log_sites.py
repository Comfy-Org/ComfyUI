import subprocess
import sys
from pathlib import Path

import pytest

from app.assets.event_log import TAG


STARTUP_SCRIPT = (
    "import runpy, comfy_kitchen; "
    "comfy_kitchen.int8_attention_is_available=lambda: False; "
    'runpy.run_path("main.py", run_name="__main__")'
)


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def run_quick_startup(tmp_path: Path, *flags: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            STARTUP_SCRIPT,
            "--cpu",
            "--quick-test-for-ci",
            "--disable-all-custom-nodes",
            "--disable-api-nodes",
            f"--base-directory={tmp_path}",
            f"--front-end-root={tmp_path}",
            f"--database-url=sqlite:///{tmp_path / 'assets.sqlite3'}",
            *flags,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    ("hashing_flag", "expected"),
    [
        pytest.param((), False, id="hashing-disabled"),
        pytest.param(("--enable-asset-hashing",), True, id="hashing-enabled"),
    ],
)
def test_enabled_assets_emits_once_with_the_hashing_flag(
    tmp_path: Path, hashing_flag: tuple[str, ...], expected: bool
) -> None:
    output = run_quick_startup(tmp_path, "--enable-assets", *hashing_flag)
    lines = [line for line in output.splitlines() if f"{TAG} assets.enabled " in line]

    assert len(lines) == 1
    assert f"hashing_enabled={str(expected).lower()}" in lines[0]


def test_noassets_emits_no_enabled_event(tmp_path: Path) -> None:
    output = run_quick_startup(tmp_path)

    assert f"{TAG} assets.enabled " not in output
