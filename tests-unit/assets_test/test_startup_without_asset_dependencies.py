import subprocess
import sys
from pathlib import Path

import pytest


# A None entry in sys.modules makes any import of that package raise ImportError.
STARTUP_SCRIPT = (
    "import sys, runpy, comfy_kitchen; "
    "sys.modules.update(dict.fromkeys(('sqlalchemy', 'alembic', 'blake3'))); "
    "comfy_kitchen.int8_attention_is_available=lambda: False; "
    'runpy.run_path("main.py", run_name="__main__")'
)


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def run_quick_startup(tmp_path: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
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
    )


def test_starts_without_asset_dependencies_when_assets_disabled(tmp_path: Path) -> None:
    stale_temp_file = tmp_path / "temp" / "stale.png"
    stale_temp_file.parent.mkdir()
    stale_temp_file.write_bytes(b"")

    result = run_quick_startup(tmp_path, "--disable-assets")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "Error importing dependencies" not in result.stdout + result.stderr
    assert not stale_temp_file.exists()


@pytest.mark.parametrize("flags", [(), ("--enable-assets",)])
def test_missing_dependencies_stop_startup_and_name_the_fix(tmp_path: Path, flags: tuple[str, ...]) -> None:
    result = run_quick_startup(tmp_path, *flags)
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert "ASSETS_STARTUP_FAILED: missing_packages" in output
    assert "The assets system needs packages that could not be imported: sqlalchemy, alembic, blake3." in output
    assert "-m pip install -r" in output
    assert "--disable-assets" in output
    assert "Traceback" not in output
