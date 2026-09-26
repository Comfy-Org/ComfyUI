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

    result = run_quick_startup(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert not stale_temp_file.exists()


def test_enable_assets_without_dependencies_names_the_missing_packages(tmp_path: Path) -> None:
    result = run_quick_startup(tmp_path, "--enable-assets")
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert (
        "--enable-assets requires packages that could not be imported: sqlalchemy, alembic, blake3"
        in output
    )
    assert "-m pip install -r" in output
    assert "Traceback" not in output
