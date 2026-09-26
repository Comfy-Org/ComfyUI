import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock

from app.assets.event_log import TAG


STARTUP_SCRIPT = (
    "import runpy, comfy_kitchen; "
    "comfy_kitchen.int8_attention_is_available=lambda: False; "
    'runpy.run_path("main.py", run_name="__main__")'
)


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def run_quick_startup_with_database_held(tmp_path: Path, *flags: str) -> subprocess.CompletedProcess:
    db_path = tmp_path / "comfyui.db"
    holder = FileLock(str(db_path) + ".lock")
    holder.acquire(timeout=0)
    try:
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
                f"--database-url=sqlite:///{db_path}",
                *flags,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        holder.release()


def test_enable_assets_starts_with_assets_disabled_when_another_process_holds_the_database(
    tmp_path: Path,
) -> None:
    result = run_quick_startup_with_database_held(tmp_path, "--enable-assets")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Another ComfyUI process is already using this database." in output
    assert (
        "Assets are disabled for this session: another ComfyUI process is using the database at "
        f"{tmp_path / 'comfyui.db'}." in output
    )
    assert f"{TAG} assets.enabled " not in output


def test_assets_disabled_still_starts_when_another_process_holds_the_database(tmp_path: Path) -> None:
    result = run_quick_startup_with_database_held(tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Another ComfyUI process is already using this database." in output
    assert "Assets are disabled" not in output
