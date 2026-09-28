import subprocess
import sys
from pathlib import Path

from filelock import FileLock


STARTUP_SCRIPT = (
    "import runpy, comfy_kitchen; "
    "comfy_kitchen.int8_attention_is_available=lambda: False; "
    'runpy.run_path("main.py", run_name="__main__")'
)


def test_enable_assets_still_exits_when_the_lock_stays_held_and_names_the_holder_first(tmp_path: Path) -> None:
    db_path = tmp_path / "comfyui.db"
    holder = FileLock(str(db_path) + ".lock")
    holder.acquire(timeout=0)
    try:
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
                f"--database-url=sqlite:///{db_path}",
                "--enable-assets",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        holder.release()
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    holder_line = output.find("Database lock held by a process that did not record itself")
    lock_message = output.find("Another ComfyUI process is already using this database.")
    assert 0 <= holder_line < lock_message, output
