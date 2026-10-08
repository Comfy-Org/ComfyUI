"""Real ComfyUI starts: assets are on unless --disable-assets, and a database that can't be
opened stops startup."""

import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPRECATED = "--enable-assets is deprecated and does nothing"


def _quick_start(base: Path, *flags: str, root: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "--cpu",
            "--quick-test-for-ci",
            "--disable-all-custom-nodes",
            "--disable-partner-nodes",
            f"--base-directory={base}",
            f"--front-end-root={base}",
            *flags,
        ],
        cwd=base,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _revision(db: Path) -> str:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def test_assets_are_on_by_default(tmp_path):
    result = _quick_start(tmp_path)

    assert result.returncode == 0, result.stderr
    assert _revision(tmp_path / "user" / "comfyui.db").startswith("0008")
    assert DEPRECATED not in result.stderr


def test_enable_assets_is_accepted_and_says_once_that_it_does_nothing(tmp_path):
    result = _quick_start(tmp_path, "--enable-assets")

    assert result.returncode == 0, result.stderr
    assert result.stderr.count(DEPRECATED) == 1
    assert (tmp_path / "user" / "comfyui.db").exists()


def test_disable_assets_leaves_the_database_alone(tmp_path):
    result = _quick_start(tmp_path, "--disable-assets")

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "user" / "comfyui.db").exists()


def test_corrupt_database_stops_startup(tmp_path):
    db = tmp_path / "user" / "comfyui.db"
    db.parent.mkdir()
    db.write_bytes(b"not a database" * 1000)

    result = _quick_start(tmp_path, f"--database-url=sqlite:///{db}")

    assert result.returncode == 1, result.stderr
    assert f"The asset database '{db}' is corrupt" in result.stderr
    assert "--disable-assets" in result.stderr
    assert "Traceback" not in result.stderr


def test_starts_with_a_percent_sign_in_the_database_path(tmp_path):
    # Alembic config values go through ConfigParser interpolation, where % is special.
    base = tmp_path / "50% data"
    base.mkdir()

    result = _quick_start(base)

    assert result.returncode == 0, result.stderr
    assert _revision(base / "user" / "comfyui.db").startswith("0008")
