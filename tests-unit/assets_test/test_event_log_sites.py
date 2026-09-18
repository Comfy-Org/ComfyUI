import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_disable_announces_itself_on_the_event_channel(caplog) -> None:
    """A degrade must reach the monitor, which only reads the tagged channel.

    PromptServer emits assets.enabled during __init__, before setup_database can
    fail, so a monitor has already been told assets are up by the time a database
    failure degrades them. Without a contradicting event it keeps believing that.
    """
    from app.assets.api import routes
    from app.assets.manager import AssetsEnabled

    manager = AssetsEnabled(SimpleNamespace(enable_assets=True, enable_asset_hashing=False))
    routes._ASSETS_ENABLED = True
    try:
        with caplog.at_level(logging.WARNING):
            manager.disable(FileNotFoundError(2, "No such file", "/home/alice/models/secret.db"))

        assert manager.enabled is False
        assert routes._ASSETS_ENABLED is False, "already-registered routes must start answering 503"

        tagged = [line for line in caplog.text.splitlines() if f"{TAG} assets.disabled" in line]
        assert len(tagged) == 1, "the monitor needs exactly one disabled event"
        assert "error_type=FileNotFoundError" in tagged[0]
        assert "/home/alice" not in tagged[0], (
            "OSError embeds its path in str(); the event must carry only the class name"
        )
    finally:
        routes._ASSETS_ENABLED = False


def test_disable_actually_stops_scanning_and_ingest_not_just_http() -> None:
    """A degrade must disarm the write paths, not only the aiohttp routes.

    `enabled` is read once, by PromptServer.__init__, which has already run by the
    time setup_database can fail. So flipping it is invisible to every later caller:
    the scanner and the three ingest entry points have to be gated directly.
    """
    from app.assets import manager as manager_mod
    from app.assets.api import routes
    from app.assets.manager import AssetsEnabled

    started: list[str] = []
    mgr = AssetsEnabled(SimpleNamespace(enable_assets=True, enable_asset_hashing=False))
    routes._ASSETS_ENABLED = True
    seeder = manager_mod.asset_seeder
    was_disabled = seeder.is_disabled()
    try:
        mgr.disable(RuntimeError("database is locked"))

        assert seeder.is_disabled(), (
            "queue_output_scan's existing gate reads the seeder, so disable() must arm it"
        )
        mgr.ensure_scan_started()
        assert started == []
        assert mgr.register_upload("/tmp/x.png", "x", "input", "", content_written=True) is None
        assert mgr.register_executed_output("/tmp/x.png", "job-1") is None
        assert mgr.register_cached_output("/tmp/x.png", "job-1") is None
    finally:
        seeder._disabled = was_disabled
        routes._ASSETS_ENABLED = False
