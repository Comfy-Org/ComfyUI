from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aiohttp import web
from aiohttp.pytest_plugin import AiohttpClient
from sqlalchemy.orm import Session

import folder_paths
from app.assets import lifecycle
from app.assets.api import routes
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.manager import NoAssets
from app.assets.mode import hashing_enabled
from app.assets.seeder import asset_seeder
from app.assets.services.hash_mode_state import read_stored_mode


class _Args:
    enable_assets = False

    def __init__(self, hashing: bool) -> None:
        self.enable_asset_hashing = hashing


def _no_assets(*, hashing: bool = False) -> NoAssets:
    return NoAssets(_Args(hashing))


@pytest.mark.asyncio
async def test_noassets_register_routes_returns_service_disabled_and_disables_seeder(
    aiohttp_client: AiohttpClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", False)
    monkeypatch.setattr(asset_seeder, "_disabled", False)
    app = web.Application()

    _no_assets().register_routes(app, None)
    client = await aiohttp_client(app)
    response = await client.get("/api/assets")

    assert response.status == 503
    assert (await response.json())["error"]["code"] == "SERVICE_DISABLED"
    assert asset_seeder.is_disabled()


def test_noassets_startup_applies_hash_mode_persists_state_and_cleans_temp_dir(
    mock_create_session: Callable[[], AbstractContextManager[Session]],
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_file = tmp_path / "startup-stale.bin"
    temp_file.write_bytes(b"stale")
    monkeypatch.setattr(lifecycle, "create_session", mock_create_session)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    _no_assets(hashing=True).startup()

    assert hashing_enabled() is True
    assert read_stored_mode(session) == "on"
    assert not temp_file.exists()


def test_noassets_shutdown_wipes_temp_rows_and_files(
    mock_create_session: Callable[[], AbstractContextManager[Session]],
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _no_assets()
    monkeypatch.setattr(lifecycle, "create_session", mock_create_session)
    monkeypatch.setattr(lifecycle, "can_create_session", lambda: True)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    manager.startup()

    tmp_path.mkdir()
    temp_file = tmp_path / "shutdown-stale.bin"
    temp_file.write_bytes(b"stale")
    content = create_content(session, str(temp_file))
    record = create_record(session, content.id, temp_file.name)
    session.commit()
    record_id = record.id
    content_id = content.id

    manager.shutdown()
    session.expire_all()

    assert session.get(Asset, record_id) is None
    assert session.get(AssetContent, content_id) is None
    assert not temp_file.exists()


def test_noassets_shutdown_runs_cleanup_after_unsuccessful_seeder_shutdown(
    mock_create_session: Callable[[], AbstractContextManager[Session]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_file = tmp_path / "seeder-timeout.bin"
    temp_file.write_bytes(b"stale")
    calls: list[str] = []
    monkeypatch.setattr(lifecycle, "create_session", mock_create_session)
    monkeypatch.setattr(lifecycle, "can_create_session", lambda: True)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    def seeder_shutdown() -> bool:
        calls.append("seeder")
        return False

    def cleanup() -> None:
        calls.append("cleanup")
        lifecycle.run_shutdown()

    with (
        patch.object(asset_seeder, "shutdown", side_effect=seeder_shutdown),
        patch("app.assets.manager.run_shutdown", side_effect=cleanup) as cleanup_spy,
    ):
        _no_assets().shutdown()

    cleanup_spy.assert_called_once_with()
    assert calls == ["seeder", "cleanup"]
    assert not temp_file.exists()


def test_noassets_shutdown_without_database_sweeps_files_without_session_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp_file = tmp_path / "no-database.bin"
    temp_file.write_bytes(b"stale")
    create_session = Mock()
    monkeypatch.setattr(lifecycle, "can_create_session", lambda: False)
    monkeypatch.setattr(lifecycle, "create_session", create_session)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    _no_assets().shutdown()

    assert not temp_file.exists()
    create_session.assert_not_called()


def test_noassets_callbacks_are_noops_without_seeder_side_effects() -> None:
    manager = _no_assets()
    with (
        patch.object(asset_seeder, "start") as start,
        patch.object(asset_seeder, "pause") as pause,
        patch.object(asset_seeder, "enqueue_enrich") as enqueue_enrich,
        patch.object(asset_seeder, "resume") as resume,
        patch.object(asset_seeder, "set_event_sink") as set_event_sink,
    ):
        assert manager.ensure_scan_started() is None
        assert manager.pause_background_scan() is None
        assert manager.queue_output_enrichment() is None
        assert manager.resume_background_scan() is None
        assert manager.set_event_sink(Mock()) is None

    start.assert_not_called()
    pause.assert_not_called()
    enqueue_enrich.assert_not_called()
    resume.assert_not_called()
    set_event_sink.assert_not_called()


def test_noassets_registration_methods_return_none() -> None:
    manager = _no_assets()

    assert manager.register_upload(
        "/tmp/output.png", "output.png", "output", "", content_written=True
    ) is None
    assert manager.register_executed_output("/tmp/output.png", "job-id") is None
    assert manager.register_cached_output("/tmp/output.png", "job-id") is None


def test_noassets_is_disabled() -> None:
    assert _no_assets().enabled is False
