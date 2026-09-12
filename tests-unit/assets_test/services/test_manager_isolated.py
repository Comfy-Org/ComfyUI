import dataclasses
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import folder_paths
import pytest
from sqlalchemy.orm import Session

from app.assets import lifecycle
from app.assets import manager as manager_module
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.manager import AssetsEnabled
from app.assets.seeder import asset_seeder
from app.assets.services.schemas import RegisteredAsset, UploadAssetView


class _ArgsStub:
    enable_assets = True
    enable_asset_hashing = False


@pytest.fixture
def enabled_manager() -> AssetsEnabled:
    return AssetsEnabled(_ArgsStub())


@pytest.fixture
def asset_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    input_dir.mkdir()
    temp_dir.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_dir))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_dir))
    return output_dir, input_dir, temp_dir


def test_executed_and_cached_outputs_share_unhashed_content(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    mock_create_session: Callable[[], AbstractContextManager[Session]],
) -> None:
    output_dir, _, _ = asset_roots
    output_path = output_dir / "executed.png"
    output_path.write_bytes(b"executed output")

    executed = enabled_manager.register_executed_output(
        str(output_path), job_id="executed-job"
    )

    assert isinstance(executed, RegisteredAsset)
    assert executed.id
    assert executed.content_id
    assert executed.name
    assert executed.job_id == "executed-job"
    assert not hasattr(executed, "asset_hash")
    assert {field.name for field in dataclasses.fields(executed)} == {
        "id",
        "content_id",
        "job_id",
        "name",
    }
    with mock_create_session() as session:
        asset = session.get(Asset, executed.id)
        content = session.get(AssetContent, executed.content_id)
        assert asset is not None
        assert asset.content_id == executed.content_id
        assert content is not None
        assert content.hash is None

    cached = enabled_manager.register_cached_output(str(output_path), job_id="cached-job")

    assert isinstance(cached, RegisteredAsset)
    assert cached.id != executed.id
    assert cached.content_id == executed.content_id
    assert cached.job_id == "cached-job"
    assert (
        enabled_manager.register_cached_output(
            str(output_dir / "unknown.png"), job_id="unknown-job"
        )
        is None
    )


def test_register_upload_hashes_and_tags_fresh_input_file(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    mock_create_session: Callable[[], AbstractContextManager[Session]],
) -> None:
    _, input_dir, _ = asset_roots
    upload_path = input_dir / "pasted" / "upload.png"
    upload_path.parent.mkdir()
    upload_path.write_bytes(b"uploaded input")

    view = enabled_manager.register_upload(
        str(upload_path),
        name=upload_path.name,
        upload_type="input",
        subfolder="pasted",
        content_written=True,
    )

    assert isinstance(view, UploadAssetView)
    assert isinstance(view.asset, RegisteredAsset)
    assert view.asset_hash
    assert "pasted" in view.tags


def test_startup_runs_against_memory_db_without_starting_a_scanner_thread(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    mock_create_session: Callable[[], AbstractContextManager[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, temp_dir = asset_roots
    (temp_dir / "stale.tmp").write_bytes(b"stale")
    seeder_start = MagicMock(return_value=False)
    monkeypatch.setattr(lifecycle, "create_session", mock_create_session)
    monkeypatch.setattr(lifecycle, "start_asset_seeder", seeder_start)

    thread_count = threading.active_count()
    enabled_manager.startup()
    assert threading.active_count() == thread_count, (
        "start_asset_seeder is mocked, so a new thread means a component other than the seeder spawned one"
    )

    assert not temp_dir.exists()
    seeder_start.assert_called_once_with()


def test_ensure_scan_started_starts_the_lazy_object_info_scan(
    enabled_manager: AssetsEnabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeder_start = MagicMock()
    monkeypatch.setattr(asset_seeder, "start", seeder_start)

    enabled_manager.ensure_scan_started()

    seeder_start.assert_called_once_with(roots=("models", "input", "output"))


def test_shutdown_runs_lifecycle_cleanup_when_seeder_shutdown_times_out(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    mock_create_session: Callable[[], AbstractContextManager[Session]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, temp_dir = asset_roots
    stale_file = temp_dir / "stale.tmp"
    stale_file.write_bytes(b"stale")
    with mock_create_session() as session:
        content = create_content(session, path=str(stale_file))
        asset = create_record(session, content_id=content.id, name=stale_file.name)
        session.commit()
        asset_id = asset.id
        content_id = content.id

    calls = Mock()
    seeder_shutdown = MagicMock(return_value=False)
    run_shutdown = MagicMock(wraps=manager_module.run_shutdown)
    calls.attach_mock(seeder_shutdown, "seeder_shutdown")
    calls.attach_mock(run_shutdown, "run_shutdown")
    monkeypatch.setattr(lifecycle, "can_create_session", lambda: True)
    monkeypatch.setattr(lifecycle, "create_session", mock_create_session)
    monkeypatch.setattr(asset_seeder, "shutdown", seeder_shutdown)
    monkeypatch.setattr(manager_module, "run_shutdown", run_shutdown)

    enabled_manager.shutdown()

    assert calls.mock_calls == [call.seeder_shutdown(), call.run_shutdown()]
    assert not temp_dir.exists()
    with mock_create_session() as session:
        assert session.get(Asset, asset_id) is None
        assert session.get(AssetContent, content_id) is None
