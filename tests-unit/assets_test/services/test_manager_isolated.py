import dataclasses
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import MagicMock, Mock, call

import folder_paths
import pytest
from aiohttp.test_utils import make_mocked_request
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.assets import lifecycle
from app.assets.api import routes
from app.assets import manager as manager_module
from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.manager import AssetsEnabled
from app.assets.seeder import ScanStatus, asset_seeder
from app.database.models import Base
from app.assets.services.schemas import RegisteredAsset, UploadAssetView


class _ArgsStub:
    enable_assets = True
    enable_asset_hashing = False


class _OutputSeeder(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...

    def get_status(self) -> ScanStatus: ...

    def shutdown(self, timeout: float = 5.0) -> bool: ...


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


@pytest.fixture
def threaded_create_session(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[], AbstractContextManager[Session]]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    @contextmanager
    def _create_session() -> Generator[SASession, None, None]:
        with SASession(engine) as session:
            yield session

    monkeypatch.setattr(seeder_module, "create_session", _create_session)
    monkeypatch.setattr(scanner, "create_session", _create_session)
    monkeypatch.setattr("app.assets.services.ingest.create_session", _create_session)
    monkeypatch.setattr("app.database.db.WriteSession", sessionmaker(bind=engine))
    yield _create_session
    engine.dispose()


@pytest.fixture
def output_seeder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_OutputSeeder]:
    seeder = asset_seeder.__class__()
    monkeypatch.setattr(manager_module, "asset_seeder", seeder)
    yield seeder
    _ = seeder.shutdown()


def test_queue_output_scan_does_not_register_undeclared_output(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
) -> None:
    output_dir, _, _ = asset_roots
    output_path = output_dir / "undeclared.bin"
    output_path.write_bytes(b"custom node output")

    enabled_manager.queue_output_scan()
    assert output_seeder.wait(timeout=5)

    with threaded_create_session() as session:
        rows = list(
            session.scalars(
                select(Asset)
                .join(AssetContent, Asset.content_id == AssetContent.id)
                .where(AssetContent.path == str(output_path.resolve()))
            )
        )
    assert rows == []


def test_queue_output_scan_does_not_duplicate_declared_output(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
) -> None:
    output_dir, _, _ = asset_roots
    output_path = output_dir / "declared.bin"
    output_path.write_bytes(b"declared custom node output")
    registered = enabled_manager.register_executed_output(
        str(output_path), job_id="declared-job"
    )
    assert registered is not None
    undeclared_path = output_dir / "alongside.bin"
    undeclared_path.write_bytes(b"undeclared custom node output")

    enabled_manager.queue_output_scan()
    assert output_seeder.wait(timeout=5)
    assert output_seeder.get_status().errors == []

    with threaded_create_session() as session:
        rows = list(
            session.scalars(
                select(Asset)
                .join(AssetContent, Asset.content_id == AssetContent.id)
                .where(AssetContent.path == str(output_path.resolve()))
            )
        )
        undeclared_rows = list(
            session.scalars(
                select(Asset)
                .join(AssetContent, Asset.content_id == AssetContent.id)
                .where(AssetContent.path == str(undeclared_path.resolve()))
            )
        )
        assert rows[0].job_id == "declared-job"
    assert len(rows) == 1
    assert undeclared_rows == []


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

    seeder_start.assert_called_once_with(roots=("input",))


@pytest.fixture
def model_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkpoint added on disk after startup, in a registered models folder."""
    checkpoints = tmp_path / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    model_path = checkpoints / "added_while_running.safetensors"
    model_path.write_bytes(b"\0" * 16)
    folders = [str(checkpoints)]
    monkeypatch.setattr(
        folder_paths, "folder_names_and_paths", {"checkpoints": (folders, {".safetensors"})}
    )
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    return model_path


def _catalogued_paths(create_session: Callable[[], AbstractContextManager[Session]]) -> set[str]:
    with create_session() as session:
        return set(session.scalars(select(AssetContent.path)))


def test_ensure_scan_started_does_not_scan_models(
    enabled_manager: AssetsEnabled,
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
    model_on_disk: Path,
) -> None:
    """GET /object_info runs this on every page load; models are left to startup and POST /seed."""
    _, input_dir, _ = asset_roots
    input_path = input_dir / "copied_in.png"
    input_path.write_bytes(b"not really a png")

    enabled_manager.ensure_scan_started()
    assert output_seeder.wait(timeout=10)

    paths = _catalogued_paths(threaded_create_session)
    assert str(input_path.resolve()) in paths
    assert str(model_on_disk.resolve()) not in paths


@pytest.mark.asyncio
async def test_seed_request_during_the_object_info_scan_still_scans_models(
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
    model_on_disk: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontend's node-definition refresh sends GET /object_info and POST /seed together."""
    seeder = cast(seeder_module._AssetSeeder, output_seeder)
    monkeypatch.setattr(routes, "asset_seeder", seeder)
    assert seeder.start(roots=("input",), _start_paused=True)

    response = await routes.seed_assets.__wrapped__(make_mocked_request("POST", "/api/assets/seed"))
    assert response.status == 202
    seeder.resume()
    assert seeder.wait(timeout=10)  # the input scan, which starts the queued one
    assert seeder.wait(timeout=10)

    assert str(model_on_disk.resolve()) in _catalogued_paths(threaded_create_session)


def test_shutdown_does_not_start_a_queued_scan(
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
    model_on_disk: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seed request queued behind a scan must not start once shutdown cancels that scan."""
    seeder = cast(seeder_module._AssetSeeder, output_seeder)
    assert seeder.start(roots=("input",), _start_paused=True)
    assert not seeder.enqueue_scan(("models",), seeder_module.ScanPhase.FULL)
    real_start = seeder.start
    later_starts: list[bool] = []

    def recording_start(**kwargs) -> bool:
        later_starts.append(real_start(**kwargs))
        return later_starts[-1]

    monkeypatch.setattr(seeder, "start", recording_start)

    assert seeder.shutdown(timeout=10)

    assert later_starts == []
    assert str(model_on_disk.resolve()) not in _catalogued_paths(threaded_create_session)


def test_a_cancel_still_starts_the_queued_scan(
    asset_roots: tuple[Path, Path, Path],
    threaded_create_session: Callable[[], AbstractContextManager[Session]],
    output_seeder: _OutputSeeder,
    model_on_disk: Path,
) -> None:
    """POST /seed answered "queued", so cancelling the scan ahead of it must not strand it."""
    seeder = cast(seeder_module._AssetSeeder, output_seeder)
    assert seeder.start(roots=("input",), _start_paused=True)
    assert not seeder.enqueue_scan(("models",), seeder_module.ScanPhase.FULL)

    assert seeder.cancel()
    assert seeder.wait(timeout=10)  # the cancelled scan, which starts the queued one
    assert seeder.wait(timeout=10)

    assert str(model_on_disk.resolve()) in _catalogued_paths(threaded_create_session)


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
