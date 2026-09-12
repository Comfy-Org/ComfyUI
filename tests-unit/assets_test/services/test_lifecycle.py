import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session as SASession

from app.assets import lifecycle
from app.assets.database.models import Asset, AssetContent, Base
from app.assets.database.queries.records import create_content, create_record
from app.assets.lifecycle import (
    cleanup_temp_filesystem,
    get_excluded_scan_roots,
    run_asset_shutdown_cleanup,
    run_asset_startup,
    run_startup,
    wipe_temp_db_rows,
)
from app.assets.scanner import get_temp_prefixes, sync_temp_references_safely
from app.assets.scanner_changes import is_path_under_prefixes
from app.assets.seeder import asset_seeder
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import clear_transition_queue, write_stored_mode

from .path_prefix_cases import prefix_case_paths


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    lifecycle._excluded_scan_roots.clear()
    yield
    lifecycle._excluded_scan_roots.clear()


@pytest.fixture(autouse=True)
def autoclean_transition_queue():
    clear_transition_queue()
    lifecycle._hash_mode_transition = None
    yield
    clear_transition_queue()
    lifecycle._hash_mode_transition = None


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def mock_create_session(session):
    engine = session.bind

    @contextmanager
    def _create_session():
        with SASession(engine) as sess:
            yield sess

    with patch("app.assets.lifecycle.create_session", _create_session), \
         patch("app.database.db.create_session", _create_session):
        yield _create_session


@pytest.fixture
def comfy_dirs():
    with tempfile.TemporaryDirectory() as base:
        temp_dir = Path(base) / "temp"
        temp_dir.mkdir()
        with patch("folder_paths.get_temp_directory", return_value=str(temp_dir)):
            yield temp_dir


def _seed_temp_rows(session: Session, temp_dir: Path) -> tuple[str, str]:
    temp_file = temp_dir / "preview.png"
    temp_file.write_bytes(b"\x00" * 10)
    content = create_content(session, path=str(temp_file))
    record = create_record(session, content_id=content.id, name="preview.png")
    session.commit()
    return record.id, content.id


def test_startup_order_wipe_before_rmtree_before_seeder(mock_create_session):
    calls: list[str] = []

    def _wipe(session):
        calls.append("wipe")
        return 0, 0

    with (
        patch("app.assets.lifecycle.wipe_temp_db_rows", side_effect=_wipe),
        patch("app.assets.lifecycle.cleanup_temp_filesystem", side_effect=lambda: calls.append("rmtree") or True),
        patch("app.assets.lifecycle.enqueue_mode_transition_work", side_effect=lambda: calls.append("enqueue")),
        patch("app.assets.lifecycle.start_asset_seeder", side_effect=lambda: calls.append("seeder") or True),
    ):
        run_asset_startup()

    assert calls == ["wipe", "rmtree", "enqueue", "seeder"]


@pytest.mark.parametrize("wipe_fails", [False, True], ids=["wipe_succeeds", "wipe_fails"])
def test_startup_defers_transition_drain_to_the_seeder(
    session, mock_create_session, tmp_path, monkeypatch, wipe_fails
):
    asset_file = tmp_path / "model.safetensors"
    asset_file.write_bytes(b"weights")
    stat = asset_file.stat()
    create_content(session, path=str(asset_file), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    session.commit()
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    wipe_side_effect = RuntimeError("db wipe failed") if wipe_fails else (lambda _session: (0, 0))

    with (
        patch("app.assets.lifecycle.wipe_temp_db_rows", side_effect=wipe_side_effect),
        patch("app.assets.lifecycle.cleanup_temp_filesystem", return_value=True),
        patch("app.assets.lifecycle.start_asset_seeder", return_value=True),
        patch.object(hash_mode_state, "drain_transition_queue") as drain_spy,
    ):
        lifecycle.record_hash_mode_transition_intent()
        run_asset_startup()

    drain_spy.assert_not_called()
    assert hash_mode_state.pending_transition_count() > 0
    assert hash_mode_state.read_stored_mode(session) == "off"


def test_db_wipe_failure_skips_rmtree_and_continues(mock_create_session):
    with (
        patch("app.assets.lifecycle.wipe_temp_db_rows", side_effect=RuntimeError("db wipe failed")),
        patch("app.assets.lifecycle.cleanup_temp_filesystem") as cleanup_mock,
        patch("app.assets.lifecycle.start_asset_seeder", return_value=True) as seeder_mock,
    ):
        run_asset_startup()

    cleanup_mock.assert_not_called()
    seeder_mock.assert_called_once()


def test_run_startup_disabled_sweeps_temp_filesystem_without_db_work(comfy_dirs):
    stale = comfy_dirs / "stale.png"
    stale.write_bytes(b"\x00" * 10)
    assert stale.exists()

    with (
        patch("app.assets.lifecycle.run_asset_startup") as asset_startup_mock,
        patch("app.assets.lifecycle.wipe_temp_db_rows") as wipe_mock,
        patch("app.assets.lifecycle.start_asset_seeder") as seeder_mock,
    ):
        run_startup(enable_assets=False)

    assert not stale.exists()
    asset_startup_mock.assert_not_called()
    wipe_mock.assert_not_called()
    seeder_mock.assert_not_called()


def test_run_startup_enabled_delegates_to_asset_startup_not_bare_sweep():
    with (
        patch("app.assets.lifecycle.run_asset_startup") as asset_startup_mock,
        patch("app.assets.lifecycle.cleanup_temp_filesystem") as cleanup_mock,
    ):
        run_startup(enable_assets=True)

    asset_startup_mock.assert_called_once_with()
    cleanup_mock.assert_not_called()


def test_run_startup_logs_and_absorbs_disabled_filesystem_failure(caplog):
    with patch("app.assets.lifecycle.cleanup_temp_filesystem", side_effect=RuntimeError("filesystem failure")):
        run_startup(enable_assets=False)

    assert "Asset startup maintenance failed" in caplog.text


def test_rmtree_failure_excludes_temp_from_scan(session, comfy_dirs, mock_create_session):
    record_id, content_id = _seed_temp_rows(session, comfy_dirs)

    wipe_temp_db_rows(session)
    session.commit()
    assert session.get(Asset, record_id) is None

    with patch("app.assets.lifecycle.shutil.rmtree", side_effect=OSError("busy")):
        assert cleanup_temp_filesystem() is False

    assert str(comfy_dirs) in get_excluded_scan_roots()
    assert get_temp_prefixes() == []

    residual = comfy_dirs / "leftover.png"
    residual.write_bytes(b"\x00" * 10)

    with patch("app.assets.scanner.create_session", mock_create_session):
        sync_temp_references_safely()

    assert session.scalars(select(Asset)).all() == []


def test_shutdown_skips_cleanup_when_seeder_join_times_out(session, comfy_dirs, mock_create_session, caplog):
    record_id, _ = _seed_temp_rows(session, comfy_dirs)

    with patch("app.assets.lifecycle.wipe_temp_db_rows") as wipe_mock:
        with patch.object(asset_seeder, "shutdown", return_value=False):
            joined = asset_seeder.shutdown()
            if joined:
                run_asset_shutdown_cleanup()

    wipe_mock.assert_not_called()
    assert session.get(Asset, record_id) is not None


def test_seeder_shutdown_logs_only_when_thread_does_not_exit(caplog: pytest.LogCaptureFixture) -> None:
    seeder = type(asset_seeder)()
    with patch.object(seeder, "cancel"), patch.object(seeder, "wait", return_value=False), caplog.at_level(logging.WARNING):
        assert seeder.shutdown(timeout=2.5) is False
    assert caplog.messages == ["Asset seeder thread did not exit within 2.5s"]
    assert "skipping temp cleanup" not in caplog.text


def test_shutdown_cleanup_wipes_rows_then_rmtree(session, comfy_dirs, mock_create_session):
    record_id, content_id = _seed_temp_rows(session, comfy_dirs)
    calls: list[str] = []

    real_wipe = wipe_temp_db_rows

    def tracking_wipe(sess):
        calls.append("wipe")
        return real_wipe(sess)

    with (
        patch("app.assets.lifecycle.wipe_temp_db_rows", side_effect=tracking_wipe),
        patch("app.assets.lifecycle.cleanup_temp_filesystem", side_effect=lambda: calls.append("rmtree") or True),
    ):
        run_asset_shutdown_cleanup()

    assert calls == ["wipe", "rmtree"]
    assert session.get(Asset, record_id) is None
    assert session.get(AssetContent, content_id) is None


def test_run_shutdown_without_session_sweeps_temp_filesystem(comfy_dirs):
    stale = comfy_dirs / "stale.png"
    stale.write_bytes(b"\x00" * 10)

    with (
        patch("app.assets.lifecycle.can_create_session", return_value=False),
        patch("app.assets.lifecycle.wipe_temp_db_rows") as wipe_mock,
    ):
        lifecycle.run_shutdown()

    assert not stale.exists()
    wipe_mock.assert_not_called()


def test_wipe_temp_db_rows_deletes_records_before_content(session, comfy_dirs):
    record_id, content_id = _seed_temp_rows(session, comfy_dirs)
    records_deleted, contents_deleted = wipe_temp_db_rows(session)
    session.commit()

    assert records_deleted == 1
    assert contents_deleted == 1
    assert session.get(Asset, record_id) is None
    assert session.get(AssetContent, content_id) is None


def test_wipe_temp_db_rows_preserves_non_temp_rows(session, comfy_dirs):
    temp_record_id, temp_content_id = _seed_temp_rows(session, comfy_dirs)
    keep_path = str(comfy_dirs) + "-sibling" + os.sep + "keep.safetensors"
    keep_content = create_content(session, path=keep_path)
    keep_record = create_record(session, content_id=keep_content.id, name="keep.safetensors")
    session.commit()

    records_deleted, contents_deleted = wipe_temp_db_rows(session)
    session.commit()

    assert (records_deleted, contents_deleted) == (1, 1)
    assert session.get(Asset, temp_record_id) is None
    assert session.get(AssetContent, temp_content_id) is None
    assert session.get(Asset, keep_record.id) is not None
    assert session.get(AssetContent, keep_content.id) is not None


def _seed_paths(session: Session, paths: list[str]) -> list[str]:
    stored: list[str] = []
    for index, path in enumerate(paths):
        content = create_content(session, path=path)
        create_record(session, content_id=content.id, name=f"case-{index}.png")
        stored.append(content.path)
    session.commit()
    return stored


def _surviving_paths(session: Session) -> set[str]:
    return set(session.scalars(select(AssetContent.path)).all())


def test_wipe_deletes_exactly_what_is_path_under_prefixes_accepts(session, comfy_dirs):
    temp_root = str(comfy_dirs)
    stored = _seed_paths(session, prefix_case_paths(temp_root))

    records_deleted, contents_deleted = wipe_temp_db_rows(session)
    session.commit()

    wiped = {p for p in stored if is_path_under_prefixes(p, [temp_root])}
    assert _surviving_paths(session) == set(stored) - wiped
    assert (records_deleted, contents_deleted) == (len(wiped), len(wiped))
    assert wiped and wiped != set(stored)
    case_different = os.path.join(
        os.path.dirname(temp_root), os.path.basename(temp_root).upper(), "case.png"
    )
    assert case_different in _surviving_paths(session)
    assert os.path.join(os.path.dirname(temp_root), "escaped.png") in _surviving_paths(session)


def test_wipe_with_metacharacter_temp_root_matches_only_literal_children(session):
    with tempfile.TemporaryDirectory() as base:
        temp_root = os.path.join(base, "a_b%c*d?e[f")
        inside = os.path.join(temp_root, "preview.png")
        decoy = os.path.join(base, "aXbYcZdWeQf", "keep.safetensors")
        _seed_paths(session, [inside, decoy])

        with patch("folder_paths.get_temp_directory", return_value=temp_root):
            wipe_temp_db_rows(session)
        session.commit()

    assert is_path_under_prefixes(decoy, [temp_root]) is False
    assert _surviving_paths(session) == {decoy}
