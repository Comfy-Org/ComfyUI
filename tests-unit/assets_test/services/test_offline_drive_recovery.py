"""A library on a drive that is offline for a scan comes back as the same records
once the drive returns. Run through the seeder's real fast phase on an in-memory
catalog, with hashing off unless a test says otherwise."""

import errno
import logging
import os
import re
import shutil
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import folder_paths
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session as SASession, sessionmaker

from app.assets import mode, scanner, seeder as seeder_module
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    ensure_tag,
    ensure_tag_link,
    mark_content_missing,
)
from app.assets.scanner import SeedAssetSpec, insert_asset_specs, seed_asset_specs
from app.assets.scanner_changes import recover_missing_content_by_stat
from app.assets.scanner_admission import _WATCH_LIST

ROOTS = ("input", "output")
EVENT_LINE = re.compile(r"^\[assets-event\] (?P<event>\S+)(?P<fields>(?: \S+)*)$")


@pytest.fixture
def drive(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A removable drive holding both the input and the output directory."""
    drive = temp_dir / "drive"
    for name in ROOTS:
        (drive / name).mkdir(parents=True)
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(drive / "output"))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(drive / "input"))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    return drive


@pytest.fixture(autouse=True)
def isolated_state(db_engine):
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield
    _WATCH_LIST.clear()


def _scan(roots=ROOTS) -> seeder_module._ScanState:
    seeder = seeder_module._AssetSeeder()
    seeder._scan_state = seeder_module._ScanState()
    seeder._phase = seeder_module.ScanPhase.FAST
    seeder._run_gate.set()
    seeder._cancel_event.clear()
    seeder._run_fast_phase(roots)
    return seeder._scan_state


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, str]]:
    found = []
    for record in caplog.records:
        match = EVENT_LINE.match(record.getMessage())
        if match is not None and match.group("event") == name:
            found.append(dict(pair.split("=", 1) for pair in match.group("fields").split()))
    return found


def _populate(drive: Path) -> list[Path]:
    files = []
    for name in ROOTS:
        for i in range(5):
            path = drive / name / f"{name}_{i}.png"
            path.write_bytes(f"{name}-{i}".encode() * (i + 1))
            files.append(path)
    return files


def _customise(session) -> dict[str, tuple[str, dict, str, set[str]]]:
    """Give every record the edits a user makes; returns them keyed by record id."""
    session.expire_all()
    edits = {}
    for i, record in enumerate(session.scalars(sa.select(Asset).order_by(Asset.name))):
        record.name = f"renamed {i}"
        record.user_metadata = {"note": i}
        record.job_id = f"job-{i}"
        ensure_tag(session, "favourite")
        ensure_tag_link(session, asset_id=record.id, tag_name="favourite", origin="manual")
    session.commit()
    for record in session.scalars(sa.select(Asset)):
        edits[record.id] = (record.name, record.user_metadata, record.job_id, _tags(session, record.id))
    return edits


def _tags(session, record_id: str) -> set[str]:
    return set(session.scalars(sa.select(AssetTag.tag_name).where(AssetTag.asset_id == record_id)))


def _records(session) -> dict[str, tuple[str, dict, str, set[str]]]:
    session.expire_all()
    return {
        record.id: (record.name, record.user_metadata, record.job_id, _tags(session, record.id))
        for record in session.scalars(sa.select(Asset))
    }


def _missing_count(session) -> int:
    session.expire_all()
    return session.scalar(
        sa.select(sa.func.count()).select_from(AssetContent).where(AssetContent.is_missing.is_(True))
    )


def _take_offline(drive: Path, variant: str) -> Path:
    parked = drive.with_name("drive.offline")
    drive.rename(parked)
    if variant == "empty":
        drive.mkdir()  # an unmounted mount point
    return parked


def _bring_back(drive: Path, parked: Path) -> None:
    shutil.rmtree(drive, ignore_errors=True)
    parked.rename(drive)


@pytest.mark.parametrize("variant", ["absent", "empty"])
def test_offline_drive_round_trip_keeps_every_record(drive, session, caplog, variant):
    files = _populate(drive)
    _scan()
    edits = _customise(session)
    assert len(edits) == len(files)

    parked = _take_offline(drive, variant)
    with caplog.at_level(logging.INFO):
        offline = _scan()
    assert offline.missing_marked == len(files)
    assert _missing_count(session) == len(files)
    assert sorted(e["root"] for e in _events(caplog, "seeder.marked_missing")) == ["input", "output"]
    assert {e["stage"] for e in _events(caplog, "seeder.marked_missing")} == {"fast_scan"}

    _bring_back(drive, parked)
    back = _scan()

    assert back.recovered == len(files)
    assert _missing_count(session) == 0
    assert _records(session) == edits
    assert session.scalar(sa.select(sa.func.count()).select_from(AssetContent)) == len(files)


def test_output_listing_rescan_recovers_a_returning_output_drive(drive, session, caplog):
    files = [path for path in _populate(drive) if path.parent.name == "output"]
    _scan(("output",))
    edits = _customise(session)

    parked = _take_offline(drive, "absent")
    with caplog.at_level(logging.INFO):
        offline = _scan(("output",))
    assert offline.missing_marked == len(files)
    assert _events(caplog, "seeder.marked_missing") == [
        {"count": str(len(files)), "root": "output", "stage": "fast_scan"}
    ]

    _bring_back(drive, parked)
    back = _scan(("output",))

    assert back.recovered == len(files)
    assert _records(session) == edits
    assert _missing_count(session) == 0


def test_an_io_error_leaves_rows_live(drive, session, monkeypatch):
    files = _populate(drive)
    _scan()
    real_stat = os.stat

    def flaky_stat(path, *args, **kwargs):
        if str(path).startswith(str(drive)):
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky_stat)
    state = _scan()
    monkeypatch.setattr(os, "stat", real_stat)

    assert state.missing_marked == 0
    assert _missing_count(session) == 0
    assert session.scalar(sa.select(sa.func.count()).select_from(AssetContent)) == len(files)


def test_a_genuinely_deleted_file_is_still_marked_missing(drive, session, caplog):
    files = _populate(drive)
    _scan()
    for path in files[:3]:
        path.unlink()

    with caplog.at_level(logging.INFO):
        state = _scan()

    assert state.missing_marked == 3
    assert state.recovered == 0
    assert _missing_count(session) == 3
    assert _events(caplog, "seeder.marked_missing") == [
        {"count": "3", "root": "input", "stage": "fast_scan"}
    ]


def test_a_returning_file_with_a_changed_mtime_gets_a_new_record(drive, session):
    files = _populate(drive)
    _scan()
    target = files[0]
    parked = _take_offline(drive, "absent")
    _scan()
    _bring_back(drive, parked)
    stat_result = target.stat()
    os.utime(target, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000))

    state = _scan()

    assert state.recovered == len(files) - 1
    session.expire_all()
    rows = list(session.scalars(sa.select(AssetContent).where(AssetContent.path == str(target))))
    assert sorted(row.is_missing for row in rows) == [False, True]


def test_hashing_on_recovers_through_the_hash_path(drive, session):
    class _HashingOn:
        enable_asset_hashing = True

    mode.init(_HashingOn())
    files = _populate(drive)
    _scan()
    edits = _customise(session)
    parked = _take_offline(drive, "absent")
    _scan()
    _bring_back(drive, parked)

    with patch(
        "app.assets.scanner.recover_missing_content_by_stat"
    ) as by_stat, patch(
        "app.assets.scanner.missing_content_ids_by_path"
    ) as prefetch:
        back = _scan()

    by_stat.assert_not_called()
    prefetch.assert_not_called()
    assert back.recovered == len(files)
    assert _records(session) == edits


def _spec(path: Path) -> SeedAssetSpec:
    stat_result = path.stat()
    return {
        "abs_path": str(path),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "info_name": path.name,
        "tags": ["input"],
        "fname": path.name,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }


def _missing_row(session, path: Path) -> AssetContent:
    stat_result = path.stat()
    content = create_content(
        session, path=str(path), size_bytes=stat_result.st_size, mtime_ns=stat_result.st_mtime_ns
    )
    create_record(session, content_id=content.id, name=path.name)
    mark_content_missing(session, content.id)
    return content


def test_the_newest_of_several_matching_missing_rows_recovers(session, temp_dir):
    path = temp_dir / "left-behind.png"
    path.write_bytes(b"one file, catalogued twice by earlier offline cycles")
    older = _missing_row(session, path)
    newer = _missing_row(session, path)
    older.created_at = newer.created_at - timedelta(days=1)
    session.commit()

    created, error = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert (created, error) == (0, None)
    assert session.get(AssetContent, newer.id).is_missing is False
    assert session.get(AssetContent, older.id).is_missing is True
    assert session.get(AssetTag, {"asset_id": session.scalar(
        sa.select(Asset.id).where(Asset.content_id == newer.id)
    ), "tag_name": "missing"}) is None


def test_stat_recovery_skips_a_path_a_live_row_already_occupies(session, temp_dir):
    path = temp_dir / "contested.png"
    path.write_bytes(b"bytes")
    missing = _missing_row(session, path)
    stat_result = path.stat()
    live = create_content(
        session, path=str(path), size_bytes=stat_result.st_size, mtime_ns=stat_result.st_mtime_ns
    )
    create_record(session, content_id=live.id, name=path.name)
    session.commit()

    created, error = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert (created, error) == (0, None)
    assert session.get(AssetContent, missing.id).is_missing is True
    assert session.get(AssetContent, live.id).is_missing is False


def test_a_recovery_rolled_back_by_a_failed_commit_is_not_counted(session, temp_dir, db_engine):
    path = temp_dir / "rolled-back.png"
    path.write_bytes(b"bytes")
    _missing_row(session, path)
    session.commit()
    progress = seeder_module._ScanState()

    @contextmanager
    def failing_write_session():
        with SASession(db_engine) as sess:
            def fail():
                raise RuntimeError("disk full")

            sess.commit = fail
            yield sess

    with patch("app.assets.scanner.create_write_session", failing_write_session):
        with pytest.raises(RuntimeError):
            insert_asset_specs([_spec(path)], set(), progress)

    assert progress.recovered == 0


@pytest.mark.parametrize("while_away", ["unregistered", "base_path_unresolved"])
def test_a_model_folder_pruned_while_unregistered_recovers_when_it_returns(
    temp_dir, session, monkeypatch, while_away
):
    """A folder a custom node registers late, or an extra_model_paths entry whose base
    path did not resolve for one launch, is pruned at startup and seen again later."""
    models = temp_dir / "lazy_models" / "checkpoints"
    models.mkdir(parents=True)
    for i in range(3):
        (models / f"ckpt_{i}.safetensors").write_bytes(b"weights" * (i + 1))
    for name in ("input", "output", "temp"):
        (temp_dir / name).mkdir()
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir / "input"))
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(temp_dir / "output"))
    monkeypatch.setattr("folder_paths.get_temp_directory", lambda: str(temp_dir / "temp"))

    def register(path: Path | None) -> None:
        paths = {} if path is None else {"checkpoints": ([str(path)], {".safetensors"})}
        monkeypatch.setattr(folder_paths, "folder_names_and_paths", paths)
        monkeypatch.setattr(folder_paths, "filename_list_cache", {})

    register(models)
    _scan(("models",))
    edits = _customise(session)
    assert len(edits) == 3

    register(None if while_away == "unregistered" else temp_dir / "$UNSET_VAR" / "checkpoints")
    assert scanner.mark_missing_outside_prefixes_safely(scanner.get_owned_prefixes()).marked == 3
    _scan(("models",))
    assert _missing_count(session) == 3

    register(models)
    back = _scan(("models",))

    assert back.recovered == 3
    assert _missing_count(session) == 0
    assert _records(session) == edits


def test_a_row_whose_records_were_deleted_while_missing_does_not_recover(session, temp_dir):
    path = temp_dir / "deleted-while-offline.png"
    path.write_bytes(b"bytes")
    orphan = _missing_row(session, path)
    session.execute(sa.delete(AssetTag))
    session.execute(sa.delete(Asset).where(Asset.content_id == orphan.id))
    session.commit()

    created, error = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert (created, error) == (1, None)
    assert session.get(AssetContent, orphan.id).is_missing is True
    live = session.scalar(
        sa.select(AssetContent).where(AssetContent.path == str(path), AssetContent.is_missing.is_(False))
    )
    assert session.scalar(sa.select(Asset.id).where(Asset.content_id == live.id)) is not None


def test_a_newer_recordless_row_does_not_win_over_the_users_record(session, temp_dir):
    path = temp_dir / "deduplicated.png"
    path.write_bytes(b"bytes")
    original = _missing_row(session, path)
    duplicate = _missing_row(session, path)
    original.created_at = duplicate.created_at - timedelta(days=1)
    session.execute(sa.delete(AssetTag).where(AssetTag.asset_id.in_(
        sa.select(Asset.id).where(Asset.content_id == duplicate.id)
    )))
    session.execute(sa.delete(Asset).where(Asset.content_id == duplicate.id))
    session.commit()

    created, error = seed_asset_specs(session, [_spec(path)])
    session.commit()

    assert (created, error) == (0, None)
    assert session.get(AssetContent, original.id).is_missing is False
    assert session.get(AssetContent, duplicate.id).is_missing is True


def test_stat_recovery_runs_no_table_scan_in_the_write_transaction(session, temp_dir, db_engine):
    """Recovery runs once per returning file inside the batch's write transaction, so a
    table scan there makes the lock window grow with the catalog."""
    path = temp_dir / "returning.png"
    path.write_bytes(b"bytes")
    _missing_row(session, path)
    session.commit()
    candidates = {str(path): [row.id for row in session.scalars(sa.select(AssetContent))]}
    statements: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("SELECT", "UPDATE", "DELETE")):
            statements.append((statement, parameters))

    sa.event.listen(db_engine, "before_cursor_execute", capture)
    try:
        assert recover_missing_content_by_stat(session, str(path), path.stat(), candidates[str(path)]) == "recovered"
        session.flush()
    finally:
        sa.event.remove(db_engine, "before_cursor_execute", capture)

    assert statements
    with db_engine.connect() as conn:
        for statement, parameters in statements:
            plan = conn.exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters).all()
            assert not any("SCAN asset_contents" in row[-1] for row in plan), (statement, plan)
