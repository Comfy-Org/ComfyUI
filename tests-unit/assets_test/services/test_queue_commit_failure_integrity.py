import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

import app.database.db as db_mod
from app.assets import mode as mode_module
from app.assets import scanner
from app.assets import scanner_admission
from app.assets import scanner_changes
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content
from app.assets.scanner_admission import _WatchEntry
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)

_LOCKED_ERROR = OperationalError("COMMIT", {}, sqlite3.OperationalError("database is locked"))


def _fail_commit_always(engine):
    real_factory = sessionmaker(bind=engine)

    def factory():
        session = real_factory()

        def commit():
            session.rollback()
            raise _LOCKED_ERROR

        session.commit = commit
        return session

    return factory


def _fail_run_write_txn_at(real_run_write_txn, fail_index: int):
    calls = {"count": -1}

    def wrapper(work):
        calls["count"] += 1
        if calls["count"] == fail_index:
            raise _LOCKED_ERROR
        return real_run_write_txn(work)

    return wrapper


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)


def test_pending_verification_queue_survives_terminal_commit_failure(
    db_engine, tmp_path: Path, session, monkeypatch
):
    path = tmp_path / "terminal.bin"
    path.write_bytes(b"terminal failure content")
    content = AssetContent(
        path=str(path), hash=None, size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
    )
    session.add(content)
    session.flush()
    session.commit()
    content_id = content.id

    scanner_changes.clear_pending_verifications()
    scanner_changes.queue_pending_verification(content_id)
    monkeypatch.setattr(db_mod, "WriteSession", _fail_commit_always(db_engine))

    with pytest.raises(OperationalError):
        scanner_changes.drain_pending_verifications()

    assert scanner_changes._pending_verification_ids == [content_id]
    scanner_changes.clear_pending_verifications()


def test_watch_list_survives_terminal_commit_failure(db_engine, tmp_path: Path, monkeypatch):
    path = tmp_path / "watched-terminal.bin"
    path.write_bytes(b"watched")
    stat = path.stat()
    entry = _WatchEntry(str(path), stat)
    scanner_admission._WATCH_LIST[:] = [entry]
    monkeypatch.setattr(db_mod, "WriteSession", _fail_commit_always(db_engine))

    with (
        patch("folder_paths.get_input_directory", return_value=str(tmp_path)),
        pytest.raises(OperationalError),
    ):
        scanner_admission.tick_watch_list()

    assert scanner_admission._WATCH_LIST == [entry]
    scanner_admission._WATCH_LIST.clear()


def test_transition_queue_and_companion_state_survive_terminal_commit_failure(
    db_engine, tmp_path: Path, session, monkeypatch
):
    path = tmp_path / "transition-terminal.bin"
    path.write_bytes(b"transition terminal")
    stat = path.stat()
    create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    session.commit()

    monkeypatch.setattr(mode_module, "hashing_enabled", lambda: True)

    clear_transition_queue()
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    session.commit()
    assert hash_mode_state._off_to_on_transition_in_flight is True

    monkeypatch.setattr(db_mod, "WriteSession", _fail_commit_always(db_engine))

    with pytest.raises(OperationalError):
        drain_transition_queue()

    assert {e.path for e in hash_mode_state._PENDING_QUEUE} == hash_mode_state._PENDING_PATHS
    assert str(path) in hash_mode_state._PENDING_PATHS
    assert hash_mode_state._off_to_on_transition_in_flight is True
    clear_transition_queue()


def test_transition_in_flight_flag_survives_a_failed_final_mode_commit(
    db_engine, tmp_path: Path, session, monkeypatch
):
    """The queue itself drains (real commits succeed) but the FINAL
    write_stored_mode('on') commit fails: the in-flight flag must stay set."""
    path = tmp_path / "final-commit.bin"
    path.write_bytes(b"final commit content")
    stat = path.stat()
    create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    session.commit()

    monkeypatch.setattr(mode_module, "hashing_enabled", lambda: True)

    clear_transition_queue()
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    session.commit()

    real_run_write_txn = scanner.run_write_txn
    monkeypatch.setattr(
        hash_mode_state,
        "run_write_txn",
        _fail_run_write_txn_at(real_run_write_txn, fail_index=1),
    )

    with pytest.raises(OperationalError):
        drain_transition_queue()

    assert list(hash_mode_state._PENDING_QUEUE) == []
    assert hash_mode_state._PENDING_PATHS == set()
    assert hash_mode_state._off_to_on_transition_in_flight is True
    assert read_stored_mode(session) == "off"
    clear_transition_queue()
