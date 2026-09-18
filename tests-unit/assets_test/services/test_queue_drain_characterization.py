from pathlib import Path

import pytest

import app.database.db as db_mod
from app.assets import scanner_admission
from app.assets import scanner_changes
from app.assets.database.models import AssetContent
from app.assets.scanner_admission import _WATCH_LIST, _WatchEntry, tick_watch_list
from app.assets.scanner_changes import drain_pending_verifications, queue_pending_verification
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    _PENDING_PATHS,
    _PENDING_QUEUE,
    _PendingEntry,
    clear_transition_queue,
    drain_transition_queue,
    read_stored_mode,
    write_stored_mode,
)


@pytest.fixture(autouse=True)
def clear_queues():
    scanner_changes.clear_pending_verifications()
    _WATCH_LIST.clear()
    clear_transition_queue()
    yield
    scanner_changes.clear_pending_verifications()
    _WATCH_LIST.clear()
    clear_transition_queue()


def _content(session, path: Path) -> AssetContent:
    stat = path.stat()
    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
    session.add(content)
    session.commit()
    return content


def test_pending_verification_requeues_after_hash_oserror(session, temp_dir, monkeypatch):
    path = temp_dir / "pending.bin"
    path.write_bytes(b"pending")
    content = _content(session, path)
    queue_pending_verification(content.id)
    monkeypatch.setattr(
        scanner_changes,
        "snapshot_hash",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )

    processed = drain_pending_verifications(session)

    assert processed == 0
    assert scanner_changes._pending_verification_ids == [content.id]


def test_watch_list_keeps_entries_when_stat_raises(session, temp_dir, monkeypatch):
    path = temp_dir / "watched.bin"
    path.write_bytes(b"watched")
    entry = _WatchEntry(str(path), path.stat())
    _WATCH_LIST.append(entry)
    monkeypatch.setattr(
        scanner_admission.os,
        "stat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(PermissionError):
        tick_watch_list(session)

    assert _WATCH_LIST == [entry]


def test_transition_queue_retries_without_losing_companion_path(
    session, monkeypatch
):
    path = "/unreadable/transition.bin"
    db_mod.run_write_txn(
        lambda write_session: write_session.add(
            AssetContent(path=path, hash=None, size_bytes=0, mtime_ns=None)
        )
    )
    entry = _PendingEntry(path)
    _PENDING_QUEUE.append(entry)
    _PENDING_PATHS.add(path)
    hash_mode_state._off_to_on_transition_in_flight = True
    monkeypatch.setattr(
        hash_mode_state,
        "snapshot_hash",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )

    drain_transition_queue(session)

    assert list(_PENDING_QUEUE) == [_PendingEntry(path, ticks=1)]
    assert _PENDING_PATHS == {path}
    assert hash_mode_state._off_to_on_transition_in_flight is True


def test_transition_queue_exhaustion_clears_companion_and_persists_mode(
    session, monkeypatch
):
    path = "/unreadable/exhausted.bin"
    _PENDING_QUEUE.append(_PendingEntry(path))
    _PENDING_PATHS.add(path)
    hash_mode_state._off_to_on_transition_in_flight = True
    write_stored_mode(session, "off")
    session.commit()
    monkeypatch.setattr(
        hash_mode_state,
        "snapshot_hash",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )

    for _ in range(3):
        drain_transition_queue(session)
        session.commit()

    assert list(_PENDING_QUEUE) == []
    assert _PENDING_PATHS == set()
    assert hash_mode_state._off_to_on_transition_in_flight is False
    assert read_stored_mode(session) == "on"
