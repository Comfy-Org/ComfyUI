from pathlib import Path

import pytest

import app.database.db as db_mod
from app.assets import scanner_changes
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content
from app.assets.services import hash_mode_state


class _StatInsideWriteTxn(AssertionError):
    pass


@pytest.fixture(autouse=True)
def clear_drain_queues():
    scanner_changes.clear_pending_verifications()
    hash_mode_state.clear_transition_queue()
    yield
    scanner_changes.clear_pending_verifications()
    hash_mode_state.clear_transition_queue()


def _seed_content(path: Path) -> str:
    stat = path.stat()
    return db_mod.run_write_txn(
        lambda session: create_content(
            session,
            str(path),
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        ).id
    )


def _guard_stat_during_write_txn(module, monkeypatch: pytest.MonkeyPatch) -> None:
    inside = False
    real_run_write_txn = module.run_write_txn
    real_stat = module.os.stat

    def guarded_stat(*args, **kwargs):
        if inside:
            raise _StatInsideWriteTxn
        return real_stat(*args, **kwargs)

    def tracked_run_write_txn(work):
        def tracked_work(session):
            nonlocal inside
            inside = True
            try:
                return work(session)
            finally:
                inside = False

        return real_run_write_txn(tracked_work)

    monkeypatch.setattr(module.os, "stat", guarded_stat)
    monkeypatch.setattr(module, "run_write_txn", tracked_run_write_txn)


def test_pending_verification_does_not_stat_inside_write_transaction(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "pending-ready.bin"
    path.write_bytes(b"ready")
    content_id = _seed_content(path)
    scanner_changes.queue_pending_verification(content_id)
    _guard_stat_during_write_txn(scanner_changes, monkeypatch)

    processed = scanner_changes.drain_pending_verifications()

    assert processed == 1


def test_pending_verification_trusts_gone_preflight_without_statting_in_transaction(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "pending-gone.bin"
    path.write_bytes(b"gone")
    content_id = _seed_content(path)
    scanner_changes.queue_pending_verification(content_id)
    path.unlink()
    _guard_stat_during_write_txn(scanner_changes, monkeypatch)

    processed = scanner_changes.drain_pending_verifications()
    session.expire_all()

    assert processed == 1
    assert session.get(AssetContent, content_id).is_missing is True


def test_pending_verification_drops_when_row_changes_after_preflight(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "pending-row-changed.bin"
    path.write_bytes(b"row changes")
    content_id = _seed_content(path)
    scanner_changes.queue_pending_verification(content_id)
    changed_mtime_ns = path.stat().st_mtime_ns + 1
    real_snapshot_hash = scanner_changes.snapshot_hash

    def mutate_row_during_hash(candidate_path: str):
        snapshot = real_snapshot_hash(candidate_path)

        def mutate_row(write_session):
            write_session.get(AssetContent, content_id).mtime_ns = changed_mtime_ns

        db_mod.run_write_txn(mutate_row)
        return snapshot

    monkeypatch.setattr(scanner_changes, "snapshot_hash", mutate_row_during_hash)

    processed = scanner_changes.drain_pending_verifications()
    session.expire_all()
    content = session.get(AssetContent, content_id)

    assert processed == 0
    assert content.hash is None
    assert content.mtime_ns == changed_mtime_ns
    assert scanner_changes._pending_verification_ids == []


def test_pending_verification_interrupts_between_entries(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content_ids: list[str] = []
    for index in range(3):
        path = temp_dir / f"pending-{index}.bin"
        path.write_bytes(f"pending-{index}".encode())
        content_id = _seed_content(path)
        content_ids.append(content_id)
        scanner_changes.queue_pending_verification(content_id)

    transaction_count = 0
    real_run_write_txn = scanner_changes.run_write_txn

    def count_run_write_txn(work):
        nonlocal transaction_count
        result = real_run_write_txn(work)
        transaction_count += 1
        return result

    monkeypatch.setattr(scanner_changes, "run_write_txn", count_run_write_txn)

    processed = scanner_changes.drain_pending_verifications(
        interrupt_check=lambda: transaction_count == 1
    )

    assert processed == 1
    assert transaction_count == 1
    assert scanner_changes._pending_verification_ids == content_ids[1:]


def test_transition_drain_does_not_stat_inside_write_transaction(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "transition-ready.bin"
    path.write_bytes(b"ready")
    _seed_content(path)
    hash_mode_state._PENDING_QUEUE.append(hash_mode_state._PendingEntry(str(path)))
    hash_mode_state._PENDING_PATHS.add(str(path))
    _guard_stat_during_write_txn(hash_mode_state, monkeypatch)

    hash_mode_state.drain_transition_queue()

    assert hash_mode_state.pending_transition_count() == 0


def test_transition_drain_trusts_gone_preflight_without_statting_in_transaction(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "transition-gone.bin"
    path.write_bytes(b"gone")
    content_id = _seed_content(path)
    hash_mode_state._PENDING_QUEUE.append(hash_mode_state._PendingEntry(str(path)))
    hash_mode_state._PENDING_PATHS.add(str(path))
    path.unlink()
    _guard_stat_during_write_txn(hash_mode_state, monkeypatch)

    hash_mode_state.drain_transition_queue()
    session.expire_all()

    assert session.get(AssetContent, content_id).is_missing is True
    assert hash_mode_state.pending_transition_count() == 0


def test_transition_drain_retries_when_row_changes_during_hashing(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "transition-row-changed.bin"
    path.write_bytes(b"row changes")
    content_id = _seed_content(path)
    changed_mtime_ns = path.stat().st_mtime_ns + 1
    hash_mode_state._PENDING_QUEUE.append(hash_mode_state._PendingEntry(str(path)))
    hash_mode_state._PENDING_PATHS.add(str(path))
    real_snapshot_hash = hash_mode_state.snapshot_hash

    def mutate_row_during_hash(candidate_path: str):
        snapshot = real_snapshot_hash(candidate_path)

        def mutate_row(write_session):
            write_session.get(AssetContent, content_id).mtime_ns = changed_mtime_ns

        db_mod.run_write_txn(mutate_row)
        return snapshot

    monkeypatch.setattr(hash_mode_state, "snapshot_hash", mutate_row_during_hash)

    hash_mode_state.drain_transition_queue()
    session.expire_all()
    content = session.get(AssetContent, content_id)

    assert content.hash is None
    assert content.mtime_ns == changed_mtime_ns
    assert list(hash_mode_state._PENDING_QUEUE) == [
        hash_mode_state._PendingEntry(str(path), ticks=1)
    ]


def test_transition_drain_interrupts_then_resumes_before_completing_mode(
    session, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(3):
        path = temp_dir / f"transition-{index}.bin"
        path.write_bytes(f"transition-{index}".encode())
        _seed_content(path)
    db_mod.run_write_txn(
        lambda write_session: hash_mode_state.write_stored_mode(write_session, "off")
    )
    hash_mode_state.enqueue_transition_work(session, "off_to_on")

    transaction_count = 0
    real_run_write_txn = hash_mode_state.run_write_txn

    def count_run_write_txn(work):
        nonlocal transaction_count
        result = real_run_write_txn(work)
        transaction_count += 1
        return result

    monkeypatch.setattr(hash_mode_state, "run_write_txn", count_run_write_txn)

    hash_mode_state.drain_transition_queue(
        interrupt_check=lambda: transaction_count == 1
    )
    session.expire_all()

    assert transaction_count == 1
    assert hash_mode_state.pending_transition_count() == 2
    assert hash_mode_state._off_to_on_transition_in_flight is True
    assert hash_mode_state.read_stored_mode(session) == "off"

    monkeypatch.setattr(hash_mode_state, "run_write_txn", real_run_write_txn)
    hash_mode_state.drain_transition_queue()
    session.expire_all()

    assert hash_mode_state.pending_transition_count() == 0
    assert hash_mode_state._PENDING_PATHS == set()
    assert hash_mode_state._off_to_on_transition_in_flight is False
    assert hash_mode_state.read_stored_mode(session) == "on"
