from pathlib import Path

import folder_paths
import pytest
from blake3 import blake3
from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)
from app.assets.services.lookup import lookup_for_view
from app.assets.services.path_utils import get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash


@pytest.fixture(autouse=True)
def transition_queue():
    clear_transition_queue()
    yield
    clear_transition_queue()


def _stored_hash(path: Path) -> str:
    snapshot = snapshot_hash(str(path))
    assert snapshot is not None
    digest, _ = snapshot
    return to_stored_hash(digest)


def test_off_to_on_transition_hashes_null_rows_and_persists_mode(session, temp_dir, monkeypatch):
    paths = [temp_dir / "first.bin", temp_dir / "second.bin"]
    for index, path in enumerate(paths):
        path.write_bytes(f"bytes-{index}".encode())
        stat = path.stat()
        create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    assert {content.hash for content in contents} == {_stored_hash(path) for path in paths}
    assert read_stored_mode(session) == "on"


def test_transition_drain_splits_changed_content(session, temp_dir, monkeypatch):
    path = temp_dir / "changed.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = path.stat()
    old_content = create_content(
        session, str(path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(session, old_content_id, "changed.bin")
    path.write_bytes(b"new bytes")

    enqueue_transition_work(session, "off_to_on")
    drain_transition_queue(session)
    session.commit()

    contents = list(session.scalars(select(AssetContent)))
    live_content = next(content for content in contents if not content.is_missing)
    records = list(session.scalars(select(Asset)))
    assert session.get(AssetContent, old_content_id).is_missing is True
    assert live_content.hash == _stored_hash(path)
    assert len(records) == 2
    assert any(record.content_id == live_content.id for record in records)


def test_transition_drain_serves_unchanged_content_whose_stored_stat_went_stale(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "unchanged.bin"
    path.write_bytes(b"bytes that outlive the mode flip")
    stat = path.stat()
    stored_hash = _stored_hash(path)
    size_before_the_restore = stat.st_size - 1
    mtime_before_the_restore = stat.st_mtime_ns - 1_000_000_000
    content = create_content(
        session, str(path), stored_hash, size_before_the_restore, mtime_before_the_restore
    )
    content_id = content.id
    create_record(session, content_id, "unchanged.bin")
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_dir / "never"))
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    assert lookup_for_view(session, stored_hash) is None, (
        "precondition: the stale stat makes the row unservable by hash"
    )

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    refreshed = session.get(AssetContent, content_id)
    assert (refreshed.size_bytes, refreshed.mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    served = lookup_for_view(session, stored_hash)
    assert served is not None and served.id == content_id, (
        "the drain verified these bytes but left the row unservable until the next full scan"
    )
    assert [row.id for row in session.scalars(select(AssetContent))] == [content_id]
    assert [row.content_id for row in session.scalars(select(Asset))] == [content_id]
    assert read_stored_mode(session) == "on"


def test_transition_drain_requeues_permission_errors_and_processes_other_paths(
    session, temp_dir, monkeypatch
):
    protected_path = temp_dir / "protected.bin"
    healthy_path = temp_dir / "healthy.bin"
    protected_path.write_bytes(b"protected")
    healthy_payload = b"healthy"
    healthy_path.write_bytes(healthy_payload)
    for path in (protected_path, healthy_path):
        stat = path.stat()
        create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    def hash_or_raise(candidate_path: str):
        if candidate_path == str(protected_path):
            raise PermissionError("denied")
        return snapshot_hash(candidate_path)

    monkeypatch.setattr(hash_mode_state, "snapshot_hash", hash_or_raise)
    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)

    drain_transition_queue(session)

    healthy_content = session.scalar(
        select(AssetContent).where(AssetContent.path == str(healthy_path))
    )
    assert healthy_content is not None
    assert healthy_content.hash == to_stored_hash(blake3(healthy_payload).hexdigest())
    assert hash_mode_state.pending_transition_count() == 1
    assert read_stored_mode(session) == "off"


def test_transition_drain_marks_deleted_path_missing_and_completes_transition(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "vanished.bin"
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    path.write_bytes(b"bytes that die during the outage")
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    record = create_record(session, content_id, "vanished.bin")
    record_id = record.id
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    path.unlink()

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    assert session.get(AssetContent, content_id).is_missing is True, (
        "a path deleted while every server was down must be marked missing, not requeued forever"
    )
    assert session.get(AssetTag, {"asset_id": record_id, "tag_name": "missing"}) is not None
    assert hash_mode_state.pending_transition_count() == 0, (
        "the queue never empties while a deleted path is requeued unconditionally"
    )
    assert read_stored_mode(session) == "on", (
        "the completion gate stays wedged at 'off' for the process lifetime when the queue "
        "cannot drain, so hash-addressed serving never gets its re-verification certificate"
    )


def test_transition_drain_requeues_transient_stat_error_without_marking_it_missing(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "transient-stat-error.bin"
    path.write_bytes(b"present the whole time")
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    monkeypatch.setattr(hash_mode_state, "snapshot_hash", lambda candidate_path: None)
    real_stat = hash_mode_state.os.stat

    def flaky_stat(candidate_path, *args, **kwargs):
        if candidate_path == str(path):
            raise NotADirectoryError("stale handle mid-drain")
        return real_stat(candidate_path, *args, **kwargs)

    monkeypatch.setattr(hash_mode_state.os, "stat", flaky_stat)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    assert session.get(AssetContent, content_id).is_missing is False, (
        "a transient stat failure on a file that is still present must not mark its row missing"
    )
    assert hash_mode_state.pending_transition_count() == 1
    assert read_stored_mode(session) == "off"


def test_transition_drain_requeues_unstable_present_file_without_marking_it_missing(
    session, temp_dir, monkeypatch
):
    path = temp_dir / "still-being-written.bin"
    path.write_bytes(b"a partial write in flight")
    stat = path.stat()
    content = create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    content_id = content.id
    create_record(session, content_id, "still-being-written.bin")
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    monkeypatch.setattr(hash_mode_state, "snapshot_hash", lambda candidate_path: None)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    assert session.get(AssetContent, content_id).is_missing is False, (
        "snapshot_hash returns None for drift too; a file still on disk is unstable, not gone"
    )
    assert hash_mode_state.pending_transition_count() == 1
    assert read_stored_mode(session) == "off"


def test_transition_drain_mixes_a_deleted_path_with_a_healthy_one(session, temp_dir, monkeypatch):
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    deleted_path = temp_dir / "deleted.bin"
    healthy_path = temp_dir / "survivor.bin"
    deleted_path.write_bytes(b"deleted during the outage")
    healthy_path.write_bytes(b"survivor")
    content_ids = {}
    for path in (deleted_path, healthy_path):
        stat = path.stat()
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        content_ids[path] = content.id
        create_record(session, content.id, path.name)
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)
    healthy_hash = _stored_hash(healthy_path)
    deleted_path.unlink()

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    drain_transition_queue(session)
    session.commit()

    assert session.get(AssetContent, content_ids[deleted_path]).is_missing is True
    healthy_content = session.get(AssetContent, content_ids[healthy_path])
    assert (healthy_content.is_missing, healthy_content.hash) == (False, healthy_hash), (
        "one dead path must not cost the healthy paths behind it their hashes"
    )
    assert hash_mode_state.pending_transition_count() == 0
    assert read_stored_mode(session) == "on"


def test_transition_drain_skips_out_of_root_path(session, temp_dir, monkeypatch, caplog):
    import logging

    outside_path = temp_dir / "orphan.bin"
    outside_path.write_bytes(b"old bytes")
    old_snapshot = snapshot_hash(str(outside_path))
    assert old_snapshot is not None
    old_digest, _ = old_snapshot
    stat = outside_path.stat()
    old_content = create_content(
        session, str(outside_path), to_stored_hash(old_digest), stat.st_size, stat.st_mtime_ns
    )
    old_content_id = old_content.id
    create_record(session, old_content_id, "orphan.bin")
    outside_path.write_bytes(b"new bytes")

    with pytest.raises(ValueError):
        get_name_and_tags_from_asset_path(str(outside_path))

    enqueue_transition_work(session, "off_to_on")

    with caplog.at_level(logging.WARNING):
        try:
            drain_transition_queue(session)
        except ValueError as error:
            pytest.fail(
                f"an out-of-root path escaped the drain as {error!r}; setup_database turns that "
                f"into sys.exit(1) when --enable-assets is set"
            )
    session.commit()

    assert session.get(AssetContent, old_content_id).is_missing is False
    assert hash_mode_state.pending_transition_count() == 0
    assert any("orphan.bin" in record.getMessage() for record in caplog.records)
