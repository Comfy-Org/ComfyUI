import os
import threading
import uuid

import folder_paths
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import app.assets.mode as mode_module
import app.database.db as db_mod
from app.assets import scanner
from app.assets import scanner_changes
from app.assets.database.queries.records import create_content, create_record
from app.assets.services import hash_mode_state
from app.assets.services import ingest

_BARRIER_TIMEOUT = 5
_PROBE_LOCK_DEADLINE_SECONDS = 0.5
_PROBE_BUSY_TIMEOUT_MS = 250
_LEASE_HELD = "writer lease was held across out-of-transaction work"


@pytest.fixture
def file_database(tmp_path, monkeypatch):
    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    if hasattr(db_mod, "WriteSession"):
        monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    for factory in (db_mod.Session, getattr(db_mod, "WriteSession", None)):
        if factory is not None:
            factory.kw["bind"].dispose()
    db_mod._db_lock.release(force=True)


@pytest.fixture
def hashing_on():
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


def _probe_write() -> None:
    name = f"probe-{uuid.uuid4().hex}"
    db_mod.run_write_txn(
        lambda session: session.execute(
            text("INSERT INTO tags (name) VALUES (:name)"), {"name": name}
        )
    )


@pytest.fixture
def impatient_probe(monkeypatch):
    """Let a probe surface a held lease as an error instead of waiting out the real deadline.

    An unheld lease is acquired on the first attempt, so these shortened deadlines are
    only ever reached when the lease really is held.
    """
    monkeypatch.setattr(
        db_mod, "_WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS", _PROBE_LOCK_DEADLINE_SECONDS
    )
    monkeypatch.setattr(db_mod, "_SQLITE_BUSY_TIMEOUT_MS", _PROBE_BUSY_TIMEOUT_MS)


def _probe_write_outcome() -> Exception | None:
    try:
        _probe_write()
    except OperationalError as exc:
        return exc
    return None


def _blocking_fake(entered: threading.Event, release: threading.Event, real_fn):
    def fake(*args, **kwargs):
        entered.set()
        release.wait()
        return real_fn(*args, **kwargs)

    return fake


class _IngestWriteTxnFsTracker:
    def __init__(self, monkeypatch) -> None:
        self.calls: list[str] = []
        self.inside = False
        real_run_write_txn = ingest.run_write_txn

        def track_write_transaction(work):
            def tracked_work(session):
                self.inside = True
                try:
                    return work(session)
                finally:
                    self.inside = False

            return real_run_write_txn(tracked_work)

        monkeypatch.setattr(ingest, "run_write_txn", track_write_transaction)
        for name in (
            "lookup_for_view",
            "refresh_qualified_content",
            "_file_signature",
            "_file_signature_matches",
        ):
            if hasattr(ingest, name):
                real = getattr(ingest, name)
                monkeypatch.setattr(ingest, name, self._track(name, real))
        monkeypatch.setattr(ingest.os, "stat", self._track("os.stat", os.stat))
        monkeypatch.setattr(
            ingest.os.path,
            "isfile",
            self._track("os.path.isfile", os.path.isfile),
        )

    def _track(self, label: str, real):
        def tracked(*args, **kwargs):
            if self.inside:
                self.calls.append(label)
            return real(*args, **kwargs)

        return tracked


@pytest.fixture
def ingest_write_txn_fs_tracker(monkeypatch) -> _IngestWriteTxnFsTracker:
    return _IngestWriteTxnFsTracker(monkeypatch)


def test_reused_upload_does_not_touch_the_filesystem_inside_write_transaction(
    file_database, tmp_path, ingest_write_txn_fs_tracker
) -> None:
    existing_path = tmp_path / "reuse-existing.bin"
    existing_path.write_bytes(b"shared bytes")
    stat_result = existing_path.stat()
    digest, _ = ingest._snapshot_hash_with_retry(str(existing_path))
    stored_hash = ingest.to_stored_hash(digest)

    db_mod.run_write_txn(
        lambda session: create_content(
            session,
            str(existing_path),
            hash=stored_hash,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    )
    upload_path = tmp_path / "reuse-upload.part"
    upload_path.write_bytes(b"shared bytes")

    result = ingest.upload_from_temp_path(str(upload_path), name="reuse.bin")

    assert result.content_id is not None
    assert ingest_write_txn_fs_tracker.calls == []


def test_new_upload_does_not_touch_the_filesystem_inside_write_transaction(
    file_database, tmp_path, monkeypatch, ingest_write_txn_fs_tracker
) -> None:
    upload_path = tmp_path / "new-upload.part"
    upload_path.write_bytes(b"new bytes")
    destination = tmp_path / "new-upload.bin"
    monkeypatch.setattr(
        ingest,
        "_hash_mode_dest_path",
        lambda *_args: str(destination),
    )

    result = ingest.upload_from_temp_path(
        str(upload_path),
        name="new-upload.bin",
        tags=["output"],
    )

    assert result.ref.file_path == str(destination)
    assert ingest_write_txn_fs_tracker.calls == []


def test_cached_registration_does_not_touch_the_filesystem_inside_write_transaction(
    file_database, ingest_write_txn_fs_tracker
) -> None:
    path = os.path.join(folder_paths.get_output_directory(), "cached-output.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(b"cached bytes")
    stat_result = os.stat(path)
    db_mod.run_write_txn(
        lambda session: create_content(
            session,
            path,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    )

    try:
        result = ingest.register_cached_output(path)
    finally:
        os.unlink(path)

    assert result is not None
    assert ingest_write_txn_fs_tracker.calls == []


def test_settle_destination_does_not_touch_the_filesystem_inside_write_transaction(
    file_database, tmp_path, ingest_write_txn_fs_tracker
) -> None:
    path = tmp_path / "settle-output.bin"
    path.write_bytes(b"incumbent bytes")
    stat_result = path.stat()
    db_mod.run_write_txn(
        lambda session: create_content(
            session,
            str(path),
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    )

    ingest._settle_destination_before_write(str(path))

    assert ingest_write_txn_fs_tracker.calls == []


def test_register_file_in_place_does_not_touch_the_filesystem_inside_write_transaction(
    file_database, tmp_path, ingest_write_txn_fs_tracker
) -> None:
    path = tmp_path / "register-in-place.bin"
    path.write_bytes(b"in-place bytes")

    result = ingest.register_file_in_place(str(path), path.name, ["output"])

    assert result.ref.file_path == str(path)
    assert ingest_write_txn_fs_tracker.calls == []


def test_seed_recovery_hashing_does_not_hold_the_write_lock(
    file_database, impatient_probe, hashing_on, tmp_path, monkeypatch
):
    path = tmp_path / "recoverable.bin"
    path.write_bytes(b"recoverable bytes")

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        scanner_changes,
        "snapshot_hash",
        _blocking_fake(entered, release, scanner_changes.snapshot_hash),
    )

    stat = path.stat()
    spec: scanner.SeedAssetSpec = {
        "abs_path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "info_name": "recoverable.bin",
        "tags": [],
        "fname": None,
        "metadata": None,
        "mime_type": None,
        "job_id": None,
    }

    result: dict[str, int] = {}

    def _seed() -> None:
        result["created"] = scanner.insert_asset_specs([spec], set())

    worker = threading.Thread(target=_seed)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()

    assert probe_error is None, _LEASE_HELD
    assert result["created"] == 1


def test_pending_verification_hashing_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    path = tmp_path / "verify-me.bin"
    path.write_bytes(b"verify me")
    stat = path.stat()

    def _seed(session):
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        create_record(session, content.id, "verify-me.bin")
        return content.id

    content_id = db_mod.run_write_txn(_seed)

    scanner_changes.clear_pending_verifications()
    scanner_changes.queue_pending_verification(content_id)

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        scanner_changes,
        "snapshot_hash",
        _blocking_fake(entered, release, scanner_changes.snapshot_hash),
    )

    result: dict[str, int] = {}

    def _drain() -> None:
        result["processed"] = scanner_changes.drain_pending_verifications()

    worker = threading.Thread(target=_drain)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()
        scanner_changes.clear_pending_verifications()

    assert probe_error is None, _LEASE_HELD
    assert result["processed"] == 1


def test_transition_hashing_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    path = tmp_path / "transition-me.bin"
    path.write_bytes(b"transition me")
    stat = path.stat()

    db_mod.run_write_txn(
        lambda session: create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
    )

    hash_mode_state.clear_transition_queue()
    hash_mode_state._PENDING_QUEUE.append(hash_mode_state._PendingEntry(str(path)))
    hash_mode_state._PENDING_PATHS.add(str(path))

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        hash_mode_state,
        "snapshot_hash",
        _blocking_fake(entered, release, hash_mode_state.snapshot_hash),
    )

    result: dict[str, bool] = {}

    def _drain() -> None:
        hash_mode_state.drain_transition_queue()
        result["done"] = True

    worker = threading.Thread(target=_drain)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()
        hash_mode_state.clear_transition_queue()

    assert probe_error is None, _LEASE_HELD
    assert result.get("done") is True


def test_enrichment_hashing_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    path = tmp_path / "enrich-hash.bin"
    path.write_bytes(b"enrich me via hash")
    stat = path.stat()

    def _seed(session):
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        record = create_record(session, content.id, "enrich-hash.bin")
        return content.id, record.id

    content_id, record_id = db_mod.run_write_txn(_seed)

    row = scanner.UnenrichedContent(
        content_id,
        record_id,
        str(path),
        True,
        observed_size_bytes=stat.st_size,
        observed_mtime_ns=stat.st_mtime_ns,
    )

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        scanner,
        "snapshot_hash",
        _blocking_fake(entered, release, scanner.snapshot_hash),
    )

    result: dict[str, object] = {}

    def _enrich() -> None:
        result["outcome"] = scanner.enrich_assets_batch(
            [row], extract_metadata=False, compute_hash=True
        )

    worker = threading.Thread(target=_enrich)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()

    assert probe_error is None, _LEASE_HELD
    assert result["outcome"] == (1, [])


def test_enrichment_metadata_extraction_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    path = tmp_path / "enrich-metadata.bin"
    path.write_bytes(b"enrich me via metadata")
    stat = path.stat()

    def _seed(session):
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        record = create_record(session, content.id, "enrich-metadata.bin")
        return content.id, record.id

    content_id, record_id = db_mod.run_write_txn(_seed)

    row = scanner.UnenrichedContent(
        content_id,
        record_id,
        str(path),
        False,
        observed_size_bytes=stat.st_size,
        observed_mtime_ns=stat.st_mtime_ns,
    )

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        scanner,
        "extract_file_metadata",
        _blocking_fake(entered, release, scanner.extract_file_metadata),
    )

    result: dict[str, object] = {}

    def _enrich() -> None:
        result["outcome"] = scanner.enrich_assets_batch(
            [row], extract_metadata=True, compute_hash=False
        )

    worker = threading.Thread(target=_enrich)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()

    assert probe_error is None, _LEASE_HELD
    assert result["outcome"] == (1, [])


def test_enrichment_apply_does_not_stat_inside_write_transaction(
    file_database, tmp_path, monkeypatch
):
    path = tmp_path / "enrich-without-in-transaction-stat.bin"
    path.write_bytes(b"prepared before the write")
    stat_result = path.stat()

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
        record = create_record(session, content.id, path.name)
        return content.id, record.id

    content_id, record_id = db_mod.run_write_txn(seed)
    row = scanner.UnenrichedContent(
        content_id,
        record_id,
        str(path),
        observed_size_bytes=stat_result.st_size,
        observed_mtime_ns=stat_result.st_mtime_ns,
    )
    real_run_write_txn = scanner.run_write_txn
    real_stat = scanner.os.stat
    inside = False

    def track_write_transaction(work):
        def tracked_work(session):
            nonlocal inside
            inside = True
            try:
                return work(session)
            finally:
                inside = False

        return real_run_write_txn(tracked_work)

    def reject_in_transaction_stat(*args, **kwargs):
        if inside:
            raise AssertionError("enrichment stat ran inside the write transaction")
        return real_stat(*args, **kwargs)

    monkeypatch.setattr(scanner, "run_write_txn", track_write_transaction)
    monkeypatch.setattr(scanner.os, "stat", reject_in_transaction_stat)

    result = scanner.enrich_assets_batch(
        [row], extract_metadata=True, compute_hash=False
    )

    assert result == (1, [])


def test_scanner_reference_stat_walk_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    root = tmp_path / "models"
    root.mkdir()
    path = root / "catalogued.bin"
    path.write_bytes(b"catalogued bytes")
    stat = path.stat()

    def _seed(session):
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        create_record(session, content.id, "catalogued.bin")

    db_mod.run_write_txn(_seed)

    entered = threading.Event()
    release = threading.Event()
    real_stat = scanner.os.stat

    def blocking_stat(target, *args, **kwargs):
        if str(target) == str(path):
            entered.set()
            assert release.wait(timeout=_BARRIER_TIMEOUT)
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(scanner, "get_scan_prefixes_for_root", lambda _root: [str(root)])
    monkeypatch.setattr(scanner.os, "stat", blocking_stat)

    survivors: dict[str, set[str]] = {}

    def _scan() -> None:
        survivors["found"] = scanner.sync_root_safely("models")

    worker = threading.Thread(target=_scan)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()

    assert probe_error is None, _LEASE_HELD
    assert survivors["found"] == {str(path)}


def test_download_hash_resolution_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    from app.assets.services import asset_management, lookup

    path = tmp_path / "servable.bin"
    path.write_bytes(b"servable bytes")
    stat = path.stat()
    digest = "b" * 64
    stored_hash = f"blake3:{digest}"

    def _seed(session):
        content = create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        create_record(session, content.id, "servable.bin")

    db_mod.run_write_txn(_seed)

    entered = threading.Event()
    release = threading.Event()
    real_stat = lookup.os.stat

    def blocking_stat(target, *args, **kwargs):
        if str(target) == str(path):
            entered.set()
            assert release.wait(timeout=_BARRIER_TIMEOUT)
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(lookup.os, "stat", blocking_stat)
    monkeypatch.setattr(lookup, "is_temp_path", lambda _path: False)

    resolved: dict[str, object] = {}

    def _resolve() -> None:
        resolved["result"] = asset_management.resolve_hash_to_path(stored_hash)

    worker = threading.Thread(target=_resolve)
    worker.start()
    try:
        assert entered.wait(timeout=_BARRIER_TIMEOUT)
        probe_error = _probe_write_outcome()
    finally:
        release.set()
        worker.join(timeout=_BARRIER_TIMEOUT)
        assert not worker.is_alive()

    assert probe_error is None, _LEASE_HELD
    assert resolved["result"] is not None
    assert resolved["result"].abs_path == str(path)
