import threading
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import app.assets.mode as mode_module
import app.database.db as db_mod
from app.assets import scanner
from app.assets import scanner_changes
from app.assets.database.queries.records import create_content, create_record
from app.assets.services import hash_mode_state

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

    with db_mod.Session() as session:
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        create_record(session, content.id, "verify-me.bin")
        session.commit()
        content_id = content.id

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

    with db_mod.Session() as session:
        create_content(session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        session.commit()

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

    with db_mod.Session() as session:
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        record = create_record(session, content.id, "enrich-hash.bin")
        session.commit()
        content_id, record_id = content.id, record.id

    row = scanner.UnenrichedContent(content_id, record_id, str(path), True)

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

    with db_mod.Session() as session:
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        record = create_record(session, content.id, "enrich-metadata.bin")
        session.commit()
        content_id, record_id = content.id, record.id

    row = scanner.UnenrichedContent(content_id, record_id, str(path), False)

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


def test_scanner_reference_stat_walk_does_not_hold_the_write_lock(
    file_database, impatient_probe, tmp_path, monkeypatch
):
    root = tmp_path / "models"
    root.mkdir()
    path = root / "catalogued.bin"
    path.write_bytes(b"catalogued bytes")
    stat = path.stat()

    with db_mod.Session() as session:
        content = create_content(
            session, str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        create_record(session, content.id, "catalogued.bin")
        session.commit()

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

    with db_mod.Session() as session:
        content = create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        create_record(session, content.id, "servable.bin")
        session.commit()

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
