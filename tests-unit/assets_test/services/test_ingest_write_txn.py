import os

import json
import logging
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import folder_paths
import pytest
from aiohttp import web
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

import app.assets.services.ingest as ingest
import app.database.db as db_mod
from app.assets.api import routes, schemas_in
from app.assets.database.queries.records import create_content, create_record

_BARRIER_TIMEOUT = 5
_PROBE_BUDGET_SECONDS = 1.0


def _output_path(name: str) -> str:
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


@pytest.fixture
def file_database(tmp_path, monkeypatch):
    """A real file-backed engine (WAL + BEGIN IMMEDIATE) for lock-hold barrier tests.

    ``mock_create_session`` binds an in-memory StaticPool engine, which never
    contends on a write lock and would make a lock-hold assertion vacuously
    true; only the production runtime engines built by ``init_db`` enforce it.
    """
    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    for factory in (db_mod.Session, db_mod.WriteSession):
        if factory is not None:
            factory.kw["bind"].dispose()
    db_mod._db_lock.release(force=True)


def _probe_write() -> None:
    db_mod.run_write_txn(
        lambda session: session.execute(
            text("INSERT INTO tags (name) VALUES (:name)"),
            {"name": f"probe-{os.urandom(8).hex()}"},
        )
    )


def _blocking_fake(entered: threading.Event, release: threading.Event, real_fn):
    def fake(*args, **kwargs):
        entered.set()
        release.wait()
        return real_fn(*args, **kwargs)

    return fake


class _GuardedHelperInsideWriteTxn(BaseException):
    """Barrier trip. Derives from BaseException so ``except Exception`` cannot eat it."""


class _StubbedWriteTxn:
    """Runs write-transaction callables without a database.

    The guards below trip only on the three named helpers, so this is a stub that
    happens to assert, not proof that a callable left the filesystem alone.
    """

    def __init__(self) -> None:
        self.depth = 0

    def run_write_txn(self, work):
        self.depth += 1
        try:
            return work(object())
        finally:
            self.depth -= 1

    def _guard(self, label: str, real):
        def guarded(*args, **kwargs):
            if self.depth:
                raise _GuardedHelperInsideWriteTxn(f"{label} ran inside run_write_txn")
            return real(*args, **kwargs)

        return guarded

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(ingest, "run_write_txn", self.run_write_txn)
        for name in ("_extract_system_metadata_sync", "_snapshot_hash_with_retry"):
            monkeypatch.setattr(ingest, name, self._guard(name, getattr(ingest, name)))
        monkeypatch.setattr(
            ingest.os.path,
            "isfile",
            self._guard("os.path.isfile", os.path.isfile),
        )


@pytest.fixture
def stubbed_write_txn(monkeypatch) -> _StubbedWriteTxn:
    barrier = _StubbedWriteTxn()
    barrier.install(monkeypatch)
    return barrier


def _reuse_path(monkeypatch, _tmp):
    preflight = SimpleNamespace(signature=object())
    prepared = object()
    monkeypatch.setattr(ingest, "_preflight_upload_record", lambda *_a: preflight)
    monkeypatch.setattr(ingest, "_prepare_upload_record", lambda _p: prepared)
    spec = ingest._UploadRecordSpec("asset", [], None, {}, None)
    return (
        "_apply_reused_upload_record",
        prepared,
        lambda: ingest._reuse_qualified_content("blake3:hash", spec),
    )


def _settle_path(monkeypatch, _tmp):
    preflight = SimpleNamespace(signature=object())
    prepared = SimpleNamespace(facts=None)
    monkeypatch.setattr(ingest, "_preflight_settle_target", lambda _d: preflight)
    monkeypatch.setattr(ingest, "_prepare_settle_target", lambda _p: prepared)
    return (
        "_apply_settle_target",
        prepared,
        lambda: ingest._settle_destination_before_write(_output_path("settle-unstable.bin")),
    )


def _cached_path(monkeypatch, _tmp):
    path = _output_path("cached-always-stale.bin")
    with open(path, "wb") as file:
        file.write(b"output")
    preflight = SimpleNamespace(
        content_id="content-always-stale",
        sibling_id=None,
        sibling_metadata=None,
        signature=None,
    )
    monkeypatch.setattr(ingest, "_preflight_cached_registration", lambda _l: preflight)
    return "_apply_cached_registration", None, lambda: ingest.register_cached_output(path)


@pytest.mark.parametrize(
    ("build_path", "raises"),
    [
        pytest.param(_reuse_path, True, id="reused-upload"),
        pytest.param(_settle_path, True, id="settle-destination"),
        pytest.param(_cached_path, False, id="cached-registration"),
    ],
)
def test_ingest_paths_refuse_after_four_stale_preflights(
    stubbed_write_txn, monkeypatch, tmp_path, build_path, raises
) -> None:
    """A preflight that never settles is refused, not persisted from mixed facts."""
    monkeypatch.setattr(ingest, "_assert_signature_current", lambda _s: None)
    apply_name, expected_prepared, invoke = build_path(monkeypatch, tmp_path)
    attempts: list[object] = []

    def stale_apply(_session, observed, *_args):
        attempts.append(observed)
        raise ingest._PreflightStale

    monkeypatch.setattr(ingest, apply_name, stale_apply)

    if raises:
        with pytest.raises(ingest.UploadUnstableError):
            invoke()
    else:
        assert invoke() is None

    assert len(attempts) == 4, "the retry budget is four attempts, then refusal"
    if expected_prepared is not None:
        assert attempts == [expected_prepared] * 4


def test_cached_registration_refusal_is_an_outcome_not_a_crash(
    stubbed_write_txn, monkeypatch, caplog
) -> None:
    path = _output_path("cached-refusal-logging.bin")
    with open(path, "wb") as file:
        file.write(b"output")
    monkeypatch.setattr(
        ingest,
        "_preflight_cached_registration",
        lambda _l: SimpleNamespace(
            content_id="c", sibling_id=None, sibling_metadata=None, signature=None
        ),
    )

    def stale_apply(_session, _preflight, *_args):
        raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_apply_cached_registration", stale_apply)

    with caplog.at_level(logging.INFO):
        assert ingest.register_cached_output(path) is None

    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "preflight changed" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == [], (
        "giving up is an expected outcome, not a crash"
    )
    assert not [
        r
        for r in caplog.records
        if r.getMessage().startswith("[assets-event] ingest.register_failed")
    ], "giving up is not a registration failure event"


@pytest.mark.asyncio
async def test_upload_route_reports_an_unsettleable_destination_as_unstable(
    stubbed_write_txn, monkeypatch, tmp_path
) -> None:
    temp_path = tmp_path / "unstable-destination.bin"
    temp_path.write_bytes(b"upload bytes")

    parsed = schemas_in.ParsedUpload(
        file_present=True,
        file_written=temp_path.stat().st_size,
        file_client_name="unstable-destination.bin",
        tmp_path=str(temp_path),
        tags_raw=["output", "unit-tests"],
        provided_name="unstable-destination.bin",
        user_metadata_raw=None,
        provided_hash=None,
        provided_hash_exists=None,
    )
    monkeypatch.setattr(routes, "_ASSETS_ENABLED", True)
    monkeypatch.setattr(
        routes, "parse_multipart_upload", AsyncMock(return_value=parsed)
    )
    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "test-user"),
    )
    preflight = SimpleNamespace(signature=object())
    prepared = SimpleNamespace(facts=None)
    monkeypatch.setattr(ingest, "_preflight_settle_target", lambda _dest: preflight)
    monkeypatch.setattr(ingest, "_prepare_settle_target", lambda _preflight: prepared)
    monkeypatch.setattr(ingest, "_assert_signature_current", lambda _signature: None)

    def stale_apply(_session, _prepared):
        raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_apply_settle_target", stale_apply)

    response = await routes.upload_asset(AsyncMock(spec=web.Request))

    assert isinstance(response, web.Response)
    assert response.status == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == "UPLOAD_UNSTABLE"


def test_cached_registration_skips_extraction_when_live_content_is_missing(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("cached-missing-no-extraction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("missing content must not trigger metadata extraction")

    monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extraction_must_not_run)
    try:
        assert ingest.register_cached_output(path) is None
    finally:
        os.unlink(path)


def test_cached_registration_skips_extraction_when_reusing_a_sibling(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("cached-sibling-no-extraction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("sibling metadata must be reused without extraction")

    with mock_create_session() as session:
        content = create_content(session, path, size_bytes=6)
        create_record(
            session,
            content.id,
            "sibling.bin",
            system_metadata={"source": "sibling"},
        )
        session.commit()

    monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extraction_must_not_run)
    try:
        result = ingest.register_cached_output(path)
        assert result is not None
    finally:
        os.unlink(path)


def test_executed_registration_uses_the_write_transaction_runner(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("executed-write-transaction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    calls: list[None] = []

    def record_call(work):
        calls.append(None)
        with mock_create_session() as session:
            result = work(session)
            session.commit()
            return result

    monkeypatch.setattr(ingest, "run_write_txn", record_call, raising=False)
    try:
        result = ingest.register_executed_output(path)
        assert result is not None
        assert len(calls) == 1
    finally:
        os.unlink(path)


def _registration_failure_event(caplog) -> str:
    events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[assets-event] ingest.register_failed")
    ]
    assert len(events) == 1
    return events[0]


def _seed_cached_content(mock_create_session, path: str) -> str:
    with mock_create_session() as session:
        content = create_content(session, path, size_bytes=os.path.getsize(path))
        session.commit()
        return content.id


def _apply_cached_preflight(session, preflight, path: str, system_metadata: dict[str, int]) -> None:
    ingest._apply_cached_registration(
        session,
        preflight,
        os.path.basename(path),
        ["output"],
        None,
        None,
        path,
        system_metadata,
    )


def test_reused_upload_apply_compares_the_row_to_preflight_row_values(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "reuse-row-behind-disk.bin"
    path.write_bytes(b"newer disk bytes")
    signature = ingest._file_signature(str(path))
    stored_hash = "blake3:" + "a" * 64

    def seed(session):
        content = create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=3,
            mtime_ns=7,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._UploadRecordPreflight(
        content_id,
        stored_hash,
        str(path),
        3,
        7,
        signature,
        ingest._UploadRecordSpec(path.name, [], None, {}, None),
    )
    prepared = ingest._PreparedUploadRecord(preflight, {})

    result = db_mod.run_write_txn(
        lambda session: ingest._apply_reused_upload_record(session, prepared)
    )

    assert result.content_id == content_id


def test_reconcile_unhashed_live_content_writes_hash_size_and_mtime_together(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "unhashed-live-content.bin"
    path.write_bytes(b"same-size bytes")
    facts = ingest._ContentFacts(
        "blake3:" + "f" * 64,
        path.stat().st_size,
        path.stat().st_mtime_ns,
    )

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=facts.size_bytes,
            mtime_ns=1,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    db_mod.run_write_txn(
        lambda session: ingest._reconcile_live_content_at_path(
            session,
            str(path),
            facts,
            content_written=False,
        )
    )

    with mock_create_session() as session:
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        assert (content.hash, content.size_bytes, content.mtime_ns) == facts


def test_reused_upload_apply_rejects_a_row_changed_after_preflight(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "reuse-row-mutated.bin"
    path.write_bytes(b"stable bytes")
    stat_result = path.stat()
    stored_hash = "blake3:" + "b" * 64

    def seed(session):
        content = create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    spec = ingest._UploadRecordSpec(path.name, [], None, {}, None)
    preflight = ingest._preflight_upload_record(stored_hash, None, spec)
    assert preflight is not None
    prepared = ingest._prepare_upload_record(preflight)

    def mutate_row(session):
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        content.mtime_ns = stat_result.st_mtime_ns + 1

    db_mod.run_write_txn(mutate_row)

    with pytest.raises(ingest._PreflightStale):
        db_mod.run_write_txn(
            lambda session: ingest._apply_reused_upload_record(session, prepared)
        )


def test_reused_upload_apply_accepts_a_qualified_row_without_mtime(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "reuse-row-without-mtime.bin"
    path.write_bytes(b"stable bytes")
    stored_hash = "blake3:" + "c" * 64

    def seed(session):
        content = create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=path.stat().st_size,
            mtime_ns=None,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    spec = ingest._UploadRecordSpec(path.name, [], None, {}, None)
    preflight = ingest._preflight_upload_record(stored_hash, None, spec)
    assert preflight is not None
    assert preflight.row_mtime_ns is None
    prepared = ingest._prepare_upload_record(preflight)

    result = db_mod.run_write_txn(
        lambda session: ingest._apply_reused_upload_record(session, prepared)
    )

    assert result.content_id == content_id


def test_upload_preflight_rejects_a_file_changed_after_qualification(
    mock_create_session, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "qualification-gap.bin"
    path.write_bytes(b"qualified bytes")
    stat_result = path.stat()
    stored_hash = "blake3:" + "d" * 64
    db_mod.run_write_txn(
        lambda session: create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    )
    real_file_signature = ingest._file_signature

    def mutate_then_sign(file_path: str):
        path.write_bytes(b"changed after qualification")
        return real_file_signature(file_path)

    monkeypatch.setattr(ingest, "_file_signature", mutate_then_sign)

    with pytest.raises(ingest._PreflightStale):
        ingest._preflight_upload_record(
            stored_hash,
            None,
            ingest._UploadRecordSpec(path.name, [], None, {}, None),
        )


def test_reused_upload_retries_when_pretransaction_signature_check_is_stale_once(
    mock_create_session, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "reuse-retry.bin"
    path.write_bytes(b"stable bytes")
    stat_result = path.stat()
    stored_hash = "blake3:" + "e" * 64
    db_mod.run_write_txn(
        lambda session: create_content(
            session,
            str(path),
            hash=stored_hash,
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
    )
    checks = 0

    def stale_once(_signature):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_assert_signature_current", stale_once, raising=False)

    result = ingest._reuse_qualified_content(
        stored_hash,
        ingest._UploadRecordSpec(path.name, [], None, {}, None),
    )

    assert result is not None
    assert checks == 2


def test_settle_apply_compares_the_row_to_preflight_row_values(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "settle-row-behind-disk.bin"
    path.write_bytes(b"newer disk bytes")
    signature = ingest._file_signature(str(path))

    def seed(session):
        content = create_content(
            session,
            str(path),
            hash="blake3:" + "1" * 64,
            size_bytes=3,
            mtime_ns=7,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._SettleTargetPreflight(
        content_id,
        "blake3:" + "1" * 64,
        3,
        7,
        signature,
    )

    db_mod.run_write_txn(
        lambda session: ingest._apply_settle_target(
            session,
            ingest._PreparedSettleTarget(preflight, None),
        )
    )

    with mock_create_session() as session:
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        assert content.is_missing is True


def test_settle_apply_rejects_a_row_changed_after_preflight(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "settle-row-mutated.bin"
    path.write_bytes(b"stable bytes")
    stat_result = path.stat()

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._preflight_settle_target(str(path))
    assert preflight is not None

    def mutate_row(session):
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        content.mtime_ns = stat_result.st_mtime_ns + 1

    db_mod.run_write_txn(mutate_row)

    with pytest.raises(ingest._PreflightStale):
        db_mod.run_write_txn(
            lambda session: ingest._apply_settle_target(
                session,
                ingest._PreparedSettleTarget(preflight, None),
            )
        )


def test_settle_apply_accepts_a_preflight_row_without_mtime(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "settle-row-without-mtime.bin"
    path.write_bytes(b"stable bytes")

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=path.stat().st_size,
            mtime_ns=None,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._preflight_settle_target(str(path))
    assert preflight is not None
    assert preflight.content_mtime_ns is None

    db_mod.run_write_txn(
        lambda session: ingest._apply_settle_target(
            session,
            ingest._PreparedSettleTarget(preflight, None),
        )
    )

    with mock_create_session() as session:
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        assert content.is_missing is True


def test_settle_retries_when_hash_facts_do_not_match_the_preflight_signature(
    monkeypatch,
) -> None:
    signature = ingest._FileSignature("/settle.bin", 4, 5)
    preflight = ingest._SettleTargetPreflight("content", None, 4, 5, signature)
    preparations = 0
    transactions = 0

    monkeypatch.setattr(ingest, "_preflight_settle_target", lambda _path: preflight)

    def prepare(_preflight):
        nonlocal preparations
        preparations += 1
        facts = ingest._ContentFacts(
            "blake3:" + "2" * 64,
            6 if preparations == 1 else signature.size_bytes,
            signature.mtime_ns,
        )
        return ingest._PreparedSettleTarget(preflight, facts)

    def run_once(_work):
        nonlocal transactions
        transactions += 1

    monkeypatch.setattr(ingest, "_prepare_settle_target", prepare)
    monkeypatch.setattr(ingest, "_assert_signature_current", lambda _signature: None)
    monkeypatch.setattr(ingest, "run_write_txn", run_once)

    ingest._settle_destination_before_write(signature.path)

    assert preparations == 2
    assert transactions == 1


def test_settle_retries_when_pretransaction_signature_check_is_stale_once(
    monkeypatch,
) -> None:
    signature = ingest._FileSignature("/settle.bin", 4, 5)
    preflight = ingest._SettleTargetPreflight("content", None, 4, 5, signature)
    prepared = ingest._PreparedSettleTarget(
        preflight,
        ingest._ContentFacts("blake3:" + "3" * 64, 4, 5),
    )
    checks = 0
    transactions = 0

    monkeypatch.setattr(ingest, "_preflight_settle_target", lambda _path: preflight)
    monkeypatch.setattr(ingest, "_prepare_settle_target", lambda _preflight: prepared)

    def stale_once(_signature):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise ingest._PreflightStale

    def run_once(_work):
        nonlocal transactions
        transactions += 1

    monkeypatch.setattr(ingest, "_assert_signature_current", stale_once)
    monkeypatch.setattr(ingest, "run_write_txn", run_once)

    ingest._settle_destination_before_write(signature.path)

    assert checks == 2
    assert transactions == 1


def test_settle_restarts_when_destination_changes_after_hash_preparation(
    mock_create_session, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "settle-changed-after-prepare.bin"
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
    real_prepare = ingest._prepare_settle_target
    preparations = 0

    def mutate_after_first_prepare(preflight):
        nonlocal preparations
        preparations += 1
        prepared = real_prepare(preflight)
        if preparations == 1:
            path.write_bytes(b"replacement bytes are different")
        return prepared

    monkeypatch.setattr(ingest, "_prepare_settle_target", mutate_after_first_prepare)

    ingest._settle_destination_before_write(str(path))

    assert preparations == 2


def test_new_content_retries_when_preflight_signature_does_not_match_hash_facts(
    mock_create_session, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "new-content-hash-snapshot.bin"
    original_bytes = b"hashed bytes"
    path.write_bytes(original_bytes)
    original_stat = path.stat()
    stored_hash = "blake3:" + "4" * 64
    facts = ingest._ContentFacts(
        stored_hash,
        original_stat.st_size,
        original_stat.st_mtime_ns,
    )
    spec = ingest._UploadRecordSpec(path.name, [], None, {}, None)
    real_preflight = ingest._preflight_upload_record
    preflights = 0

    def change_before_first_preflight(*args):
        nonlocal preflights
        preflights += 1
        if preflights == 1:
            path.write_bytes(b"newer bytes with another size")
        else:
            path.write_bytes(original_bytes)
            os.utime(
                path,
                ns=(original_stat.st_mtime_ns, original_stat.st_mtime_ns),
            )
        return real_preflight(*args)

    monkeypatch.setattr(ingest, "_preflight_upload_record", change_before_first_preflight)

    result = ingest._create_content_and_upload_record(
        stored_hash,
        str(path),
        facts,
        True,
        spec,
    )

    assert result.content_id is not None
    assert preflights == 2


def test_new_content_retries_when_pretransaction_signature_check_is_stale_once(
    monkeypatch,
) -> None:
    signature = ingest._FileSignature("/new-content.bin", 4, 5)
    spec = ingest._UploadRecordSpec("new-content.bin", [], None, {}, None)
    preflight = ingest._UploadRecordPreflight(
        None,
        None,
        signature.path,
        None,
        None,
        signature,
        spec,
    )
    prepared = ingest._PreparedUploadRecord(preflight, {})
    facts = ingest._ContentFacts("blake3:" + "5" * 64, 4, 5)
    preflights = 0
    checks = 0
    result = object()

    def preflight_once(*_args):
        nonlocal preflights
        preflights += 1
        return preflight

    def stale_once(_signature):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_preflight_upload_record", preflight_once)
    monkeypatch.setattr(ingest, "_prepare_upload_record", lambda _preflight: prepared)
    monkeypatch.setattr(ingest, "_assert_signature_current", stale_once)
    monkeypatch.setattr(ingest, "run_write_txn", lambda _work: result)

    observed = ingest._create_content_and_upload_record(
        facts.stored_hash,
        signature.path,
        facts,
        True,
        spec,
    )

    assert observed is result
    assert preflights == 2
    assert checks == 2


def _raise_locked(_work):
    raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))


def _raise_integrity(_work):
    raise IntegrityError("INSERT", {}, sqlite3.IntegrityError("constraint failed"))


@pytest.mark.parametrize(
    ("failure", "kind", "job_id", "expected"),
    [
        pytest.param(
            _raise_locked, "executed", "job-locked",
            "error_type=OperationalError job_id=job-locked output_kind=executed",
            id="executed-exhausted-lock-retries",
        ),
        pytest.param(
            _raise_integrity, "executed", "job-integrity",
            "error_type=IntegrityError job_id=job-integrity output_kind=executed",
            id="executed-non-retryable-write",
        ),
        pytest.param(
            _raise_integrity, "cached", None,
            "error_type=IntegrityError output_kind=cached",
            id="cached-terminal-write",
        ),
    ],
)
def test_registration_failure_is_reported_and_never_raised(
    mock_create_session, monkeypatch, caplog, failure, kind, job_id, expected
) -> None:
    """A terminal write failure yields no asset id and one structured event."""
    path = _output_path(f"registration-failure-{kind}-{job_id}.bin")
    with open(path, "wb") as file:
        file.write(b"output")
    if kind == "cached":
        with mock_create_session() as session:
            create_content(session, path, size_bytes=6)
            session.commit()

    monkeypatch.setattr(ingest, "run_write_txn", failure)
    register = (
        ingest.register_executed_output if kind == "executed" else ingest.register_cached_output
    )
    try:
        with caplog.at_level(logging.INFO):
            assert register(path, job_id=job_id) is None
        assert _registration_failure_event(caplog) == (
            f"[assets-event] ingest.register_failed {expected}"
        )
    finally:
        os.unlink(path)


def test_executed_registration_reports_preflight_os_error(monkeypatch, caplog) -> None:
    monkeypatch.setattr(ingest.os, "stat", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("gone")))

    with caplog.at_level(logging.INFO):
        assert ingest.register_executed_output("/missing/output.bin", job_id="job-preflight") is None
    assert _registration_failure_event(caplog) == (
        "[assets-event] ingest.register_failed error_type=OSError job_id=job-preflight output_kind=executed"
    )


def test_cached_registration_apply_rejects_changed_content_row_with_sibling(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "cached-sibling-row-mutated.bin"
    path.write_bytes(b"cached bytes")
    stat_result = path.stat()

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
        create_record(session, content.id, "sibling.bin", system_metadata={"source": 1})
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._preflight_cached_registration(str(path))
    assert preflight is not None
    assert preflight.sibling_id is not None

    def mutate_row(session):
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        content.mtime_ns = stat_result.st_mtime_ns + 1

    db_mod.run_write_txn(mutate_row)

    with mock_create_session() as session:
        with pytest.raises(ingest._PreflightStale):
            _apply_cached_preflight(session, preflight, str(path), {"source": 1})


def test_cached_registration_apply_rejects_changed_content_row_without_sibling(
    mock_create_session, tmp_path
) -> None:
    path = tmp_path / "cached-no-sibling-row-mutated.bin"
    path.write_bytes(b"cached bytes")
    stat_result = path.stat()

    def seed(session):
        content = create_content(
            session,
            str(path),
            size_bytes=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
        )
        return content.id

    content_id = db_mod.run_write_txn(seed)
    preflight = ingest._preflight_cached_registration(str(path))
    assert preflight is not None
    assert preflight.sibling_id is None

    def mutate_row(session):
        content = session.get(ingest.AssetContent, content_id)
        assert content is not None
        content.mtime_ns = stat_result.st_mtime_ns + 1

    db_mod.run_write_txn(mutate_row)

    with mock_create_session() as session:
        with pytest.raises(ingest._PreflightStale):
            _apply_cached_preflight(session, preflight, str(path), {})


def test_cached_registration_retries_when_pretransaction_signature_check_is_stale_once(
    monkeypatch
) -> None:
    path = _output_path("cached-retry.bin")
    with open(path, "wb") as file:
        file.write(b"cached bytes")
    signature = ingest._file_signature(path)
    preflight = SimpleNamespace(
        content_id="content",
        sibling_id="sibling",
        sibling_metadata={"source": 1},
        signature=signature,
        row_size_bytes=signature.size_bytes,
        row_mtime_ns=signature.mtime_ns,
    )
    preflights = 0
    checks = 0
    result = object()

    def preflight_once(_locator):
        nonlocal preflights
        preflights += 1
        return preflight

    def stale_once(_signature):
        nonlocal checks
        checks += 1
        if checks == 1:
            raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_preflight_cached_registration", preflight_once)
    monkeypatch.setattr(ingest, "_assert_signature_current", stale_once)
    monkeypatch.setattr(ingest, "run_write_txn", lambda _work: result)

    try:
        observed = ingest.register_cached_output(path)
    finally:
        os.unlink(path)

    assert observed is result
    assert preflights == 2
    assert checks == 2


def test_cached_registration_restarts_when_content_vanishes_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-content-vanished.bin")
    public_path = _output_path("cached-public-content-vanished.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"output")
    try:
        direct_content_id = _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        with mock_create_session() as session:
            content = session.get(ingest.AssetContent, direct_content_id)
            assert content is not None
            ingest.mark_content_missing(session, content.id)
            session.commit()
        with mock_create_session() as session:
            with pytest.raises(ingest._PreflightStale):
                _apply_cached_preflight(session, direct_preflight, direct_path, {})

        public_content_id = _seed_cached_content(mock_create_session, public_path)
        real_apply = ingest._apply_cached_registration
        mutated = False

        def vanish_then_apply(session, *args):
            nonlocal mutated
            if not mutated:
                mutated = True
                with mock_create_session() as mutation_session:
                    content = mutation_session.get(ingest.AssetContent, public_content_id)
                    assert content is not None
                    ingest.mark_content_missing(mutation_session, content.id)
                    mutation_session.commit()
            return real_apply(session, *args)

        monkeypatch.setattr(ingest, "_apply_cached_registration", vanish_then_apply)
        assert ingest.register_cached_output(public_path) is None
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)


def test_cached_registration_restarts_when_sibling_appears_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-sibling-appeared.bin")
    public_path = _output_path("cached-public-sibling-appeared.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"output")
    try:
        direct_content_id = _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        with mock_create_session() as session:
            create_record(
                session,
                direct_content_id,
                "sibling.bin",
                system_metadata={"generation": 1},
            )
            session.commit()
        with mock_create_session() as session:
            with pytest.raises(ingest._PreflightStale):
                _apply_cached_preflight(session, direct_preflight, direct_path, {})

        public_content_id = _seed_cached_content(mock_create_session, public_path)
        real_apply = ingest._apply_cached_registration
        extraction_count = 0
        mutated = False

        def extract_metadata(*_args, **_kwargs):
            nonlocal extraction_count
            extraction_count += 1
            return {"generation": 0}

        def add_sibling_then_apply(session, *args):
            nonlocal mutated
            if not mutated:
                mutated = True
                with mock_create_session() as mutation_session:
                    create_record(
                        mutation_session,
                        public_content_id,
                        "sibling.bin",
                        system_metadata={"generation": 2},
                    )
                    mutation_session.commit()
            return real_apply(session, *args)

        monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extract_metadata)
        monkeypatch.setattr(ingest, "_apply_cached_registration", add_sibling_then_apply)
        result = ingest.register_cached_output(public_path)
        assert result is not None
        with mock_create_session() as session:
            record = session.get(ingest.Asset, result.id)
            assert record is not None
            assert record.system_metadata == {"generation": 2}
        assert extraction_count == 1
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)


def test_cached_registration_restarts_when_file_changes_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-stat-changed.bin")
    public_path = _output_path("cached-public-stat-changed.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"old")
    try:
        _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        assert direct_preflight.signature is not None
        with open(direct_path, "wb") as file:
            file.write(b"new bytes")
        with pytest.raises(ingest._PreflightStale):
            ingest._assert_signature_current(direct_preflight.signature)

        _seed_cached_content(mock_create_session, public_path)
        real_assert_signature_current = ingest._assert_signature_current
        extraction_sizes: list[int] = []
        mutated = False

        def extract_metadata(path, *_args, **_kwargs):
            size = os.path.getsize(path)
            extraction_sizes.append(size)
            return {"size": size}

        def rewrite_then_assert(signature):
            nonlocal mutated
            if not mutated:
                mutated = True
                with open(public_path, "wb") as file:
                    file.write(b"new public bytes")
            return real_assert_signature_current(signature)

        monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extract_metadata)
        monkeypatch.setattr(ingest, "_assert_signature_current", rewrite_then_assert)
        result = ingest.register_cached_output(public_path)
        assert result is not None
        with mock_create_session() as session:
            record = session.get(ingest.Asset, result.id)
            assert record is not None
            assert record.system_metadata == {"size": len(b"new public bytes")}
        assert extraction_sizes == [len(b"old"), len(b"new public bytes")]
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)





def test_unsettled_new_upload_persists_nothing_rather_than_mixing_facts(
    file_database, monkeypatch, caplog
) -> None:
    from sqlalchemy import select

    from app.assets.database.models import Asset, AssetContent

    path = _output_path("unsettled-upload.bin")
    with open(path, "wb") as handle:
        handle.write(b"first bytes")

    stored_hash = "blake3:" + "c" * 64
    stat_result = os.stat(path)
    facts = ingest._ContentFacts(stored_hash, stat_result.st_size, stat_result.st_mtime_ns)
    spec = ingest._UploadRecordSpec("unsettled-upload.bin", [], None, {}, None)

    real_assert = ingest._assert_signature_current
    attempts = 0

    def always_stale(_signature):
        nonlocal attempts
        attempts += 1
        raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_assert_signature_current", always_stale)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ingest.UploadUnstableError, match="did not settle"):
            ingest._create_content_and_upload_record(
                stored_hash, path, facts, True, spec
            )

    monkeypatch.setattr(ingest, "_assert_signature_current", real_assert)

    with db_mod.Session() as session:
        contents = list(session.scalars(select(AssetContent).where(AssetContent.path == path)))
        records = list(session.scalars(select(Asset)))

    assert contents == [], "refused upload must not leave a content row behind"
    assert records == [], "refused upload must not leave an asset record behind"
    assert attempts == 4
    os.unlink(path)
