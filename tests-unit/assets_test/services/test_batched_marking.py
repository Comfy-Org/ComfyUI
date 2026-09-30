"""The prune and the offline marking write in short batches rather than one long
transaction, so a foreground write gets the lock between batches. These tests pin the
batch boundaries, what a failure or a stop leaves behind, the pause between batches,
what a batch does with rows another writer changed, and the set-based mark."""

import asyncio
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from aiohttp.test_utils import make_mocked_request
from sqlalchemy import event
from sqlalchemy.orm import Session as SASession, sessionmaker
from sqlalchemy.pool import StaticPool

from app.assets import scanner, seeder as seeder_module
from app.assets.api import routes
from app.assets.database.models import Asset, AssetContent, AssetTag, Base, Tag
from app.assets.database.queries.records import (
    create_content,
    create_record,
    ensure_tag,
    ensure_tag_link,
    mark_content_missing,
    mark_contents_missing,
)
from app.assets.scanner_admission import _WATCH_LIST
from app.assets.seeder import State


@pytest.fixture
def db_engine():
    """One in-memory database every thread shares, for the tests that prune on a worker."""
    engine = sa.create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(autouse=True)
def no_yield(monkeypatch):
    monkeypatch.setattr(scanner, "WRITE_YIELD_MIN_SECONDS", 0.0)
    monkeypatch.setattr(scanner, "WRITE_YIELD_MAX_SECONDS", 0.0)


@pytest.fixture
def catalog(db_engine):
    """Routes the scanner's sessions to the test engine and counts write transactions."""
    opened: list[int] = []
    write_factory = sessionmaker(bind=db_engine)

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    def _create_write_session():
        opened.append(1)
        return write_factory()

    _WATCH_LIST.clear()
    with patch("app.assets.scanner.create_session", _create_session), \
         patch("app.assets.seeder.create_session", _create_session), \
         patch("app.assets.scanner.create_write_session", _create_write_session):
        yield opened
    _WATCH_LIST.clear()


def _rows(session, directory: Path, count: int, *, files: bool = False) -> list[str]:
    ids = []
    for i in range(count):
        path = directory / f"f{i:05d}.png"
        if files:
            path.write_bytes(f"bytes-{i}".encode())
        stat = path.stat() if files else None
        content = create_content(
            session,
            path=str(path),
            size_bytes=stat.st_size if stat else 1,
            mtime_ns=stat.st_mtime_ns if stat else 1,
        )
        create_record(session, content_id=content.id, name=path.name, tags=["output"])
        ids.append(content.id)
    session.commit()
    return ids


def _live_ids(session) -> set[str]:
    session.expire_all()
    return set(session.scalars(sa.select(AssetContent.id).where(AssetContent.is_missing == sa.false())))


def _prune(owned: list[str]) -> int | None:
    return scanner.mark_missing_outside_prefixes_safely(owned)


@pytest.mark.parametrize("count, batches", [(0, 0), (1, 1), (256, 1), (257, 2), (600, 3)])
def test_prune_commits_one_batch_per_256_rows(session, catalog, temp_dir, count, batches):
    _rows(session, temp_dir, count)

    assert _prune([]) == count
    assert len(catalog) == batches
    assert _live_ids(session) == set()


def test_rows_under_an_owned_prefix_are_not_pruned(session, catalog, temp_dir):
    owned, gone = temp_dir / "owned", temp_dir / "gone"
    owned.mkdir()
    gone.mkdir()
    kept = set(_rows(session, owned, 5))
    _rows(session, gone, 5)

    assert _prune([str(owned)]) == 5
    assert _live_ids(session) == kept


def test_a_row_re_registered_between_batches_is_left_to_its_new_owner(session, catalog, temp_dir):
    """register_executed_output retires the live row at a path and inserts a new one.
    Doing that between two batches must leave exactly the new row live."""
    ids = _rows(session, temp_dir, 300)
    last = session.get(AssetContent, ids[-1])
    replacement: list[str] = []

    def between_batches() -> bool:
        if len(catalog) == 1:
            mark_content_missing(session, last.id)
            new = create_content(session, path=last.path, size_bytes=2, mtime_ns=2)
            session.commit()
            replacement.append(new.id)
        return False

    marked = scanner.mark_missing_outside_prefixes_safely([], between_batches)

    # The replacement was never a candidate; the retired row is not counted twice.
    assert marked == len(ids) - 1
    assert _live_ids(session) == set(replacement)


def test_a_foreground_write_gets_the_lock_between_batches(tmp_path, monkeypatch):
    """On a real file database with the production write-session setup, another
    connection can take the write lock at every point between two batches."""
    db = tmp_path / "catalog.db"
    read_engine = sa.create_engine(f"sqlite:///{db}")
    write_engine = sa.create_engine(f"sqlite:///{db}")

    @event.listens_for(write_engine, "connect")
    def _connect(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(write_engine, "begin")
    def _begin(connection):
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    with read_engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(read_engine)
    with SASession(read_engine) as session:
        _rows(session, tmp_path, 600)

    foreground: list[bool] = []

    def between_batches() -> bool:
        other = sqlite3.connect(db, timeout=0, isolation_level=None)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("INSERT INTO tags (name) VALUES (?)", (f"fg-{len(foreground)}",))
            other.execute("COMMIT")
            foreground.append(True)
        except sqlite3.OperationalError:
            foreground.append(False)
        finally:
            other.close()
        return False

    monkeypatch.setattr(scanner, "create_session", lambda: SASession(read_engine))
    monkeypatch.setattr(scanner, "create_write_session", sessionmaker(bind=write_engine))
    assert scanner.mark_missing_outside_prefixes_safely([], between_batches) == 600

    assert foreground == [True, True, True]


def test_a_failed_batch_keeps_the_batches_before_it(session, catalog, temp_dir):
    _rows(session, temp_dir, 600)
    real = scanner.mark_contents_missing

    def fail_in_the_second_batch(sess, ids):
        if len(catalog) == 2:
            raise RuntimeError("disk I/O error")
        return real(sess, ids)

    with patch("app.assets.scanner.mark_contents_missing", fail_in_the_second_batch):
        assert _prune([]) is None

    assert len(_live_ids(session)) == 600 - scanner.WRITE_BATCH_ROWS


def test_sync_root_counts_the_batches_committed_before_a_failure(session, catalog, temp_dir, monkeypatch):
    output = temp_dir / "output"
    output.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(output))
    _rows(session, output, 300)
    real = scanner.mark_contents_missing

    def fail_in_the_second_batch(sess, ids):
        if len(catalog) == 2:
            raise RuntimeError("disk I/O error")
        return real(sess, ids)

    progress = seeder_module._ScanState()
    with patch("app.assets.scanner.mark_contents_missing", fail_in_the_second_batch):
        assert scanner.sync_root_safely("output", progress) == set()

    assert progress.missing_marked == scanner.WRITE_BATCH_ROWS


def _gone_observations(session, temp_dir: Path, count: int) -> list[scanner._ReferenceObservation]:
    ids = _rows(session, temp_dir, count, files=True)
    observations = []
    for content_id in ids:
        content = session.get(AssetContent, content_id)
        os.remove(content.path)
        observations.append(
            scanner._ReferenceObservation(content.id, content.size_bytes, content.mtime_ns, None)
        )
    return observations


def test_stop_between_batches_leaves_the_rest_live(session, catalog, temp_dir):
    observations = _gone_observations(session, temp_dir, 600)

    committed: list[int] = []
    scanner._write_in_batches(observations, scanner.apply_reference_observations, lambda: bool(committed), committed)

    assert committed == [scanner.WRITE_BATCH_ROWS]
    assert len(_live_ids(session)) == 600 - scanner.WRITE_BATCH_ROWS


def test_a_scan_waits_between_batches_while_a_prompt_runs(session, catalog, temp_dir, monkeypatch):
    output = temp_dir / "output"
    output.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(output))
    _rows(session, output, 600)
    instance = seeder_module._AssetSeeder()
    instance._state = State.RUNNING
    instance._scan_state = seeder_module._ScanState()
    instance._run_gate.set()
    real = scanner.mark_contents_missing
    first_batch = threading.Event()

    def mark(sess, ids):
        if not first_batch.is_set():
            assert instance.pause()  # a prompt starts during the first batch
            first_batch.set()
        return real(sess, ids)

    def sync():
        scanner.sync_root_safely(
            "output",
            instance._scan_state,
            lambda: instance._check_pause_and_cancel(seeder_module._ScanStage.FAST_SCAN),
        )

    with patch("app.assets.scanner.mark_contents_missing", mark):
        worker = threading.Thread(target=sync)
        worker.start()
        assert first_batch.wait(5)
        time.sleep(0.2)
        assert len(catalog) == 1  # no batch while paused
        assert instance.resume()
        worker.join(5)

    assert len(catalog) == 3
    assert instance._scan_state.missing_marked == 600


def test_offline_rows_retired_across_batches_recover_when_the_drive_returns(
    session, catalog, temp_dir, monkeypatch
):
    """#16646's hashing-off recovery with the marking split over several batches: files
    that come back while the marking is part way through are retired by the batches
    that follow, then recovered by the walk that follows in the same scan, every record
    keeping its id."""
    output = temp_dir / "output"
    (temp_dir / "input").mkdir()
    output.mkdir()
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(output))
    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir / "input"))
    monkeypatch.setattr(scanner, "get_comfy_models_folders", lambda: [])
    ids = _rows(session, output, 700, files=True)
    records = {record.id: record.content_id for record in session.scalars(sa.select(Asset))}
    parked = temp_dir / "parked"
    output.rename(parked)
    output.mkdir()

    instance = seeder_module._AssetSeeder()
    instance._scan_state = seeder_module._ScanState()
    instance._phase = seeder_module.ScanPhase.FAST
    instance._run_gate.set()
    real = scanner.mark_contents_missing

    def mark(sess, content_ids):
        marked = real(sess, content_ids)
        if len(catalog) == 1:
            # The drive comes back after the first batch.
            for name in os.listdir(parked):
                os.rename(parked / name, output / name)
        return marked

    with patch("app.assets.scanner.mark_contents_missing", mark):
        instance._run_fast_phase(("input", "output"))

    session.expire_all()
    assert instance._scan_state.missing_marked == 700
    assert instance._scan_state.recovered == 700
    assert _live_ids(session) == set(ids)
    assert {record.id: record.content_id for record in session.scalars(sa.select(Asset))} == records
    live_paths = session.scalars(sa.select(AssetContent.path).where(AssetContent.is_missing == sa.false())).all()
    assert len(live_paths) == len(set(live_paths)) == 700


def _link_state(session) -> tuple[dict[str, bool], set[tuple[str, str, str]]]:
    session.expire_all()
    contents = {c.path: c.is_missing for c in session.scalars(sa.select(AssetContent))}
    links = {
        (record.name, link.tag_name, link.origin)
        for record in session.scalars(sa.select(Asset))
        for link in session.scalars(sa.select(AssetTag).where(AssetTag.asset_id == record.id))
    }
    return contents, links


def _equivalence_fixture(session) -> list[str]:
    """Rows covering each case the mark handles: no record, one, two, a record already
    tagged missing by hand, an already-missing row, and an id that does not exist."""
    none = create_content(session, path="/c/none.png")
    one = create_content(session, path="/c/one.png")
    create_record(session, content_id=one.id, name="one", tags=["output"])
    two = create_content(session, path="/c/two.png")
    create_record(session, content_id=two.id, name="two-a")
    create_record(session, content_id=two.id, name="two-b", tags=["input"])
    tagged = create_content(session, path="/c/tagged.png")
    record = create_record(session, content_id=tagged.id, name="tagged")
    ensure_tag(session, "missing")
    ensure_tag_link(session, asset_id=record.id, tag_name="missing", origin="manual")
    already = create_content(session, path="/c/already.png")
    create_record(session, content_id=already.id, name="already")
    mark_content_missing(session, already.id)
    session.commit()
    return [none.id, one.id, two.id, tagged.id, already.id, "no-such-id"]


def test_the_set_mark_matches_marking_row_by_row(session):
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with SASession(engine) as per_row:
        # Same fixture in both catalogs, then the ids made to match by path.
        ids = _equivalence_fixture(session)
        _equivalence_fixture(per_row)
        by_path = {c.path: c.id for c in per_row.scalars(sa.select(AssetContent))}
        paths = {c.id: c.path for c in session.scalars(sa.select(AssetContent))}

        marked = mark_contents_missing(session, ids)
        session.commit()
        for content_id in ids:
            path = paths.get(content_id)
            content = per_row.get(AssetContent, by_path[path]) if path else None
            if content is not None and not content.is_missing:
                mark_content_missing(per_row, content.id)
        per_row.commit()

        # create_content stores os.path.abspath(path), so compare in that form (a drive letter on Windows).
        expected = sorted(os.path.abspath(f"/c/{name}.png") for name in ("none", "one", "tagged", "two"))
        assert sorted(paths[i] for i in marked) == expected
        assert _link_state(session) == _link_state(per_row)
        assert session.get(Tag, "missing") is not None


def test_the_set_mark_settles_a_link_race_row_by_row(session, monkeypatch):
    content = create_content(session, path="/c/raced.png")
    record = create_record(session, content_id=content.id, name="raced")
    session.commit()
    real_execute = session.execute

    def insert_loses_the_race(statement, *args, **kwargs):
        if isinstance(statement, sa.Insert) and statement.table is AssetTag.__table__:
            raise sa.exc.IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))
        return real_execute(statement, *args, **kwargs)

    monkeypatch.setattr(session, "execute", insert_loses_the_race)
    assert mark_contents_missing(session, [content.id]) == [content.id]
    session.commit()

    link = session.get(AssetTag, (record.id, "missing"))
    assert link is not None and link.origin == "automatic"


def test_the_batched_writes_run_no_table_scan_inside_their_transactions(session, db_engine, temp_dir, monkeypatch):
    """Each statement a batch runs while it holds the write lock is looked up by key or
    index, so the lock window grows with the batch, not the catalog. (The prune's
    candidate read is a full read by design, and runs before any write transaction.)"""
    in_write: list[bool] = []

    class _Tracked(SASession):
        def __enter__(self):
            in_write.append(True)
            return super().__enter__()

        def __exit__(self, *exc_info):
            in_write.clear()
            return super().__exit__(*exc_info)

    monkeypatch.setattr(scanner, "create_write_session", sessionmaker(bind=db_engine, class_=_Tracked))
    monkeypatch.setattr(scanner, "create_session", lambda: SASession(db_engine))
    # A size change splits the row, and the new record's tags come from its root.
    monkeypatch.setattr("folder_paths.get_output_directory", lambda: str(temp_dir))
    gone_dir, changed_dir, pruned_dir = (temp_dir / name for name in ("gone", "changed", "pruned"))
    for directory in (gone_dir, changed_dir, pruned_dir):
        directory.mkdir()
    observations = _gone_observations(session, gone_dir, 50)
    for content_id in _rows(session, changed_dir, 2, files=True):
        content = session.get(AssetContent, content_id)
        if content.path.endswith("0.png"):
            Path(content.path).write_bytes(b"a different size")
        os.utime(content.path, ns=(10**18, 10**18))
        observations.append(
            scanner._ReferenceObservation(content.id, content.size_bytes, content.mtime_ns, os.stat(content.path))
        )
    statements: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if in_write and statement.lstrip().upper().startswith(("SELECT", "UPDATE", "INSERT", "DELETE")):
            statements.append((statement, parameters))

    event.listen(db_engine, "before_cursor_execute", capture)
    try:
        scanner._write_in_batches(observations, scanner.apply_reference_observations, lambda: False, [])
        _rows(session, pruned_dir, 50)
        assert scanner.mark_missing_outside_prefixes_safely([str(gone_dir), str(changed_dir)]) == 50
    finally:
        event.remove(db_engine, "before_cursor_execute", capture)

    kinds = {statement.split()[0].upper() for statement, _ in statements}
    assert {"SELECT", "UPDATE", "INSERT"} <= kinds
    with db_engine.connect() as conn:
        for statement, parameters in statements:
            plan = conn.exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters).all()
            scans = [row[-1] for row in plan if row[-1].startswith("SCAN")]
            assert not scans, (statement, plan)


@pytest.mark.asyncio
async def test_the_prune_endpoint_keeps_the_event_loop_serving(monkeypatch):
    def slow_prune() -> int:
        time.sleep(0.5)
        return 3

    monkeypatch.setattr(routes.asset_seeder, "mark_missing_outside_prefixes", slow_prune)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticking = asyncio.create_task(ticker())
    response = await routes.mark_missing_assets.__wrapped__(make_mocked_request("POST", "/api/assets/prune"))
    ticking.cancel()

    assert json.loads(response.body) == {"status": "completed", "marked": 3}
    assert ticks >= 20



def test_the_standalone_prune_starts_the_scan_queued_while_it_ran(session, catalog, temp_dir, monkeypatch):
    """The API runs the prune off the event loop, so a prompt can finish meanwhile and
    queue its output rescan, which cannot start while the prune holds the seeder."""
    _rows(session, temp_dir, 10)
    instance = seeder_module._AssetSeeder()
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    started: list[dict] = []

    def start(**kwargs) -> bool:
        if instance._state is not State.IDLE:
            return False
        started.append(kwargs)
        return True

    monkeypatch.setattr(instance, "start", start)
    real = scanner.mark_contents_missing

    def mark(sess, ids):
        assert instance.enqueue_scan(roots=("output",), phase=seeder_module.ScanPhase.FULL) is False
        return real(sess, ids)

    with patch("app.assets.scanner.mark_contents_missing", mark):
        assert instance.mark_missing_outside_prefixes() == 10

    assert [kwargs["roots"] for kwargs in started] == [("output",)]
    assert instance._pending_scan is None
