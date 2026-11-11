"""Every "live row at this path" lookup must be served by the partial index
uq_asset_contents_path_live. Written as ``is_missing IS 0`` it is not, and the lookup scans
asset_contents, which inside a write transaction holds the lock for as long as the scan runs."""

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa

import folder_paths
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import (
    create_content,
    create_content_reporting_insert,
    get_record_by_path_or_none,
)
from app.assets.scanner_changes import recover_missing_content
from app.assets.services import ingest
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
)

# SQLite before 3.36 words it "SCAN TABLE asset_contents".
_CONTENT_SCAN = re.compile(r"\bSCAN (TABLE )?asset_contents\b")


@contextmanager
def no_content_scans(db_engine):
    """Fail if any statement run inside the block scans asset_contents."""
    statements: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(("SELECT", "UPDATE", "DELETE")):
            statements.append((statement, parameters))

    sa.event.listen(db_engine, "before_cursor_execute", capture)
    try:
        yield
    finally:
        sa.event.remove(db_engine, "before_cursor_execute", capture)

    path_lookups = [s for s in statements if "asset_contents.path = ?" in s[0]]
    assert path_lookups, "the block ran no live-path lookup"
    with db_engine.connect() as conn:
        for statement, parameters in statements:
            plan = conn.exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters).all()
            assert not any(_CONTENT_SCAN.search(row[-1]) for row in plan), (statement, plan)


def _output_file(name: str) -> str:
    path = Path(folder_paths.get_output_directory()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"output bytes")
    return str(path)


@pytest.fixture
def output_file():
    paths: list[str] = []

    def make(name: str) -> str:
        paths.append(_output_file(name))
        return paths[-1]

    yield make
    for path in paths:
        Path(path).unlink(missing_ok=True)


def test_register_executed_output(mock_create_session, db_engine, output_file):
    path = output_file("live-index-executed.png")
    ingest.register_executed_output(path, job_id=None)
    with no_content_scans(db_engine):
        assert ingest.register_executed_output(path, job_id=None) is not None


def test_register_cached_output(mock_create_session, db_engine, output_file):
    path = output_file("live-index-cached.png")
    ingest.register_executed_output(path, job_id=None)
    with no_content_scans(db_engine):
        assert ingest.register_cached_output(path, job_id=None) is not None


def test_register_file_in_place(mock_create_session, db_engine, output_file):
    path = output_file("live-index-in-place.png")
    ingest.register_file_in_place(path, "in-place.png", ["output"])
    with no_content_scans(db_engine):
        ingest.register_file_in_place(path, "in-place.png", ["output"])


def test_upload_onto_an_existing_destination(mock_create_session, db_engine):
    def upload():
        # Staged where the upload route stages it: the move into input/ is an os.replace,
        # which fails across drives. The upload removes this directory along with the file.
        uploads = Path(folder_paths.get_temp_directory()) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=uploads)) / "upload.tmp"
        temp.write_bytes(b"upload bytes")
        return ingest.upload_from_temp_path(str(temp), name="live-index-upload.png", tags=["input"])

    result = upload()
    try:
        with no_content_scans(db_engine):
            upload()
    finally:
        with mock_create_session() as session:
            for content in session.scalars(sa.select(AssetContent)):
                Path(content.path).unlink(missing_ok=True)
    assert result is not None


def test_insert_race_fallback(session, db_engine, temp_dir):
    path = str(temp_dir / "raced.png")
    create_content(session, path)
    with no_content_scans(db_engine):
        _, inserted = create_content_reporting_insert(session, path)
    assert inserted is False


def test_get_record_by_path_or_none(session, db_engine, temp_dir):
    with no_content_scans(db_engine):
        get_record_by_path_or_none(session, str(temp_dir / "absent.png"))


def test_hash_on_recovery_occupied_check(session, db_engine, temp_dir):
    path = str(temp_dir / "returning.png")
    with no_content_scans(db_engine):
        recover_missing_content(session, path, None, hashing_is_enabled=True)


@pytest.mark.parametrize("outcome", ["hashed", "vanished", "retired"])
def test_off_to_on_transition_drain(session, db_engine, temp_dir, outcome):
    path = temp_dir / "transition.png"
    path.write_bytes(b"bytes")
    create_content(session, str(path))
    session.commit()
    enqueue_transition_work(session, "off_to_on")
    if outcome == "vanished":
        path.unlink()
    try:
        with no_content_scans(db_engine):
            if outcome == "retired":
                with patch("app.assets.services.hash_mode_state.snapshot_hash", side_effect=OSError):
                    for _ in range(3):
                        drain_transition_queue(session)
            else:
                drain_transition_queue(session)
            session.commit()
    finally:
        clear_transition_queue()
    content = session.scalars(sa.select(AssetContent).where(AssetContent.path == os.path.abspath(path))).one()
    assert (content.hash is not None, content.is_missing) == {
        "hashed": (True, False),
        "vanished": (False, True),
        "retired": (False, False),
    }[outcome]
