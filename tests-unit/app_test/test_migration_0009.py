import os
import sqlite3
from datetime import datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config

_BASELINE_0008 = "0008_drop_asset_meta"
_REVISION_0009 = "0009_created_at_from_mtime"

# Content and record take separate clock reads in one scan transaction, so they differ by µs.
_CONTENT_TIME = "2026-10-01 12:00:00.000000"
_SCAN_TIME = "2026-10-01 12:00:00.000004"
_UPDATED_TIME = "2026-10-01 12:00:00.000007"
_HALF_SECOND_LATER = "2026-10-01 12:00:00.500000"
_TWO_SECONDS_LATER = "2026-10-01 12:00:02.000000"
_LATER = "2026-10-03 09:00:00.000000"
_MTIME = datetime(2025, 3, 4, 5, 6, 7, 123456)
_MTIME_TEXT = "2025-03-04 05:06:07.123456"


def _ns(when: datetime) -> int:
    return (when - datetime(1970, 1, 1)) // timedelta(microseconds=1) * 1000


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _add_content(conn: sqlite3.Connection, content_id: str, mtime_ns: int | None) -> None:
    conn.execute(
        "INSERT INTO asset_contents (id, size_bytes, path, mtime_ns, is_missing, created_at) "
        "VALUES (?, 1, ?, ?, 0, ?)",
        (content_id, f"/out/{content_id}.png", mtime_ns, _CONTENT_TIME),
    )


def _add_record(
    conn: sqlite3.Connection,
    record_id: str,
    content_id: str,
    created_at: str = _SCAN_TIME,
    job_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO assets (id, content_id, name, job_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (record_id, content_id, record_id, job_id, created_at, _UPDATED_TIME),
    )


def _created(db_path: str) -> dict[str, str]:
    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute("SELECT id, created_at FROM assets"))


@pytest.fixture
def db_at_0008(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _BASELINE_0008)
    with sqlite3.connect(db_path) as conn:
        mtimes = {
            "scanned": _ns(_MTIME) + 789,  # sub-µs part is floored
            "generated": _ns(_MTIME),
            "shared": _ns(_MTIME),
            "reuploaded": _ns(_MTIME),
            "uploaded": _ns(_MTIME),
            "half-second": _ns(_MTIME),
            "two-seconds": _ns(_MTIME),
            "touched": _ns(datetime(2026, 10, 2)),  # mtime refreshed after the scan
            "future": _ns(datetime(2200, 1, 1)),
            "no-mtime": None,
        }
        for content_id, mtime_ns in mtimes.items():
            _add_content(conn, content_id, mtime_ns)
        _add_record(conn, "scanned", "scanned")
        _add_record(conn, "generated", "generated", job_id="job-1")
        _add_record(conn, "shared-job", "shared", job_id="job-2")
        _add_record(conn, "shared-later", "shared", created_at=_LATER)
        # Its scanned sibling was deleted before the upgrade; this one came later.
        _add_record(conn, "reuploaded", "reuploaded", created_at=_LATER)
        # Uploads create content and record together too; an /upload/image duplicate of an
        # older file keeps that file's mtime.
        _add_record(conn, "uploaded", "uploaded")
        conn.execute("INSERT INTO tags (name) VALUES ('uploaded')")
        conn.execute(
            "INSERT INTO asset_tags (asset_id, tag_name, origin, added_at) VALUES (?, ?, ?, ?)",
            ("uploaded", "uploaded", "manual", _SCAN_TIME),
        )
        _add_record(conn, "half-second", "half-second", created_at=_HALF_SECOND_LATER)
        _add_record(conn, "two-seconds", "two-seconds", created_at=_TWO_SECONDS_LATER)
        _add_record(conn, "touched", "touched")
        _add_record(conn, "future", "future")
        _add_record(conn, "no-mtime", "no-mtime")
        conn.commit()
    yield cfg, db_path


def test_0009_dates_scanned_records_by_mtime(db_at_0008):
    cfg, db_path = db_at_0008

    command.upgrade(cfg, _REVISION_0009)
    created = _created(db_path)

    assert created.pop("scanned") == _MTIME_TEXT
    assert created.pop("half-second") == _MTIME_TEXT
    assert created == {
        "generated": _SCAN_TIME,
        "shared-job": _SCAN_TIME,
        "shared-later": _LATER,
        "reuploaded": _LATER,
        "uploaded": _SCAN_TIME,
        "two-seconds": _TWO_SECONDS_LATER,
        "touched": _SCAN_TIME,
        "future": _SCAN_TIME,
        "no-mtime": _SCAN_TIME,
    }
    with sqlite3.connect(db_path) as conn:
        assert {row[0] for row in conn.execute("SELECT updated_at FROM assets")} == {_UPDATED_TIME}


def test_0009_rerun_sets_the_same_dates(db_at_0008):
    cfg, db_path = db_at_0008
    command.upgrade(cfg, _REVISION_0009)
    first = _created(db_path)

    command.downgrade(cfg, _BASELINE_0008)
    assert _created(db_path) == first, "downgrade leaves the dates"
    command.upgrade(cfg, _REVISION_0009)

    assert _created(db_path) == first
