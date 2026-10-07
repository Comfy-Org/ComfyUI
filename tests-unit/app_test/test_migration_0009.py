import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config

_REVISION_0008 = "0008_drop_asset_meta"
_REVISION_0009 = "0009_add_missing_since"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _columns(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(asset_contents)")}


def _indexes(db_path: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA index_list(asset_contents)")}


@pytest.fixture
def db_at_0008_with_a_missing_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _REVISION_0008)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO asset_contents (id, size_bytes, path, is_missing, created_at) "
            "VALUES ('c1', 1, '/models/a.safetensors', 1, '2026-10-01 00:00:00')"
        )
    yield cfg, db_path


def test_0009_adds_missing_since_and_leaves_existing_rows_unstamped(db_at_0008_with_a_missing_row):
    cfg, db_path = db_at_0008_with_a_missing_row

    command.upgrade(cfg, _REVISION_0009)

    assert "missing_since" in _columns(db_path)
    with sqlite3.connect(db_path) as conn:
        (index_sql,) = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'ix_asset_contents_missing_since'"
        ).fetchone()
        assert "WHERE missing_since IS NOT NULL" in index_sql
        assert conn.execute("SELECT is_missing, missing_since FROM asset_contents").fetchall() == [(1, None)]


def test_0009_downgrade_removes_the_column_and_keeps_the_row(db_at_0008_with_a_missing_row):
    cfg, db_path = db_at_0008_with_a_missing_row
    command.upgrade(cfg, _REVISION_0009)

    command.downgrade(cfg, _REVISION_0008)

    assert "missing_since" not in _columns(db_path)
    assert "ix_asset_contents_missing_since" not in _indexes(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id, is_missing FROM asset_contents").fetchall() == [("c1", 1)]
