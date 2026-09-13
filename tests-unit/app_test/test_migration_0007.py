import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config

_BASELINE_0006 = "0006_add_loader_path"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


@pytest.fixture
def db_at_0006(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _BASELINE_0006)
    yield cfg, db_path


def test_0007_upgrade_from_0006(db_at_0006):
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "assets" in tables
    assert "asset_contents" in tables
    assert "asset_system_state" in tables
    assert "asset_references" not in tables


def test_0007_schema_has_expected_columns(db_at_0006):
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        asset_cols = {r[1] for r in conn.execute("PRAGMA table_info(assets)")}
        content_cols = {r[1] for r in conn.execute("PRAGMA table_info(asset_contents)")}
    assert {"id", "content_id", "name", "loader_path", "updated_at", "last_access_time"} <= asset_cols
    assert {"id", "hash", "path", "is_missing", "mtime_ns"} <= content_cols


def test_0007_downgrade_restores_0006_schema(db_at_0006):
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    command.downgrade(cfg, _BASELINE_0006)
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "asset_references" in tables
    assert "asset_contents" not in tables


def test_0007_invariants_on_migrated_db(db_at_0006):
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
            "VALUES ('c1', '/tmp/f1', 0, 0, '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO assets(id, content_id, name, created_at, updated_at) "
            "VALUES ('a1', 'c1', 'test', '2024-01-01', '2024-01-01')"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
                "VALUES ('c2', '/tmp/f1', 0, 0, '2024-01-01')"
            )
            conn.commit()
        conn.rollback()

        conn.execute(
            "INSERT INTO asset_contents(id, path, is_missing, size_bytes, created_at) "
            "VALUES ('c3', '/tmp/f1', 1, 0, '2024-01-01')"
        )
        conn.commit()

        conn.execute("UPDATE asset_contents SET hash='abc123' WHERE id='c1'")
        conn.execute(
            "INSERT INTO asset_contents(id, path, hash, is_missing, size_bytes, created_at) "
            "VALUES ('c4', '/tmp/f2', 'abc123', 0, 0, '2024-01-01')"
        )
        conn.commit()


def test_0007_orm_parity(db_at_0006, tmp_path):
    from sqlalchemy import create_engine, inspect

    import app.assets.database.models as asset_models
    from app.database.models import Base

    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    with sqlite3.connect(db_path) as conn:
        alembic_tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'alembic%' AND name NOT LIKE 'sqlite%'"
            )
        }

    alembic_engine = create_engine(f"sqlite:///{db_path}")
    orm_db = str(tmp_path / "orm.db")
    engine = create_engine(f"sqlite:///{orm_db}")
    alembic_inspector = inspect(alembic_engine)
    Base.metadata.create_all(engine)
    orm_inspector = inspect(engine)
    orm_tables = set(orm_inspector.get_table_names())
    assert asset_models.AssetMeta.__tablename__ == "asset_meta"

    alembic_indexes = {
        (index["name"], tuple(index["column_names"]))
        for index in alembic_inspector.get_indexes("asset_meta")
    }
    orm_indexes = {
        (index.name, tuple(index.columns.keys()))
        for index in Base.metadata.tables["asset_meta"].indexes
    }

    assert alembic_tables == orm_tables, f"Mismatch: alembic={alembic_tables}, orm={orm_tables}"
    assert alembic_indexes == orm_indexes, f"Mismatch: alembic={alembic_indexes}, orm={orm_indexes}"


def test_0007_downgrade_chain_past_0003_succeeds(db_at_0006):
    cfg, db_path = db_at_0006
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001_assets")
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "assets_info" in tables
    assert "asset_references" not in tables
    assert "asset_contents" not in tables


def test_0007_downgrade_restores_legacy_assets_definition(db_at_0006, tmp_path):
    from sqlalchemy import create_engine, inspect

    cfg, db_path = db_at_0006

    reference_db = str(tmp_path / "reference_0006.db")
    reference_cfg = _make_config(reference_db)
    command.upgrade(reference_cfg, _BASELINE_0006)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, _BASELINE_0006)

    def _assets_shape(path: str):
        engine = create_engine(f"sqlite:///{path}")
        try:
            inspector = inspect(engine)
            indexes = {
                (index["name"], tuple(index["column_names"]), bool(index["unique"]))
                for index in inspector.get_indexes("assets")
            }
            checks = {
                (check["name"], check["sqltext"])
                for check in inspector.get_check_constraints("assets")
            }
            defaults = {
                column["name"]: column["default"]
                for column in inspector.get_columns("assets")
            }
            return indexes, checks, defaults
        finally:
            engine.dispose()

    restored_indexes, restored_checks, restored_defaults = _assets_shape(db_path)
    expected_indexes, expected_checks, expected_defaults = _assets_shape(reference_db)

    assert restored_indexes == expected_indexes, (
        f"index drift: restored={restored_indexes}, expected={expected_indexes}"
    )
    assert restored_checks == expected_checks, (
        f"check drift: restored={restored_checks}, expected={expected_checks}"
    )
    assert restored_defaults == expected_defaults, (
        f"default drift: restored={restored_defaults}, expected={expected_defaults}"
    )


def test_0007_downgrade_restores_0006_asset_references_indexes(db_at_0006, tmp_path):
    from sqlalchemy import create_engine, inspect

    cfg, db_path = db_at_0006

    reference_db = str(tmp_path / "reference_0006.db")
    reference_cfg = _make_config(reference_db)
    command.upgrade(reference_cfg, _BASELINE_0006)

    command.upgrade(cfg, "head")
    command.downgrade(cfg, _BASELINE_0006)

    def _asset_reference_indexes(path: str) -> set[tuple[str, tuple[str, ...]]]:
        engine = create_engine(f"sqlite:///{path}")
        try:
            return {
                (index["name"], tuple(index["column_names"]))
                for index in inspect(engine).get_indexes("asset_references")
            }
        finally:
            engine.dispose()

    restored = _asset_reference_indexes(db_path)
    expected = _asset_reference_indexes(reference_db)
    assert restored == expected, f"index drift: restored={restored}, expected={expected}"
