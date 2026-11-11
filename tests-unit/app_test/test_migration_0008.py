import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

_BASELINE_0007 = "0007_record_content_split"
_REVISION_0008 = "0008_drop_asset_meta"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _has_asset_meta(db_path: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_meta'"
        ).fetchone() is not None


def _asset_meta_shape(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        columns = {
            (column["name"], str(column["type"]))
            for column in inspector.get_columns("asset_meta")
        }
        indexes = {
            (index["name"], tuple(index["column_names"]))
            for index in inspector.get_indexes("asset_meta")
        }
        return columns, indexes
    finally:
        engine.dispose()


@pytest.fixture
def db_at_0007(tmp_path):
    db_path = str(tmp_path / "test.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _BASELINE_0007)
    yield cfg, db_path


def test_0008_upgrade_drops_asset_meta(db_at_0007):
    cfg, db_path = db_at_0007
    assert _has_asset_meta(db_path), "0007 creates asset_meta, so 0008 has something to drop"

    command.upgrade(cfg, _REVISION_0008)

    assert not _has_asset_meta(db_path)


def test_0008_upgrade_tolerates_a_database_without_asset_meta(db_at_0007):
    cfg, db_path = db_at_0007
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE asset_meta")
        conn.commit()

    command.upgrade(cfg, _REVISION_0008)

    assert not _has_asset_meta(db_path)


def test_0008_downgrade_restores_the_0007_asset_meta(db_at_0007, tmp_path):
    cfg, db_path = db_at_0007

    reference_db = str(tmp_path / "reference_0007.db")
    command.upgrade(_make_config(reference_db), _BASELINE_0007)

    command.upgrade(cfg, _REVISION_0008)
    command.downgrade(cfg, _BASELINE_0007)

    restored_columns, restored_indexes = _asset_meta_shape(db_path)
    expected_columns, expected_indexes = _asset_meta_shape(reference_db)

    assert restored_columns == expected_columns, (
        f"column drift: restored={restored_columns}, expected={expected_columns}"
    )
    assert restored_indexes == expected_indexes, (
        f"index drift: restored={restored_indexes}, expected={expected_indexes}"
    )
    assert len(restored_indexes) == 4, f"0007 creates four asset_meta indexes, got {restored_indexes}"


def test_head_has_no_asset_meta(tmp_path):
    db_path = str(tmp_path / "head.db")

    command.upgrade(_make_config(db_path), "head")

    assert not _has_asset_meta(db_path)
