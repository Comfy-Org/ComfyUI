
import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _user_tables(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'alembic%' AND name NOT LIKE 'sqlite%'"
            )
        }
    finally:
        conn.close()


@pytest.fixture
def db_at_head(tmp_path):
    db_path = str(tmp_path / "roundtrip.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, "head")
    yield cfg, db_path


def test_downgrade_to_base_empties_the_schema(db_at_head):
    cfg, db_path = db_at_head
    assert _user_tables(db_path), "fixture did not build a schema"

    command.downgrade(cfg, "base")

    assert _user_tables(db_path) == set()


def test_upgrade_to_head_again_after_downgrade_to_base(db_at_head):
    cfg, db_path = db_at_head
    expected = _user_tables(db_path)

    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    assert _user_tables(db_path) == expected
