"""Test that Alembic migrations run cleanly on a file-backed SQLite DB.

This catches problems like unnamed FK constraints that prevent batch-mode
drop_constraint from working on real SQLite files (see MB-2).

Migrations 0001 and 0002 are already shipped, so we only exercise
upgrade/downgrade for 0003+.
"""

import os
import sqlite3

import pytest
from alembic import command
from alembic.config import Config


# Oldest shipped revision — we upgrade to here as a baseline and never
# downgrade past it.
_BASELINE = "0002_merge_to_asset_references"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    config_path = os.path.abspath(os.path.join(root, "alembic.ini"))
    scripts_path = os.path.abspath(os.path.join(root, "alembic_db"))

    cfg = Config(config_path)
    cfg.set_main_option("script_location", scripts_path)
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _sqlite_path(cfg: Config) -> str:
    url = cfg.get_main_option("sqlalchemy.url")
    assert url is not None and url.startswith("sqlite:///")
    return url.removeprefix("sqlite:///")


@pytest.fixture
def migration_db(tmp_path):
    """Yield an alembic Config pre-upgraded to the baseline revision."""
    db_path = str(tmp_path / "test_migration.db")
    cfg = _make_config(db_path)
    command.upgrade(cfg, _BASELINE)
    yield cfg


def test_upgrade_to_head(migration_db):
    """Upgrade from baseline to head must succeed on a file-backed DB."""
    command.upgrade(migration_db, "head")


def test_downgrade_to_baseline(migration_db):
    """Upgrade to head then downgrade back to baseline."""
    command.upgrade(migration_db, "head")
    command.downgrade(migration_db, _BASELINE)


def test_upgrade_downgrade_cycle(migration_db):
    """Full cycle: upgrade → downgrade → upgrade again."""
    command.upgrade(migration_db, "head")
    command.downgrade(migration_db, _BASELINE)
    command.upgrade(migration_db, "head")


def test_case_sensitive_tags_downgrade_normalizes_existing_tags(migration_db):
    """Downgrading 0005 folds mixed-case tag vocabulary before restoring CHECK."""
    command.upgrade(migration_db, "0005_allow_case_sensitive_tags")

    db_path = _sqlite_path(migration_db)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO tags(name) VALUES (?)", ("NewTag",))
        conn.execute("INSERT INTO tags(name) VALUES (?)", ("newtag",))
        conn.execute("INSERT INTO tags(name) VALUES (?)", ("model_type:LLM",))

    command.downgrade(migration_db, "0004_drop_tag_type")

    with sqlite3.connect(db_path) as conn:
        tags = {row[0] for row in conn.execute("SELECT name FROM tags")}
        assert "newtag" in tags
        assert "model_type:llm" in tags
        assert "NewTag" not in tags
        assert "model_type:LLM" not in tags

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO tags(name) VALUES (?)", ("Upper",))
