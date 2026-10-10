"""An upgrade that fails part way must not leave a schema the next upgrade can't run on."""

import os
import sqlite3
from contextlib import closing

import pytest
from alembic import command
from alembic.config import Config
from alembic.ddl.impl import DefaultImpl
from alembic.script import ScriptDirectory


def _make_config(db_path) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _schema(db_path):
    # Batch-mode table rebuilds emit constraints in no fixed order, so compare definition lines as a set.
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        )
        return sorted((kind, name, sorted(line.strip().rstrip(",") for line in (sql or "").splitlines()))
                      for kind, name, sql in rows)


def _version(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        try:
            return [row[0] for row in conn.execute("SELECT version_num FROM alembic_version")]
        except sqlite3.OperationalError:  # no version table: nothing was ever stamped
            return []


def _revisions_and_parents():
    script = ScriptDirectory.from_config(_make_config("unused.db"))
    return [(rev.down_revision, rev.revision) for rev in script.walk_revisions()]


@pytest.fixture(scope="module")
def fresh_head_schema(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("fresh") / "head.db"
    command.upgrade(_make_config(db_path), "head")
    return _schema(db_path)


@pytest.mark.parametrize("start, revision", _revisions_and_parents())
def test_migration_failing_at_its_last_statement_leaves_a_consistent_revision(
    tmp_path, monkeypatch, start, revision, fresh_head_schema
):
    db_path = tmp_path / "comfyui.db"
    cfg = _make_config(db_path)
    if start is not None:
        command.upgrade(cfg, start)
    before = _schema(db_path)

    # The version stamp is a migration's last statement, so all of its own statements have run.
    real_exec = DefaultImpl._exec

    def fail_at_version_stamp(self, construct, *args, **kwargs):
        if getattr(getattr(construct, "table", None), "name", None) == "alembic_version":
            raise RuntimeError("upgrade interrupted")
        return real_exec(self, construct, *args, **kwargs)

    monkeypatch.setattr(DefaultImpl, "_exec", fail_at_version_stamp)
    with pytest.raises(RuntimeError, match="upgrade interrupted"):
        command.upgrade(cfg, "head")
    monkeypatch.undo()

    assert _version(db_path) == ([start] if start else [])
    assert _schema(db_path) == before
    command.upgrade(cfg, "head")
    assert _schema(db_path) == fresh_head_schema


def test_ensure_version_keeps_the_version_table(tmp_path):
    db_path = tmp_path / "comfyui.db"
    command.ensure_version(_make_config(db_path))

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'alembic_version'").fetchone()
