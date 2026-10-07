"""An upgrade that fails part way must not leave a schema the next upgrade can't run on."""

import os
import sqlite3

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
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        )
        return sorted((kind, name, sorted(line.strip().rstrip(",") for line in (sql or "").splitlines()))
                      for kind, name, sql in rows)


def _version(db_path):
    with sqlite3.connect(db_path) as conn:
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


def _count_statements(tmp_path, monkeypatch, start, revision):
    cfg = _make_config(tmp_path / "count.db")
    if start is not None:
        command.upgrade(cfg, start)
    count = 0
    real_exec = DefaultImpl._exec

    def counting(self, *args, **kwargs):
        nonlocal count
        count += 1
        return real_exec(self, *args, **kwargs)

    monkeypatch.setattr(DefaultImpl, "_exec", counting)
    command.upgrade(cfg, revision)
    monkeypatch.undo()
    return count


@pytest.mark.parametrize("start, revision", _revisions_and_parents())
def test_migration_failing_at_its_last_statement_leaves_a_consistent_revision(
    tmp_path, monkeypatch, start, revision, fresh_head_schema
):
    last = _count_statements(tmp_path, monkeypatch, start, revision)
    db_path = tmp_path / "comfyui.db"
    cfg = _make_config(db_path)
    if start is not None:
        command.upgrade(cfg, start)
    before = _schema(db_path)

    # The last statement is the migration's version stamp, so all of its own statements have run.
    seen = 0
    real_exec = DefaultImpl._exec

    def fail_last(self, *args, **kwargs):
        nonlocal seen
        seen += 1
        if seen == last:
            raise RuntimeError("upgrade interrupted")
        return real_exec(self, *args, **kwargs)

    monkeypatch.setattr(DefaultImpl, "_exec", fail_last)
    with pytest.raises(RuntimeError, match="upgrade interrupted"):
        command.upgrade(cfg, "head")
    monkeypatch.undo()

    assert _version(db_path) == ([start] if start else [])
    assert _schema(db_path) == before
    command.upgrade(cfg, "head")
    assert _schema(db_path) == fresh_head_schema
