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
        return [row[0] for row in conn.execute("SELECT version_num FROM alembic_version")]


def _revisions_with_a_parent():
    script = ScriptDirectory.from_config(_make_config("unused.db"))
    return [(rev.down_revision, rev.revision) for rev in script.walk_revisions() if rev.down_revision]


def _fresh_schema(tmp_path, revision):
    db_path = tmp_path / f"fresh-{revision}.db"
    command.upgrade(_make_config(db_path), revision)
    return _schema(db_path)


def _count_statements(tmp_path, monkeypatch, start, revision):
    cfg = _make_config(tmp_path / "count.db")
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


@pytest.mark.parametrize("start, revision", _revisions_with_a_parent())
def test_migration_failing_at_its_last_statement_leaves_a_consistent_revision(
    tmp_path, monkeypatch, start, revision
):
    last = _count_statements(tmp_path, monkeypatch, start, revision)
    db_path = tmp_path / "comfyui.db"
    cfg = _make_config(db_path)
    command.upgrade(cfg, start)

    # Every earlier statement of the migration has run when its last one fails.
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

    assert _version(db_path) == [start]
    assert _schema(db_path) == _fresh_schema(tmp_path, start)
    command.upgrade(cfg, "head")
    assert _schema(db_path) == _fresh_schema(tmp_path, "head")
