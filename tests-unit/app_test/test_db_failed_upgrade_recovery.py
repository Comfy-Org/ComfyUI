import glob
import logging
import os
import sqlite3
import sys
from contextlib import closing

import pytest

from app.database import db as db_module
from app_test import test_db_corruption_recovery as corruption_recovery
from app_test.test_db_corruption_recovery import _boot, _head, _make_db, _revision

# The boot fixtures of the corruption-recovery tests: the autouse ones reset boot state between tests.
boot_state = corruption_recovery.boot_state
boot_events = corruption_recovery.boot_events
startup_warnings = corruption_recovery.startup_warnings  # autouse: isolates the summary list
default_db = corruption_recovery.default_db
explicit_db = corruption_recovery.explicit_db


def _moved_aside(db_path):
    return sorted(p for p in glob.glob(db_path + ".failed-upgrade-*") if not p.endswith(("-wal", "-shm", "-journal")))


def _execute(db_path, *statements):
    with closing(sqlite3.connect(db_path)) as conn:
        for statement in statements:
            conn.execute(statement)
        conn.commit()


# Left by an upgrade killed part way through a migration, before it stamped its revision.
_INTERRUPTED_UPGRADES = {
    "0002 dropped its first index": (
        "0001_assets",
        ["DROP INDEX ix_asset_info_meta_key_val_bool"],
        "no such index: ix_asset_info_meta_key_val_bool",
    ),
    "0002 dropped all the old tables": (
        "0001_assets",
        [f"DROP TABLE {t}" for t in ("asset_info_meta", "asset_info_tags", "asset_cache_state", "assets_info")],
        "no such index: ix_asset_info_meta_key_val_bool",
    ),
    "0003 added its first column": (
        "0002_merge_to_asset_references",
        ["ALTER TABLE asset_references ADD COLUMN system_metadata JSON"],
        "duplicate column name: system_metadata",
    ),
    "0005 created tags_new": (
        "0004_drop_tag_type",
        ["CREATE TABLE tags_new (name VARCHAR(512) NOT NULL, CONSTRAINT pk_tags PRIMARY KEY (name))"],
        "table tags_new already exists",
    ),
    "0001 created its tables, no version stamp": (
        "0001_assets",
        ["DROP TABLE alembic_version"],
        "table assets already exists",
    ),
    "0007 created asset_system_state": (
        "0006_add_loader_path",
        ["CREATE TABLE asset_system_state (key VARCHAR(256) NOT NULL PRIMARY KEY, value TEXT NOT NULL)"],
        "table asset_system_state already exists",
    ),
}


@pytest.mark.parametrize(
    "revision, statements, error", _INTERRUPTED_UPGRADES.values(), ids=_INTERRUPTED_UPGRADES.keys()
)
def test_database_an_interrupted_upgrade_left_unupgradable_is_recreated(
    default_db, caplog, startup_warnings, revision, statements, error
):
    _make_db(default_db, revision=revision)
    _execute(default_db, *statements)
    old_file = os.stat(default_db).st_ino

    _boot()

    assert _revision(default_db) == _head()
    [moved] = _moved_aside(default_db)
    assert os.stat(moved).st_ino == old_file  # moved aside, not deleted
    # A handled failure is one warning naming the moved file: no ERROR, no traceback above DEBUG.
    shown = [r for r in caplog.records if r.levelno >= logging.INFO and r.name == "root"]
    [warning] = [r for r in shown if r.levelno >= logging.WARNING and "Database upgrade failed" in r.getMessage()]
    assert error in warning.getMessage() and moved in warning.getMessage()
    assert "[SQL:" not in warning.getMessage()  # the driver's message, not SQLAlchemy's wrapper
    assert not [r for r in shown if r.levelno >= logging.ERROR or r.exc_info]
    assert not any("Database upgrade failed" in w for w in startup_warnings)  # not repeated in the summary


def _failing_upgrade(monkeypatch):
    def fail(config, revision):
        raise sqlite3.OperationalError("upgrade failed")

    monkeypatch.setattr(db_module.command, "upgrade", fail)


def test_failed_upgrade_that_keeps_the_catalog_is_not_recreated(default_db, monkeypatch):
    _make_db(default_db, revision="0007_record_content_split")
    old_file = os.stat(default_db).st_ino
    _failing_upgrade(monkeypatch)

    with pytest.raises(SystemExit):
        _boot()

    assert _moved_aside(default_db) == []
    assert os.stat(default_db).st_ino == old_file
    assert _revision(default_db) == "0007_record_content_split"


def test_failed_upgrade_of_an_explicit_database_url_is_not_recreated(explicit_db, monkeypatch):
    _make_db(explicit_db, revision="0006_add_loader_path")
    _failing_upgrade(monkeypatch)

    with pytest.raises(SystemExit):
        _boot()

    assert _moved_aside(explicit_db) == []
    assert _revision(explicit_db) == "0006_add_loader_path"


@pytest.mark.parametrize("hold", [
    ["BEGIN IMMEDIATE"],  # others may still read (the pre-upgrade backup), not write
    ["BEGIN", "SELECT count(*) FROM sqlite_master"],  # a reader: the upgrade can't commit, the probe could
])
def test_database_locked_by_another_process_is_not_recreated(default_db, hold):
    _make_db(default_db, revision="0006_add_loader_path")
    holder = sqlite3.connect(default_db, isolation_level=None)
    for statement in hold:
        holder.execute(statement)
    try:
        with pytest.raises(SystemExit):
            _boot()
    finally:
        holder.close()

    assert _moved_aside(default_db) == []
    assert _revision(default_db) == "0006_add_loader_path"


def test_revision_from_a_newer_release_reports_the_upgrade_error(default_db):
    _make_db(default_db, revision="0006_add_loader_path")
    _execute(default_db, "UPDATE alembic_version SET version_num = '9999_from_a_newer_release'")

    with pytest.raises(Exception, match="Can't locate revision"):
        db_module.init_db()

    assert _moved_aside(default_db) == []


def test_failed_first_upgrade_of_a_new_database_is_not_moved_aside(default_db, monkeypatch, caplog):
    _failing_upgrade(monkeypatch)

    with pytest.raises(SystemExit):
        _boot()

    assert _moved_aside(default_db) == []
    assert "Database upgrade failed" not in caplog.text


def test_explicit_database_url_naming_the_default_file_is_recreated(default_db, monkeypatch):
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{default_db}")
    _make_db(default_db, revision="0001_assets")
    _execute(default_db, "DROP INDEX ix_asset_info_meta_key_val_bool")

    _boot()

    assert _revision(default_db) == _head()
    assert len(_moved_aside(default_db)) == 1


def _recreate_with_a_journal_left_behind(monkeypatch):
    real_recreate = db_module._recreate_after_failed_upgrade

    def recreate(db_path, error):
        open(db_path + "-journal", "wb").close()  # a sidecar another connection left
        real_recreate(db_path, error)

    monkeypatch.setattr(db_module, "_recreate_after_failed_upgrade", recreate)


def test_sidecars_move_before_the_database(default_db, monkeypatch):
    _make_db(default_db, revision="0001_assets")
    _execute(default_db, "DROP INDEX ix_asset_info_meta_key_val_bool")
    real_replace, moves = os.replace, []

    def recording_replace(src, dst):
        if dst.startswith(default_db + ".failed-upgrade-"):
            moves.append((src, dst))
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        _recreate_with_a_journal_left_behind(patch)
        patch.setattr(db_module.os, "replace", recording_replace)
        _boot()

    [moved] = _moved_aside(default_db)
    assert moves == [(default_db + "-journal", moved + "-journal"), (default_db, moved)]
    assert os.path.exists(moved + "-journal")


def test_failed_move_puts_every_file_back(default_db, monkeypatch):
    _make_db(default_db, revision="0001_assets")
    _execute(default_db, "DROP INDEX ix_asset_info_meta_key_val_bool")
    real_replace = os.replace

    def database_cannot_move(src, dst):
        if src == default_db:
            raise PermissionError("held open")
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        _recreate_with_a_journal_left_behind(patch)
        patch.setattr(db_module.os, "replace", database_cannot_move)
        with pytest.raises(SystemExit):
            _boot()

    assert os.path.exists(default_db + "-journal")
    assert glob.glob(default_db + ".failed-upgrade-*") == []
    assert _revision(default_db) == "0001_assets"


def test_unupgradable_database_held_by_another_process_is_not_moved(default_db, caplog):
    # The upgrade fails on the schema before it needs a lock, so its error says nothing about the holder.
    _make_db(default_db, revision="0001_assets")
    _execute(default_db, "DROP INDEX ix_asset_info_meta_key_val_bool")
    holder = sqlite3.connect(default_db)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SystemExit):
            _boot()
    finally:
        holder.close()

    assert "no such index: ix_asset_info_meta_key_val_bool" in caplog.text  # failed on the schema, not the lock
    assert glob.glob(default_db + ".failed-upgrade-*") == []
    assert _revision(default_db) == "0001_assets"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows refuses to rename an open file, so the move is undone")
def test_unupgradable_database_open_only_for_reading_is_moved(default_db):
    # Known limitation: only a writer stops the move. A reader keeps reading the moved file.
    _make_db(default_db, revision="0001_assets", marker="original")
    _execute(default_db, "DROP INDEX ix_asset_info_meta_key_val_bool")
    reader = sqlite3.connect(default_db, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM sqlite_master")
    try:
        _boot()
        assert reader.execute("SELECT value FROM marker").fetchone() == ("original",)
    finally:
        reader.close()

    assert _revision(default_db) == _head()
    [moved] = _moved_aside(default_db)
    with closing(sqlite3.connect(moved)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("original",)


def test_upgrade_failure_that_is_not_recovered_is_logged_as_an_error(explicit_db, monkeypatch, caplog):
    _make_db(explicit_db, revision="0001_assets")
    _execute(explicit_db, "DROP INDEX ix_asset_info_meta_key_val_bool")

    with pytest.raises(SystemExit):
        _boot()

    [error] = [r for r in caplog.records if r.levelno == logging.ERROR and "Error upgrading database" in r.getMessage()]
    assert error.exc_info
