import glob
import os
import sqlite3
from contextlib import closing

import pytest

from app.database import db as db_module
from app_test import test_db_corruption_recovery as corruption_recovery
from app_test.test_db_corruption_recovery import _boot, _head, _make_db, _revision

# The boot fixtures of the corruption-recovery tests: the autouse ones reset boot state between tests.
boot_state = corruption_recovery.boot_state
boot_events = corruption_recovery.boot_events
startup_warnings = corruption_recovery.startup_warnings
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
    "0002 dropped its first index": ("0001_assets", ["DROP INDEX ix_asset_info_meta_key_val_bool"]),
    "0002 dropped the old tables": (
        "0001_assets",
        [f"DROP TABLE {t}" for t in ("asset_info_meta", "asset_info_tags", "asset_cache_state", "assets_info")],
    ),
    "0003 added its first column": (
        "0002_merge_to_asset_references",
        ["ALTER TABLE asset_references ADD COLUMN system_metadata JSON"],
    ),
}


@pytest.mark.parametrize("revision, statements", _INTERRUPTED_UPGRADES.values(), ids=_INTERRUPTED_UPGRADES.keys())
def test_database_an_interrupted_upgrade_left_unupgradable_is_recreated(
    default_db, startup_warnings, revision, statements
):
    _make_db(default_db, revision=revision)
    _execute(default_db, *statements)
    old_file = os.stat(default_db).st_ino

    _boot()

    assert _revision(default_db) == _head()
    [moved] = _moved_aside(default_db)
    assert os.stat(moved).st_ino == old_file  # moved aside, not deleted
    assert any("Database upgrade failed" in w and moved in w for w in startup_warnings)


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


def test_failed_upgrade_of_an_explicit_database_url_is_not_recreated(explicit_db, monkeypatch):
    _make_db(explicit_db, revision="0006_add_loader_path")
    _failing_upgrade(monkeypatch)

    with pytest.raises(SystemExit):
        _boot()

    assert _moved_aside(explicit_db) == []
