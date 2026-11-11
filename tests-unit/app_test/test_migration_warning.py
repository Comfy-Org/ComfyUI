import os

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

import app.database.db as db_module

_BASELINE_0006 = "0006_add_loader_path"
_BASELINE_0002 = "0002_merge_to_asset_references"
_DESTRUCTIVE = "0007_record_content_split"

_DISCARDED_CLASSES = (
    "manual tags",
    "user metadata",
    "previews",
    "renames",
    "API-created records",
    "job_id",
)


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


@pytest.fixture
def captured_warnings(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(db_module, "log_startup_warning", warnings.append)
    return warnings


@pytest.fixture
def db_url_for(monkeypatch):
    def _set(db_path: str) -> str:
        url = f"sqlite:///{db_path}"
        monkeypatch.setattr(db_module.args, "database_url", url)
        return url

    return _set


def _migrate(db_path: str, db_url: str, db_exists: bool) -> None:
    db_module._migrate_and_bind(db_url, db_path, db_exists)


def test_upgrade_across_the_destructive_revision_warns_and_names_the_backup(
    tmp_path, captured_warnings, db_url_for
):
    db_path = str(tmp_path / "existing.db")
    command.upgrade(_make_config(db_path), _BASELINE_0006)
    db_url = db_url_for(db_path)
    assert os.path.exists(db_path)

    _migrate(db_path, db_url, db_exists=True)

    assert len(captured_warnings) == 1, (
        "the destructive rebuild is one startup event, not one warning per traversed revision"
    )
    message = captured_warnings[0]
    assert db_path + ".bkp" in message, "the warning must name the backup it left behind"
    for discarded in _DISCARDED_CLASSES:
        assert discarded in message, (
            f"the warning must inventory what the rebuild discarded; {discarded!r} is missing"
        )


def test_fresh_install_warns_nothing(tmp_path, captured_warnings, db_url_for):
    db_path = str(tmp_path / "fresh.db")
    db_url = db_url_for(db_path)
    assert not os.path.exists(db_path)

    _migrate(db_path, db_url, db_exists=False)

    assert captured_warnings == [], (
        "a fresh install discarded nothing and took no backup, so it has nothing to warn about"
    )


def test_upgrade_that_does_not_cross_the_destructive_revision_warns_nothing(
    tmp_path, captured_warnings, db_url_for, monkeypatch
):
    db_path = str(tmp_path / "partial.db")
    command.upgrade(_make_config(db_path), _BASELINE_0002)
    db_url = db_url_for(db_path)
    monkeypatch.setattr(ScriptDirectory, "get_current_head", lambda _self: _BASELINE_0006)

    _migrate(db_path, db_url, db_exists=True)

    assert captured_warnings == [], (
        "a multi-revision upgrade that never traverses the rebuild keeps every row it had"
    )


def test_database_already_at_head_warns_nothing(tmp_path, captured_warnings, db_url_for):
    db_path = str(tmp_path / "current.db")
    command.upgrade(_make_config(db_path), "head")
    db_url = db_url_for(db_path)

    _migrate(db_path, db_url, db_exists=True)

    assert captured_warnings == [], (
        "an up-to-date database migrates nothing, so it must stay silent on every boot"
    )
