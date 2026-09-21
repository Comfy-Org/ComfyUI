import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, Session as SASession, sessionmaker

from app.assets import mode
from app.assets.database.models import Base
from app.assets.scanner import SeedAssetSpec, seed_asset_specs, stat_seed_specs
from app.assets.scanner_changes import PreparedRecovery, prepare_missing_content_recovery


def seed_with_recovery(
    session: Session, specs: list[SeedAssetSpec]
) -> tuple[int, list[str]]:
    """Seed the way insert_asset_specs does, minus its write transaction.

    Recovery reads the stat and the hash taken before the transaction opened, so a
    caller that hands seed_asset_specs bare specs gets no recovery at all.
    """
    stats = stat_seed_specs(specs)
    prepared: dict[str, PreparedRecovery | None] = {}
    if mode.hashing_enabled():
        for path, stat_result in stats.items():
            if stat_result is None:
                prepared[path] = None
                continue
            try:
                prepared[path] = prepare_missing_content_recovery(path, stat_result)
            except OSError:
                prepared[path] = None
    pending: list[str] = []
    created = seed_asset_specs(session, specs, stats, prepared, pending)
    return created, pending


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override parent autouse fixture - service unit tests don't need server cleanup."""
    yield


@pytest.fixture(autouse=True)
def initialised_hash_mode():
    class _HashingOff:
        enable_asset_hashing = False

    mode.init(_HashingOff())
    yield
    mode.init(None)


@pytest.fixture
def db_engine():
    """In-memory SQLite engine for fast unit tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_engine_fk():
    """In-memory SQLite engine with foreign key enforcement enabled."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(db_engine, monkeypatch):
    """Session fixture for tests that need direct DB access."""
    factory = sessionmaker(bind=db_engine)
    monkeypatch.setattr("app.database.db.Session", factory)
    monkeypatch.setattr("app.database.db.WriteSession", factory)
    with Session(db_engine) as sess:
        yield sess


@pytest.fixture
def mock_create_session(db_engine):
    """Patch create_session to use our in-memory database."""
    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    with patch("app.assets.services.ingest.create_session", _create_session), \
         patch("app.assets.services.asset_management.create_session", _create_session), \
         patch("app.assets.services.tagging.create_session", _create_session), \
         patch("app.database.db.WriteSession", sessionmaker(bind=db_engine)):
        yield _create_session


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def production_writer_database(tmp_path, monkeypatch):
    """Bind the real runtime engines so rollback and savepoint paths run under
    BEGIN IMMEDIATE semantics rather than the in-memory fixture's looser ones.
    """
    import app.database.db as db_mod

    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    for factory in (db_mod.Session, db_mod.WriteSession):
        if factory is not None:
            factory.kw["bind"].dispose()
    db_mod._db_lock.release(force=True)
