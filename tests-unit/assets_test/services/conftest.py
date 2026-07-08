import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.assets.database.models import Base


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override parent autouse fixture - service unit tests don't need server cleanup."""
    yield


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
def session(db_engine):
    """Session fixture for tests that need direct DB access."""
    with Session(db_engine) as sess:
        yield sess


@pytest.fixture
def mock_create_session(db_engine):
    """Patch create_session to use our in-memory database."""
    from contextlib import contextmanager
    from sqlalchemy.orm import Session as SASession

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    with patch("app.assets.services.ingest.create_session", _create_session), \
         patch("app.assets.services.asset_management.create_session", _create_session), \
         patch("app.assets.services.tagging.create_session", _create_session):
        yield _create_session


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
