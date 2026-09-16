import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session as SASession, sessionmaker

import app.database.db as db_mod
from app.assets import lifecycle
from app.assets.database.models import Base
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import clear_transition_queue, write_stored_mode


@pytest.fixture(autouse=True)
def clear_lifecycle_transition_state():
    clear_transition_queue()
    lifecycle._hash_mode_transition = None
    yield
    clear_transition_queue()
    lifecycle._hash_mode_transition = None


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


@pytest.fixture
def mock_create_session(session):
    engine = session.bind

    @contextmanager
    def create_test_session():
        with SASession(engine) as database_session:
            yield database_session

    with (
        patch("app.assets.lifecycle.create_session", create_test_session),
        patch("app.database.db.create_session", create_test_session),
        patch("app.database.db.WriteSession", sessionmaker(bind=engine)),
    ):
        yield create_test_session


def test_transition_intent_keeps_global_unchanged_until_retry_succeeds(
    session, mock_create_session, monkeypatch
):
    write_stored_mode(session, "on")
    session.commit()
    lifecycle._hash_mode_transition = "off_to_on"
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: False)
    real_run_write_txn = db_mod.run_write_txn
    attempts = 0

    def retry_after_first_attempt(work):
        def flaky_work(writer_session):
            nonlocal attempts

            transition = work(writer_session)
            attempts += 1
            if attempts == 1:
                assert lifecycle._hash_mode_transition == "off_to_on"
                raise OperationalError(
                    "UPDATE", {}, sqlite3.OperationalError("database is locked")
                )
            return transition

        return real_run_write_txn(flaky_work)

    monkeypatch.setattr(lifecycle, "run_write_txn", retry_after_first_attempt)
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)

    lifecycle.record_hash_mode_transition_intent()

    assert attempts == 2
    assert lifecycle._hash_mode_transition == "on_to_off"
    session.expire_all()
    assert hash_mode_state.read_stored_mode(session) == "off"


def test_transition_intent_terminal_failure_preserves_global(
    mock_create_session, monkeypatch
):
    lifecycle._hash_mode_transition = "off_to_on"

    def fail_write(_work):
        raise OperationalError("UPDATE", {}, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(lifecycle, "run_write_txn", fail_write)

    with pytest.raises(OperationalError, match="database is locked"):
        lifecycle.record_hash_mode_transition_intent()

    assert lifecycle._hash_mode_transition == "off_to_on"
