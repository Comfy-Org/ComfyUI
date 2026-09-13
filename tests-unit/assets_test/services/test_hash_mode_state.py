from unittest.mock import patch

from app.assets.database.models import AssetContent
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    drain_transition_queue,
    enqueue_transition_work,
    pending_transition_count,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)


def test_absent_row_off_mode_no_transition(session):
    with patch("app.assets.services.hash_mode_state._mode.hashing_enabled", return_value=False):
        assert record_transition_intent(session) is None
    assert read_stored_mode(session) == "off"


def test_empty_drain_keeps_off_mode_ready_for_a_later_on_transition(session):
    write_stored_mode(session, "off")

    drain_transition_queue(session)

    with patch("app.assets.services.hash_mode_state._mode.hashing_enabled", return_value=True):
        assert record_transition_intent(session) == "off_to_on"


def test_off_to_on_enqueues_null_rows(session):
    session.add(AssetContent(path="/tmp/null", hash=None))
    session.add(AssetContent(path="/tmp/hashed", hash="abc"))
    write_stored_mode(session, "off")
    with patch("app.assets.services.hash_mode_state._mode.hashing_enabled", return_value=True):
        transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    assert transition == "off_to_on"
    assert pending_transition_count() == 2
    assert read_stored_mode(session) == "off"
    clear_transition_queue()


def test_on_to_off_freezes(session):
    write_stored_mode(session, "on")
    with patch("app.assets.services.hash_mode_state._mode.hashing_enabled", return_value=False):
        transition = record_transition_intent(session)
    assert transition == "on_to_off"
    assert read_stored_mode(session) == "off"
    assert pending_transition_count() == 0


def test_mode_stays_off_during_transition(session):
    session.add(AssetContent(path="/tmp/null", hash=None))
    write_stored_mode(session, "off")
    with patch("app.assets.services.hash_mode_state._mode.hashing_enabled", return_value=True):
        transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    assert read_stored_mode(session) == "off"
