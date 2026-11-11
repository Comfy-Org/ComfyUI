from sqlalchemy.orm import Session

from app.assets.database.queries import (
    create_content,
    create_record,
    fetch_record_tags,
    mark_content_missing,
    unset_content_missing,
)


def test_missing_state_lives_on_content_and_projects_to_records(session: Session) -> None:
    content = create_content(session, "/models/model.safetensors")
    first = create_record(session, content.id, "model-a")
    second = create_record(session, content.id, "model-b")

    mark_content_missing(session, content.id)

    assert content.is_missing is True
    assert fetch_record_tags(session, first.id) == ["missing"]
    assert fetch_record_tags(session, second.id) == ["missing"]


def test_recovery_clears_the_automatic_missing_marker(session: Session) -> None:
    content = create_content(session, "/models/model.safetensors")
    record = create_record(session, content.id, "model")
    mark_content_missing(session, content.id)

    unset_content_missing(session, content.id)

    assert content.is_missing is False
    assert fetch_record_tags(session, record.id) == []
