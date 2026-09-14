from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, delete_record, get_record_by_id


def test_delete_removes_only_the_requested_delivery_record(session: Session) -> None:
    content = create_content(session, "/output/shared.png")
    first = create_record(session, content.id, "first.png")
    second = create_record(session, content.id, "second.png")

    delete_record(session, first.id)

    assert get_record_by_id(session, first.id) is None
    assert get_record_by_id(session, second.id) == second
