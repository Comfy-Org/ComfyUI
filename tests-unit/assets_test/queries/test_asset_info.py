from sqlalchemy.orm import Session

from app.assets.database.queries import (
    create_content,
    create_record,
    delete_record,
    get_record_by_id,
    rename_record,
)


def test_record_operations_do_not_mutate_content(session: Session) -> None:
    content = create_content(session, "/output/result.png", hash="digest", size_bytes=3)
    record = create_record(session, content.id, "before.png", mime_type="image/png")

    renamed = rename_record(session, record.id, "after.png")

    assert renamed.content_id == content.id
    assert renamed.name == "after.png"
    assert content.hash == "digest"


def test_delete_record_is_hard_delete(session: Session) -> None:
    content = create_content(session, "/output/result.png")
    record = create_record(session, content.id, "result.png")

    delete_record(session, record.id)

    assert get_record_by_id(session, record.id) is None
