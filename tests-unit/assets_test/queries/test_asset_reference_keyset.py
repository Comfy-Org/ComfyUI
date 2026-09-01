from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, list_records_page
from app.assets.database.queries.records import RecordCursorBoundary, RecordPageSpec


def test_record_keyset_cursor_pages_in_creation_order(session: Session) -> None:
    records = [
        create_record(session, create_content(session, f"/output/{name}").id, name)
        for name in ("one.png", "two.png", "three.png")
    ]
    for index, record in enumerate(records, start=1):
        record.id = f"00000000-0000-0000-0000-{index:012d}"
    session.flush()

    first_page, _, _ = list_records_page(
        session,
        RecordPageSpec(limit=2, order="asc"),
    )
    boundary_record = first_page[-1]
    second_page, _, _ = list_records_page(
        session,
        RecordPageSpec(
            limit=2,
            order="asc",
            after=RecordCursorBoundary(
                value=boundary_record.created_at,
                id=boundary_record.id,
            ),
        ),
    )

    assert [record.id for record in first_page + second_page] == [record.id for record in records]
