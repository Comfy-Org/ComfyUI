import pytest
from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, list_records_page
from app.assets.database.queries.records import (
    RecordCursorBoundary,
    RecordPageSpec,
    RecordSortOrder,
)


@pytest.mark.parametrize(
    ("order", "expected_indexes"),
    [("asc", (0, 1, 2)), ("desc", (2, 1, 0))],
)
def test_record_keyset_cursor_pages_in_creation_order(
    session: Session, order: RecordSortOrder, expected_indexes: tuple[int, int, int]
) -> None:
    records = [
        create_record(session, create_content(session, f"/output/{name}").id, name)
        for name in ("one.png", "two.png", "three.png")
    ]
    shared_created_at = records[0].created_at
    for index, record in enumerate(records, start=1):
        record.id = f"00000000-0000-0000-0000-{index:012d}"
        record.created_at = shared_created_at
    session.flush()

    first_page, _, _ = list_records_page(
        session,
        RecordPageSpec(limit=2, order=order),
    )
    boundary_record = first_page[-1]
    second_page, _, _ = list_records_page(
        session,
        RecordPageSpec(
            limit=2,
            order=order,
            after=RecordCursorBoundary(
                value=boundary_record.created_at,
                id=boundary_record.id,
            ),
        ),
    )

    assert [record.id for record in first_page + second_page] == [
        records[index].id for index in expected_indexes
    ]
