from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record, list_records_page
from app.assets.database.queries.records import RecordPageSpec


def test_tag_selection_counts_records_not_content_rows(session: Session) -> None:
    content = create_content(session, "/models/shared.safetensors")
    create_record(session, content.id, "one", tags=["model"])
    create_record(session, content.id, "two", tags=["model"])

    records, _, _ = list_records_page(
        session,
        RecordPageSpec(all_tags=("model",)),
    )

    assert len(records) == 2
