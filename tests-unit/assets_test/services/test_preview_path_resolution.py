from sqlalchemy import event

from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.services.asset_management import get_preview_file_paths


def test_preview_paths_resolve_preview_record_content(session, mock_create_session) -> None:
    preview_content = create_content(session, "/output/preview.png")
    preview = create_record(session, preview_content.id, "preview.png")
    record_content = create_content(session, "/output/record.png")
    record = create_record(session, record_content.id, "record.png")
    record.preview_id = preview.id
    session.commit()

    paths = get_preview_file_paths([preview.id])

    assert paths == {preview.id: "/output/preview.png"}


def test_preview_paths_exclude_missing_preview_content(session, mock_create_session) -> None:
    preview_content = create_content(session, "/output/missing-preview.png")
    preview = create_record(session, preview_content.id, "missing-preview.png")
    record_content = create_content(session, "/output/record.png")
    record = create_record(session, record_content.id, "record.png")
    record.preview_id = preview.id
    mark_content_missing(session, preview_content.id)
    session.commit()

    paths = get_preview_file_paths([preview.id])

    assert paths == {}


def test_preview_paths_resolve_a_page_in_one_query(session, mock_create_session, db_engine) -> None:
    preview_ids: list[str] = []
    expected_paths: dict[str, str] = {}
    for index in range(3):
        preview_path = f"/output/preview-{index}.png"
        preview_content = create_content(session, preview_path)
        preview = create_record(session, preview_content.id, f"preview-{index}.png")
        record_content = create_content(session, f"/output/record-{index}.png")
        record = create_record(session, record_content.id, f"record-{index}.png")
        record.preview_id = preview.id
        preview_ids.append(preview.id)
        expected_paths[preview.id] = preview_path
    session.commit()

    statements: list[str] = []

    def count_statements(_, __, statement, ___, ____, _____) -> None:
        statements.append(statement)

    event.listen(db_engine, "before_cursor_execute", count_statements)
    try:
        paths = get_preview_file_paths(preview_ids)
    finally:
        event.remove(db_engine, "before_cursor_execute", count_statements)

    assert paths == expected_paths
    assert len(statements) == 1
