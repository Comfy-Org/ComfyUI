from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record


def test_bulk_discovery_creates_a_record_for_each_path(session: Session) -> None:
    discovered = [
        ("/models/a/model.safetensors", "a/model.safetensors"),
        ("/models/b/model.safetensors", "b/model.safetensors"),
    ]

    records = [
        create_record(session, create_content(session, path).id, name, loader_path=name)
        for path, name in discovered
    ]

    assert [record.loader_path for record in records] == [name for _, name in discovered]
    assert records[0].content_id != records[1].content_id
