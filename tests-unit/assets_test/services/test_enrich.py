from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record


def test_unenriched_content_is_represented_by_a_null_hash(session: Session) -> None:
    content = create_content(session, "/models/pending.safetensors", hash=None)
    record = create_record(session, content.id, "pending.safetensors")

    assert record.content_id == content.id
    assert content.hash is None


def test_enrichment_does_not_merge_equal_hash_content_rows(session: Session) -> None:
    first = create_content(session, "/models/one.safetensors", hash="digest")
    second = create_content(session, "/models/two.safetensors", hash="digest")

    assert first.id != second.id
