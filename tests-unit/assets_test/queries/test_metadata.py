from sqlalchemy.orm import Session

from app.assets.database.queries import create_content, create_record


def test_interpretation_metadata_is_record_specific(session: Session) -> None:
    content = create_content(session, "/output/shared-bytes.bin", hash="digest")
    image = create_record(session, content.id, "image", mime_type="image/png")
    text = create_record(session, content.id, "text", mime_type="text/plain")
    image.system_metadata = {"width": 8}
    text.system_metadata = {"encoding": "utf-8"}

    assert image.content_id == text.content_id
    assert image.system_metadata == {"width": 8}
    assert text.system_metadata == {"encoding": "utf-8"}
