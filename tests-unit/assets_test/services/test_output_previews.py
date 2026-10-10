"""An output registers already linked to the preview its save node named."""

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.services.ingest import register_executed_output

from .preview_helpers import write_exr, write_preview


def test_the_output_row_is_never_committed_without_its_preview(session, mock_create_session, roots):
    ref = write_preview(roots)
    committed: list = []

    def before_commit(s):
        if s.in_nested_transaction():
            return  # a savepoint, not the commit that makes the row visible
        s.flush()
        committed.extend(a.preview_id for a in s.identity_map.values() if isinstance(a, Asset) and a.name == "frame.exr")

    event.listen(Session, "before_commit", before_commit)
    try:
        result = register_executed_output(str(write_exr(roots / "output" / "frame.exr", 8, 8)), "job", ref)
    finally:
        event.remove(Session, "before_commit", before_commit)

    assert committed and None not in committed
    preview = session.get(Asset, result.preview_id)
    assert preview.name == ref["filename"] and preview.mime_type == "image/jpeg"
    assert [t.name for t in preview.tags] == ["preview"]
    assert preview.system_metadata == {"kind": "image", "width": 4, "height": 3}
    assert session.get(AssetContent, preview.content_id).hash == f"blake3:{ref['filename'][:64]}"
