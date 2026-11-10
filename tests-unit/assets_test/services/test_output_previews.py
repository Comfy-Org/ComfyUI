"""An output registers already linked to the preview its save node named."""

from unittest.mock import patch

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.services import ingest
from app.assets.services.ingest import register_executed_output

from .preview_helpers import write_exr, write_preview, write_prores


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


def test_a_prores_output_registers_linked_to_its_proxy_made_before_the_write_lock(session, mock_create_session, roots):
    source = write_prores(roots / "output" / "clip.mov", 64, 48)
    calls: list[str] = []
    write_session, write_proxy = ingest.create_write_session, ingest.write_video_proxy

    def proxy(path):
        calls.append("proxy")
        return write_proxy(path)

    def locked():
        calls.append("write_session")
        return write_session()

    with patch.object(ingest, "write_video_proxy", proxy), patch.object(ingest, "create_write_session", locked):
        result = register_executed_output(str(source), "job")

    assert calls == ["proxy", "write_session"]
    preview = session.get(Asset, result.preview_id)
    assert preview.name.endswith(".mp4") and preview.mime_type == "video/mp4"
    assert [t.name for t in preview.tags] == ["preview"]
    assert preview.system_metadata == {"kind": "video", "width": 64, "height": 48}


def test_a_video_browsers_play_gets_no_proxy(session, mock_create_session, roots):
    import av

    path = roots / "output" / "clip.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        frame = av.VideoFrame(64, 48, "yuv420p")
        frame.pts = 0
        out.mux(stream.encode(frame))
        out.mux(stream.encode(None))

    with patch.object(ingest, "write_video_proxy") as proxy:
        result = register_executed_output(str(path), "job")

    assert not proxy.called and result.preview_id is None
