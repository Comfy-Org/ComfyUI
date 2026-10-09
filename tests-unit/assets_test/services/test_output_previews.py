"""An output registers already linked to the preview its save node named."""

import builtins
from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.services import ingest
from app.assets.services.asset_management import delete_asset_reference
from app.assets.services.ingest import register_cached_output, register_executed_output

from .preview_helpers import write_exr, write_preview


@pytest.fixture
def db_engine(db_engine_fk):
    return db_engine_fk


def _output(roots, name="frame.exr"):
    return str(write_exr(roots / "output" / name, 8, 8))


def _previews(session) -> list[Asset]:
    session.expire_all()
    return [a for a in session.query(Asset).all() if a.name.endswith((".jpg", ".webp"))]


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
        result = register_executed_output(_output(roots), "job", ref)
    finally:
        event.remove(Session, "before_commit", before_commit)

    assert committed and None not in committed
    preview = session.get(Asset, result.preview_id)
    assert preview.name == ref["filename"] and preview.mime_type == "image/jpeg"
    assert [t.name for t in preview.tags] == ["preview"]
    assert preview.system_metadata == {"kind": "image", "width": 4, "height": 3}
    assert session.get(AssetContent, preview.content_id).hash == f"blake3:{ref['filename'][:64]}"


def _bad_name(roots, ref):
    (roots / "previews" / "not-a-hash.jpg").write_bytes(b"jpg")
    return {**ref, "filename": "not-a-hash.jpg"}


def _escaping(roots, ref):
    (roots / ref["filename"]).write_bytes(b"jpg")
    return {**ref, "filename": f"../{ref['filename']}"}


def test_a_webp_preview_registers_as_webp(session, mock_create_session, roots):
    ref = write_preview(roots, alpha=True)

    preview = session.get(Asset, register_executed_output(_output(roots), "job", ref).preview_id)

    assert preview.name == ref["filename"] and preview.mime_type == "image/webp"


@pytest.mark.parametrize(
    "bad",
    [
        _bad_name,
        _escaping,
        lambda roots, ref: {**ref, "filename": f"{'0' * 64}.jpg"},  # well named, but no such file
        lambda roots, ref: {**ref, "width": "wide"},
        lambda roots, ref: {"name": ref["filename"]},
    ],
)
def test_a_bad_preview_ref_registers_the_output_without_one(session, mock_create_session, roots, bad):
    result = register_executed_output(_output(roots), "job", bad(roots, write_preview(roots)))

    assert result is not None and result.preview_id is None
    assert _previews(session) == []


def test_a_failed_link_registers_the_output_without_one(session, mock_create_session, roots):
    real = ingest.create_record

    def fail_for_previews(session, *a, tags=None, **k):
        if tags == ["preview"]:
            raise RuntimeError("boom")
        return real(session, *a, tags=tags, **k)

    with patch.object(ingest, "create_record", fail_for_previews):
        result = register_executed_output(_output(roots), "job", write_preview(roots))

    assert result is not None and result.preview_id is None
    assert session.get(Asset, result.id) is not None
    assert session.query(AssetContent).filter(AssetContent.path.startswith(str(roots / "previews"))).count() == 0


def test_the_preview_file_is_never_read(session, mock_create_session, roots):
    ref = write_preview(roots)
    real_open = builtins.open
    opened: list = []

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    with patch("builtins.open", spy):
        assert register_executed_output(_output(roots), "job", ref).preview_id is not None

    assert not [path for path in opened if ref["filename"] in path]


def test_identical_previews_share_one_file_content_and_record(session, mock_create_session, roots):
    ref = write_preview(roots)
    first = register_executed_output(_output(roots, "a.exr"), "job", ref)
    second = register_executed_output(_output(roots, "b.exr"), "job", ref)

    assert first.preview_id == second.preview_id is not None
    assert len(_previews(session)) == 1
    assert session.query(AssetContent).filter(AssetContent.path.startswith(str(roots / "previews"))).count() == 1

    delete_asset_reference(first.id)
    assert (roots / "previews" / ref["filename"]).exists(), "the other output still uses it"
    delete_asset_reference(second.id)
    assert not (roots / "previews" / ref["filename"]).exists()
    assert _previews(session) == []


def test_a_cached_rerun_reuses_the_preview(session, mock_create_session, roots):
    path = _output(roots)
    executed = register_executed_output(path, "job-1", write_preview(roots))

    cached = register_cached_output(path, "job-2")

    assert cached.preview_id == executed.preview_id
    assert len(_previews(session)) == 1


def test_a_preview_removed_with_its_last_user_is_rewritten_and_linked_again(session, mock_create_session, roots):
    first = register_executed_output(_output(roots, "a.exr"), "job", write_preview(roots))
    delete_asset_reference(first.id)

    again = register_executed_output(_output(roots, "b.exr"), "job", write_preview(roots))

    assert again.preview_id is not None and again.preview_id != first.preview_id
    assert session.get(Asset, again.preview_id) is not None


def test_a_preview_deleted_after_the_probe_is_not_linked(session, mock_create_session, roots):
    ref = write_preview(roots)
    probe = ingest._probe_output_preview

    def probe_then_delete(preview_ref):
        probed = probe(preview_ref)
        (roots / "previews" / ref["filename"]).unlink()  # a delete of its last user, between probe and link
        return probed

    with patch.object(ingest, "_probe_output_preview", probe_then_delete):
        result = register_executed_output(_output(roots), "job", ref)

    assert result is not None and result.preview_id is None, "never a link to a missing file"
